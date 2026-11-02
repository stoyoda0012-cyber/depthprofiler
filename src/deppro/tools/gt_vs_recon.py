"""真値と再構成を並べる 2x2 図（GT vs reconstruction）。

なぜ要るか
----------
再構成の 3D 等値面を単独で眺めても良し悪しは読めない。真値（合成データ）
と**同じ視点・同じクロップ・同じ配色**で横に並べ、その下に (x, y) 平均の
深さプロファイルを添えて初めて、層位置のずれと濃度のずれが見分けられる。
その並べ方を毎回書き直さないための薄い合成層で、描画そのものは
:mod:`deppro.visualize_4d`（``MatlabIsosurfaceRenderer``）、視点・クロップの
規約は :mod:`deppro.tools.triptych` のものをそのまま使う。

配置（1 フレーム 1 図、複数フレームなら mp4）::

    +---------------------------+---------------------------+
    |  3D ground truth          |  3D reconstruction        |
    +---------------------------+---------------------------+
    |  GT の (x,y) 平均プロファイル |  再構成の (x,y) 平均プロファイル |
    |  conc (%) vs depth (Å)    |  同じ元素色               |
    +---------------------------+---------------------------+
    見出し: <title> — frame t/nt | RMSE x.xx % | W1 median-px x.x Å

指標は公開済みの定義だけを使う: RMSE[%] は
:func:`deppro.tools.triptych.rmse_percent`（全画素・全深さ・全元素）、
W1[Å] は :func:`deppro.depth_profile.wasserstein_distance_1d` を画素×元素ごとに
取り、真値が空の画素は NaN、画素方向の中央値を元素で平均したもの。
膜厚線は描かない（膜厚推定は配布物に含まれない ``benchmarks`` 側にある）。

入力
----
真値は ``/Dstk`` を持つ H5（MATLAB 生成、または
:class:`deppro.simulate.writer.H5Writer` の出力）、再構成は
:func:`deppro.app.write_depth_estimate` が書く ``/D_est/data`` の H5。
:func:`load_profile` はどちらの dialect かをファイルから判別する。両者は
grid（nx, ny, nt）・深さ軸・元素名（順序まで）が一致していなければならず、
違えば :func:`check_compatible` が何が違うかを挙げて拒否する。

使い方
------
ライブラリとして::

    from deppro.tools.gt_vs_recon import load_profile, render_gt_vs_recon
    gt = load_profile("gt.h5")
    recon = load_profile("depth_est_NN_20260101_000000.h5")
    render_gt_vs_recon(gt, recon, output="gt_vs_nn.mp4")

コマンドラインから::

    python -m deppro.tools.gt_vs_recon --gt gt.h5 --recon depth_est_NN.h5 \\
        --label "NN surrogate" --title "Checkered 3x2" -o gt_vs_nn.mp4
    python -m deppro.tools.gt_vs_recon --gt gt.h5 --recon est.h5 --frame 0   # PNG 1 枚

出力は絶対パスならその場所、相対パスなら
``deppro.paths.ensure_outputs_dir("gt_vs_recon")`` の下。描画には pyvista と
imageio（動画は imageio-ffmpeg）が要り、無ければ使う時点で ImportError。
指標の計算と入力の検査は描画ライブラリ無しで動く。

ヘッドレスのソフトウェア OpenGL（Xvfb + llvmpipe）では、レンダラの depth
peeling が面をすべて落として真っ白な図になる。その環境では
``--no-depth-peeling``（``GtVsReconConfig(depth_peeling=False)``）を付ける。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from deppro.depth_profile import wasserstein_distance_1d
from deppro.paths import ensure_outputs_dir
from deppro.tools.triptych import (
    TriptychConfig,
    _common_crop,
    _make_renderer_config,
    _shoot,
    rmse_percent,
)
from deppro.visualize_4d import (
    HAS_PYVISTA,
    DepthProfile4DData,
    MatlabIsosurfaceRenderer,
    load_depth_estimate,
    load_gt_4d,
)

__all__ = [
    "GtVsReconConfig",
    "load_profile",
    "check_compatible",
    "mean_profile",
    "frame_metrics",
    "render_gt_vs_recon",
    "build_parser",
    "main",
]


@dataclass
class GtVsReconConfig(TriptychConfig):
    """2x2 図の見た目。3D パネルの既定は triptych（EzDeP4D の本番動画）と同じ。"""

    gt_label: str = "Ground truth"
    recon_label: str = "Reconstruction"
    profile_row: bool = True       # 下段のプロファイルを描く
    show_metrics: bool = True      # 見出しに RMSE / W1 を出す
    profile_row_ratio: float = 0.55  # 下段の高さ（3D パネル高さに対する比）
    fig_width_in: float = 8.4      # 2 枚なので triptych より狭い
    # 半透明面の順序を depth peeling で解く（MATLAB の SortMethod='depth' 相当）。
    # ソフトウェア OpenGL（llvmpipe 等、ヘッドレス CI）では peeling が面を
    # すべて落として真っ白になるので、そこでは False にする（面は不透明に描かれる）。
    depth_peeling: bool = True


# ---------------------------------------------------------------------------
# 入力
# ---------------------------------------------------------------------------


def load_profile(
    path: str,
    nx: Optional[int] = None,
    ny: Optional[int] = None,
    *,
    color_source: str = "auto",
    palette: Optional[Dict[str, List[float]]] = None,
) -> DepthProfile4DData:
    """H5 の dialect を見て真値ローダか推定ローダかを選んで読む。

    ``/D_est/data`` があれば :func:`deppro.visualize_4d.load_depth_estimate`、
    無くて ``/Dstk`` があれば :func:`deppro.visualize_4d.load_gt_4d`。
    どちらも無ければ ``ValueError``。
    """
    import h5py

    path = str(path)
    with h5py.File(path, "r") as f:
        has_est = "D_est" in f and "data" in f["D_est"]
        has_gt = "Dstk" in f
        top = sorted(f.keys())
    if has_est:
        return load_depth_estimate(path, nx=nx, ny=ny,
                                   color_source=color_source, palette=palette)
    if has_gt:
        return load_gt_4d(path, nx=nx, ny=ny,
                          color_source=color_source, palette=palette)
    raise ValueError(
        f"{path}: neither /D_est/data (estimate) nor /Dstk (ground truth); "
        f"top-level keys are {top}"
    )


def check_compatible(gt: DepthProfile4DData, recon: DepthProfile4DData) -> None:
    """真値と再構成が同じ grid・深さ軸・元素列であることを確かめる。

    並べて比較する前提そのものなので、違えば何が違うかをすべて挙げて
    ``ValueError`` にする（最初の不一致で止めない）。
    """
    problems = []
    if (gt.nx, gt.ny, gt.nt) != (recon.nx, recon.ny, recon.nt):
        problems.append(
            f"grid (nx, ny, nt): gt {(gt.nx, gt.ny, gt.nt)} vs recon "
            f"{(recon.nx, recon.ny, recon.nt)}"
        )
    if gt.n_depth != recon.n_depth or not np.allclose(gt.depth, recon.depth):
        problems.append(
            f"depth axis: gt {gt.n_depth} points "
            f"[{gt.depth[0]:g}..{gt.depth[-1]:g}] vs recon {recon.n_depth} points "
            f"[{recon.depth[0]:g}..{recon.depth[-1]:g}]"
        )
    if list(gt.element_names) != list(recon.element_names):
        problems.append(
            f"elements: gt {list(gt.element_names)} vs recon {list(recon.element_names)}"
        )
    if problems:
        raise ValueError(
            "ground truth and reconstruction are not comparable:\n  - "
            + "\n  - ".join(problems)
        )


# ---------------------------------------------------------------------------
# 指標・プロファイル（描画ライブラリ不要）
# ---------------------------------------------------------------------------


def mean_profile(data: DepthProfile4DData, t: int) -> np.ndarray:
    """フレーム *t* の (x, y) 平均プロファイル、濃度 % で ``(n_depth, n_elem)``。"""
    frame = np.asarray(data.get_frame(t), dtype=np.float64)   # (nx, ny, n_depth, n_elem)
    return frame.mean(axis=(0, 1)) * 100.0


def frame_metrics(gt: DepthProfile4DData, recon: DepthProfile4DData, t: int) -> Dict[str, float]:
    """フレーム *t* の指標。

    Returns
    -------
    dict
        ``rmse_percent``: :func:`deppro.tools.triptych.rmse_percent`。
        ``w1_median_px_A``: 画素×元素ごとの
        :func:`deppro.depth_profile.wasserstein_distance_1d`（真値が空の画素は
        NaN）を画素方向に中央値、元素で平均した Å。全元素が NaN なら NaN。
        ``w1_per_element_A``: 元素ごとの画素中央値（``n_elem`` 長の配列）。
    """
    G = np.asarray(gt.get_frame(t), dtype=np.float64).reshape(-1, gt.n_depth, gt.n_elem)
    D = np.asarray(recon.get_frame(t), dtype=np.float64).reshape(-1, recon.n_depth, recon.n_elem)
    n_px = G.shape[0]
    w1 = np.full((n_px, gt.n_elem), np.nan)
    for px in range(n_px):
        for e in range(gt.n_elem):
            w1[px, e] = wasserstein_distance_1d(gt.depth, G[px, :, e], D[px, :, e])
    with np.errstate(all="ignore"):
        per_elem = np.array([
            np.nanmedian(w1[:, e]) if np.isfinite(w1[:, e]).any() else np.nan
            for e in range(gt.n_elem)
        ])
        w1_mean = float(np.nanmean(per_elem)) if np.isfinite(per_elem).any() else float("nan")
    return {
        "rmse_percent": rmse_percent(recon, gt, t),
        "w1_median_px_A": w1_mean,
        "w1_per_element_A": per_elem,
    }


# ---------------------------------------------------------------------------
# 描画
# ---------------------------------------------------------------------------


def _resolve_output(output: Optional[str], default_name: str) -> Path:
    if output is None:
        return ensure_outputs_dir("gt_vs_recon") / default_name
    p = Path(output).expanduser()
    if p.is_absolute():
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
    return ensure_outputs_dir("gt_vs_recon") / p


def render_gt_vs_recon(
    gt: DepthProfile4DData,
    recon: DepthProfile4DData,
    config: Optional[GtVsReconConfig] = None,
    *,
    output: Optional[str] = None,
    frames: Optional[Sequence[int]] = None,
) -> Path:
    """真値と再構成を 2x2 に並べ、動画（複数フレーム）か PNG（1 フレーム）を書く。

    Parameters
    ----------
    gt, recon
        :func:`load_profile` で読んだもの。:func:`check_compatible` を通ること。
    config
        見た目。既定は triptych と同じ視点・照明オフ。
    output
        出力ファイル名。絶対パスはそのまま、相対パスは
        ``ensure_outputs_dir("gt_vs_recon")`` の下。既定 ``gt_vs_recon.mp4`` /
        1 フレームなら ``gt_vs_recon.png``。動画のときは先頭フレームの
        PNG（poster）を同名 ``.png`` で添える。
    frames
        描くフレーム番号。既定は全フレーム。

    Returns
    -------
    Path
        書き出した動画または PNG。
    """
    if not HAS_PYVISTA:
        raise ImportError("PyVista required: pip install pyvista")
    try:
        import imageio
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise ImportError(
            "imageio required to write the figure/video: pip install imageio imageio-ffmpeg"
        ) from e
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pyvista as pv

    check_compatible(gt, recon)
    cfg = config or GtVsReconConfig()
    frames = list(range(gt.nt) if frames is None else frames)
    if not frames:
        raise ValueError("frames is empty")

    pv.OFF_SCREEN = True
    render_cfg = replace(_make_renderer_config(cfg), matlab_depth_peeling=cfg.depth_peeling)
    pairs = []
    for data in (gt, recon):
        renderer = MatlabIsosurfaceRenderer(data, render_cfg)
        plotter = pv.Plotter(off_screen=True, window_size=cfg.panel_px)
        plotter.set_background("white")
        pairs.append((renderer, plotter))

    try:
        crop, touches_edge = _common_crop(pairs, gt.nt, cfg.panel_px, cfg.crop_pad)
        if touches_edge:
            print(
                "warning: 立方体がパネル端に接している。camera_zoom を下げるか "
                "crop_pad を減らすと切れなくなる"
            )
        ch = crop[0].stop - crop[0].start
        cw = crop[1].stop - crop[1].start

        # 上段は共通クロップの縦横比、下段はその profile_row_ratio 倍の高さ。
        # 左余白は下段の目盛と軸ラベルの分（上段だけなら 0.02 で足りる）
        lo, hi, top, bot = (0.075 if cfg.profile_row else 0.02), 0.985, 0.90, 0.09
        slot_w = cfg.fig_width_in * (hi - lo) / (2 + cfg.panel_gap)
        img_h = slot_w * ch / cw
        rows = 2 if cfg.profile_row else 1
        ratios = [1.0, cfg.profile_row_ratio] if cfg.profile_row else [1.0]
        fig_h = img_h * sum(ratios) / (top - bot)

        fig, axes = plt.subplots(
            rows, 2, figsize=(cfg.fig_width_in, fig_h), dpi=cfg.dpi,
            squeeze=False, gridspec_kw={"height_ratios": ratios},
        )
        images, titles = [], []
        for ax, label in zip(axes[0], (cfg.gt_label, cfg.recon_label)):
            images.append(ax.imshow(np.zeros((ch, cw, 3), np.uint8)))
            titles.append(ax.set_title(label, fontsize=11, fontweight="bold", pad=6))
            ax.axis("off")

        visible = [i for i in range(gt.n_elem) if i not in cfg.invisible_elements]
        lines: list = []
        if cfg.profile_row:
            for col, ax in enumerate(axes[1]):
                row_lines = []
                for i in range(gt.n_elem):
                    (ln,) = ax.plot(gt.depth, np.zeros_like(gt.depth), color=gt.colors[i],
                                    lw=1.6, label=gt.element_names[i])
                    row_lines.append(ln)
                lines.append(row_lines)
                ax.set_xlim(gt.depth[0], gt.depth[-1])
                ax.set_ylim(0, 100)
                ax.set_xlabel("Depth (Å)", fontsize=9)
                ax.tick_params(labelsize=8)
                ax.grid(True, alpha=0.3)
                if col == 0:
                    ax.set_ylabel("Concentration (%)", fontsize=9)
                else:
                    ax.tick_params(labelleft=False)
            if cfg.show_legend:
                axes[1][1].legend(
                    handles=[lines[0][i] for i in visible], fontsize=7,
                    loc="upper right", frameon=False, ncol=2,
                )
        elif cfg.show_legend:
            handles = [
                plt.Line2D([], [], marker="s", ls="", markersize=9,
                           color=gt.colors[i], label=gt.element_names[i])
                for i in visible
            ]
            fig.legend(handles=handles, loc="lower center", ncol=len(handles),
                       frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.0))

        suptitle = fig.suptitle("", fontsize=11, y=0.995)
        fig.subplots_adjust(left=lo, right=hi, top=top, bottom=bot,
                            wspace=cfg.panel_gap, hspace=0.28)

        def draw(t: int) -> np.ndarray:
            for (renderer, plotter), im in zip(pairs, images):
                im.set_data(_shoot(renderer, plotter, t)[crop])
            if cfg.profile_row:
                for data, row_lines in zip((gt, recon), lines):
                    prof = mean_profile(data, t)
                    for i, ln in enumerate(row_lines):
                        ln.set_ydata(prof[:, i])
            head = cfg.title or (Path(gt.source_path).stem if gt.source_path else "")
            text = f"{head}  —  frame {t + 1}/{gt.nt}" if head else f"frame {t + 1}/{gt.nt}"
            if cfg.show_metrics:
                m = frame_metrics(gt, recon, t)
                text += (f"   |   RMSE {m['rmse_percent']:.2f} %"
                         f"   |   W1 median-px {m['w1_median_px_A']:.1f} Å")
            suptitle.set_text(text)
            fig.canvas.draw()
            return np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()

        if len(frames) == 1:
            path = _resolve_output(output, "gt_vs_recon.png")
            imageio.imwrite(path, draw(frames[0]))
        else:
            path = _resolve_output(output, "gt_vs_recon.mp4")
            writer = imageio.get_writer(
                str(path), fps=cfg.fps, codec="libx264",
                pixelformat="yuv420p", quality=cfg.video_quality,
            )
            try:
                for i, t in enumerate(frames):
                    img = draw(t)
                    if i == 0:
                        imageio.imwrite(path.with_suffix(".png"), img)
                    writer.append_data(img)
                    if (i + 1) % 50 == 0 or (i + 1) == len(frames):
                        print(f"  frame {i + 1}/{len(frames)}", flush=True)
            finally:
                writer.close()
        plt.close(fig)
    finally:
        for _, plotter in pairs:
            plotter.close()

    print(f"saved: {path}")
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="真値と再構成を 2x2 に並べる（3D 等値面 + (x,y) 平均プロファイル）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("使い方")[-1],
    )
    parser.add_argument("--gt", required=True, help="真値の H5（/Dstk）")
    parser.add_argument("--recon", required=True,
                        help="再構成の H5（/D_est/data、または /Dstk dialect）")
    parser.add_argument("--label", default="Reconstruction", help="右パネルの見出し")
    parser.add_argument("--gt-label", default="Ground truth", help="左パネルの見出し")
    parser.add_argument("-o", "--output", default=None,
                        help="出力ファイル名（絶対パスはそのまま、相対は outputs/gt_vs_recon/ の下）")
    parser.add_argument("--frame", type=int, default=None, help="このフレームだけ PNG で書く")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--title", default="")
    parser.add_argument("--nx", type=int, default=None, help="dxdydt を上書きする grid")
    parser.add_argument("--ny", type=int, default=None)
    parser.add_argument("--hide", nargs="*", type=int, default=[],
                        help="3D パネルで隠す元素の 0 始まり index")
    parser.add_argument("--no-metrics", action="store_true", help="見出しの RMSE / W1 を出さない")
    parser.add_argument("--no-profile", action="store_true", help="下段のプロファイルを描かない")
    parser.add_argument("--elevation", type=float, default=None)
    parser.add_argument("--azimuth", type=float, default=None)
    parser.add_argument("--zoom", type=float, default=None)
    parser.add_argument("--crop-pad", type=int, default=None)
    parser.add_argument("--lighting", action="store_true",
                        help="陰影をつける（MATLAB の 3D Lighting on 相当）")
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--no-depth-peeling", action="store_true",
                        help="半透明面の depth peeling を切る（ソフトウェア OpenGL では必要）")
    return parser


def config_from_args(args: argparse.Namespace) -> GtVsReconConfig:
    cfg = GtVsReconConfig(
        gt_label=args.gt_label,
        recon_label=args.label,
        title=args.title,
        invisible_elements=list(args.hide),
        show_metrics=not args.no_metrics,
        profile_row=not args.no_profile,
        lighting=args.lighting,
        depth_peeling=not args.no_depth_peeling,
    )
    for name, value in (("elevation", args.elevation), ("azimuth", args.azimuth),
                        ("camera_zoom", args.zoom), ("crop_pad", args.crop_pad),
                        ("fps", args.fps)):
        if value is not None:
            setattr(cfg, name, value)
    return cfg


def main(argv: Optional[Sequence[str]] = None) -> Path:
    args = build_parser().parse_args(argv)
    gt = load_profile(args.gt, nx=args.nx, ny=args.ny)
    recon = load_profile(args.recon, nx=args.nx, ny=args.ny)
    check_compatible(gt, recon)
    if args.frame is not None:
        frames: Sequence[int] = [args.frame]
    else:
        frames = range(gt.nt if args.max_frames is None else min(gt.nt, args.max_frames))
    return render_gt_vs_recon(gt, recon, config_from_args(args),
                              output=args.output, frames=frames)


if __name__ == "__main__":
    main()
