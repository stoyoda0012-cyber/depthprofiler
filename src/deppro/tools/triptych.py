"""三点比較図（triptych）— 真値 / ベースライン / 評価対象を横に並べる。

なぜ要るか
----------
逆問題の結果を人間が見て判断するとき、単独の再構成を眺めても良し悪しが
分からない。**同じ縮尺・同じ視点・同じ配色で並べて初めて**差が読める。
その並べ方を毎回書き直していると、パネルごとにクロップや zoom が変わって
「片方だけ大きい」比較になりやすい。ここではその枠を固定する。

規約は 3 枚組:

===========  =========================================================
左           真値（合成データ）または外部参照（触針計・RBS・SIMS など）
中           ベースライン手法（既存ソルバ・従来法）
右           評価対象（新しいソルバ・NN・チューニング後）
===========  =========================================================

真値が無い実データでも、左を外部参照に読み替えれば同じ枠で使える。

描画は :class:`deppro.visualize_4d.MatlabIsosurfaceRenderer`（MATLAB
EzDeP4D の ``SurfacePatch == 2`` 忠実モード）。既定値は EzDeP4D の本番
動画に揃えてある（照明オフ・横長プロットボックス）。値は MATLAB の描画と
比べて実測した（記録は公開版に含めない）。

パスは環境変数で解決する。出力は
:func:`deppro.paths.ensure_outputs_dir`、入力の H5 は
:func:`deppro.paths.simulation_root`。

使い方
------
ライブラリとして::

    from deppro.tools.triptych import TriptychConfig, render_triptych

    render_triptych(
        [("Ground truth", gt), ("SLSQP", slsqp), ("NN", nn)],
        TriptychConfig(title="Checkered 10x10, 7 elements, noisy"),
    )

コマンドラインから（MATLAB 生成の H5 三つ組）::

    export DEPPRO_SIMULATION_ROOT=/path/to/Simulation/DepthProfiler
    python -m deppro.tools.triptych \\
        --h5 Checkered/10x10x200_7elem_noisy/<main>.h5:Dstk:"Ground truth" \\
        --h5 Checkered/10x10x200_7elem_noisy_SLSQP/<main>.h5:Drlt:SLSQP \\
        --h5 Checkered/10x10x200_7elem_noisy_NN/<main>.h5:Drlt:NN \\
        --hide-oxygen -o triptych.mp4
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from deppro.paths import ensure_outputs_dir, simulation_root
from deppro.visualize_4d import (
    HAS_PYVISTA,
    AnimationConfig,
    DepthProfile4DData,
    MatlabIsosurfaceRenderer,
)

__all__ = [
    "TriptychConfig",
    "render_triptych",
    "load_matlab_h5",
    "rmse_percent",
]

# EzDeP4D の 2x2 タイル（327x220 px の axes）で実測した PlotBoxAspectRatio。
# 方位・仰角・データ範囲に依らず一定で、axes 矩形の縦横比だけで決まる
# （MATLAB の描画と比べた実測）。None にすると立方体（正方 axes 相当）。
MATLAB_TILE_PLOT_BOX = (0.7127, 1.0, 0.6855)


@dataclass
class TriptychConfig:
    """三点比較図の見た目。既定は EzDeP4D の本番動画に合わせてある。"""

    # --- 視点 ---
    # MATLAB 既定は view([-105 5]) だが、全体を見渡すには少し上から見る方が
    # 読みやすい。方位は MATLAB のまま。
    azimuth: float = -105.0
    elevation: float = 15.0
    camera_zoom: float = 1.28
    plot_box_aspect: Optional[Tuple[float, float, float]] = MATLAB_TILE_PLOT_BOX

    # --- 質感 ---
    # EzDeP4D は GUI の 3D Lighting トグルが off のとき lightangle を呼ばない。
    # 本番動画はその状態で書き出されていた（実測、docs 参照）ので既定を off に
    # する。True にすると陰影がつく（MATLAB の Lighting on 相当）。
    lighting: bool = False
    face_alpha: float = 1.0          # 実効不透明度 = 0.75 * これ（MATLAB 既定 0.75）
    isovalue_degree: float = 0.5

    # --- レイアウト ---
    panel_px: Tuple[int, int] = (640, 440)
    crop_pad: int = 26               # 立方体まわりの余白 px。大きいほど「引き」
    panel_gap: float = 0.09          # パネル間の隙間（スロット幅に対する比）
    fig_width_in: float = 11.4
    dpi: int = 110

    # --- 中身 ---
    title: str = ""
    invisible_elements: List[int] = field(default_factory=list)
    show_legend: bool = True

    # --- 動画 ---
    fps: int = 12
    video_quality: int = 8


def rmse_percent(est: DepthProfile4DData, ref: DepthProfile4DData, t: int) -> float:
    """フレーム *t* の RMSE[%]。可視元素だけでなく全元素で取る。"""
    return float(
        np.sqrt(np.mean((est.get_frame(t) - ref.get_frame(t)) ** 2))
    ) * 100.0


def load_matlab_h5(
    rel_path: str,
    dataset: str = "Drlt",
    *,
    color_source: str = "auto",
) -> DepthProfile4DData:
    """``DEPPRO_SIMULATION_ROOT`` 配下の MATLAB H5 を 5D で読む。

    ``Dstk``（設定＝真値）/ ``Dtrn``（学習）/ ``Drlt``（再構成）のどれでも
    読める。整形は :func:`deppro.visualize_4d.load_gt_4d` と同じで、MATLAB が
    列優先なのでフレーム内は x が最速。

    ``ezdeprof/dxdydt`` の 3 要素目（時間）は実データと食い違うことがあるので
    使わず、画素数から割り出す。
    """
    import h5py

    from deppro.visualize_4d import _resolve_colors

    path = Path(rel_path)
    if not path.is_absolute():
        path = simulation_root() / path
    if not path.exists():
        raise FileNotFoundError(
            f"{path} が無い。DEPPRO_SIMULATION_ROOT を設定したか確認する "
            f"（現在: {simulation_root()}）"
        )

    with h5py.File(path, "r") as f:
        if dataset not in f:
            raise KeyError(
                f"{path} に {dataset!r} が無い。あるのは "
                f"{[k for k in f.keys() if k.startswith('D')]}"
            )
        raw = np.array(f[dataset], dtype=np.float64)   # (n_px, n_elem, n_depth)
        n_px, n_elem, n_depth = raw.shape
        arr = np.transpose(raw, (0, 2, 1))             # (n_px, n_depth, n_elem)

        g = f["ezdeprof"]
        depth = np.array(g["depth"], dtype=np.float64).ravel()
        step = float(np.array(g["stepsize"], dtype=np.float64).ravel()[0])
        names = [
            e.decode() if isinstance(e, bytes) else str(e)
            for e in g["element"][:].ravel()
        ]
        dxdydt = np.array(g["dxdydt"], dtype=np.float64).ravel()
        nx, ny = int(dxdydt[0]), int(dxdydt[1])
        h5col = np.array(g["Col"], dtype=np.float64) if "Col" in g else None
        colors = _resolve_colors(names, h5col, n_elem, color_source=color_source)

    dxdy = nx * ny
    nt = n_px // dxdy
    idx_f = np.arange(dxdy).reshape(nx, ny, order="F")
    data5d = arr.reshape(nt, dxdy, n_depth, n_elem)[:, idx_f, :, :]

    return DepthProfile4DData(
        data=data5d, depth=depth, element_names=names, colors=colors,
        step_size=step, nx=nx, ny=ny, nt=nt, n_depth=n_depth,
        n_elem=n_elem, source_path=str(path),
    )


def _make_renderer_config(cfg: TriptychConfig) -> AnimationConfig:
    # MATLAB の lightangle は view に対して方位 -10 / 仰角 +20 のオフセット。
    # カメラを動かしたら光源も同じオフセットで動かす（固定すると逆光になる）。
    return AnimationConfig(
        render_mode="matlab",
        depth_profile="none",
        invisible_elements=list(cfg.invisible_elements),
        azimuth=cfg.azimuth,
        elevation=cfg.elevation,
        light_azimuth=cfg.azimuth - 10.0,
        light_elevation=cfg.elevation + 20.0,
        lighting=cfg.lighting,
        plot_box_aspect=cfg.plot_box_aspect,
        camera_zoom=cfg.camera_zoom,
        face_alpha=cfg.face_alpha,
        isovalue_degree=cfg.isovalue_degree,
        window_size=cfg.panel_px,
    )


def _shoot(renderer, plotter, t: int) -> np.ndarray:
    renderer.render_frame(t, plotter)
    for name in ("matlab_iso_title",):
        try:
            plotter.remove_actor(name)
        except Exception:
            pass
    try:
        plotter.hide_axes()
    except Exception:
        pass
    plotter.render()
    return plotter.screenshot(return_img=True)


def _common_crop(pairs, n_frames: int, panel_px, pad: int):
    """3 面**共通**のクロップ枠を決める。

    枠を面ごとに変えると縮尺が変わって比較にならないので、必ず和を取る。
    立方体の外形はフレームに依らないはずだが、念のため複数フレームで取る。
    """
    w, h = panel_px
    y0 = x0 = 10 ** 9
    y1 = x1 = -1
    for t in sorted({0, n_frames // 2, n_frames - 1}):
        for renderer, plotter in pairs:
            mask = (_shoot(renderer, plotter, t)[:, :, :3] < 245).any(axis=2)
            ys, xs = np.where(mask)
            if not len(ys):
                continue
            y0, y1 = min(y0, ys.min()), max(y1, ys.max())
            x0, x1 = min(x0, xs.min()), max(x1, xs.max())
    if y1 < 0:
        raise RuntimeError("全フレームが空だった（閾値に届く元素が無い？）")
    y0, y1 = max(0, y0 - pad), min(h - 1, y1 + pad)
    x0, x1 = max(0, x0 - pad), min(w - 1, x1 + pad)
    touches_edge = (y0 <= 0 or x0 <= 0 or y1 >= h - 1 or x1 >= w - 1)
    return (slice(y0, y1 + 1), slice(x0, x1 + 1)), touches_edge


def render_triptych(
    panels: Sequence[Tuple[str, DepthProfile4DData]],
    config: Optional[TriptychConfig] = None,
    *,
    output: Optional[str] = None,
    frames: Optional[Sequence[int]] = None,
    metric: Optional[Callable[[DepthProfile4DData, DepthProfile4DData, int], float]]
    = rmse_percent,
    metric_label: str = "RMSE",
) -> Path:
    """3 枚組を描いて動画（複数フレーム）か PNG（単一フレーム）を書く。

    Parameters
    ----------
    panels
        ``(ラベル, データ)`` を左から 3 つ。左が真値／外部参照。
    config
        見た目。既定は EzDeP4D の本番動画相当。
    output
        出力ファイル名。``ensure_outputs_dir("triptych")`` の下に置かれる。
        既定は ``triptych.mp4`` / 単一フレームなら ``triptych.png``。
    frames
        描くフレーム番号。既定は全フレーム。
    metric
        パネル 2 以降の見出しに出す指標。左のデータを参照として呼ばれる。
        ``None`` で無効。

    Returns
    -------
    Path
        書き出したファイル。
    """
    if not HAS_PYVISTA:
        raise ImportError("PyVista required: pip install pyvista")

    import imageio
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pyvista as pv

    if len(panels) != 3:
        raise ValueError(f"3 枚組なので 3 つ渡す（受け取った: {len(panels)}）")
    cfg = config or TriptychConfig()
    ref = panels[0][1]
    if frames is None:
        frames = range(ref.nt)
    frames = list(frames)

    pv.OFF_SCREEN = True
    render_cfg = _make_renderer_config(cfg)
    pairs = []
    for _, data in panels:
        renderer = MatlabIsosurfaceRenderer(data, render_cfg)
        plotter = pv.Plotter(off_screen=True, window_size=cfg.panel_px)
        plotter.set_background("white")
        # 半透明にするなら depth peeling が要る（MATLAB は SortMethod='depth'
        # で解決しているが VTK は並べ替えない）。レンダラ側が面倒を見る。
        pairs.append((renderer, plotter))

    crop, touches_edge = _common_crop(pairs, ref.nt, cfg.panel_px, cfg.crop_pad)
    if touches_edge:
        print(
            "warning: 立方体がパネル端に接している。camera_zoom を下げるか "
            "crop_pad を減らすと切れなくなる"
        )
    ch = crop[0].stop - crop[0].start
    cw = crop[1].stop - crop[1].start

    # クロップ後の縦横比にスロットを合わせると自動の余白（letterbox）が
    # 消え、パネル間の空きは panel_gap だけになる。
    lo, hi, top, bot = 0.005, 0.995, 0.875, 0.10
    slot_w = cfg.fig_width_in * (hi - lo) / (3 + 2 * cfg.panel_gap)
    fig_h = (slot_w * ch / cw) / (top - bot)

    fig, axes = plt.subplots(
        1, 3, figsize=(cfg.fig_width_in, fig_h), dpi=cfg.dpi
    )
    images, titles = [], []
    for ax, (label, _) in zip(axes, panels):
        images.append(ax.imshow(np.zeros((ch, cw, 3), np.uint8)))
        titles.append(ax.set_title(label, fontsize=11, fontweight="bold", pad=6))
        ax.axis("off")

    if cfg.show_legend:
        visible = [
            i for i in range(ref.n_elem) if i not in cfg.invisible_elements
        ]
        handles = [
            plt.Line2D([], [], marker="s", ls="", markersize=9,
                       color=ref.colors[i], label=ref.element_names[i])
            for i in visible
        ]
        fig.legend(handles=handles, loc="lower center", ncol=len(handles),
                   frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.0))
    suptitle = fig.suptitle("", fontsize=11, y=0.995)
    fig.subplots_adjust(left=lo, right=hi, top=top, bottom=bot,
                        wspace=cfg.panel_gap)

    def draw(t: int) -> np.ndarray:
        for (renderer, plotter), im in zip(pairs, images):
            im.set_data(_shoot(renderer, plotter, t)[crop])
        for idx, (label, data) in enumerate(panels):
            if idx == 0 or metric is None:
                titles[idx].set_text(label)
            else:
                titles[idx].set_text(
                    f"{label}   ({metric_label} {metric(data, ref, t):.2f}%)"
                )
        head = cfg.title or (Path(ref.source_path).stem if ref.source_path else "")
        suptitle.set_text(
            f"{head}  —  frame {t + 1}/{ref.nt}"
            f"   |   view([{cfg.azimuth:.0f} {cfg.elevation:.0f}])"
            + ("" if cfg.lighting else "   |   lighting off")
        )
        fig.canvas.draw()
        return np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()

    out_dir = ensure_outputs_dir("triptych")
    try:
        if len(frames) == 1:
            path = out_dir / (output or "triptych.png")
            imageio.imwrite(path, draw(frames[0]))
        else:
            path = out_dir / (output or "triptych.mp4")
            writer = imageio.get_writer(
                str(path), fps=cfg.fps, codec="libx264",
                pixelformat="yuv420p", quality=cfg.video_quality,
            )
            for i, t in enumerate(frames):
                writer.append_data(draw(t))
                if (i + 1) % 50 == 0 or (i + 1) == len(frames):
                    print(f"  frame {i + 1}/{len(frames)}", flush=True)
            writer.close()
    finally:
        for _, plotter in pairs:
            plotter.close()
        plt.close(fig)

    print(f"saved: {path}")
    return path


def _parse_spec(spec: str) -> Tuple[str, str, Optional[str]]:
    """``path:dataset:label`` を分解する（後ろ 2 つは省略可）。"""
    parts = spec.split(":")
    path = parts[0]
    dataset = parts[1] if len(parts) > 1 and parts[1] else "Drlt"
    label = ":".join(parts[2:]) if len(parts) > 2 else None
    return path, dataset, label


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="三点比較図（真値 / ベースライン / 評価対象）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("使い方")[-1],
    )
    parser.add_argument(
        "--h5", action="append", required=True, metavar="PATH[:DATASET[:LABEL]]",
        help="3 回指定する。DEPPRO_SIMULATION_ROOT からの相対パスでよい。"
        " DATASET 既定 Dstk/Drlt。左から 真値 / ベースライン / 評価対象",
    )
    parser.add_argument("-o", "--output", default=None, help="出力ファイル名")
    parser.add_argument("--frame", type=int, default=None,
                        help="このフレームだけ PNG で書く")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--title", default="")
    parser.add_argument("--elevation", type=float, default=None)
    parser.add_argument("--azimuth", type=float, default=None)
    parser.add_argument("--zoom", type=float, default=None)
    parser.add_argument("--crop-pad", type=int, default=None)
    parser.add_argument("--panel-gap", type=float, default=None)
    parser.add_argument("--lighting", action="store_true",
                        help="陰影をつける（MATLAB の 3D Lighting on 相当）")
    parser.add_argument("--face-alpha", type=float, default=None,
                        help="実効不透明度 = 0.75 x これ。4/3 で完全不透明")
    parser.add_argument("--cube", action="store_true",
                        help="立方体にする（既定は EzDeP4D タイルの横長）")
    parser.add_argument("--hide-oxygen", action="store_true")
    parser.add_argument("--hide", nargs="*", type=int, default=[])
    parser.add_argument("--fps", type=int, default=None)
    args = parser.parse_args(argv)

    if len(args.h5) != 3:
        parser.error(f"--h5 は 3 回必要（受け取った: {len(args.h5)}）")

    panels: List[Tuple[str, DepthProfile4DData]] = []
    default_labels = ["Ground truth", "Baseline", "Method"]
    for spec, fallback in zip(args.h5, default_labels):
        path, dataset, label = _parse_spec(spec)
        data = load_matlab_h5(path, dataset)
        panels.append((label or fallback, data))

    hidden = list(args.hide)
    if args.hide_oxygen:
        for i, name in enumerate(panels[0][1].element_names):
            if name.startswith("O(") or name == "O":
                if i not in hidden:
                    hidden.append(i)

    cfg = TriptychConfig(title=args.title, invisible_elements=hidden)
    if args.elevation is not None:
        cfg.elevation = args.elevation
    if args.azimuth is not None:
        cfg.azimuth = args.azimuth
    if args.zoom is not None:
        cfg.camera_zoom = args.zoom
    if args.crop_pad is not None:
        cfg.crop_pad = args.crop_pad
    if args.panel_gap is not None:
        cfg.panel_gap = args.panel_gap
    if args.face_alpha is not None:
        cfg.face_alpha = args.face_alpha
    if args.fps is not None:
        cfg.fps = args.fps
    if args.lighting:
        cfg.lighting = True
    if args.cube:
        cfg.plot_box_aspect = None

    frames = None
    if args.frame is not None:
        frames = [args.frame]
    elif args.max_frames is not None:
        frames = range(min(args.max_frames, panels[0][1].nt))

    render_triptych(panels, cfg, output=args.output, frames=frames)


if __name__ == "__main__":
    main()
