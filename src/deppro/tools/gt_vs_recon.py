"""Ground truth and reconstruction side by side in a 2x2 figure (GT vs reconstruction).

Why
----------
A reconstructed 3D isosurface on its own does not show whether it is good. Only
next to the ground truth (synthetic data) **from the same viewpoint, with the same
crop and the same colors**, with the (x, y)-averaged depth profiles underneath, can
a shift in layer position be told from an error in concentration. This is a thin
composition layer so that arrangement is not rewritten each time: rendering is
:mod:`deppro.visualize_4d` (``MatlabIsosurfaceRenderer``), and the view and crop conventions are those of :mod:`deppro.tools.triptych`.

Layout (one figure per frame; an mp4 for several frames)::

    +---------------------------+---------------------------+
    |  3D ground truth          |  3D reconstruction        |
    +---------------------------+---------------------------+
    |  GT (x,y)-mean profile    |  reconstruction (x,y)-mean |
    |  conc (%) vs depth (Å)    |  same element colors      |
    +---------------------------+---------------------------+
    heading: <title> — frame t/nt | RMSE x.xx % | W1 median-px x.x Å

Only published metric definitions are used: RMSE [%] is
:func:`deppro.tools.triptych.rmse_percent` (all pixels, depths and elements), and
W1 [Å] is :func:`deppro.depth_profile.wasserstein_distance_1d` per pixel and
element, NaN where the ground truth is empty, median over pixels, mean over elements.
No thickness line is drawn (the thickness estimator lives in ``benchmarks``, which is not shipped).

Input
----
The ground truth is an H5 with ``/Dstk`` (MATLAB-generated, or written by
:class:`deppro.simulate.writer.H5Writer`); the reconstruction is an H5 with
``/D_est/data`` as written by :func:`deppro.app.write_depth_estimate`.
:func:`load_profile` tells the two dialects apart from the file. The two must
agree in grid (nx, ny, nt), depth axis and element names (including order);
otherwise :func:`check_compatible` lists what differs and refuses.

Usage
------
As a library::

    from deppro.tools.gt_vs_recon import load_profile, render_gt_vs_recon
    gt = load_profile("gt.h5")
    recon = load_profile("depth_est_NN_20260101_000000.h5")
    render_gt_vs_recon(gt, recon, output="gt_vs_nn.mp4")

From the command line::

    python -m deppro.tools.gt_vs_recon --gt gt.h5 --recon depth_est_NN.h5 \\
        --label "NN surrogate" --title "Checkered 3x2" -o gt_vs_nn.mp4
    python -m deppro.tools.gt_vs_recon --gt gt.h5 --recon est.h5 --frame 0   # one PNG

An absolute output path is used as given; a relative one goes under
``deppro.paths.ensure_outputs_dir("gt_vs_recon")``. Rendering needs pyvista and
imageio (imageio-ffmpeg for video); without them an ImportError is raised at the point of use.
The metrics and the input checks work without the rendering libraries.

Under headless software OpenGL (Xvfb + llvmpipe) the renderer's depth peeling
drops every face and the figure comes out blank. In that environment pass
``--no-depth-peeling`` (``GtVsReconConfig(depth_peeling=False)``).
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
    """Appearance of the 2x2 figure. The 3D panel defaults are those of the triptych (production EzDeP4D videos)."""

    gt_label: str = "Ground truth"
    recon_label: str = "Reconstruction"
    profile_row: bool = True       # draw the profile row
    show_metrics: bool = True      # show RMSE / W1 in the heading
    profile_row_ratio: float = 0.55  # height of the profile row relative to the 3D panels
    fig_width_in: float = 8.4      # two panels, so narrower than the triptych
    # Resolve translucent faces with depth peeling (MATLAB's SortMethod='depth').
    # Under software OpenGL (llvmpipe and the like, headless CI) peeling drops every
    # face and leaves the panel blank, so set this to False there (faces are drawn opaque).
    depth_peeling: bool = True


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------


def load_profile(
    path: str,
    nx: Optional[int] = None,
    ny: Optional[int] = None,
    *,
    color_source: str = "auto",
    palette: Optional[Dict[str, List[float]]] = None,
) -> DepthProfile4DData:
    """Pick the ground-truth or the estimate loader from the dialect of the H5 and read it.

    With ``/D_est/data``, :func:`deppro.visualize_4d.load_depth_estimate`; otherwise
    with ``/Dstk``, :func:`deppro.visualize_4d.load_gt_4d`.
    With neither, ``ValueError``.
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
    """Check that ground truth and reconstruction share the grid, the depth axis and the element list.

    This is the premise of a side-by-side comparison, so every difference is listed
    and raised as ``ValueError`` (not only the first).
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
# Metrics and profiles (no rendering library needed)
# ---------------------------------------------------------------------------


def mean_profile(data: DepthProfile4DData, t: int) -> np.ndarray:
    """(x, y)-mean profile of frame *t* in concentration %, shape ``(n_depth, n_elem)``."""
    frame = np.asarray(data.get_frame(t), dtype=np.float64)   # (nx, ny, n_depth, n_elem)
    return frame.mean(axis=(0, 1)) * 100.0


def frame_metrics(gt: DepthProfile4DData, recon: DepthProfile4DData, t: int) -> Dict[str, float]:
    """Metrics of frame *t*.

    Returns
    -------
    dict
        ``rmse_percent``: :func:`deppro.tools.triptych.rmse_percent`。
        ``w1_median_px_A``: :func:`deppro.depth_profile.wasserstein_distance_1d`
        per pixel and element (NaN where the ground truth is empty), median over
        pixels, mean over elements, in Å. NaN if every element is NaN.
        ``w1_per_element_A``: the per-pixel median of each element (array of length ``n_elem``).
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
# Rendering
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
    """Put ground truth and reconstruction in a 2x2 figure and write a video (several frames) or a PNG (one frame).

    Parameters
    ----------
    gt, recon
        As read by :func:`load_profile`; must pass :func:`check_compatible`.
    config
        Appearance. The default is the triptych view with lighting off.
    output
        Output file name. An absolute path is used as given; a relative one goes under
        ``ensure_outputs_dir("gt_vs_recon")``. Defaults to ``gt_vs_recon.mp4``, or
        ``gt_vs_recon.png`` for one frame. For a video, the first frame is also written
        as a poster PNG with the same stem.
    frames
        Frame numbers to draw. Defaults to every frame.

    Returns
    -------
    Path
        The written video or PNG.
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
                "warning: the cube touches the panel edge. Lower camera_zoom or "
                "reduce crop_pad to stop it being cut off"
            )
        ch = crop[0].stop - crop[0].start
        cw = crop[1].stop - crop[1].start

        # The top row keeps the aspect ratio of the common crop; the bottom row is profile_row_ratio times its height.
        # The left margin makes room for the ticks and axis label of the profile row (0.02 is enough without it)
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
        description="Ground truth and reconstruction side by side in 2x2 (3D isosurfaces + (x,y)-mean profiles)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage")[-1],
    )
    parser.add_argument("--gt", required=True, help="ground-truth H5 (/Dstk)")
    parser.add_argument("--recon", required=True,
                        help="reconstruction H5 (/D_est/data, or the /Dstk dialect)")
    parser.add_argument("--label", default="Reconstruction", help="heading of the right panel")
    parser.add_argument("--gt-label", default="Ground truth", help="heading of the left panel")
    parser.add_argument("-o", "--output", default=None,
                        help="output file name (absolute as given; relative under outputs/gt_vs_recon/)")
    parser.add_argument("--frame", type=int, default=None, help="write only this frame, as a PNG")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--title", default="")
    parser.add_argument("--nx", type=int, default=None, help="grid override for dxdydt")
    parser.add_argument("--ny", type=int, default=None)
    parser.add_argument("--hide", nargs="*", type=int, default=[],
                        help="0-based indices of elements to hide in the 3D panels")
    parser.add_argument("--no-metrics", action="store_true", help="do not show RMSE / W1 in the heading")
    parser.add_argument("--no-profile", action="store_true", help="do not draw the profile row")
    parser.add_argument("--elevation", type=float, default=None)
    parser.add_argument("--azimuth", type=float, default=None)
    parser.add_argument("--zoom", type=float, default=None)
    parser.add_argument("--crop-pad", type=int, default=None)
    parser.add_argument("--lighting", action="store_true",
                        help="add shading (MATLAB's 3D Lighting on)")
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--no-depth-peeling", action="store_true",
                        help="turn off depth peeling of translucent faces (needed under software OpenGL)")
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
