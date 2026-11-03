"""Triptych: ground truth / baseline / candidate side by side.

Why
----------
Looking at one reconstruction on its own, a person cannot tell whether an
inverse-problem result is good. **Only side by side, at the same scale, from the
same viewpoint and in the same colors**, does the difference become readable.
crop or zoom differs per panel and "one side just looks bigger". This module fixes the frame.

The convention is three panels:

===========  =========================================================
left         ground truth (synthetic data) or an external reference (profilometer, RBS, SIMS, ...)
middle       the baseline method (existing solver, conventional method)
right        the candidate (new solver, NN, after tuning)
===========  =========================================================

For measured data without a ground truth, read the left panel as the external reference and the frame still applies.

Rendering is :class:`deppro.visualize_4d.MatlabIsosurfaceRenderer` (the MATLAB
EzDeP4D ``SurfacePatch == 2`` fidelity mode). The defaults match the production
EzDeP4D videos (lighting off, wide plot box). The values were measured against
MATLAB's rendering (the record is not part of the public release).

Paths are resolved through environment variables: output under
:func:`deppro.paths.ensure_outputs_dir`, input H5 under
:func:`deppro.paths.simulation_root`。

Usage
------
As a library::

    from deppro.tools.triptych import TriptychConfig, render_triptych

    render_triptych(
        [("Ground truth", gt), ("SLSQP", slsqp), ("NN", nn)],
        TriptychConfig(title="Checkered 10x10, 7 elements, noisy"),
    )

From the command line (a triple of MATLAB-generated H5 files)::

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

# PlotBoxAspectRatio measured on the EzDeP4D 2x2 tile (327x220 px axes).
# Constant regardless of azimuth, elevation and data range; set only by the
# aspect ratio of the axes rectangle (measured against MATLAB). None gives a cube (square axes).
MATLAB_TILE_PLOT_BOX = (0.7127, 1.0, 0.6855)


@dataclass
class TriptychConfig:
    """Appearance of the triptych. Defaults match the production EzDeP4D videos."""

    # --- Viewpoint ---
    # MATLAB's default is view([-105 5]); a slightly higher elevation reads better
    # for an overview. The azimuth is MATLAB's.
    azimuth: float = -105.0
    elevation: float = 15.0
    camera_zoom: float = 1.28
    plot_box_aspect: Optional[Tuple[float, float, float]] = MATLAB_TILE_PLOT_BOX

    # --- Surface ---
    # EzDeP4D does not call lightangle when the GUI's 3D Lighting toggle is off.
    # The production videos were written in that state (measured, see docs), so the
    # default is off. True adds shading (MATLAB's Lighting on).
    lighting: bool = False
    face_alpha: float = 1.0          # effective opacity = 0.75 * this (MATLAB default 0.75)
    isovalue_degree: float = 0.5

    # --- Layout ---
    panel_px: Tuple[int, int] = (640, 440)
    crop_pad: int = 26               # margin around the cube in px; larger = "further away"
    panel_gap: float = 0.09          # gap between panels (fraction of the slot width)
    fig_width_in: float = 11.4
    dpi: int = 110

    # --- Content ---
    title: str = ""
    invisible_elements: List[int] = field(default_factory=list)
    show_legend: bool = True

    # --- Video ---
    fps: int = 12
    video_quality: int = 8


def rmse_percent(est: DepthProfile4DData, ref: DepthProfile4DData, t: int) -> float:
    """RMSE [%] of frame *t*, over all elements, not only the visible ones."""
    return float(
        np.sqrt(np.mean((est.get_frame(t) - ref.get_frame(t)) ** 2))
    ) * 100.0


def load_matlab_h5(
    rel_path: str,
    dataset: str = "Drlt",
    *,
    color_source: str = "auto",
) -> DepthProfile4DData:
    """Read a MATLAB H5 under ``DEPPRO_SIMULATION_ROOT`` as 5D data.

    Reads any of ``Dstk`` (setting = ground truth) / ``Dtrn`` (training) / ``Drlt``
    (reconstruction). Shaping is the same as :func:`deppro.visualize_4d.load_gt_4d`:
    MATLAB is column-major, so x varies fastest within a frame.

    The third entry of ``ezdeprof/dxdydt`` (time) can disagree with the actual data,
    so it is not used; the time dimension comes from the pixel count.
    """
    import h5py

    from deppro.visualize_4d import _resolve_colors

    path = Path(rel_path)
    if not path.is_absolute():
        path = simulation_root() / path
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Check that DEPPRO_SIMULATION_ROOT is set "
            f"(currently: {simulation_root()})"
        )

    with h5py.File(path, "r") as f:
        if dataset not in f:
            raise KeyError(
                f"{path} has no {dataset!r}; it has "
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
    # MATLAB's lightangle is an offset of azimuth -10 / elevation +20 from the view.
    # When the camera moves, move the light by the same offset (a fixed light goes backlit).
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
    """Decide the crop frame **shared** by the three panels.

    Different frames per panel would change the scale and break the comparison, so
    the union is always taken. The cube's outline should not depend on the frame, but several frames are sampled to be safe.
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
        raise RuntimeError("every frame was empty (no element reaches the threshold?)")
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
    """Draw the triptych and write a video (several frames) or a PNG (one frame).

    Parameters
    ----------
    panels
        ``(label, data)`` three times, left to right. The left one is the ground truth / external reference.
    config
        Appearance. Defaults match the production EzDeP4D videos.
    output
        Output file name, placed under ``ensure_outputs_dir("triptych")``.
        Defaults to ``triptych.mp4``, or ``triptych.png`` for a single frame.
    frames
        Frame numbers to draw. Defaults to every frame.
    metric
        Metric shown in the headings of panels 2 and 3, called with the left data as the reference.
        ``None`` disables it.

    Returns
    -------
    Path
        The written file.
    """
    if not HAS_PYVISTA:
        raise ImportError("PyVista required: pip install pyvista")

    import imageio
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pyvista as pv

    if len(panels) != 3:
        raise ValueError(f"a triptych takes 3 panels (got {len(panels)})")
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
        # Translucency needs depth peeling (MATLAB resolves it with SortMethod='depth';
        # VTK does not sort). The renderer takes care of it.
        pairs.append((renderer, plotter))

    crop, touches_edge = _common_crop(pairs, ref.nt, cfg.panel_px, cfg.crop_pad)
    if touches_edge:
        print(
            "warning: the cube touches the panel edge. Lower camera_zoom or "
            "reduce crop_pad to stop it being cut off"
        )
    ch = crop[0].stop - crop[0].start
    cw = crop[1].stop - crop[1].start

    # Matching the slots to the cropped aspect ratio removes the automatic
    # letterbox, leaving only panel_gap between panels.
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
    """Split ``path:dataset:label`` (the last two are optional)."""
    parts = spec.split(":")
    path = parts[0]
    dataset = parts[1] if len(parts) > 1 and parts[1] else "Drlt"
    label = ":".join(parts[2:]) if len(parts) > 2 else None
    return path, dataset, label


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Triptych (ground truth / baseline / candidate)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage")[-1],
    )
    parser.add_argument(
        "--h5", action="append", required=True, metavar="PATH[:DATASET[:LABEL]]",
        help="Give it 3 times. May be relative to DEPPRO_SIMULATION_ROOT."
        " DATASET defaults to Dstk/Drlt. Left to right: ground truth / baseline / candidate",
    )
    parser.add_argument("-o", "--output", default=None, help="Output file name")
    parser.add_argument("--frame", type=int, default=None,
                        help="Write only this frame, as a PNG")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--title", default="")
    parser.add_argument("--elevation", type=float, default=None)
    parser.add_argument("--azimuth", type=float, default=None)
    parser.add_argument("--zoom", type=float, default=None)
    parser.add_argument("--crop-pad", type=int, default=None)
    parser.add_argument("--panel-gap", type=float, default=None)
    parser.add_argument("--lighting", action="store_true",
                        help="Add shading (MATLAB's 3D Lighting on)")
    parser.add_argument("--face-alpha", type=float, default=None,
                        help="Effective opacity = 0.75 x this; 4/3 is fully opaque")
    parser.add_argument("--cube", action="store_true",
                        help="Use a cube (the default is the wide EzDeP4D tile)")
    parser.add_argument("--hide-oxygen", action="store_true")
    parser.add_argument("--hide", nargs="*", type=int, default=[])
    parser.add_argument("--fps", type=int, default=None)
    args = parser.parse_args(argv)

    if len(args.h5) != 3:
        parser.error(f"--h5 is needed 3 times (got {len(args.h5)})")

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
