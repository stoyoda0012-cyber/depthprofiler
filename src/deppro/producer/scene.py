"""Scene: FeatureVector modulation per stack.

MATLAB counterpart (`+deppro/+simulation/SimulationUpdater.m::updateFeatureVector`):

    FV = FV0
    bottom layer  (Linear | Exponential | Trigonometric)   -> FV(sk, dx, dy)
    top layer     (Checkered)                              -> FV(sk, dx, dy)
    interf redox  (Linear | Exponential | Trigonometric)   -> FV(sk, dxdy, dxdydt)
    etch/grow     (Linear | Exponential | Trigonometric)   -> FV(sk, dxdy, dxdydt)

**A scene is a function of the flat stack number sk**; there is no triple loop and no frame boundary.
`dx`,`dy` are used only to set the period of the pattern. So `stack` need not be a multiple
of `nx*ny`; a prime number is defined just as well.

FeatureVector order (1-based, MATLAB):
    col 1        thickness
    col 2        abruptness
    col 3k       position of the k-th free component
    col 3k+1     width
    col 3k+2     height
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

_BOTTOM = ("Linear", "Exponential", "Trigonometric")
_REDOX = ("Linear", "Exponential", "Trigonometric")
_ETCH = ("Linear", "Exponential", "Trigonometric")


@dataclass(frozen=True)
class SceneFlags:
    """Choice of distribution functions in cfg. An empty string / None means "do not apply that group"."""

    bottom_layer: Optional[str] = None
    top_layer: Optional[str] = None       # "Checkered" only
    interf_redox: Optional[str] = None
    etch_grow: Optional[str] = None

    def validate(self) -> None:
        """Unported branches are **refused here**. They are not silently replaced by an existing branch.

        The MATLAB side has 3 groups × 3 shapes (Linear/Exponential/Trigonometric), but this
        producer ported only one per group. Existing as a name and being buildable here are
        different things, so the message tells them apart.
        """
        for name, value, known, table in (
            ("bottom_layer", self.bottom_layer, _BOTTOM, _BOTTOM_FN),
            ("interf_redox", self.interf_redox, _REDOX, _REDOX_FN),
            ("etch_grow", self.etch_grow, _ETCH, _ETCH_FN),
        ):
            if not value:
                continue
            if value not in known:
                raise NotImplementedError(
                    f"{name}={value!r} such a branch does not exist in MATLAB either "
                    f"(MATLAB has {known}); implemented here: {tuple(table)}")
            if value not in table:
                raise NotImplementedError(
                    f"{name}={value!r} exists in MATLAB (+deppro/+simulation) but is NOT ported "
                    f"to this producer. Implemented here: {tuple(table)}. "
                    f"Generate it with the MATLAB CLI, or port the branch first — "
                    f"do not substitute another branch.")
        if self.top_layer and self.top_layer != "Checkered":
            raise NotImplementedError(
                f"top_layer={self.top_layer!r} is not ported (implemented here: ('Checkered',))")


def _n_free(fv: np.ndarray) -> int:
    return (fv.shape[1] - 2) // 3


def distrib_bottom_layer_trigonometric(fv: np.ndarray, sk: np.ndarray, dx: int, dy: int) -> np.ndarray:
    out = fv.copy()
    w = (2 - np.cos(2 * np.pi * (sk - 1) / (dx / 3)) - np.cos(2 * np.pi * np.floor((sk - 1) / dy) / (dy / 3))) / 4
    out[:, [0, 2]] = fv[:, [0, 2]] + 50 * w[:, None]
    out[:, 3] = fv[:, 3] + 75 * w
    return out


def distrib_top_layer_checkered(fv: np.ndarray, sk: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Checkerboard. Swaps the **height** of the second-to-last and last free elements.

    The period is `dx/6` in the fast direction and `dx*dy/6` in the slow one (= 6 blocks per side).
    """
    out = fv.copy()
    nfree = _n_free(fv)
    c1 = 3 * (nfree - 1) + 1          # MATLAB 3*(SizeofFree-1)+2 -> 0-based
    c2 = 3 * nfree + 1                # MATLAB 3*SizeofFree+2     -> 0-based
    id1 = np.mod(np.ceil(6 * sk / dx), 2) == 0
    id2 = np.mod(np.ceil(6 * sk / (dx * dy)), 2) == 0
    same = (id1 & id2) | (~id1 & ~id2)
    out[same, c1] = 0
    out[same, c2] = 100
    out[~same, c1] = 100
    out[~same, c2] = 0
    return out


def phenom_interf_redox_exponential(fv: np.ndarray, sk: np.ndarray, dxdy: int, dxdydt: int) -> np.ndarray:
    out = fv.copy()
    cyc = np.floor((sk - 1) / dxdydt)
    rising = np.mod(cyc, 2) == 0
    within = sk - 1 - cyc * dxdydt
    up = 1 - np.exp(-np.floor(within / dxdy) * dxdy / (dxdydt / 5))
    down = np.exp(np.ceil((-sk + 1 + cyc * dxdydt) / dxdy) * dxdy / (dxdydt / 5))
    w = np.where(rising, up, down)
    out[:, [0, 2]] = fv[:, [0, 2]] + 100 * w[:, None]
    out[:, 3] = fv[:, 3] + 150 * w
    return out


def phenom_etch_grow_linear(fv: np.ndarray, sk: np.ndarray, dxdy: int, dxdydt: int) -> np.ndarray:
    out = fv.copy()
    nfree = _n_free(fv)
    cols = [0] + [3 * k - 1 for k in range(1, nfree + 1)]   # MATLAB [1, 3*(1:SizeofFree)]
    cyc = np.floor((sk - 1) / dxdydt)
    rising = np.mod(cyc, 2) == 0
    within = sk - 1 - cyc * dxdydt
    shift = np.where(rising,
                     -np.floor(within / (2 * dxdy)),
                     -50 - np.ceil((-sk + 1 + cyc * dxdydt) / (2 * dxdy)))
    out[:, cols] = fv[:, cols] + shift[:, None]
    return out


_BOTTOM_FN = {"Trigonometric": distrib_bottom_layer_trigonometric}
_REDOX_FN = {"Exponential": phenom_interf_redox_exponential}
_ETCH_FN = {"Linear": phenom_etch_grow_linear}


def feature_vector_for_stacks(
    fv0: np.ndarray, sk: np.ndarray, nx: int, ny: int, time_size: int, flags: SceneFlags
) -> np.ndarray:
    """Return the FeatureVector for the flat stack number `sk` (1-based).

    `sk` may be a vector. The MATLAB side also handles sk as a vector, so chunking
    does not change the result (stacks are independent).
    """
    flags.validate()
    sk = np.asarray(sk, dtype=float).ravel()
    fv = np.repeat(np.asarray(fv0, dtype=float).reshape(1, -1), sk.size, axis=0)
    dxdy = nx * ny
    dxdydt = nx * ny * time_size

    if flags.bottom_layer:
        fn = _BOTTOM_FN.get(flags.bottom_layer)
        if fn is None:
            raise NotImplementedError(f"bottom_layer={flags.bottom_layer!r} not implemented yet")
        fv = fn(fv, sk, nx, ny)
    if flags.top_layer:
        fv = distrib_top_layer_checkered(fv, sk, nx, ny)
    if flags.interf_redox:
        fn = _REDOX_FN.get(flags.interf_redox)
        if fn is None:
            raise NotImplementedError(f"interf_redox={flags.interf_redox!r} not implemented yet")
        fv = fn(fv, sk, dxdy, dxdydt)
    if flags.etch_grow:
        fn = _ETCH_FN.get(flags.etch_grow)
        if fn is None:
            raise NotImplementedError(f"etch_grow={flags.etch_grow!r} not implemented yet")
        fv = fn(fv, sk, dxdy, dxdydt)
    return fv


#: Frame interval at which a temporal phenomenon reverses direction. Same for all 3 shapes in `+deppro/+simulation`.
#:   *Linear / *Exponential : where the parity of id = ~mod(floor((sk-1)/(nx*ny*dt)),2) changes
#:   *Trigonometric         : where (1-cos(pi*f/dt))/2 turns around
#: In both, 0-based frames f = dt, 2dt, ... (1-based: dt+1, 2dt+1, ...).
TURNING_RULE = "per-phenomenon direction reverses at 1-based frame dt+1, 2dt+1, ... (dt = timeSize)"

#: Difference lag for counting extrema after composition. `*Linear` is a staircase of one step per
#: 2 frames, so with lag 1 every frame looks like a turning point. **Do not use a naive sign(diff).**
_EXTREMA_LAG = 2


def scene_timing(fv: np.ndarray, nx: int, ny: int, n_stack: int, time_size: int,
                 column: int = 0) -> dict:
    """Record of the time axis. Returns **the reversal of each modulation** and **the peaks and troughs after composition** as separate items.

    The former follows from the rule (`TURNING_RULE`). The latter is the sum of the phenomena and does not coincide:
    with etch (falling) + redox (rising), for example, each phenomenon reverses at the frame the rule gives,
    but the extremum of the composed thickness can fall on another frame. **Do not confuse the two.**

    `column` is the FeatureVector column looked at (default 0 = thickness). Only the first pixel of
    each frame is used, so the spatial modulation is fixed and only the time direction remains.
    """
    fv = np.atleast_2d(np.asarray(fv, dtype=float))
    dxdy = int(nx) * int(ny)
    dt = int(time_size)
    nt_full, rem = divmod(int(n_stack), dxdy) if dxdy else (0, 0)
    nt = nt_full + (1 if rem else 0)
    head = fv[::dxdy, int(column)] if dxdy else fv[:, int(column)]

    turning = list(range(dt + 1, nt + 1, dt)) if dt > 0 else []
    if dt > 0 and nt <= dt:
        kind_run = "one-way"
    elif dt > 0:
        kind_run = f"round-trip x {nt / (2 * dt):g}"
    else:
        kind_run = "undefined (timeSize <= 0)"

    # Peaks and troughs after composition. **Not written as a single frame number.**
    # The `*Linear` staircase (one step per 2 frames) sits on top of a smooth component, so near an
    # extremum there is a plateau with a sawtooth of about 0.5 A, and "which frame is the peak" is undetermined to ±1.
    # A moving average (window 2) removes the period-2 component to find the extremum, and the frames
    # of the raw series within the tolerance are returned as a band.
    extrema: list = []
    tol = 0.0
    if head.size >= 4:
        sm = 0.5 * (head[:-1] + head[1:])                 # moving average, window 2
        p2p = float(sm.max() - sm.min())
        tol = max(1e-3 * p2p, 1e-9)
        d = np.diff(sm)
        d[np.abs(d) < 1e-12] = 0.0
        nz = np.flatnonzero(d)
        if nz.size > 1:
            sgn = np.sign(d[nz])
            for k in np.flatnonzero(np.diff(sgn) != 0):
                c = int(nz[k]) + 1                        # 0-based position of the extremum in sm
                kind = "max" if sgn[k] > 0 else "min"
                lo = hi = c
                while lo > 0 and abs(sm[lo - 1] - sm[c]) <= tol:
                    lo -= 1
                while hi < sm.size - 1 and abs(sm[hi + 1] - sm[c]) <= tol:
                    hi += 1
                extrema.append({"kind": kind,
                                "frameBand": [lo + 1, hi + 2],   # 1-based, the 2 frames of the moving average
                                "value": float(head[c:c + 2].max() if kind == "max"
                                               else head[c:c + 2].min())})
    return {
        "nt": int(nt),
        "timeSize": dt,
        "lastFrameCount": int(rem) if rem else None,
        "halfCycles": (nt / dt) if dt > 0 else None,
        "runKind": kind_run,
        "perPhenomenonTurningFrames": turning,
        "perPhenomenonTurningRule": TURNING_RULE,
        "compositeExtrema": extrema,
        "compositeExtremaColumn": int(column),
        "compositeExtremaTolerance": tol,
        "compositeExtremaMethod":
            "window-2 moving average of the first pixel of each frame (kills the 2-frame "
            "Linear staircase exactly), then sign changes of its difference. Reported as a "
            "frame BAND, not a single frame: near an extremum the raw series is a plateau "
            "with a ~0.5 A sawtooth, so the peak frame is not defined to better than +/-1.",
        "note": "perPhenomenonTurningFrames and compositeExtrema are DIFFERENT things. "
                "The first is where each modulation reverses (a rule from the source); the "
                "second is where the summed profile actually peaks (a measurement). They "
                "coincide only for a single-phenomenon scene; with several phenomena the "
                "composite extrema can fall at other frames than the turning frames.",
        "headRange": [float(head.min()), float(head.max())] if head.size else None,
    }
