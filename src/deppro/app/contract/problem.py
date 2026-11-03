"""P1 contract: ``ProblemSpec``, which holds only observations, and ``GroundTruth`` as a separate object.

The GT is isolated from the solver input (design document §5). ``ProblemSpec`` holds no D at all.
Reading is h5py read-only. Arrays are in Python layout ``(px, depth, elem)``, T is
``(angle, depth, elem)`` (the same transposition as ``deppro.io.dstk.load_dstk_data``).
Role indices are converted to 0-based.

The roles (Dstk = GT, Xstk = observation, Drlt = existing reconstruction) are **assumed** from the names;
``roles_declared=False`` is carried until the producer manifest's dataset_roles settles them. The GT is not normalized to sum 1.
"""
from __future__ import annotations

import dataclasses
import json
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np

from .dataset_files import (
    decode_strings, discover_files, matlab_index_list, parse_dataset_folder,
)

ROLES_ASSUMED = {
    "Dstk": "ground truth D(z) (assumed from name; producer declaration pending)",
    "Xstk": "observed X(theta), total-normalised per angle (assumed)",
    "Drlt": "existing reconstruction written by a previous run (assumed)",
    "ezdeprof/TransMatObs": "transfer matrix of pixel 0 (Xobs/Dobs identity); "
                            "not the full generator forward",
}


# --------------------------------------------------------------------------
@dataclass
class DatasetRef:
    folder: str
    folder_rel: Optional[str]
    main_h5: str
    family: Optional[str]
    nx: int
    ny: int
    nt: int
    ncomp: int
    variant: Optional[str]
    fingerprint: str                # cheap stat fingerprint, NOT a content hash
    fingerprint_kind: str
    manifest_present: bool
    main_rule: str
    layout: str = "rectangular_grid"      # rectangular_grid | rectangular_grid_partial | flat_samples


@dataclass
class Axes:
    nx: int
    ny: int
    nt: int
    n_depth: int
    n_angles: int
    n_elem: int
    depth_A: np.ndarray
    angles_deg: np.ndarray
    step_A: float
    flatten_order: str = ("frame-major assumed: pixel = t*nx*ny + within-frame index; "
                          "within-frame order unverified (no producer sentinel)")
    # Pixel count of the last frame. px_per_frame (complete grid) if stack is a multiple of nx*ny; otherwise
    # (prime stack and the like) the remainder. nt counts that truncated frame too (same convention as the producer's layout).
    last_frame_count: Optional[int] = None
    # Kind of grid (same vocabulary as the producer's layout.type). flat_samples has no grid (measured data:
    # stack = number of measurement points). The whole sample sequence is then merely **represented** as one
    # frame (nx = stack, ny = 1, nt = 1); that is not a claim of a grid.
    layout: str = "rectangular_grid"

    @property
    def px_per_frame(self) -> int:
        return self.nx * self.ny

    @property
    def complete(self) -> bool:
        return self.last_frame_count is None or self.last_frame_count == self.px_per_frame

    @property
    def n_px_total(self) -> int:
        last = self.px_per_frame if self.last_frame_count is None else self.last_frame_count
        return (self.nt - 1) * self.px_per_frame + last if self.nt else 0


@dataclass
class Elements:
    symbols: List[str]
    charges: np.ndarray
    free: List[int]          # 0-based
    linked: List[int]
    const: List[int]
    subst: List[int]
    linked_const: Optional[np.ndarray]
    orbitals: Optional[List[str]] = None
    compounds: Optional[List[str]] = None
    binding_energies: Optional[np.ndarray] = None

    def element_configs(self) -> list:
        """List of ``ElementConfig`` for the solver (built as in deppro.io.dstk.load_dstk_data)."""
        from deppro.depth_profile import ElementConfig
        if self.orbitals is None or self.compounds is None or self.binding_energies is None:
            raise ValueError("orbital/compound/bindingenergy were not read, so ElementConfig cannot be built")
        return [ElementConfig(symbol=self.symbols[i], orbital=self.orbitals[i],
                              compound=self.compounds[i],
                              binding_energy=float(self.binding_energies[i]),
                              oxidation_state=float(self.charges[i]))
                for i in range(len(self.symbols))]

    @property
    def labels(self) -> List[str]:
        out = []
        for s, q in zip(self.symbols, self.charges):
            qi = int(round(float(q)))
            out.append(f"{s}({'+' if qi > 0 else ''}{qi})")
        return out

    @property
    def n_elem(self) -> int:
        return len(self.symbols)


@dataclass
class ProblemSpec:
    """What may be passed to a solver. Contains no GT."""
    ref: DatasetRef
    axes: Axes
    elements: Elements
    X: np.ndarray                    # (n_px, n_angles, n_elem)
    pixel_ids: np.ndarray            # (n_px,) flat indices into the file
    stored_T: np.ndarray             # (n_angles, n_depth, n_elem)
    lambda_basic: Optional[np.ndarray]
    gamma_eff: Optional[float]
    noise: Optional[float]
    photon_energy: Optional[float] = None    # first value of ezdeprof/photonenergy (as in load_dstk_data)
    work_function: Optional[float] = None
    roles_declared: bool = False
    roles_assumed: Dict[str, str] = field(default_factory=lambda: dict(ROLES_ASSUMED))

    @property
    def n_px(self) -> int:
        return int(self.pixel_ids.size)

    def frame_of(self, pixel_ids: Optional[np.ndarray] = None) -> np.ndarray:
        ids = self.pixel_ids if pixel_ids is None else np.asarray(pixel_ids)
        return ids // self.axes.px_per_frame

    def summary(self) -> Dict[str, Any]:
        """Summary for the manifest (arrays by shape only)."""
        return {
            "dataset": dataclasses.asdict(self.ref),
            "axes": {"nx": self.axes.nx, "ny": self.axes.ny, "nt": self.axes.nt,
                     "n_depth": self.axes.n_depth, "n_angles": self.axes.n_angles,
                     "n_elem": self.axes.n_elem, "step_A": self.axes.step_A,
                     "depth_A": self.axes.depth_A.tolist(),
                     "angles_deg": self.axes.angles_deg.tolist(),
                     "flatten_order": self.axes.flatten_order,
                     # pixel count of the last frame and the total (differs from nt*nx*ny on a partial grid)
                     "last_frame_count": self.axes.last_frame_count,
                     "complete": self.axes.complete, "n_px_total": self.axes.n_px_total,
                     "layout": self.axes.layout},
            "elements": {"labels": self.elements.labels, "free": self.elements.free,
                         "linked": self.elements.linked, "const": self.elements.const,
                         "subst": self.elements.subst},
            "n_px": self.n_px, "X_shape": list(self.X.shape),
            "stored_T_shape": list(self.stored_T.shape),
            "noise": self.noise, "gamma_eff": self.gamma_eff,
            "roles_declared": self.roles_declared, "roles_assumed": self.roles_assumed,
        }


@dataclass
class GroundTruth:
    """Received only by the evaluator. Not passed to solvers or routers."""
    pixel_ids: np.ndarray
    D: np.ndarray                    # (n_px, n_depth, n_elem), as stored (not renormalised)
    source: str = "Dstk (assumed GT; producer declaration pending)"


# --------------------------------------------------------------------------
def cheap_fingerprint(path: Path) -> Tuple[str, str]:
    """Cheap fingerprint from name, size and mtime. Not a content hash (that is done separately in qualification)."""
    st = Path(path).stat()
    h = hashlib.sha256(f"{Path(path).name}|{st.st_size}|{st.st_mtime_ns}".encode()).hexdigest()
    return h[:16], "sha256(name|size|mtime_ns)[:16]"


def select_pixel_ids(n_px_total: int, px_per_frame: int, *, frames: Optional[Sequence[int]] = None,
                     pixel_ids: Optional[Sequence[int]] = None,
                     max_px_per_frame: Optional[int] = None) -> np.ndarray:
    """Deterministic selection of pixels for checks and experiments. frames are contiguous from the frame start; pixel_ids are explicit.

    When the last frame is incomplete (n_px_total is not a multiple of px_per_frame), that frame's
    range is cut at the existing pixels. A frame that does not exist is an IndexError, not silently empty.
    """
    if pixel_ids is not None:
        ids = np.unique(np.asarray(pixel_ids, dtype=np.int64))
    elif frames is not None:
        cnt = px_per_frame if max_px_per_frame is None else min(px_per_frame, max_px_per_frame)
        parts = []
        for f in frames:
            start = int(f) * px_per_frame
            if f < 0 or start >= n_px_total:
                raise IndexError(f"frame {f} out of range: {n_px_total} pixels hold "
                                 f"{-(-n_px_total // px_per_frame)} frames of {px_per_frame}")
            parts.append(np.arange(start, min(start + cnt, n_px_total), dtype=np.int64))
        ids = np.unique(np.concatenate(parts)) if parts else np.zeros(0, np.int64)
    else:
        ids = np.arange(n_px_total, dtype=np.int64)
    if ids.size and (ids.min() < 0 or ids.max() >= n_px_total):
        raise IndexError(f"pixel ids out of range [0, {n_px_total})")
    return ids


def _read_rows(dset: h5py.Dataset, ids: np.ndarray) -> np.ndarray:
    """Read sorted unique ids as slices per contiguous run (faster than point selection, and order is preserved)."""
    if ids.size == 0:
        return np.zeros((0,) + dset.shape[1:], dtype=np.float64)
    breaks = np.nonzero(np.diff(ids) != 1)[0] + 1
    runs = np.split(ids, breaks)
    parts = [np.asarray(dset[int(r[0]):int(r[-1]) + 1], dtype=np.float64) for r in runs]
    return np.concatenate(parts, axis=0)


def _scalar(g: h5py.Group, key: str) -> Optional[float]:
    return float(np.asarray(g[key][()]).ravel()[0]) if key in g else None


def _declared_layout(manifest_path: Path) -> Optional[Dict[str, Any]]:
    """The layout of the co-located manifest (None when missing, unreadable, or not an object)."""
    try:
        data = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    layout = data.get("layout") if isinstance(data, dict) else None
    return layout if isinstance(layout, dict) else None


def _declared_partial_matches(layout: Optional[Dict[str, Any]], nx: int, ny: int, n_px: int):
    """Check the partial-grid declaration against the contents. None = not declared, True = matches, str = reason for the mismatch."""
    if not layout or layout.get("type") != "rectangular_grid_partial":
        return None
    ppf = nx * ny
    n_full, rem = divmod(n_px, ppf)
    want = {"nx": nx, "ny": ny, "nt": n_full + 1, "lastFrameCount": rem}
    bad = {k: (layout.get(k), v) for k, v in want.items() if k in layout and int(layout[k]) != v}
    return True if not bad else ", ".join(f"{k}={got} but file implies {v}" for k, (got, v) in bad.items())


def load_problem(folder: Path, *, frames: Optional[Sequence[int]] = None,
                 pixel_ids: Optional[Sequence[int]] = None,
                 max_px_per_frame: Optional[int] = None, root: Optional[Path] = None,
                 with_ground_truth: bool = True,
                 allow_partial: bool = False,
                 allow_flat: bool = False) -> Tuple[ProblemSpec, Optional[GroundTruth]]:
    """Read ProblemSpec (observations) and GroundTruth (separate) from a dataset folder. Read-only.

    Data whose stack is not a multiple of nx*ny (incomplete last frame) is read only when the co-located
    manifest declares ``layout.type == "rectangular_grid_partial"`` consistently with the contents, or the
    caller says ``allow_partial=True`` explicitly. Otherwise ValueError (no silent truncation, no silent
    squeezing into a rectangle; the same convention as stack_coordinates in the producer adapter).

    Data without a grid (measured data, SimOrExp=0: stack is the number of measurement points and the dxdydt grid is unrelated)
    is read when the manifest declares ``layout.type == "flat_samples"`` or the caller says ``allow_flat=True``,
    and the whole sample sequence is represented as one frame (nx = stack, ny = 1, nt = 1).
    """
    folder = Path(folder).expanduser().resolve()
    spec = parse_dataset_folder(folder)
    files = discover_files(folder)
    if len(files["main"]) != 1:
        raise FileNotFoundError(
            f"main H5 not identifiable in {folder} (rule={files['rule']}, "
            f"candidates={[p.name for p in files['main'] + files['derived']]})")
    main = files["main"][0]
    fp, fp_kind = cheap_fingerprint(main)
    rel = None
    if root is not None:
        r = Path(root).expanduser().resolve()
        if folder.is_relative_to(r):
            rel = str(folder.relative_to(r))
    with h5py.File(main, "r") as f:
        g = f["ezdeprof"]
        D_ds, X_ds = f["Dstk"], f["Xstk"]
        P, E, Z = D_ds.shape
        A = X_ds.shape[2]
        dxdydt = [int(v) for v in np.asarray(g["dxdydt"][()]).ravel()]
        layout = _declared_layout(files["manifest"])
        ltype = (layout or {}).get("type")
        if allow_flat and ltype not in (None, "flat_samples"):
            raise ValueError(f"allow_flat=True but the manifest declares layout {ltype!r} (a grid); "
                             "do not override a declared grid")
        if ltype == "flat_samples" or allow_flat:
            # No grid: represent the whole sample sequence as one frame (not a claim of a grid)
            nx, ny, nt, last, lay = P, 1, 1, P, "flat_samples"
        else:
            nx, ny = (spec.nx, spec.ny) if spec is not None else (dxdydt[0], dxdydt[1])
            if nx * ny == 0:
                raise ValueError(f"grid nx*ny is zero (folder={spec}, dxdydt={dxdydt}); "
                                 "for grid-less data pass allow_flat=True or declare layout flat_samples")
            n_full, rem = divmod(P, nx * ny)     # nt from the contents. dxdydt[2] is TimeSize, not nt
            if rem == 0:
                nt, last, lay = n_full, nx * ny, "rectangular_grid"
            else:
                declared = _declared_partial_matches(layout, nx, ny, P)
                if declared is not None and declared is not True:
                    raise ValueError(f"manifest layout declares a partial grid that does not match the file: {declared}")
                if declared is None and not allow_partial:
                    raise ValueError(
                        f"pixel count {P} is not a multiple of nx*ny={nx * ny} (last frame would hold {rem}). "
                        "No manifest layout 'rectangular_grid_partial' declares this; pass allow_partial=True "
                        "to read it as a partial grid explicitly, or allow_flat=True if it is grid-less "
                        "experiment data (SimulationOrExperiment=0: stack = number of measured points)")
                nt, last, lay = n_full + 1, rem, "rectangular_grid_partial"
        depth = np.asarray(g["depth"][()], dtype=np.float64).ravel()
        theta = np.asarray(g["thetaobs"][()], dtype=np.float64).ravel()
        step = _scalar(g, "stepsize") or float(np.median(np.diff(depth)))
        symbols = list(decode_strings(g["elementsymbol"][()]))
        charges = np.asarray(g["charge"][()], dtype=np.float64).ravel()
        roles = {k: [i - 1 for i in matlab_index_list(g[k][()])] if k in g else []
                 for k in ("free", "linked", "const", "subst")}
        linked_const = (np.asarray(g["linkedconst"][()], dtype=np.float64).ravel()
                        if "linkedconst" in g else None)
        T = np.transpose(np.asarray(g["TransMatObs"][()], dtype=np.float64), (2, 1, 0))
        lam_b = (np.asarray(g["lambdabasic"][()], dtype=np.float64).reshape(E, E)
                 if "lambdabasic" in g else None)
        orbitals = list(decode_strings(g["orbital"][()])) if "orbital" in g else None
        compounds = list(decode_strings(g["compound"][()])) if "compound" in g else None
        be = (np.asarray(g["bindingenergy"][()], dtype=np.float64).ravel()
              if "bindingenergy" in g else None)
        photon = (float(np.asarray(g["photonenergy"][()], dtype=np.float64).ravel()[0])
                  if "photonenergy" in g else None)
        wf = _scalar(g, "workfunction")
        ids = select_pixel_ids(P, nx * ny, frames=frames, pixel_ids=pixel_ids,
                               max_px_per_frame=max_px_per_frame)
        X = np.transpose(_read_rows(X_ds, ids), (0, 2, 1))          # (n, A, E)
        gt = None
        if with_ground_truth:
            D = np.transpose(_read_rows(D_ds, ids), (0, 2, 1))      # (n, Z, E), as stored
            gt = GroundTruth(pixel_ids=ids.copy(), D=D)
        ref = DatasetRef(
            folder=str(folder), folder_rel=rel, main_h5=main.name,
            family=spec.family if spec else None, nx=nx, ny=ny, nt=nt, ncomp=E,
            variant=spec.variant if spec else None, fingerprint=fp, fingerprint_kind=fp_kind,
            manifest_present=files["manifest"].exists(), main_rule=files["rule"], layout=lay)
        axes = Axes(nx=nx, ny=ny, nt=nt, n_depth=Z, n_angles=A, n_elem=E,
                    depth_A=depth, angles_deg=theta, step_A=step, last_frame_count=last, layout=lay)
        elements = Elements(symbols=symbols, charges=charges, free=roles["free"],
                            linked=roles["linked"], const=roles["const"], subst=roles["subst"],
                            linked_const=linked_const, orbitals=orbitals, compounds=compounds,
                            binding_energies=be)
        problem = ProblemSpec(ref=ref, axes=axes, elements=elements, X=X, pixel_ids=ids,
                              stored_T=T, lambda_basic=lam_b, gamma_eff=_scalar(g, "gammaeff"),
                              noise=_scalar(g, "noise"), photon_energy=photon, work_function=wf,
                              roles_declared=False)   # the manifest's dataset_roles contract is not agreed yet
    return problem, gt


def load_existing_result(folder: Path, pixel_ids: np.ndarray, dataset: str = "Drlt") -> np.ndarray:
    """Read an existing reconstruction (``Drlt`` and the like) in Python layout (n, depth, elem). Not an independent sample."""
    folder = Path(folder).expanduser().resolve()
    files = discover_files(folder)
    if len(files["main"]) != 1:
        raise FileNotFoundError(f"main H5 not identifiable in {folder}")
    ids = np.unique(np.asarray(pixel_ids, dtype=np.int64))
    with h5py.File(files["main"][0], "r") as f:
        if dataset not in f:
            raise KeyError(f"{dataset} not in {files['main'][0].name}")
        return np.transpose(_read_rows(f[dataset], ids), (0, 2, 1))
