"""MATLAB-generated benchmark HDF5 (``Dstk`` / ``Xstk`` / ``Drlt``): reader and containers.

These used to live in ``deppro.benchmarks.benchmark_dstk``. They moved into the
body because body modules depend on them (``benchmark_h5_writer`` writes a
``BenchmarkResult``; the Fisher / multi-energy tools document
``load_dstk_data`` + ``create_tm_from_h5`` as their entry point), while
``deppro.benchmarks`` is excluded from the wheel (``pyproject.toml``) and is not
part of the publishable body. ``benchmark_dstk`` re-exports every name below,
so ``from deppro.benchmarks.benchmark_dstk import ...`` keeps working; its own
``load_dstk_data`` additionally supplies the benchmark default path.

Layout conventions (unchanged):

- ``Dstk`` / ``Xstk`` / ``Drlt`` are transposed from MATLAB ``(pixels, elements,
  depth)`` to Python ``(pixels, depth, elements)``; the transfer matrix from
  ``(elements, depth, angles)`` to ``(angles, depth, elements)``.
- ``free`` / ``subst`` / ``linked`` keep MATLAB's 1-based indices; the
  ``*_indices`` properties convert to 0-based.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import h5py
import numpy as np

from deppro.depth_profile import ElementConfig
from deppro.transfer_matrix import TransferMatrix

__all__ = [
    "DstkBenchmarkData",
    "BenchmarkResult",
    "load_dstk_data",
    "create_tm_from_h5",
    "estimate_thickness",
]


@dataclass
class DstkBenchmarkData:
    """Ground truth data loaded from HDF5."""
    Dstk: np.ndarray           # (n_pixels, n_depth, n_elements) ground truth
    Xstk: np.ndarray           # (n_pixels, n_angles, n_elements) input intensities
    Drlt: np.ndarray           # (n_pixels, n_depth, n_elements) MATLAB results
    trans_mat: np.ndarray      # (n_angles, n_depth, n_elements) transfer matrix
    depth: np.ndarray          # (n_depth,)
    angles: np.ndarray         # (n_angles,)
    elements: List[ElementConfig]
    element_names: List[str]
    charges: np.ndarray
    linkedconst: np.ndarray
    step_size: float
    gamma_eff: float
    photon_energy: float
    work_function: float
    n_pixels: int
    n_depth: int
    n_angles: int
    n_elements: int
    # MATLAB 1-based index arrays from H5
    free_matlab: np.ndarray        # e.g. [2,3,4,5] or [2,3,4,5,6]
    subst_matlab: np.ndarray       # e.g. [1]
    linked_matlab: np.ndarray      # e.g. [6] or [7]

    @property
    def free_indices(self) -> List[int]:
        """0-based free element indices for SemiFreeSolverConfig."""
        return [int(f) - 1 for f in self.free_matlab]

    @property
    def linked_indices(self) -> List[int]:
        """0-based linked element indices for SemiFreeSolverConfig."""
        return [int(l) - 1 for l in self.linked_matlab]

    @property
    def substrate_indices(self) -> List[int]:
        """0-based substrate element indices for SemiFreeSolverConfig."""
        return [int(s) - 1 for s in self.subst_matlab]

    @property
    def film_idx(self) -> List[int]:
        """0-based film element indices (free elements) for Dopt extraction."""
        return self.free_indices


@dataclass
class BenchmarkResult:
    """Benchmark metrics."""
    method: str
    n_pixels: int
    # Per-pixel
    rmse_per_pixel: np.ndarray
    thickness_opt: np.ndarray
    thickness_true: np.ndarray
    thickness_error: np.ndarray
    chi2_per_pixel: np.ndarray
    success_per_pixel: np.ndarray
    warm_restarts_per_pixel: np.ndarray
    # Per-element
    rmse_per_element: np.ndarray
    w1_per_element: Optional[np.ndarray]   # W1 distance per element (Å)
    element_names: List[str]
    # Aggregate
    total_rmse: float
    total_w1: Optional[float]              # Mean W1 distance (Å)
    mean_thickness_error: float
    success_rate: float
    # Timing
    total_time: float
    # Optional: full profiles for plotting
    D_opt_all: Optional[np.ndarray] = None
    # Phase 1 (pre-MEM) profiles — Dtrn equivalent
    D_opt_phase1: Optional[np.ndarray] = None
    # Per-pixel Q values for full H5 save
    abruptness_opt: Optional[np.ndarray] = None       # (n_px,) final abruptness
    objective_per_pixel: Optional[np.ndarray] = None   # (n_px,) final objective value
    # Phase 1 Q values (separate from final, for Qtrn vs Qrlt)
    thickness_phase1: Optional[np.ndarray] = None
    abruptness_phase1: Optional[np.ndarray] = None
    chi2_phase1: Optional[np.ndarray] = None
    objective_phase1: Optional[np.ndarray] = None
    # Total iterations across all pixels
    n_iters_total: Optional[int] = None


# ==============================================================================
# Data loading
# ==============================================================================

def load_dstk_data(h5_path: Union[str, Path]) -> DstkBenchmarkData:
    """Load ground truth data from HDF5 and transpose to Python layout."""
    with h5py.File(h5_path, "r") as f:
        # Core data — transpose from MATLAB (pixels, elements, depth) to Python (pixels, depth, elements)
        Dstk = np.transpose(f["Dstk"][:].astype(np.float64), (0, 2, 1))
        Xstk = np.transpose(f["Xstk"][:].astype(np.float64), (0, 2, 1))
        Drlt = np.transpose(f["Drlt"][:].astype(np.float64), (0, 2, 1))

        # Transfer matrix: MATLAB (elements, depth, angles) → Python (angles, depth, elements)
        tm_raw = f["ezdeprof/TransMatObs"][:].astype(np.float64)
        trans_mat = np.transpose(tm_raw, (2, 1, 0))

        # Metadata
        depth = f["ezdeprof/depth"][:].flatten().astype(np.float64)
        angles = f["ezdeprof/thetaobs"][:].flatten().astype(np.float64)
        charges = f["ezdeprof/charge"][:].flatten().astype(np.float64)
        linkedconst = f["ezdeprof/linkedconst"][:].flatten().astype(np.float64)
        step_size = float(f["ezdeprof/stepsize"][:].flatten()[0])
        gamma_eff = float(f["ezdeprof/gammaeff"][:].flatten()[0])
        photon_energy = float(f["ezdeprof/photonenergy"][:].flatten()[0])
        work_function = float(f["ezdeprof/workfunction"][:].flatten()[0])

        # Element info
        elem_names_raw = f["ezdeprof/element"][:].flatten()
        element_names = [x.decode() if isinstance(x, bytes) else str(x) for x in elem_names_raw]

        symbols_raw = f["ezdeprof/elementsymbol"][:].flatten()
        symbols = [x.decode() if isinstance(x, bytes) else str(x) for x in symbols_raw]

        orbitals_raw = f["ezdeprof/orbital"][:].flatten()
        orbitals = [x.decode() if isinstance(x, bytes) else str(x) for x in orbitals_raw]

        compounds_raw = f["ezdeprof/compound"][:].flatten()
        compounds = [x.decode() if isinstance(x, bytes) else str(x) for x in compounds_raw]

        binding_energies = f["ezdeprof/bindingenergy"][:].flatten().astype(np.float64)

        # Index arrays (MATLAB 1-based)
        free_matlab = f["ezdeprof/free"][:].flatten().astype(int)
        subst_matlab = f["ezdeprof/subst"][:].flatten().astype(int)
        linked_matlab = f["ezdeprof/linked"][:].flatten().astype(int)

    # Build ElementConfig list
    elements = []
    for i in range(len(symbols)):
        elements.append(ElementConfig(
            symbol=symbols[i],
            orbital=orbitals[i],
            compound=compounds[i],
            binding_energy=binding_energies[i],
            oxidation_state=charges[i],
        ))

    n_pixels, n_depth, n_elements = Dstk.shape
    n_angles = angles.shape[0]

    return DstkBenchmarkData(
        Dstk=Dstk, Xstk=Xstk, Drlt=Drlt,
        trans_mat=trans_mat,
        depth=depth, angles=angles,
        elements=elements, element_names=element_names,
        charges=charges, linkedconst=linkedconst,
        step_size=step_size, gamma_eff=gamma_eff,
        photon_energy=photon_energy, work_function=work_function,
        n_pixels=n_pixels, n_depth=n_depth,
        n_angles=n_angles, n_elements=n_elements,
        free_matlab=free_matlab, subst_matlab=subst_matlab,
        linked_matlab=linked_matlab,
    )


# ==============================================================================
# TransferMatrix injection
# ==============================================================================

def create_tm_from_h5(data: DstkBenchmarkData) -> TransferMatrix:
    """Create TransferMatrix with pre-computed matrix injected (bypass compute()).

    If data has been remapped to a non-uniform grid (via remap_data_to_grid),
    the resulting TransferMatrix will have step_sizes set for non-uniform support.
    """
    # Check if grid is non-uniform by looking at step size variance
    diffs = np.diff(data.depth)
    is_nonuniform = len(diffs) > 1 and (np.max(diffs) - np.min(diffs)) > 0.01

    if is_nonuniform:
        tm = TransferMatrix.from_nonuniform_grid(
            depth_grid=data.depth,
            angles=data.angles,
            photon_energy=data.photon_energy,
            work_function=data.work_function,
            gamma_eff=data.gamma_eff,
        )
    else:
        tm = TransferMatrix(
            depth=data.depth,
            step_size=data.step_size,
            angles=data.angles,
            photon_energy=data.photon_energy,
            work_function=data.work_function,
            gamma_eff=data.gamma_eff,
        )

    # Inject pre-computed transfer matrix directly
    tm.trans_mat = data.trans_mat  # (n_angles, n_depth, n_elements)
    tm.n_elements = data.n_elements
    return tm


# ==============================================================================
# Thickness estimation
# ==============================================================================

def estimate_thickness(D: np.ndarray, depth: np.ndarray, substrate_idx: int = 0) -> float:
    """Estimate film thickness from substrate crossing 0.5.

    Args:
        D: Depth profile (n_depth, n_elements)
        depth: Depth array (n_depth,)
        substrate_idx: Index of substrate element

    Returns:
        Estimated thickness in Angstroms
    """
    D_sub = D[:, substrate_idx]
    crossings = np.where(D_sub >= 0.5)[0]
    if len(crossings) == 0:
        return depth[-1]
    idx = crossings[0]
    if idx > 0:
        z0, z1 = depth[idx - 1], depth[idx]
        d0, d1 = D_sub[idx - 1], D_sub[idx]
        denom = d1 - d0
        if abs(denom) < 1e-12:
            return (z0 + z1) / 2
        return z0 + (0.5 - d0) / denom * (z1 - z0)
    return depth[0]
