"""
HAXPES Analysis Pipeline
=========================

High-level orchestration: Config -> Load -> Preprocess -> Solve.
"""
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from time import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from deppro.depth_profile import ElementConfig
from deppro.inverse_solver import (
    SemiFreeSolver,
    SemiFreeSolverConfig,
    SemiFreeSolverResult,
    SolverConfig,
)
from deppro.transfer_matrix import TransferMatrix

from .config import (
    HAXPESExperimentConfig,
    HAXPESInstrumentConfig,
    HAXPESSolverConfig,
    CALTIOSI_EXPERIMENT,
    FIX_IBW_INSTRUMENT,
)
from .preprocessor import (
    compute_L_factors,
    compute_si_normalized_rrmse,
    subtract_background,
)

logger = logging.getLogger(__name__)


# ============================================================
# Result dataclass
# ============================================================

@dataclass
class PipelineResult:
    """Result from HAXPES depth profiling pipeline.

    Attributes:
        D_opt: (n_depth, n_elements) optimized depth profile.
        I_opt: (n_angles, n_elements) forward model prediction.
        thickness: Optimized film thickness (Angstroms).
        abruptness: Optimized interface width (Angstroms).
        rrmse: Si-normalized relative RMSE.
        rmse: Raw RMSE.
        depth: (n_depth,) depth grid.
        angles_used: (n_angles,) angles used in solver.
        Ismp_raw: (n_elements, n_bins) raw peak areas.
        Xobs: (n_elements, n_bins) cross-section corrected Ismp.
        sigmas: (n_elements,) cross sections.
        elapsed_seconds: Processing time.
        solver_result: Original SemiFreeSolverResult.
    """
    D_opt: np.ndarray
    I_opt: np.ndarray
    thickness: float
    abruptness: float
    rrmse: float
    rmse: float
    depth: np.ndarray
    angles_used: np.ndarray
    Ismp_raw: np.ndarray
    Xobs: np.ndarray
    sigmas: np.ndarray
    elapsed_seconds: float
    solver_result: SemiFreeSolverResult = field(repr=False)


# ============================================================
# Core: single-pixel depth profiling
# ============================================================

def process_single_pixel(
    pixel_data: Dict[str, np.ndarray],
    elements: List[ElementConfig],
    angles: np.ndarray,
    sigmas: np.ndarray,
    alpha: float = 0.01,
    thickness_init: float = 240.0,
    elem_names: Optional[List[str]] = None,
    n_depth: int = 20,
    totaldepth: float = 600.0,
    photon_energy: float = 9251.74,
    free_indices: Optional[List[int]] = None,
    linked_indices: Optional[List[int]] = None,
    substrate_indices: Optional[List[int]] = None,
    non_dipole_L: Optional[np.ndarray] = None,
) -> Tuple[Optional[np.ndarray], float, float]:
    """Run full depth profiling pipeline for one pseudo-pixel.

    Includes BG subtraction, peak integration, cross section correction,
    TransferMatrix construction, and SemiFreeSolver optimization.

    Parameters
    ----------
    pixel_data : dict
        {elem: (n_energy, n_angles)} binned spectra.
    elements : list
        ElementConfig list.
    angles : ndarray
        (n_angles,) emission angles.
    sigmas : ndarray
        (n_elements,) pre-computed total cross sections.
    alpha : float
        L1 regularization strength.
    thickness_init : float
        Initial thickness estimate (Angstroms).
    elem_names : list, optional
        Ordered element names. Default: ['Si', 'Ti', 'Al', 'C', 'O'].
    n_depth : int
        Number of depth grid points.
    totaldepth : float
        Maximum analysis depth (Angstroms).
    photon_energy : float
        Excitation energy (eV).
    free_indices : list, optional
        Element indices freely optimized. Default: [1, 2, 3].
    linked_indices : list, optional
        Charge-balanced element indices. Default: [4].
    substrate_indices : list, optional
        Substrate element indices. Default: [0].
    non_dipole_L : ndarray, optional
        (n_elements, n_angles) non-dipole correction factors.
        If provided, sigma_eff = sigmas * L for angle-dependent correction.

    Returns
    -------
    D_opt : ndarray or None
        (n_depth, n_elements) depth profile, None on failure.
    thickness : float
        Optimized thickness.
    rrmse : float
        Si-normalized rRMSE, 999.0 on failure.
    """
    if elem_names is None:
        elem_names = ['Si', 'Ti', 'Al', 'C', 'O']
    if free_indices is None:
        free_indices = [1, 2, 3]
    if linked_indices is None:
        linked_indices = [4]
    if substrate_indices is None:
        substrate_indices = [0]

    try:
        # Background subtract + integrate → Ismp
        Ismp_list = []
        for elem in elem_names:
            if elem not in pixel_data:
                Ismp_list.append(np.zeros(angles.shape[0]))
                continue
            spectrum = pixel_data[elem]  # (n_energy, n_angles)
            bg_sub = subtract_background(spectrum, 'linear_endpoints', 0.05)
            peak_areas = bg_sub.sum(axis=0)
            Ismp_list.append(peak_areas)

        Ismp = np.array(Ismp_list)  # (n_elements, n_angles)

        # Cross section correction (sigma only — L goes to transfer matrix)
        Xobs = Ismp / sigmas[:, np.newaxis]

        if np.all(Xobs == 0) or np.any(np.isnan(Xobs)):
            return None, 0.0, 999.0

        # Transfer matrix (with non-dipole L if provided)
        stepsize = totaldepth / n_depth
        depth = np.arange(1, n_depth + 1) * stepsize

        # non_dipole_L: (n_elements, n_angles) → transpose for TM
        L_tm = non_dipole_L.T if non_dipole_L is not None else None

        tm = TransferMatrix(
            depth=depth,
            step_size=stepsize,
            angles=angles,
            photon_energy=photon_energy,
        )
        D_init = np.ones((n_depth, len(elements))) / len(elements)
        tm.compute(D_init, elements, L_correction=L_tm)

        solver_config = SolverConfig(
            method='L1',
            alpha=alpha,
            scipy_method='SLSQP',
            enable_fallback=False,
            use_xreg_weighting=True,
            tol=1e-6,
            max_iter=200,
            degen_depth_threshold=0.01,
        )
        semi_config = SemiFreeSolverConfig(
            free_indices=free_indices,
            linked_indices=linked_indices,
            substrate_indices=substrate_indices,
            thickness_init=thickness_init,
            thickness_bounds=(10.0, 500.0),
            abruptness_init=5.0,
            optimize_abruptness=False,
            use_legacy_sum_constraint=True,
        )
        solver = SemiFreeSolver(
            transfer_matrix=tm,
            config=solver_config,
            semi_free_config=semi_config,
            D_prior=None,
            elements=elements,
        )

        I_obs = Xobs.T  # (n_angles, n_elements)
        n_film = len(free_indices)
        film_conc_uniform = np.ones((n_depth, n_film)) / (n_film + 1)
        x0 = solver.build_x0(film_conc=film_conc_uniform)

        result = solver.solve(I_obs, verbose=False, x0_override=x0)

        rrmse = compute_si_normalized_rrmse(
            I_obs, result.I_opt,
            substrate_index=substrate_indices[0])

        return result.D_opt, result.thickness, rrmse

    except Exception:
        return None, 0.0, 999.0


def _worker(args):
    """Multiprocessing worker wrapper for process_single_pixel."""
    idx, pixel_data, elements, angles, sigmas, alpha, thk_init = args
    D_opt, thickness, rrmse = process_single_pixel(
        pixel_data, elements, angles, sigmas,
        alpha=alpha, thickness_init=thk_init,
    )
    return idx, D_opt, thickness, rrmse


# ============================================================
# Run depth profiling (single-point, from pre-computed Ismp)
# ============================================================

def run_depth_profiling(
    Xobs: np.ndarray,
    angles: np.ndarray,
    elements: List[ElementConfig],
    solver_config: Optional[HAXPESSolverConfig] = None,
    experiment: Optional[HAXPESExperimentConfig] = None,
    photon_energy: float = 9251.74,
    alpha: Optional[float] = None,
    thickness_init: Optional[float] = None,
    angle_lo: Optional[float] = None,
    angle_hi: Optional[float] = None,
    verbose: bool = True,
    L_correction: Optional[np.ndarray] = None,
) -> Tuple['SemiFreeSolverResult', np.ndarray, np.ndarray, float]:
    """Run SemiFreeSolver on pre-computed cross-section-corrected Ismp.

    Parameters
    ----------
    Xobs : ndarray
        (n_elements, n_angles) cross-section corrected intensities.
    angles : ndarray
        (n_angles,) emission angles in degrees.
    elements : list
        ElementConfig list.
    solver_config : HAXPESSolverConfig, optional
        Solver configuration. Uses defaults if None.
    experiment : HAXPESExperimentConfig, optional
        Experiment config for element indices.
    photon_energy : float
        Excitation energy (eV).
    alpha : float, optional
        Override solver_config.alpha.
    thickness_init : float, optional
        Override solver_config.thickness_init.
    angle_lo, angle_hi : float, optional
        Angle trimming range.
    verbose : bool
        Print solver progress.

    Returns
    -------
    result : SemiFreeSolverResult
    depth : ndarray
    angles_used : ndarray
    rrmse : float
    """
    if solver_config is None:
        solver_config = HAXPESSolverConfig()
    if experiment is None:
        experiment = CALTIOSI_EXPERIMENT

    _alpha = alpha if alpha is not None else solver_config.alpha
    _thk = (thickness_init if thickness_init is not None
            else solver_config.thickness_init)

    # Angle trimming
    L_trim = L_correction
    if angle_lo is not None or angle_hi is not None:
        lo = angle_lo if angle_lo is not None else angles.min()
        hi = angle_hi if angle_hi is not None else angles.max()
        mask = (angles >= lo) & (angles <= hi)
        angles_trim = angles[mask]
        Xobs_trim = Xobs[:, mask]
        if L_correction is not None:
            L_trim = L_correction[mask]
        logger.info(f"Angle trim: {lo:.0f}-{hi:.0f} deg "
                    f"({mask.sum()}/{len(angles)} bins)")
    elif solver_config.angle_trim is not None:
        lo, hi = solver_config.angle_trim
        mask = (angles >= lo) & (angles <= hi)
        angles_trim = angles[mask]
        Xobs_trim = Xobs[:, mask]
        if L_correction is not None:
            L_trim = L_correction[mask]
    else:
        angles_trim = angles
        Xobs_trim = Xobs

    n_depth = solver_config.n_depth
    stepsize = solver_config.stepsize
    depth = solver_config.depth

    tm = TransferMatrix(
        depth=depth,
        step_size=stepsize,
        angles=angles_trim,
        photon_energy=photon_energy,
    )
    D_init = np.ones((n_depth, len(elements))) / len(elements)
    tm.compute(D_init, elements, L_correction=L_trim)

    sc = SolverConfig(
        method=solver_config.method,
        alpha=_alpha,
        scipy_method=solver_config.scipy_method,
        enable_fallback=False,
        use_xreg_weighting=True,
        tol=solver_config.tol,
        max_iter=solver_config.max_iter,
    )

    semi_config = SemiFreeSolverConfig(
        free_indices=experiment.free_indices,
        linked_indices=experiment.linked_indices,
        substrate_indices=experiment.substrate_indices,
        thickness_init=_thk,
        thickness_bounds=solver_config.thickness_bounds,
        abruptness_init=solver_config.abruptness_init,
        optimize_abruptness=solver_config.optimize_abruptness,
    )

    solver = SemiFreeSolver(
        transfer_matrix=tm,
        config=sc,
        semi_free_config=semi_config,
        D_prior=None,
        elements=elements,
    )

    I_obs = Xobs_trim.T  # (n_angles, n_elements)

    n_film = experiment.n_film
    film_conc_uniform = np.ones((n_depth, n_film)) / (n_film + 1)
    x0 = solver.build_x0(film_conc=film_conc_uniform)

    logger.info(f"SemiFreeSolver {solver_config.method}: "
                f"alpha={_alpha}, thk_init={_thk:.0f}A")

    result = solver.solve(I_obs, verbose=verbose, x0_override=x0)

    rrmse = compute_si_normalized_rrmse(
        I_obs, result.I_opt,
        substrate_index=experiment.substrate_indices[0])

    logger.info(f"  Thickness: {result.thickness:.1f} A, "
                f"rRMSE: {rrmse:.2%}")

    return result, depth, angles_trim, rrmse


# ============================================================
# 3D Map Pipeline
# ============================================================

def run_3d_map(
    pixel_args: List[tuple],
    nx: int,
    ny: int,
    n_depth: int = 20,
    n_elements: int = 5,
    n_workers: int = 4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Process spatial map with multiprocessing.

    Parameters
    ----------
    pixel_args : list
        List of (idx, pixel_data, elements, angles, sigmas, alpha, thk_init).
    nx, ny : int
        Spatial grid dimensions.
    n_depth : int
        Number of depth grid points.
    n_elements : int
        Number of elements.
    n_workers : int
        Number of parallel workers.

    Returns
    -------
    D_map : ndarray
        (nx*ny, n_depth, n_elements).
    thk_map : ndarray
        (nx*ny,).
    rrmse_map : ndarray
        (nx*ny,).
    """
    n_total = nx * ny
    D_map = np.zeros((n_total, n_depth, n_elements))
    thk_map = np.zeros(n_total)
    rrmse_map = np.full(n_total, 999.0)

    t0 = time()

    if n_workers <= 1:
        for args in pixel_args:
            idx, D_opt, thickness, rrmse = _worker(args)
            if D_opt is not None:
                D_map[idx] = D_opt
                thk_map[idx] = thickness
                rrmse_map[idx] = rrmse
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_worker, args): args[0]
                       for args in pixel_args}
            done_count = 0
            for future in as_completed(futures):
                idx, D_opt, thickness, rrmse = future.result()
                if D_opt is not None:
                    D_map[idx] = D_opt
                    thk_map[idx] = thickness
                    rrmse_map[idx] = rrmse
                done_count += 1
                if done_count % max(1, n_total // 10) == 0:
                    logger.info(f"  {done_count}/{n_total} pixels done")

    elapsed = time() - t0
    n_converged = np.sum(rrmse_map < 900)
    logger.info(f"3D map: {n_converged}/{n_total} converged in {elapsed:.1f}s")

    return D_map, thk_map, rrmse_map


# ============================================================
# HAXPESPipeline class
# ============================================================

class HAXPESPipeline:
    """High-level pipeline for HAXPES depth profiling.

    Usage::

        from deppro.haxpes import (
            HAXPESPipeline, FIX_IBW_INSTRUMENT, CALTIOSI_EXPERIMENT,
        )

        pipeline = HAXPESPipeline(
            instrument=FIX_IBW_INSTRUMENT,
            experiment=CALTIOSI_EXPERIMENT,
        )
        result = pipeline.run_fix_ibw(FIX_IBW_DIR)
    """

    def __init__(
        self,
        instrument: Optional[HAXPESInstrumentConfig] = None,
        experiment: Optional[HAXPESExperimentConfig] = None,
        solver_config: Optional[HAXPESSolverConfig] = None,
    ):
        self.instrument = instrument or FIX_IBW_INSTRUMENT
        self.experiment = experiment or CALTIOSI_EXPERIMENT
        self.solver_config = solver_config or HAXPESSolverConfig()
        self._elements = self.experiment.make_elements()
        self._sigmas: Optional[np.ndarray] = None

    def _ensure_sigmas(self) -> np.ndarray:
        """Lazily compute cross sections."""
        if self._sigmas is None:
            from .preprocessor import apply_cross_section_correction
            dummy = np.ones((len(self._elements), 1))
            _, self._sigmas = apply_cross_section_correction(
                dummy, self._elements,
                self.instrument.photon_energy,
                source=self.solver_config.cross_section_source,
                poly_order=self.solver_config.cross_section_poly_order,
            )
        return self._sigmas

    def run_fix_ibw(
        self,
        base_path: Path,
        measurements: Tuple[int, ...] = (1, 2),
    ) -> PipelineResult:
        """End-to-end: IBW files -> depth profile.

        Parameters
        ----------
        base_path : Path
            Path to IBW data directory.
        measurements : tuple
            Measurement indices.

        Returns
        -------
        result : PipelineResult
        """
        from .loader import load_fix_ibw, sum_and_bin
        from .preprocessor import (
            integrate_peak_areas,
            apply_cross_section_correction,
        )

        t0 = time()
        sc = self.solver_config

        # Load
        data_dict = load_fix_ibw(base_path, self.experiment, measurements)

        # Sum and bin
        (binned_spectra, energy_axes, bin_edges, bin_centers,
         emission_angles, channels_per_bin) = sum_and_bin(
            data_dict, self.experiment.elem_names,
            n_bins=sc.n_bins,
            angle_lo=sc.angle_range[0],
            angle_hi=sc.angle_range[1],
            polar_offset=self.instrument.polar_offset,
        )

        # BG subtract + integrate
        Ismp, _ = integrate_peak_areas(
            binned_spectra, energy_axes,
            self.experiment.elem_names,
            bg_method=sc.bg_method,
            bg_margin_frac=sc.bg_margin_frac,
        )

        # Cross section correction (sigma only — L goes to transfer matrix)
        Xobs, sigmas = apply_cross_section_correction(
            Ismp, self._elements,
            self.instrument.photon_energy,
            source=sc.cross_section_source,
            poly_order=sc.cross_section_poly_order,
            non_dipole=False,  # L handled separately
        )
        self._sigmas = sigmas

        # Non-dipole L factors → transfer matrix
        L = None
        if sc.non_dipole:
            L = compute_L_factors(
                self._elements, self.instrument.photon_energy,
                bin_centers,
                xray_from_normal_deg=sc.xray_from_normal_deg,
                polarized=sc.polarized,
            )

        # Solve
        result, depth, angles_used, rrmse = run_depth_profiling(
            Xobs, bin_centers, self._elements,
            solver_config=sc,
            experiment=self.experiment,
            photon_energy=self.instrument.photon_energy,
            L_correction=L,
        )

        return PipelineResult(
            D_opt=result.D_opt,
            I_opt=result.I_opt,
            thickness=result.thickness,
            abruptness=result.abruptness,
            rrmse=rrmse,
            rmse=result.rmse,
            depth=depth,
            angles_used=angles_used,
            Ismp_raw=Ismp,
            Xobs=Xobs,
            sigmas=sigmas,
            elapsed_seconds=time() - t0,
            solver_result=result,
        )

    def run_from_ismp(
        self,
        Ismp: np.ndarray,
        angles: np.ndarray,
    ) -> PipelineResult:
        """Run from pre-computed Ismp (n_elements, n_angles).

        For cases where loading and preprocessing is already done.
        """
        from .preprocessor import apply_cross_section_correction

        t0 = time()
        sc = self.solver_config

        Xobs, sigmas = apply_cross_section_correction(
            Ismp, self._elements,
            self.instrument.photon_energy,
            source=sc.cross_section_source,
            poly_order=sc.cross_section_poly_order,
            non_dipole=False,  # L handled separately
        )
        self._sigmas = sigmas

        L = None
        if sc.non_dipole:
            L = compute_L_factors(
                self._elements, self.instrument.photon_energy,
                angles,
                xray_from_normal_deg=sc.xray_from_normal_deg,
                polarized=sc.polarized,
            )

        result, depth, angles_used, rrmse = run_depth_profiling(
            Xobs, angles, self._elements,
            solver_config=sc,
            experiment=self.experiment,
            photon_energy=self.instrument.photon_energy,
            L_correction=L,
        )

        return PipelineResult(
            D_opt=result.D_opt,
            I_opt=result.I_opt,
            thickness=result.thickness,
            abruptness=result.abruptness,
            rrmse=rrmse,
            rmse=result.rmse,
            depth=depth,
            angles_used=angles_used,
            Ismp_raw=Ismp,
            Xobs=Xobs,
            sigmas=sigmas,
            elapsed_seconds=time() - t0,
            solver_result=result,
        )
