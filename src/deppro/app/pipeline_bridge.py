"""Bridge B+C: Read ezdeprof metadata and batch-solve depth profiles.

橋B — ``read_ezdeprof()`` reads TransferMatrix params + ElementConfig from Dstk H5.
橋C — ``batch_solve()`` applies a solver (free/semifree/nn/hybrid) to all pixels.
Glue — ``run_pipeline()`` chains 橋A→B→C→D in a single call.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import h5py
import numpy as np
from numpy.typing import NDArray

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 橋B: read_ezdeprof
# ---------------------------------------------------------------------------


def read_ezdeprof(h5_path: str) -> dict[str, Any]:
    """Read /ezdeprof/ metadata from a Dstk HDF5 file.

    Returns a dict consumable by :func:`batch_solve`:

    - ``tm_params`` — kwargs for ``TransferMatrix.__init__``
    - ``trans_mat`` — pre-computed matrix ``(n_angles, n_depth, n_elem)`` or *None*
    - ``elements`` — ``list[ElementConfig]`` (requires deppro)
    - ``element_names`` — ``list[str]``
    - ``free_indices`` — 0-based
    - ``substrate_indices`` — 0-based
    - ``linked_indices`` — 0-based
    - ``linkedconst`` — ``ndarray``
    - ``depth_axis`` — ``ndarray``
    - ``charges`` — ``ndarray``
    """
    with h5py.File(h5_path, "r") as f:
        ez = f["ezdeprof"]

        # Depth / step
        depth = np.asarray(ez["depth"], dtype=np.float64).flatten()
        step_size = float(np.asarray(ez["stepsize"], dtype=np.float64).flatten()[0])
        angles = np.asarray(ez["thetaobs"], dtype=np.float64).flatten()

        # Physics scalars (with defaults matching TransferMatrix)
        photon_energy = _read_scalar(ez, "photonenergy", 1486.6)
        work_function = _read_scalar(ez, "workfunction", 4.5)
        gamma_eff = _read_scalar(ez, "gammaeff", 1.0)

        # Pre-computed transfer matrix (optional)
        trans_mat: NDArray[np.float64] | None = None
        if "TransMatObs" in ez:
            # MATLAB layout (n_elem, n_depth, n_angles) → Python (n_angles, n_depth, n_elem)
            tm_raw = np.asarray(ez["TransMatObs"], dtype=np.float64)
            trans_mat = np.transpose(tm_raw, (2, 1, 0)).copy()

        # Element info
        element_names = _read_str_array(ez, "element")
        symbols = _read_str_array(ez, "elementsymbol")
        orbitals = _read_str_array(ez, "orbital")
        compounds = _read_str_array(ez, "compound")
        binding_energies = np.asarray(ez["bindingenergy"], dtype=np.float64).flatten()
        charges = np.asarray(ez["charge"], dtype=np.float64).flatten()

        # Index arrays (MATLAB 1-based → Python 0-based)
        free_indices = _matlab_to_python_indices(ez, "free")
        substrate_indices = _matlab_to_python_indices(ez, "subst")
        linked_indices = _matlab_to_python_indices(ez, "linked")

        linkedconst: NDArray[np.float64] = np.zeros(0, dtype=np.float64)
        if "linkedconst" in ez:
            linkedconst = np.asarray(ez["linkedconst"], dtype=np.float64).flatten()

    # Build ElementConfig list (deppro dependency deferred)
    elements = _build_element_configs(
        symbols, orbitals, compounds, binding_energies, charges
    )

    return {
        "tm_params": {
            "depth": depth,
            "step_size": step_size,
            "angles": angles,
            "photon_energy": photon_energy,
            "work_function": work_function,
            "gamma_eff": gamma_eff,
        },
        "trans_mat": trans_mat,
        "elements": elements,
        "element_names": element_names,
        "free_indices": free_indices,
        "substrate_indices": substrate_indices,
        "linked_indices": linked_indices,
        "linkedconst": linkedconst,
        "depth_axis": depth,
        "charges": charges,
        "binding_energies": binding_energies,
    }


# ---------------------------------------------------------------------------
# 橋C: batch_solve
# ---------------------------------------------------------------------------


def batch_solve(
    I_elem: NDArray[np.floating],
    ezdeprof: dict[str, Any],
    solver: str = "free",
    method: str = "LLS",
    alpha: float = 0.0,
    n_workers: int = 0,
    nn_model_path: str | None = None,
    maxiter: int = 100,
    progress_callback: Callable[[int, int], None] | None = None,
) -> NDArray[np.float32]:
    """Solve depth profiles for all pixels.

    Parameters
    ----------
    I_elem : (n_px, n_angles, n_elem) — peak-area intensities from 橋A.
    ezdeprof : dict from :func:`read_ezdeprof`.
    solver : ``"free"`` | ``"semifree"`` | ``"nn"`` | ``"hybrid"``.
    method : SolverConfig method (``"LLS"``, ``"L2"``, etc.).
    alpha : Regularization strength.
    n_workers : 0 = auto.
    nn_model_path : Path to ``.npz`` weights (required for ``"nn"``/``"hybrid"``).
    maxiter : Max iterations per pixel.
    progress_callback : Optional ``(current, total)`` callback for progress reporting.

    Returns
    -------
    depth_map : (n_px, n_depth, n_elem) float32 — ready for 橋D.
    """
    from deppro.app import DEPPRO_AVAILABLE

    if not DEPPRO_AVAILABLE:
        raise ImportError(
            "deppro package required for batch_solve. "
            "Install this repository (pip install -e .) to provide it."
        )

    from deppro import (
        CompoundDatabase,
        SolverConfig,
        TransferMatrix,
    )

    n_px, n_angles, n_elem = I_elem.shape
    tm_params = ezdeprof["tm_params"]
    n_depth = len(tm_params["depth"])
    elements = ezdeprof["elements"]

    # --- Build TransferMatrix ---
    tm = TransferMatrix(
        depth=tm_params["depth"],
        step_size=tm_params["step_size"],
        angles=tm_params["angles"],
        photon_energy=tm_params["photon_energy"],
        work_function=tm_params["work_function"],
        gamma_eff=tm_params["gamma_eff"],
    )

    if ezdeprof["trans_mat"] is not None:
        # Pre-computed: inject directly.
        # IMPORTANT: also set n_elements — TransferMatrix.__init__ defaults it
        # to 0 and only compute() updates it, but we skip compute() here.
        tm.trans_mat = ezdeprof["trans_mat"]
        tm.n_elements = ezdeprof["trans_mat"].shape[2]  # (n_angles, n_depth, n_elem)
    else:
        # Compute from uniform initial guess
        D_init = np.full((n_depth, n_elem), 1.0 / n_elem, dtype=np.float64)
        tm.compute(D_init, elements, CompoundDatabase())

    # --- SolverConfig ---
    config = SolverConfig(
        method=method,
        alpha=alpha,
        max_iter=maxiter,
        n_workers=n_workers if n_workers > 0 else None,
    )

    # --- Dispatch by solver type ---
    if solver == "free":
        return _solve_free(I_elem, tm, config, elements, n_depth, n_elem, progress_callback)
    elif solver == "semifree":
        # SemiFreeSolver needs a method that handles equality constraints;
        # SLSQP is named here explicitly (it is also SolverConfig's default).
        # Use benchmark-quality settings: warm restarts, constraints.
        semi_config = SolverConfig(
            method=method,
            alpha=alpha,
            max_iter=max(maxiter, 200),
            n_workers=n_workers if n_workers > 0 else None,
            scipy_method="SLSQP",
            use_sum_constraint=True,
            use_charge_constraint=True,
            max_warm_restarts=2,
            warm_restart_threshold=0.2,
            enable_fallback=False,
        )
        return _solve_semifree(I_elem, tm, semi_config, ezdeprof, elements, n_depth, n_elem, progress_callback)
    elif solver == "l1mem":
        return _solve_l1_mem(I_elem, tm, ezdeprof, elements, n_depth, n_elem, alpha, maxiter, progress_callback)
    elif solver == "nn":
        return _solve_nn(I_elem, nn_model_path, n_depth, n_elem, progress_callback)
    elif solver == "hybrid":
        return _solve_hybrid(
            I_elem, tm, config, ezdeprof, elements, nn_model_path, n_depth, n_elem,
            maxiter, n_workers, progress_callback,
        )
    else:
        raise ValueError(f"Unknown solver: {solver!r}")


# ---------------------------------------------------------------------------
# Element alignment
# ---------------------------------------------------------------------------


def _align_elements(
    I_elem: NDArray[np.floating],
    meta_a: dict[str, Any],
    ezdeprof: dict[str, Any],
) -> tuple[NDArray[np.floating], dict[str, Any]]:
    """Align fitpara components with ezdeprof chemical states.

    Uses binding energy matching to map fitpara components (e.g. Si1s comp0
    @ 99.3 eV, Si1s comp1 @ 103.3 eV) to ezdeprof states (Si(0), Si(+4)).

    When binding energies are unavailable, falls back to element-symbol matching.

    Returns updated (I_elem, ezdeprof) with matching column counts.
    """
    import re

    fp_centers: list[float] = meta_a.get("component_centers_be", [])
    fp_elements: list[str] = meta_a.get("elements", [])
    ez_names: list[str] = ezdeprof["element_names"]
    n_fp = I_elem.shape[2]
    n_ez = len(ez_names)

    # --- Binding energy matching (preferred) ---
    if fp_centers and len(fp_centers) == n_fp:
        # ezdeprof binding energies
        ez_be = np.asarray(ezdeprof.get("binding_energies", []), dtype=np.float64)

        if len(ez_be) == n_ez:
            # Match each ezdeprof state to the closest fitpara component
            # by binding energy.  Multiple ez states can map to components
            # from different element files.
            keep_ez: list[int] = []
            fp_order: list[int] = []
            fp_arr = np.array(fp_centers)

            for ez_idx in range(n_ez):
                dists = np.abs(fp_arr - ez_be[ez_idx])
                best_fp = int(np.argmin(dists))
                if dists[best_fp] < 20.0:  # 20 eV tolerance (same orbital)
                    keep_ez.append(ez_idx)
                    fp_order.append(best_fp)

            if keep_ez:
                dropped = [ez_names[i] for i in range(n_ez) if i not in keep_ez]
                log.info(
                    "BE alignment: %d ezdeprof → %d matched (dropped %s)",
                    n_ez, len(keep_ez), dropped,
                )
                return _apply_alignment(I_elem, ezdeprof, keep_ez, fp_order)

    # --- Fallback: element-symbol matching ---
    def _base(name: str) -> str:
        m = re.match(r"([A-Z][a-z]?)", name)
        return m.group(1) if m else name

    fp_bases = [_base(n) for n in fp_elements]
    ez_bases = [_base(n) for n in ez_names]

    log.debug("Symbol align: fp=%s→%s, ez=%s→%s", fp_elements, fp_bases, ez_names, ez_bases)

    if n_fp == n_ez:
        return I_elem, ezdeprof

    keep_ez = []
    fp_order = []
    for ez_idx, ez_b in enumerate(ez_bases):
        if ez_b in fp_bases:
            keep_ez.append(ez_idx)
            fp_order.append(fp_bases.index(ez_b))

    if not keep_ez:
        raise ValueError(
            f"No element overlap: fitpara has {fp_elements}, ezdeprof has {ez_names}"
        )

    dropped = [ez_names[i] for i in range(n_ez) if i not in keep_ez]
    log.info("Symbol alignment: %d ezdeprof → %d matched (dropped %s)", n_ez, len(keep_ez), dropped)
    return _apply_alignment(I_elem, ezdeprof, keep_ez, fp_order)


def _apply_alignment(
    I_elem: NDArray[np.floating],
    ezdeprof: dict[str, Any],
    keep_ez: list[int],
    fp_order: list[int],
) -> tuple[NDArray[np.floating], dict[str, Any]]:
    """Apply column reorder + filter to I_elem and ezdeprof."""
    ez_names = ezdeprof["element_names"]

    # Reorder I_elem columns to match ezdeprof order
    I_aligned = I_elem[:, :, fp_order].copy()

    # Filter ezdeprof arrays
    ez_new = dict(ezdeprof)
    ez_new["element_names"] = [ez_names[i] for i in keep_ez]
    ez_new["elements"] = [ezdeprof["elements"][i] for i in keep_ez]
    if ezdeprof.get("charges") is not None and len(ezdeprof["charges"]) > 0:
        ez_new["charges"] = np.asarray(ezdeprof["charges"])[keep_ez]

    # Filter TransferMatrix: (n_angles, n_depth, n_elem) → subset columns
    if ezdeprof["trans_mat"] is not None:
        ez_new["trans_mat"] = ezdeprof["trans_mat"][:, :, keep_ez].copy()

    # Filter index arrays — remap to new indices
    old_to_new = {old: new for new, old in enumerate(keep_ez)}
    for key in ("free_indices", "substrate_indices", "linked_indices"):
        old_arr = ezdeprof.get(key, np.array([], dtype=int))
        if len(old_arr) > 0:
            ez_new[key] = np.array(
                [old_to_new[i] for i in old_arr if i in old_to_new], dtype=int
            )

    return I_aligned, ez_new


# ---------------------------------------------------------------------------
# run_pipeline — A→B→C→D
# ---------------------------------------------------------------------------


def run_pipeline(
    fitpara_paths: list[str],
    dstk_path: str,
    solver: str = "free",
    output_dir: str | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
    **solver_kwargs: Any,
) -> str:
    """End-to-end pipeline: fitpara → depth profile H5.

    Parameters
    ----------
    fitpara_paths : H5 paths for 橋A (element-per-file or single Dstk).
    dstk_path : Dstk H5 with /ezdeprof metadata (橋B).
    solver : ``"free"`` | ``"semifree"`` | ``"nn"`` | ``"hybrid"``.
    output_dir : Output directory (default: ``"output"``).
    progress_callback : Optional ``(current, total)`` callback for progress reporting.
    **solver_kwargs : Forwarded to :func:`batch_solve`.

    Returns
    -------
    str — Path of the saved output HDF5 file.
    """
    import time as _time

    from deppro.app.depth_result_writer import write_depth_estimate
    from deppro.app.fitpara_reader import read_fitpara_areas

    # 橋A: fitpara → I_elem
    _t0 = _time.perf_counter()
    I_elem, meta_a = read_fitpara_areas(fitpara_paths)
    log.info("橋A: I_elem %s from %d files (%.1fs)", I_elem.shape, len(fitpara_paths), _time.perf_counter() - _t0)

    # 橋B: Dstk H5 → ezdeprof dict
    _t0 = _time.perf_counter()
    ezdeprof = read_ezdeprof(dstk_path)
    log.info("橋B: %d elements (%.1fs)", len(ezdeprof.get("element_names", [])), _time.perf_counter() - _t0)

    # ── Element alignment ──
    # fitpara may have fewer elements than ezdeprof (e.g. flat spectra skipped).
    # Align ezdeprof to match the fitpara element subset.
    I_elem, ezdeprof = _align_elements(I_elem, meta_a, ezdeprof)

    # 橋C: I_elem + ezdeprof → depth_map
    _t0 = _time.perf_counter()
    depth_map = batch_solve(
        I_elem, ezdeprof, solver=solver,
        progress_callback=progress_callback, **solver_kwargs,
    )
    log.info("橋C: depth_map %s (solver=%s, %.1fs)", depth_map.shape, solver, _time.perf_counter() - _t0)

    # 橋D: depth_map → output H5
    if output_dir is None:
        output_dir = "output"
    metadata = {
        "pixel_shape": meta_a["pixel_shape"],
        "elements": ezdeprof["element_names"],
        "depth_axis": ezdeprof["depth_axis"],
        "step_size": ezdeprof["tm_params"]["step_size"],
        "angles_deg": meta_a.get("angles_deg"),
        "trans_mat": ezdeprof.get("trans_mat"),
    }
    output_path = write_depth_estimate(
        depth_map, metadata,
        output_dir=output_dir,
        solver=solver.upper(),
        I_obs=I_elem,
    )
    log.info("橋D: %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Solver dispatch helpers
# ---------------------------------------------------------------------------


def _solve_free(
    I_elem: NDArray,
    tm,
    config,
    elements: list,
    n_depth: int,
    n_elem: int,
    progress_callback: Callable[[int, int], None] | None = None,
) -> NDArray[np.float32]:
    """Free solver via BatchInverseSolver."""
    from deppro import BatchInverseSolver

    n_px = I_elem.shape[0]
    D_prior = np.full((n_depth, n_elem), 1.0 / n_elem, dtype=np.float64)
    batch_solver = BatchInverseSolver(tm, config, D_prior, elements)
    log.info("FreeSolver: %d pixels, n_depth=%d, n_elem=%d", n_px, n_depth, n_elem)
    results = batch_solver.solve_batch(I_elem)

    depth_map = np.zeros((n_px, n_depth, n_elem), dtype=np.float32)
    for i, r in enumerate(results):
        depth_map[i] = r.D_opt.astype(np.float32)
        if progress_callback:
            progress_callback(i + 1, n_px)
    return depth_map


def _solve_semifree(
    I_elem: NDArray,
    tm,
    config,
    ezdeprof: dict,
    elements: list,
    n_depth: int,
    n_elem: int,
    progress_callback: Callable[[int, int], None] | None = None,
) -> NDArray[np.float32]:
    """SemiFreeSolver — sequential per-pixel (no batch API)."""
    import time as _time

    from deppro import SemiFreeSolver, SemiFreeSolverConfig

    n_px = I_elem.shape[0]

    log.info(
        "SemiFreeSolver: %d pixels, n_depth=%d, n_elem=%d, free=%s, subst=%s",
        n_px, n_depth, n_elem,
        ezdeprof["free_indices"], ezdeprof["substrate_indices"],
    )

    # Derive thickness bounds from the depth axis. The fraction used for
    # thickness_init below is an empirical default, not a measured optimum.
    tm_params = ezdeprof["tm_params"]
    depth_axis = np.asarray(tm_params["depth"])
    total_depth = float(depth_axis.max())
    step_size = float(tm_params["step_size"])
    thickness_init = total_depth * 0.4
    thickness_bounds = (max(50.0, step_size), min(500.0, total_depth * 0.9))
    abruptness_init = max(step_size, 20.0)
    abruptness_bounds = (1.0, 100.0)

    # Convert numpy arrays to Python lists — deppro uses `if indices:` checks
    # which fail with "ambiguous truth value" for multi-element numpy arrays.
    # Pass linkedconst as free_linked_const_override for correct stoichiometry.
    linkedconst_raw = ezdeprof.get("linkedconst", np.array([]))
    free_linked_const_override = (
        list(float(x) for x in linkedconst_raw)
        if len(linkedconst_raw) > 0
        else None
    )

    semi_cfg = SemiFreeSolverConfig(
        free_indices=list(int(x) for x in ezdeprof["free_indices"]),
        substrate_indices=list(int(x) for x in ezdeprof["substrate_indices"]),
        linked_indices=list(int(x) for x in ezdeprof["linked_indices"]),
        free_linked_const_override=free_linked_const_override,
        thickness_init=thickness_init,
        thickness_bounds=thickness_bounds,
        abruptness_init=abruptness_init,
        abruptness_bounds=abruptness_bounds,
    )

    solver = SemiFreeSolver(tm, config, semi_cfg, elements=elements, use_numba=False)
    log.info(
        "SemiFreeSolver created: thickness_init=%.1f, bounds=%s, linkedconst=%s",
        thickness_init, thickness_bounds, free_linked_const_override,
    )

    depth_map = np.zeros((n_px, n_depth, n_elem), dtype=np.float32)
    _t0 = _time.perf_counter()
    for i in range(n_px):
        result = solver.solve(I_elem[i])
        depth_map[i] = result.D_opt.astype(np.float32)
        if progress_callback:
            progress_callback(i + 1, n_px)
        if i == 0:
            log.info("SemiFreeSolver: pixel 0 done in %.3fs", _time.perf_counter() - _t0)
        elif (i + 1) % 500 == 0:
            elapsed = _time.perf_counter() - _t0
            log.info("SemiFreeSolver: %d/%d px in %.1fs (%.0f px/s)", i + 1, n_px, elapsed, (i + 1) / elapsed)
    log.info("SemiFreeSolver: done %d px in %.1fs", n_px, _time.perf_counter() - _t0)
    return depth_map


def _solve_l1_mem(
    I_elem: NDArray,
    tm,
    ezdeprof: dict,
    elements: list,
    n_depth: int,
    n_elem: int,
    alpha: float = 1.0,
    maxiter: int = 200,
    progress_callback: Callable[[int, int], None] | None = None,
) -> NDArray[np.float32]:
    """L1 → MEM 2-phase pipeline — matches benchmark_dstk.run_l1_mem_pipeline().

    Phase 1: L1 for coarse sparsity-promoting reconstruction.
    Phase 2: MEM refinement using L1 result as D_prior and thickness_init.
    """
    import time as _time

    from deppro import SemiFreeSolver, SemiFreeSolverConfig, SolverConfig

    n_px = I_elem.shape[0]

    # Derive bounds from depth axis
    tm_params = ezdeprof["tm_params"]
    depth_axis = np.asarray(tm_params["depth"])
    total_depth = float(depth_axis.max())
    step_size = float(tm_params["step_size"])
    thickness_init = total_depth * 0.4
    thickness_bounds = (max(50.0, step_size), min(500.0, total_depth * 0.9))
    abruptness_init = max(step_size, 20.0)
    abruptness_bounds = (1.0, 100.0)

    # Convert index arrays
    linkedconst_raw = ezdeprof.get("linkedconst", np.array([]))
    free_linked_const_override = (
        list(float(x) for x in linkedconst_raw)
        if len(linkedconst_raw) > 0
        else None
    )

    # Shared semi-free config template
    semi_base = dict(
        free_indices=list(int(x) for x in ezdeprof["free_indices"]),
        substrate_indices=list(int(x) for x in ezdeprof["substrate_indices"]),
        linked_indices=list(int(x) for x in ezdeprof["linked_indices"]),
        free_linked_const_override=free_linked_const_override,
        thickness_bounds=thickness_bounds,
        abruptness_init=abruptness_init,
        abruptness_bounds=abruptness_bounds,
        optimize_thickness=True,
        optimize_abruptness=False,
    )

    # Phase 1: L1 config
    l1_config = SolverConfig(
        method="L1",
        alpha=alpha,
        max_iter=maxiter,
        scipy_method="SLSQP",
        use_sum_constraint=True,
        use_charge_constraint=True,
        max_warm_restarts=2,
        warm_restart_threshold=0.2,
        enable_fallback=False,
    )

    # Phase 2: MEM config
    mem_alpha = alpha * 0.0003 if alpha > 0 else 0.0003
    mem_config = SolverConfig(
        method="MEM",
        alpha=mem_alpha,
        max_iter=maxiter,
        scipy_method="SLSQP",
        use_sum_constraint=True,
        use_charge_constraint=True,
        max_warm_restarts=2,
        warm_restart_threshold=0.2,
        enable_fallback=False,
    )

    log.info(
        "L1→MEM pipeline: %d pixels, n_depth=%d, n_elem=%d, L1 α=%.4f, MEM α=%.6f",
        n_px, n_depth, n_elem, alpha, mem_alpha,
    )

    depth_map = np.zeros((n_px, n_depth, n_elem), dtype=np.float32)
    _t0 = _time.perf_counter()

    for i in range(n_px):
        try:
            # Phase 1: L1 coarse
            semi_l1 = SemiFreeSolverConfig(thickness_init=thickness_init, **semi_base)
            solver_l1 = SemiFreeSolver(
                transfer_matrix=tm, config=l1_config,
                semi_free_config=semi_l1, D_prior=None,
                elements=elements, use_numba=False,
            )
            result_l1 = solver_l1.solve(I_elem[i], verbose=False)

            # Phase 2: MEM refinement from L1
            semi_mem = SemiFreeSolverConfig(
                thickness_init=result_l1.thickness, **semi_base,
            )
            solver_mem = SemiFreeSolver(
                transfer_matrix=tm, config=mem_config,
                semi_free_config=semi_mem,
                D_prior=result_l1.D_opt,
                elements=elements, use_numba=False,
            )
            # Warm-start from L1 solution
            try:
                x0 = solver_mem.build_x0(
                    thickness=result_l1.thickness,
                    abruptness=result_l1.abruptness,
                    substrate_ratios=result_l1.substrate_ratios,
                )
                result = solver_mem.solve(I_elem[i], verbose=False, x0_override=x0)
            except (AttributeError, TypeError):
                # Fallback if build_x0 not available
                result = solver_mem.solve(I_elem[i], verbose=False)

            depth_map[i] = result.D_opt.astype(np.float32)

        except Exception as exc:
            log.warning("L1→MEM pixel %d failed: %s — using L1 result", i, exc)
            try:
                depth_map[i] = result_l1.D_opt.astype(np.float32)
            except Exception:
                pass  # Leave zeros

        if progress_callback:
            progress_callback(i + 1, n_px)
        if i == 0:
            log.info("L1→MEM: pixel 0 done in %.3fs", _time.perf_counter() - _t0)
        elif (i + 1) % 500 == 0:
            elapsed = _time.perf_counter() - _t0
            log.info("L1→MEM: %d/%d px in %.1fs (%.0f px/s)", i + 1, n_px, elapsed, (i + 1) / elapsed)

    log.info("L1→MEM: done %d px in %.1fs", n_px, _time.perf_counter() - _t0)
    return depth_map


def _solve_nn(
    I_elem: NDArray,
    nn_model_path: str | None,
    n_depth: int,
    n_elem: int,
    progress_callback: Callable[[int, int], None] | None = None,
) -> NDArray[np.float32]:
    """NN-only inference."""
    if nn_model_path is None:
        raise ValueError("nn_model_path required for solver='nn'")

    from deppro.hybrid_solver import _nn_infer_batch
    from deppro.inference import load_model

    model = load_model(nn_model_path)
    D_nn = _nn_infer_batch(I_elem, model, n_depth, n_elem)
    if progress_callback:
        progress_callback(I_elem.shape[0], I_elem.shape[0])
    return D_nn.astype(np.float32)


def _solve_hybrid(
    I_elem: NDArray,
    tm,
    config,
    ezdeprof: dict,
    elements: list,
    nn_model_path: str | None,
    n_depth: int,
    n_elem: int,
    maxiter: int,
    n_workers: int,
    progress_callback: Callable[[int, int], None] | None = None,
) -> NDArray[np.float32]:
    """Hybrid NN init + SLSQP refinement."""
    if nn_model_path is None:
        raise ValueError("nn_model_path required for solver='hybrid'")

    from deppro import SemiFreeSolver, SemiFreeSolverConfig
    from deppro.hybrid_solver import solve_hybrid_batch
    from deppro.inference import load_model

    D_prior = np.full((n_depth, n_elem), 1.0 / n_elem, dtype=np.float64)
    semi_cfg = SemiFreeSolverConfig(
        free_indices=ezdeprof["free_indices"],
        substrate_indices=ezdeprof["substrate_indices"],
        linked_indices=ezdeprof["linked_indices"],
    )

    def solver_factory():
        return SemiFreeSolver(tm, config, semi_cfg, D_prior, elements)

    model = load_model(nn_model_path)
    results = solve_hybrid_batch(
        I_elem, solver_factory, model,
        maxiter=maxiter, mode="hybrid", n_workers=n_workers,
    )

    n_px = I_elem.shape[0]
    depth_map = np.zeros((n_px, n_depth, n_elem), dtype=np.float32)
    for i, r in enumerate(results):
        depth_map[i] = r.D_opt.astype(np.float32)
        if progress_callback:
            progress_callback(i + 1, n_px)
    return depth_map


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _read_scalar(group: h5py.Group, key: str, default: float) -> float:
    """Read a scalar float from H5 group with fallback."""
    if key in group:
        return float(np.asarray(group[key], dtype=np.float64).flatten()[0])
    return default


def _read_str_array(group: h5py.Group, key: str) -> list[str]:
    """Read a string array from H5 group."""
    raw = group[key][:].flatten()
    return [x.decode() if isinstance(x, bytes) else str(x) for x in raw]


def _matlab_to_python_indices(group: h5py.Group, key: str) -> list[int]:
    """Read MATLAB 1-based index array → Python 0-based."""
    if key not in group:
        return []
    raw = np.asarray(group[key], dtype=int).flatten()
    return [int(x) - 1 for x in raw if x > 0]


def _build_element_configs(
    symbols: list[str],
    orbitals: list[str],
    compounds: list[str],
    binding_energies: NDArray,
    charges: NDArray,
) -> list:
    """Build ElementConfig list.  Returns plain dicts if deppro unavailable."""
    try:
        from deppro import ElementConfig

        return [
            ElementConfig(
                symbol=symbols[i],
                orbital=orbitals[i],
                compound=compounds[i],
                binding_energy=float(binding_energies[i]),
                oxidation_state=float(charges[i]),
            )
            for i in range(len(symbols))
        ]
    except ImportError:
        # Return plain dicts as fallback (read_ezdeprof works without deppro)
        return [
            {
                "symbol": symbols[i],
                "orbital": orbitals[i],
                "compound": compounds[i],
                "binding_energy": float(binding_energies[i]),
                "oxidation_state": float(charges[i]),
            }
            for i in range(len(symbols))
        ]
