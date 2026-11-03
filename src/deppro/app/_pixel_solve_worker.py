"""ProcessPoolExecutor worker for pixel-level depth solving.

This module is intentionally separate from the (archived) Qt GUI so that
worker processes never import PySide6/Qt.  On macOS, importing PySide6 in a
"spawn"-ed subprocess crashes the child process because Qt expects the main
thread to own the event loop.

All functions here are module-level (picklable) and use only:
  numpy, scipy, deppro — no Qt, no pyqtgraph.
"""

from __future__ import annotations

import numpy as np

# ── Per-process globals (set by initializer) ──────────────────────────────────

_worker_xstk_3d: np.ndarray | None = None
_worker_trans_mat_np: np.ndarray | None = None
_worker_warm_drlt: np.ndarray | None = None
_worker_config: dict | None = None
# SharedMemory objects must be kept alive to prevent the numpy buffer from
# being invalidated when the GC collects the local variable.
_worker_shm_refs: list = []


def pixel_solve_initializer(
    xstk_shm_name: str,
    xstk_shape: tuple,
    xstk_dtype: str,
    tm_shm_name: str,
    tm_shape: tuple,
    tm_dtype: str,
    config: dict,
) -> None:
    """Called once per worker process — attaches to shared memory arrays."""
    from multiprocessing.shared_memory import SharedMemory

    global _worker_xstk_3d, _worker_trans_mat_np, _worker_warm_drlt
    global _worker_config, _worker_shm_refs

    shm_x = SharedMemory(name=xstk_shm_name, create=False)
    _worker_xstk_3d = np.ndarray(xstk_shape, dtype=xstk_dtype, buffer=shm_x.buf)

    shm_t = SharedMemory(name=tm_shm_name, create=False)
    _worker_trans_mat_np = np.ndarray(tm_shape, dtype=tm_dtype, buffer=shm_t.buf)

    shm_refs = [shm_x, shm_t]

    # Optional warm-start Drlt via SharedMemory
    warm_shm_name = config.get("warm_start_drlt_shm")
    warm_shape = config.get("warm_start_drlt_shape")
    warm_dtype = config.get("warm_start_drlt_dtype")
    if warm_shm_name and warm_shape and warm_dtype:
        shm_w = SharedMemory(name=warm_shm_name, create=False)
        _worker_warm_drlt = np.ndarray(warm_shape, dtype=warm_dtype, buffer=shm_w.buf)
        shm_refs.append(shm_w)
    else:
        _worker_warm_drlt = None

    # Keep SharedMemory refs alive — prevents GC from unmapping the buffer
    _worker_shm_refs = shm_refs

    _worker_config = config


def pixel_solve(px_idx: int) -> np.ndarray | None:
    """Solve one pixel in a worker process (no GIL contention with GUI).

    Returns:
        (n_elem, n_depth) float32 array, or None on failure.
    """
    try:
        from deppro import (
            FreeSolver,
            SemiFreeSolver,
            SemiFreeSolverConfig,
            SolverConfig,
            TransferMatrix,
        )
    except ImportError:
        return None

    cfg = _worker_config
    xstk_3d = _worker_xstk_3d
    trans_mat_np = _worker_trans_mat_np

    if cfg is None or xstk_3d is None or trans_mat_np is None:
        return None

    try:
        x_obs = xstk_3d[px_idx]  # (n_elem, n_angles)
        intensity = x_obs.T.astype(np.float64)  # (n_angles, n_elem)

        depth_axis = np.array(cfg["depth_axis"], dtype=np.float64)
        angles = np.array(cfg["angles"], dtype=np.float64)
        step_size = float(np.mean(np.diff(depth_axis))) if len(depth_axis) > 1 else 1.0

        tm = TransferMatrix(
            depth=depth_axis,
            step_size=step_size,
            angles=angles,
            photon_energy=cfg.get("photon_energy", 1486.6),
            work_function=cfg.get("work_function", 4.5),
            gamma_eff=cfg.get("gamma_eff", 1.0),
        )
        tm.trans_mat = trans_mat_np.copy()
        tm.n_elements = trans_mat_np.shape[2]

        solver_text = cfg["solver_text"]
        # Auto (Pipeline) → SemiFreeSolver in batch mode
        if solver_text == "Auto (Pipeline)":
            solver_text = "SemiFreeSolver"
        method = cfg["method"]
        alpha = cfg["alpha"]

        if solver_text == "FreeSolver":
            config = SolverConfig(method=method, alpha=alpha, max_iter=200)
            s = FreeSolver(tm, config)
            result = s.solve(intensity, verbose=False)
            return result.D_opt.T.astype(np.float32)

        elif solver_text in ("SemiFreeSolver", "L1→MEM"):
            semi_indices = cfg.get("semifree_indices", {})
            semi_config = SemiFreeSolverConfig()
            if semi_indices:
                free = semi_indices.get("free", [])
                linked = semi_indices.get("linked", [])
                subst = semi_indices.get("subst", [])
                const = semi_indices.get("const", [])
                linkedconst = semi_indices.get("linkedconst")
                if free or linked or subst or const:
                    semi_config = SemiFreeSolverConfig(
                        free_indices=free or None,
                        linked_indices=linked or None,
                        substrate_indices=subst or None,
                        const_indices=const or None,
                        free_linked_const_override=linkedconst,
                    )

            semi_config.normalize_free_conc = False
            semi_config.use_sum_constraint = True
            semi_config.substrate_cutoff = None
            depth_max = float(np.max(depth_axis))
            semi_config.thickness_bounds = (max(50.0, 1.0), min(500.0, depth_max))
            semi_config.optimize_thickness = True
            semi_config.optimize_abruptness = False

            if not semi_config.substrate_indices:
                quick_config = SolverConfig(method="LLS", alpha=0.0)
                quick_s = FreeSolver(tm, quick_config)
                quick_r = quick_s.solve(intensity, verbose=False)
                sub_idx = int(np.argmax(quick_r.D_opt[:, -1]))
                semi_config.substrate_indices = [sub_idx]

            semi_init = cfg.get("semifree_init_params", {})
            semi_config.thickness_init = semi_init.get("thickness_init", depth_max * 0.4)
            semi_config.abruptness_init = semi_init.get(
                "abruptness_init",
                max(float(np.mean(np.diff(depth_axis))), 20.0),
            )
            semi_config.abruptness_bounds = (1.0, 100.0)

            # Batch mode: use relaxed ftol for speed + implicit regularization
            batch_ftol = cfg.get("batch_ftol", 1e-4)

            bench_config = SolverConfig(
                method=method,
                alpha=alpha,
                scipy_method="SLSQP",
                max_iter=200,
                tol=batch_ftol,
                use_sum_constraint=True,
                use_charge_constraint=True,
                max_warm_restarts=2,
                warm_restart_threshold=0.2,
                enable_fallback=False,
            )
            # Xreg weighting: normalize chi² by per-element mean intensity.
            # Meant to keep weak (trace) elements from being swamped by strong ones.
            bench_config.use_xreg_weighting = True

            elements_config = cfg.get("elements_config")

            if solver_text == "L1→MEM":
                l1_config = SolverConfig(
                    method="L1",
                    alpha=alpha,
                    scipy_method="SLSQP",
                    max_iter=200,
                    tol=batch_ftol,
                    use_sum_constraint=True,
                    use_charge_constraint=True,
                    max_warm_restarts=2,
                    warm_restart_threshold=0.2,
                    enable_fallback=False,
                )
                l1_config.use_xreg_weighting = True
                s_l1 = SemiFreeSolver(
                    tm, l1_config, semi_config, elements=elements_config,
                )
                r_l1 = s_l1.solve(intensity, verbose=False)
                mem_alpha = alpha * 0.0003 if alpha > 0 else 0.0003
                mem_config = SolverConfig(
                    method="MEM",
                    alpha=mem_alpha,
                    scipy_method="SLSQP",
                    max_iter=200,
                    tol=batch_ftol,
                    use_sum_constraint=True,
                    use_charge_constraint=True,
                    max_warm_restarts=2,
                    warm_restart_threshold=0.2,
                    enable_fallback=False,
                )
                mem_config.use_xreg_weighting = True
                semi_config.thickness_init = r_l1.thickness
                s_mem = SemiFreeSolver(
                    tm, mem_config, semi_config,
                    D_prior=r_l1.D_opt, elements=elements_config,
                )
                try:
                    x0 = s_mem.build_x0(
                        thickness=r_l1.thickness,
                        abruptness=r_l1.abruptness,
                        substrate_ratios=r_l1.substrate_ratios,
                    )
                    result = s_mem.solve(intensity, verbose=False, x0_override=x0)
                except (AttributeError, TypeError):
                    result = s_mem.solve(intensity, verbose=False)
            else:
                s = SemiFreeSolver(
                    tm, bench_config, semi_config, elements=elements_config,
                )
                result = s.solve(intensity, verbose=False)

            return result.D_opt.T.astype(np.float32)

        else:
            config = SolverConfig(method=method, alpha=alpha, max_iter=200)
            s = FreeSolver(tm, config)
            result = s.solve(intensity, verbose=False)
            return result.D_opt.T.astype(np.float32)

    except Exception:
        return None


# ── Helpers for strip-based solving ──────────────────────────────────────────


def _make_tm():
    """Build a TransferMatrix from worker globals (call once per strip)."""
    from deppro import TransferMatrix

    cfg = _worker_config
    depth_axis = np.array(cfg["depth_axis"], dtype=np.float64)
    angles = np.array(cfg["angles"], dtype=np.float64)
    step_size = float(np.mean(np.diff(depth_axis))) if len(depth_axis) > 1 else 1.0

    tm = TransferMatrix(
        depth=depth_axis,
        step_size=step_size,
        angles=angles,
        photon_energy=cfg.get("photon_energy", 1486.6),
        work_function=cfg.get("work_function", 4.5),
        gamma_eff=cfg.get("gamma_eff", 1.0),
    )
    tm.trans_mat = _worker_trans_mat_np.copy()
    tm.n_elements = _worker_trans_mat_np.shape[2]
    return tm, depth_axis


def _make_semi_config(cfg, depth_axis, tm, first_intensity):
    """Build SemiFreeSolverConfig (call once per strip).

    Handles substrate auto-detection using the first pixel in the strip.
    """
    from deppro import FreeSolver, SemiFreeSolverConfig, SolverConfig

    semi_indices = cfg.get("semifree_indices", {})
    semi_config = SemiFreeSolverConfig()
    if semi_indices:
        free = semi_indices.get("free", [])
        linked = semi_indices.get("linked", [])
        subst = semi_indices.get("subst", [])
        const = semi_indices.get("const", [])
        linkedconst = semi_indices.get("linkedconst")
        if free or linked or subst or const:
            semi_config = SemiFreeSolverConfig(
                free_indices=free or None,
                linked_indices=linked or None,
                substrate_indices=subst or None,
                const_indices=const or None,
                free_linked_const_override=linkedconst,
            )

    semi_config.normalize_free_conc = False
    semi_config.use_sum_constraint = True
    semi_config.substrate_cutoff = None
    depth_max = float(np.max(depth_axis))
    semi_config.thickness_bounds = (max(50.0, 1.0), min(500.0, depth_max))
    semi_config.optimize_thickness = True
    semi_config.optimize_abruptness = False

    # Substrate auto-detection: once per strip using first pixel
    if not semi_config.substrate_indices:
        quick_config = SolverConfig(method="LLS", alpha=0.0)
        quick_s = FreeSolver(tm, quick_config)
        quick_r = quick_s.solve(first_intensity, verbose=False)
        sub_idx = int(np.argmax(quick_r.D_opt[:, -1]))
        semi_config.substrate_indices = [sub_idx]

    semi_init = cfg.get("semifree_init_params", {})
    semi_config.thickness_init = semi_init.get("thickness_init", depth_max * 0.4)
    semi_config.abruptness_init = semi_init.get(
        "abruptness_init",
        max(float(np.mean(np.diff(depth_axis))), 20.0),
    )
    semi_config.abruptness_bounds = (1.0, 100.0)
    return semi_config


def _make_solver_config(cfg):
    """Build the SolverConfig for batch SemiFreeSolver."""
    from deppro import SolverConfig

    batch_ftol = cfg.get("batch_ftol", 1e-4)
    sc = SolverConfig(
        method=cfg["method"],
        alpha=cfg["alpha"],
        scipy_method="SLSQP",
        max_iter=200,
        tol=batch_ftol,
        use_sum_constraint=True,
        use_charge_constraint=True,
        max_warm_restarts=2,
        warm_restart_threshold=0.2,
        enable_fallback=False,
    )
    # Xreg weighting: normalize chi² by per-element mean intensity.
    # Meant to keep weak (trace) elements from being swamped by strong ones.
    sc.use_xreg_weighting = True
    return sc


def _get_warm_start(px_idx, substrate_indices, depth_axis, step_size):
    """Extract warm-start parameters from existing Drlt for a pixel.

    Returns (D_prior, thickness, substrate_ratios) or (None, None, None)
    if no warm-start data is available for this pixel.
    """
    warm_drlt = _worker_warm_drlt
    if warm_drlt is None or px_idx >= warm_drlt.shape[0]:
        return None, None, None

    drlt = warm_drlt[px_idx]  # (n_elem, n_depth)
    if not np.any(drlt != 0):
        return None, None, None

    D_prior = drlt.T.astype(np.float64)  # (n_depth, n_elem)

    # Estimate thickness from substrate 0.5 crossing
    thickness = None
    sub_ratios = None
    if substrate_indices:
        sub_conc = D_prior[:, substrate_indices[0]]
        half = 0.5 * sub_conc.max()
        if half > 1e-8:
            above = np.where(sub_conc >= half)[0]
            if len(above) > 0:
                idx = above[0]
                if idx > 0 and sub_conc[idx - 1] < half:
                    frac = (half - sub_conc[idx - 1]) / max(
                        sub_conc[idx] - sub_conc[idx - 1], 1e-10
                    )
                    thickness = float(depth_axis[idx - 1] + frac * step_size)
                elif idx < len(depth_axis):
                    thickness = float(depth_axis[idx])

        # Estimate substrate ratios from deep region
        n_depth = D_prior.shape[0]
        start = max(0, int(n_depth * 0.75))
        sub_conc_deep = D_prior[start:, substrate_indices]
        mean_c = np.mean(sub_conc_deep, axis=0)
        total = mean_c.sum()
        if total > 1e-10:
            sub_ratios = mean_c / total

    return D_prior, thickness, sub_ratios


# ── Strip-based solving with warm-start ──────────────────────────────────────


def pixel_solve_strip(
    strip_indices: list[int],
) -> list[tuple[int, np.ndarray | None]]:
    """Solve a strip of pixels with solver reuse + cross-pixel warm-start.

    Within a strip, each pixel's optimized parameter vector (x_opt) is passed
    as the starting point for the next pixel.  This dramatically reduces
    SLSQP iterations when neighboring pixels have similar depth profiles.

    The TransferMatrix, SemiFreeSolverConfig, and Solver object are created
    once and reused for the entire strip.

    Returns:
        List of ``(px_idx, D_opt_or_None)`` tuples.
        *D_opt* is ``(n_elem, n_depth)`` float32 array.
    """
    try:
        from deppro import (
            FreeSolver,
            SemiFreeSolver,
            SolverConfig,
        )
    except ImportError:
        return [(i, None) for i in strip_indices]

    cfg = _worker_config
    xstk_3d = _worker_xstk_3d
    trans_mat_np = _worker_trans_mat_np

    if cfg is None or xstk_3d is None or trans_mat_np is None:
        return [(i, None) for i in strip_indices]

    try:
        tm, depth_axis = _make_tm()
    except Exception:
        return [(i, None) for i in strip_indices]

    solver_text = cfg["solver_text"]
    # Auto (Pipeline) → SemiFreeSolver in batch mode
    if solver_text == "Auto (Pipeline)":
        solver_text = "SemiFreeSolver"
    method = cfg["method"]
    alpha = cfg["alpha"]
    elements_config = cfg.get("elements_config")
    results: list[tuple[int, np.ndarray | None]] = []

    # ── FreeSolver path (no warm-start, just solver reuse) ────────────────
    if solver_text == "FreeSolver":
        sc = SolverConfig(method=method, alpha=alpha, max_iter=200)
        solver = FreeSolver(tm, sc)
        for px_idx in strip_indices:
            try:
                intensity = xstk_3d[px_idx].T.astype(np.float64)
                r = solver.solve(intensity, verbose=False)
                results.append((px_idx, r.D_opt.T.astype(np.float32)))
            except Exception:
                results.append((px_idx, None))
        return results

    # ── SemiFreeSolver / L1→MEM path ──────────────────────────────────────
    if solver_text in ("SemiFreeSolver", "L1→MEM"):
        # First pixel intensity (for substrate auto-detection)
        first_intensity = xstk_3d[strip_indices[0]].T.astype(np.float64)

        try:
            semi_config = _make_semi_config(cfg, depth_axis, tm, first_intensity)
        except Exception:
            return [(i, None) for i in strip_indices]

        bench_config = _make_solver_config(cfg)
        batch_ftol = cfg.get("batch_ftol", 1e-4)
        step_size = float(np.mean(np.diff(depth_axis))) if len(depth_axis) > 1 else 1.0
        substrate_indices = semi_config.substrate_indices or []

        if solver_text == "L1→MEM":
            for px_idx in strip_indices:
                try:
                    intensity = xstk_3d[px_idx].T.astype(np.float64)

                    # Check for warm-start Drlt
                    warm_D_prior, warm_thick, warm_sub = _get_warm_start(
                        px_idx, substrate_indices, depth_axis, step_size,
                    )

                    # L1 phase (warm-started if Drlt available)
                    l1_config = SolverConfig(
                        method="L1",
                        alpha=alpha,
                        scipy_method="SLSQP",
                        max_iter=200,
                        tol=batch_ftol,
                        use_sum_constraint=True,
                        use_charge_constraint=True,
                        max_warm_restarts=2,
                        warm_restart_threshold=0.2,
                        enable_fallback=False,
                    )
                    l1_config.use_xreg_weighting = True

                    if warm_D_prior is not None and warm_thick is not None:
                        semi_config.thickness_init = warm_thick
                    solver_l1 = SemiFreeSolver(
                        tm, l1_config, semi_config,
                        D_prior=warm_D_prior,
                        elements=elements_config,
                    )

                    if warm_D_prior is not None and warm_thick is not None:
                        try:
                            x0_l1 = solver_l1.build_x0(
                                thickness=warm_thick,
                                abruptness=semi_config.abruptness_init,
                                substrate_ratios=warm_sub,
                            )
                            r_l1 = solver_l1.solve(
                                intensity, verbose=False, x0_override=x0_l1,
                            )
                        except (AttributeError, TypeError, ValueError):
                            r_l1 = solver_l1.solve(intensity, verbose=False)
                    else:
                        r_l1 = solver_l1.solve(intensity, verbose=False)

                    # MEM phase (D_prior = L1 result)
                    mem_alpha = alpha * 0.0003 if alpha > 0 else 0.0003
                    mem_config = SolverConfig(
                        method="MEM",
                        alpha=mem_alpha,
                        scipy_method="SLSQP",
                        max_iter=200,
                        tol=batch_ftol,
                        use_sum_constraint=True,
                        use_charge_constraint=True,
                        max_warm_restarts=2,
                        warm_restart_threshold=0.2,
                        enable_fallback=False,
                    )
                    mem_config.use_xreg_weighting = True
                    semi_config.thickness_init = r_l1.thickness
                    s_mem = SemiFreeSolver(
                        tm, mem_config, semi_config,
                        D_prior=r_l1.D_opt, elements=elements_config,
                    )
                    try:
                        x0 = s_mem.build_x0(
                            thickness=r_l1.thickness,
                            abruptness=r_l1.abruptness,
                            substrate_ratios=r_l1.substrate_ratios,
                        )
                        result = s_mem.solve(
                            intensity, verbose=False, x0_override=x0,
                        )
                    except (AttributeError, TypeError):
                        result = s_mem.solve(intensity, verbose=False)

                    results.append(
                        (px_idx, result.D_opt.T.astype(np.float32))
                    )
                except Exception:
                    results.append((px_idx, None))

        else:
            # Plain SemiFreeSolver: per-pixel warm-start from Drlt if available
            # Cross-pixel warm-start is not used,
            # but NN-based warm-start uses D_prior + x0_override per pixel.
            for px_idx in strip_indices:
                try:
                    intensity = xstk_3d[px_idx].T.astype(np.float64)

                    # Check for warm-start Drlt
                    warm_D_prior, warm_thick, warm_sub = _get_warm_start(
                        px_idx, substrate_indices, depth_axis, step_size,
                    )

                    if warm_D_prior is not None and warm_thick is not None:
                        semi_config.thickness_init = warm_thick
                    solver = SemiFreeSolver(
                        tm, bench_config, semi_config,
                        D_prior=warm_D_prior,
                        elements=elements_config,
                    )

                    if warm_D_prior is not None and warm_thick is not None:
                        try:
                            x0 = solver.build_x0(
                                thickness=warm_thick,
                                abruptness=semi_config.abruptness_init,
                                substrate_ratios=warm_sub,
                            )
                            result = solver.solve(
                                intensity, verbose=False, x0_override=x0,
                            )
                        except (AttributeError, TypeError, ValueError):
                            result = solver.solve(intensity, verbose=False)
                    else:
                        result = solver.solve(intensity, verbose=False)

                    results.append(
                        (px_idx, result.D_opt.T.astype(np.float32))
                    )
                except Exception:
                    results.append((px_idx, None))

        return results

    # ── Fallback ──────────────────────────────────────────────────────────
    sc = SolverConfig(method=method, alpha=alpha, max_iter=200)
    solver = FreeSolver(tm, sc)
    for px_idx in strip_indices:
        try:
            intensity = xstk_3d[px_idx].T.astype(np.float64)
            r = solver.solve(intensity, verbose=False)
            results.append((px_idx, r.D_opt.T.astype(np.float32)))
        except Exception:
            results.append((px_idx, None))
    return results
