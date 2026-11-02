#!/usr/bin/env python
"""CPU-only smoke benchmark: synthetic FreeSolver / SLSQP throughput.

Self-contained (no external data, no MLX). Intended for environments where
only the classical CPU path exists — e.g. a Linux container or a CI runner —
so that "does the solver still run, and how fast on this box" can be
answered quickly.

The numbers are only meaningful together with the provenance block the
script prints (CPU, cores, Python / numpy / scipy / numba versions, git
commit). They are not comparable across machines and must not be copied
into tracked files as fixed values.

Usage:
    python -m deppro.benchmarks.cpu_smoke_bench                 # 20 px, L2
    python -m deppro.benchmarks.cpu_smoke_bench --n 50 --method LLS
    python -m deppro.benchmarks.cpu_smoke_bench --out /path/result.json
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], stderr=subprocess.DEVNULL, text=True,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        ).strip()
    except Exception:  # noqa: BLE001 — provenance only
        return "unknown"


def provenance() -> dict:
    import scipy
    try:
        import numba
        numba_v = numba.__version__
    except ImportError:
        numba_v = None
    cpu = platform.processor() or platform.machine()
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_commit": _git("rev-parse", "--short", "HEAD"),
        "git_dirty": _git("status", "--porcelain") != "",
        "platform": platform.platform(),
        "cpu": cpu,
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "numba": numba_v,
        "remote": os.environ.get("CLAUDE_CODE_REMOTE") == "true",
    }


def build_problem(n_pixels: int, seed: int, noise: float):
    """Five-layer profile (C/Al2O3/TiO2/SiO2/Si), 4 angles, step 4 A, 1 % noise.

    Same construction as src/deppro/tests/solver/test_freesolver_gradient_consistency.py
    so the smoke result stays tied to a problem the regression suite covers.
    """
    from deppro.depth_profile import create_five_layer_profile
    from deppro.transfer_matrix import TransferMatrix

    profile = create_five_layer_profile(depth_end=100, step_size=4.0)
    angles = np.array([0, 20, 40, 60])
    tm = TransferMatrix(
        depth=profile.depth,
        step_size=profile.depth[1] - profile.depth[0],
        angles=angles,
        photon_energy=1486.6,
    )
    tm.compute(profile.concentrations, profile.elements)
    I_true = tm.forward(profile.concentrations, normalize=True)

    rng = np.random.default_rng(seed)
    pixels = []
    for _ in range(n_pixels):
        I_obs = np.maximum(I_true + rng.normal(0.0, noise, I_true.shape) * I_true, 0)
        pixels.append(I_obs / I_obs.sum(axis=1, keepdims=True))
    return profile, tm, pixels


def run(n_pixels: int, method: str, alpha: float, seed: int, noise: float,
        max_iter: int | None = None, tol: float | None = None) -> dict:
    from deppro.inverse_solver import FreeSolver, SolverConfig

    profile, tm, pixels = build_problem(n_pixels, seed, noise)
    n_depth, n_elements = profile.concentrations.shape
    D_prior = np.ones((n_depth, n_elements)) / n_elements
    overrides = {k: v for k, v in (("max_iter", max_iter), ("tol", tol)) if v is not None}
    config = SolverConfig(method=method, alpha=alpha, scipy_method="SLSQP", **overrides)
    solver = FreeSolver(tm, config, D_prior)

    # warm-up (numba JIT, first-call overhead) — excluded from timing
    solver.solve(pixels[0])

    objectives, rmses, successes, iters, results = [], [], [], [], []
    t0 = time.perf_counter()
    for I_obs in pixels:
        r = solver.solve(I_obs)
        results.append(r)
        objectives.append(r.objective)
        successes.append(bool(r.success))
        iters.append(int(r.n_iterations))
        rmses.append(float(np.sqrt(np.mean((r.D_opt - profile.concentrations) ** 2))))
    elapsed = time.perf_counter() - t0

    # Stationarity at the returned point (outside the timed loop): relative
    # KKT residual with the exact analytic gradient, the same quantity the
    # solver's false-convergence guard uses. Large values mean SLSQP stopped
    # on ftol short of a stationary point — a fast px/s is then not a result.
    T_mats = [tm.get_matrix(e) for e in range(n_elements)]
    cons = solver._build_constraints(n_depth, n_elements)
    lb = np.full(n_depth * n_elements, config.min_conc)
    ub = np.full(n_depth * n_elements, config.max_conc)
    kkt = []
    for I_obs, r in zip(pixels, results):
        I_norm = I_obs / np.maximum(I_obs.sum(axis=1, keepdims=True), 1e-10)
        args = (I_norm, T_mats, n_depth, n_elements, config.degree)
        kkt.append(float(solver._kkt_stationarity_residual(
            r.D_opt.flatten(), args, cons, lb, ub)))

    return {
        "problem": {
            "profile": "five_layer", "n_depth": int(n_depth), "n_elements": int(n_elements),
            "n_angles": 4, "step_A": 4.0, "noise_rel": noise, "seed": seed,
            "method": method, "alpha": alpha, "scipy_method": "SLSQP",
            "max_iter": config.max_iter, "tol": config.tol,
            "n_pixels": n_pixels,
        },
        "elapsed_s": elapsed,
        "px_per_s": n_pixels / elapsed,
        "ms_per_px": 1e3 * elapsed / n_pixels,
        "success_fraction": float(np.mean(successes)),
        "iterations_median": float(np.median(iters)),
        "objective_median": float(np.median(objectives)),
        "kkt_residual_median": float(np.median(kkt)),
        "kkt_residual_max": float(np.max(kkt)),
        "profile_rmse_median": float(np.median(rmses)),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--n", type=int, default=20, help="pixels (default 20)")
    # L1 is excluded: FreeSolver runs it without an analytic gradient and never
    # evaluates its KKT residual, so the stationarity column would not be the
    # solver's own quantity for that method.
    p.add_argument("--method", default="L2", choices=["L2", "LLS", "MEM"])
    p.add_argument("--alpha", type=float, default=0.01,
                   help="regularisation strength (default 0.01; 0.1 is prior-"
                        "dominated and converges in a handful of iterations)")
    p.add_argument("--max-iter", type=int, default=None,
                   help="SolverConfig.max_iter (default: solver default)")
    p.add_argument("--tol", type=float, default=None,
                   help="SolverConfig.tol / SLSQP ftol (default: solver default)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--noise", type=float, default=0.01)
    p.add_argument("--out", help="write JSON here (untracked location)")
    a = p.parse_args(argv)

    record = {"provenance": provenance(),
              "result": run(a.n, a.method, a.alpha, a.seed, a.noise, a.max_iter, a.tol)}
    r, pv = record["result"], record["provenance"]
    print(f"cpu_smoke_bench  {pv['cpu']}  x{pv['cpu_count']}  "
          f"py{pv['python']} numpy {pv['numpy']} scipy {pv['scipy']} numba {pv['numba']}  "
          f"git {pv['git_commit']}{' (dirty)' if pv['git_dirty'] else ''}")
    print(f"  {r['problem']['method']} alpha={r['problem']['alpha']}  "
          f"{r['problem']['n_pixels']} px: {r['px_per_s']:.2f} px/s  "
          f"({r['ms_per_px']:.1f} ms/px)  success {r['success_fraction']:.0%}  "
          f"iters med {r['iterations_median']:.0f}  "
          f"objective med {r['objective_median']:.3e}  "
          f"KKT med {r['kkt_residual_median']:.1e} max {r['kkt_residual_max']:.1e}  "
          f"profile RMSE med {r['profile_rmse_median']:.4f}")
    if a.out:
        with open(a.out, "w") as f:
            json.dump(record, f, indent=2)
        print(f"  -> {a.out}")
    return 0 if r["success_fraction"] > 0.5 else 1


if __name__ == "__main__":
    sys.exit(main())
