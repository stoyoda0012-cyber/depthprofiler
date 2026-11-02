"""PoC: regenerate the /Drlt reconstruction of a DepthProfiler ezdeprof H5
with the MLX batched solver, writing into a COPY (never the original).

The 4D "Reconstructed" videos (EzDeP4D.m) read /Dstk (GT) and /Drlt
(reconstruction). /Drlt generation is the bottleneck for large stacks with
per-pixel neighbor-MEM SLSQP. This tool streams Xstk
through MLXBatchSemiFreeSolver on the GPU and overwrites /Drlt in a copy,
so MATLAB reads the file identically (same Dstk / ezdeprof / metadata).

Read side (Xstk + metadata) = the original file (read-only).
Write side (/Drlt) = the copy. Different files -> no handle conflict.

Usage:
    python -m deppro.tools.mlx_reconstruct_drlt SRC.h5 \
        [--out OUT.h5] [--method L1] [--chunk 50000] [--max-pixels N] \
        [--thickness-init 210] [--abruptness-init 20]

--max-pixels N limits to the first N pixels (validation); default = all.
"""
import argparse
import shutil
import time
from pathlib import Path

import numpy as np
import h5py

from deppro.streaming_io import StreamingDepthReader
from deppro.transfer_matrix import TransferMatrix
from deppro.inverse_solver import SolverConfig, SemiFreeSolverConfig
from deppro.solvers_mlx import MLXBatchSemiFreeSolver, MLXBatchFreeSolver
from deppro.simulate.d2s2d import estimate_init_from_drlt


def build_tm(meta):
    tm = TransferMatrix(
        depth=meta.depth, step_size=meta.step_size, angles=meta.angles,
        photon_energy=meta.photon_energy, work_function=meta.work_function,
        gamma_eff=meta.gamma_eff)
    tm.trans_mat = np.asarray(meta.trans_mat, dtype=np.float64)
    tm.n_elements = meta.n_elements
    return tm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('--out', default=None,
                    help='output copy (default: SRC with _mlx suffix)')
    ap.add_argument('--solver', default='semifree',
                    choices=['semifree', 'free'],
                    help="'free' = per-depth free simplex (fast, but NO "
                         "structural prior: substrate interface smears on "
                         "structured multilayers — counterexample only).")
    ap.add_argument('--method', default='L1')
    ap.add_argument('--alpha', type=float, default=0.01)
    ap.add_argument('--chunk', type=int, default=50_000)
    ap.add_argument('--max-pixels', type=int, default=0)
    ap.add_argument('--thickness-init', type=float, default=210.0)
    ap.add_argument('--abruptness-init', type=float, default=20.0)
    ap.add_argument('--warmstart', action='store_true',
                    help="per-pixel warm-start from the source's existing "
                         "/Drlt (e.g. an NN reconstruction) via "
                         "estimate_init_from_drlt; narrows the t bracket.")
    ap.add_argument('--t-radius', type=float, default=50.0,
                    help='warm-start thickness search half-width [A]')
    ap.add_argument('--outer-rounds', type=int, default=0,
                    help='0 = solver default (8 scratch / 6 warm)')
    ap.add_argument('--no-copy', action='store_true',
                    help='skip the file copy (out must already exist)')
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out) if args.out else src.with_name(
        src.stem + '_mlx' + src.suffix)

    # 1. Copy the whole file (preserve Dstk / Xstk / ezdeprof / everything).
    if not args.no_copy:
        if out.exists():
            raise SystemExit(f"refusing to overwrite existing {out}; "
                             f"delete it or pass --no-copy")
        print(f"[copy] {src.name} -> {out.name} "
              f"({src.stat().st_size/1e9:.1f} GB) ...", flush=True)
        t0 = time.time()
        shutil.copy2(src, out)
        print(f"[copy] done in {time.time()-t0:.0f}s", flush=True)

    # 2. Metadata + transfer matrix from the ORIGINAL (read-only).
    with StreamingDepthReader(str(src)) as reader:
        meta = reader.read_metadata()
        n_px = meta.n_pixels
        if args.max_pixels > 0:
            n_px = min(n_px, args.max_pixels)
        tm = build_tm(meta)

        semi_cfg = SemiFreeSolverConfig(
            thickness_init=args.thickness_init,
            abruptness_init=args.abruptness_init,
            thickness_bounds=(50.0, 500.0), abruptness_bounds=(1.0, 100.0),
            optimize_thickness=True, optimize_abruptness=False,
            free_indices=meta.free_indices,
            linked_indices=meta.linked_indices,
            substrate_indices=meta.substrate_indices,
            free_linked_const_override=meta.linkedconst.tolist())
        if args.solver == 'free':
            # per-depth free simplex — NO substrate/erf/charge-balance prior.
            sc = SolverConfig(method=args.method, alpha=args.alpha,
                              use_charge_constraint=False)
            solver = MLXBatchFreeSolver(tm, sc, elements=None)
        else:
            sc = SolverConfig(method=args.method, alpha=args.alpha,
                              use_xreg_weighting=True, scipy_method='SLSQP')
            solver_kw = {}
            if args.warmstart:
                solver_kw['t_radius_init'] = args.t_radius
                solver_kw['outer_rounds'] = args.outer_rounds or 6
            elif args.outer_rounds:
                solver_kw['outer_rounds'] = args.outer_rounds
            solver = MLXBatchSemiFreeSolver(tm, sc, semi_cfg, D_prior=None,
                                            elements=meta.elements, **solver_kw)
        flc = np.maximum(np.asarray(meta.linkedconst, dtype=np.float64), 1.0)

        print(f"[setup] {n_px:,} px, {meta.n_elements} elem, "
              f"{meta.n_depth} depth, {meta.n_angles} ang | "
              f"solver={args.solver} method={args.method} "
              f"{'warm-start(src Drlt)' if args.warmstart else f'init thk={args.thickness_init}'}",
              flush=True)

        # 3. Stream: read Xstk chunk (original) -> MLX solve -> write Drlt (copy).
        src_f = h5py.File(str(src), 'r') if args.warmstart else None
        with h5py.File(out, 'r+') as of:
            drlt = of['/Drlt']          # h5py shape (n_pixels, n_elem, n_depth)
            assert drlt.shape[0] == meta.n_pixels, drlt.shape
            t_solve = 0.0
            t_start = time.time()
            done = 0
            for off in range(0, n_px, args.chunk):
                cnt = min(args.chunk, n_px - off)
                X = reader.read_xstk_chunk(off, cnt)      # (cnt, n_ang, n_elem)
                ts = time.time()
                if args.solver == 'free':
                    b = solver.solve_batch(X)             # no structural init
                elif args.warmstart:
                    # per-pixel init from the source's existing Drlt
                    D_src = np.transpose(
                        src_f['/Drlt'][off:off + cnt], (0, 2, 1))  # (cnt,depth,elem)
                    est = estimate_init_from_drlt(
                        D_src.astype(np.float64), meta.depth,
                        meta.substrate_indices, meta.free_indices, flc)
                    b = solver.solve_batch(
                        X, thickness_init=est['thickness'],
                        abruptness_init=est['abruptness'],
                        film_conc_init=est['film_conc'])
                else:
                    b = solver.solve_batch(
                        X, thickness_init=args.thickness_init,
                        abruptness_init=args.abruptness_init)
                t_solve += time.time() - ts
                # D_opt (cnt, n_depth, n_elem) -> Drlt layout (cnt, n_elem, n_depth)
                drlt[off:off + cnt] = np.transpose(
                    b.D_opt, (0, 2, 1)).astype(np.float32)
                done += cnt
                el = time.time() - t_start
                print(f"  [{done:>9,}/{n_px:,}] {done/el:7.0f} px/s "
                      f"(solve {done/t_solve:7.0f} px/s) "
                      f"eta {(n_px-done)/(done/el)/60:5.1f} min", flush=True)
        if src_f is not None:
            src_f.close()

    print(f"\n[done] wrote /Drlt for {n_px:,} px to {out}")
    print(f"  wall {time.time()-t_start:.0f}s | solve {t_solve:.0f}s "
          f"({n_px/t_solve:.0f} px/s solve-only)")
    if args.max_pixels > 0:
        print(f"  NOTE: only first {n_px:,} px written; rest of /Drlt is the "
              f"original reconstruction (partial validation file).")


if __name__ == '__main__':
    main()
