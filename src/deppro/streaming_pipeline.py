"""
Streaming Depth Profiling Pipeline
====================================

Processes large-scale ARXPS depth profiling data (up to 200M pixels)
by reading/writing HDF5 files in chunks, with parallel per-pixel solving.

Architecture:
    1. Read metadata from input H5 (once)
    2. Set up TransferMatrix, solver config, lambda_basic (once)
    3. For each temporal chunk:
       a. Read Xstk chunk (with prefetch)
       b. Parallel solve (multiprocessing Pool)
       c. Write Dtrn chunk (async)
    4. (Optional) Phase 2: neighbor-MEM refinement
    5. Finalize output H5

Usage:
    # Python API
    from deppro.streaming_pipeline import StreamingDepthPipeline, StreamingPipelineConfig
    config = StreamingPipelineConfig(method='LLS', n_loops=10, use_xreg=True)
    pipeline = StreamingDepthPipeline('input.h5', 'output.h5', config)
    pipeline.run()

    # CLI
    python -m deppro.streaming_pipeline --h5 input.h5 --output output.h5 \\
        --method LLS --loops 10 --xreg --workers 0

Date: 2026-02-14
"""

import numpy as np
import time
import sys
import multiprocessing as mp
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from pathlib import Path

from deppro import (
    TransferMatrix,
    CompoundDatabase,
    SolverConfig,
    SemiFreeSolverConfig,
)
from deppro.streaming_io import (
    StreamingDepthReader,
    StreamingDepthWriter,
    DepthProfileMetadata,
    auto_chunk_size,
    estimate_memory_budget,
)
from deppro.solver_workers import (
    worker_initializer,
    worker_phase1,
    worker_phase2,
    build_first_neighbor_indices,
    neighbor_average_prior,
    serialize_transfer_matrix,
)
from deppro.derived_metrics import (
    DerivedMetricsConfig,
    compute_all_derived_batch,
)


# ==============================================================================
# Configuration
# ==============================================================================

@dataclass
class StreamingPipelineConfig:
    """Configuration for the streaming depth profiling pipeline.

    Attributes:
        method: Optimization method ('LLS', 'L1', or 'NN')
        alpha: Regularization strength
        n_loops: Number of iterative loops per pixel
        degree_decay: Degree decay factor per loop
        use_xreg: Use Xreg weighting
        n_workers: Number of parallel workers (0 = auto)
        nx: Spatial grid width (0 = auto-detect)
        ny: Spatial grid height (0 = auto-detect)
        mem_alpha: MEM regularization strength (Phase 2)
        mem_loops: MEM refinement loops (0 = Phase 1 only)
        mem_scipy_method: SciPy method for MEM solver
        thickness_init: Initial film thickness (Å)
        abruptness_init: Initial abruptness (Å)
        solver_max_iter: Max solver iterations per pixel
        solver_tol: Solver tolerance
        memory_budget_gb: Memory budget in GB
        chunk_pixels: Pixels per chunk (0 = auto from budget)
        compression: Output compression ('gzip' or None)
        compression_opts: Compression level (1-9)
        nn_model_path: Path to NN model weights (.npz) for method='NN'
        nn_batch_size: Batch size for NN inference (default: 65536)
    """
    method: str = 'LLS'
    alpha: float = 1.0
    n_loops: int = 10
    degree_decay: float = 0.8
    use_xreg: bool = True
    n_workers: int = 0
    nx: int = 0
    ny: int = 0
    mem_alpha: float = 0.01
    mem_loops: int = 0
    mem_scipy_method: str = 'SLSQP'
    n_ring: int = 1
    thickness_init: float = 210.0
    abruptness_init: float = 20.0
    solver_max_iter: int = 200
    solver_tol: float = 1e-6
    memory_budget_gb: float = 4.0
    chunk_pixels: int = 0
    max_chunks: int = 0           # 0 = all chunks, >0 = stop after N chunks (for testing)
    full_gui_compat: bool = True  # Create all 24 MATLAB GUI datasets
    compression: str = 'gzip'
    compression_opts: int = 1
    nn_model_path: str = ''
    nn_batch_size: int = 65536
    output_dtype: str = 'float32'   # 'float16' halves D write volume (throughput)

    def to_dict(self) -> dict:
        """Serialize to dict for JSON storage."""
        return {k: v for k, v in self.__dict__.items()}


# ==============================================================================
# Pipeline
# ==============================================================================

class StreamingDepthPipeline:
    """Streaming depth profiling pipeline.

    Processes large H5 files in chunks with parallel per-pixel solving.
    Supports Phase 1 (LLS/L1 iterative) and Phase 2 (neighbor-MEM).
    """

    def __init__(
        self,
        h5_path: str,
        output_path: str,
        config: StreamingPipelineConfig,
    ):
        self.h5_path = str(h5_path)
        self.output_path = str(output_path)
        self.config = config

        # Resolved at setup time
        self._meta: Optional[DepthProfileMetadata] = None
        self._n_workers: int = 0
        self._chunk_size: int = 0
        self._tm_data: Optional[dict] = None
        self._lambda_basic: Optional[np.ndarray] = None
        self._semi_config_base: Optional[dict] = None
        self._derived_config: Optional[DerivedMetricsConfig] = None
        self._n_angles_opt: int = 0
        self._n_fv: int = 0
        self._n_sv: int = 0

    def run(self, resume: bool = False):
        """Full pipeline: Phase 1 → (optional) Phase 2 → finalize.

        Args:
            resume: If True, resume from checkpoint in existing output file.
        """
        t0 = time.time()

        with StreamingDepthReader(self.h5_path) as reader:
            self._setup(reader)

            if resume:
                resume_info = self._check_resume()
                if resume_info is None:
                    print("No checkpoint found, starting fresh...")
                    resume = False

            self._print_header()

            if resume:
                # Resume mode: open existing output for append
                import h5py as _h5py
                writer_f = _h5py.File(self.output_path, 'a')
                writer = StreamingDepthWriter.__new__(StreamingDepthWriter)
                writer.output_path = self.output_path
                writer.meta = self._meta
                writer.async_write = True
                writer.compression = self.config.compression
                writer.compression_opts = self.config.compression_opts
                writer._f = writer_f
                writer._executor = None
                writer._write_future = None
                nxy = self._meta.nxy if self._meta.nxy > 0 else min(self._meta.n_pixels, 250000)
                writer._h5_chunk_d = (nxy, self._meta.n_elements, self._meta.n_depth)
                writer._h5_chunk_q = (3, nxy)

                from concurrent.futures import ThreadPoolExecutor as _TPE
                writer._executor = _TPE(max_workers=1)

                try:
                    start_chunk = resume_info['last_chunk_phase1'] + 1
                    print(f"\n[Resume] Resuming Phase 1 from chunk {start_chunk}")
                    self.run_phase1(reader, writer, start_chunk=start_chunk)

                    if self.config.mem_loops > 0:
                        self.run_phase2(reader, writer)
                    else:
                        print("\n[Phase 2] Skipped (mem_loops=0), copying Dtrn → Drlt...")
                        writer.copy_dtrn_to_drlt()

                    writer.finalize()
                finally:
                    if writer._executor is not None:
                        writer._executor.shutdown(wait=True)
                    writer_f.close()
            else:
                # Fresh run
                with StreamingDepthWriter(
                    self.output_path, self._meta,
                    compression=self.config.compression,
                    compression_opts=self.config.compression_opts,
                    full_gui_compat=self.config.full_gui_compat,
                    n_angles_opt=self._n_angles_opt,
                    n_fv=self._n_fv,
                    n_sv=self._n_sv,
                    output_dtype=self.config.output_dtype,
                ) as writer:
                    writer.copy_metadata_from(reader)
                    writer.write_config(self.config.to_dict())

                    drlt_inline = (self.config.method == 'NN'
                                   and not self.config.full_gui_compat)
                    if self.config.method == 'NN':
                        self.run_phase1_nn(reader, writer)
                    else:
                        self.run_phase1(reader, writer)

                    if self.config.mem_loops > 0:
                        self.run_phase2(reader, writer)
                    elif drlt_inline:
                        print("\n[Phase 2] Skipped; Drlt written inline during Phase 1 "
                              "(no read-back copy).")
                    else:
                        print("\n[Phase 2] Skipped (mem_loops=0), copying Dtrn → Drlt...")
                        writer.copy_dtrn_to_drlt()

                    writer.finalize()

        total_time = time.time() - t0
        rate = self._meta.n_pixels / total_time
        print(f"\n{'='*72}")
        print(f"Pipeline complete!")
        print(f"  Output: {self.output_path}")
        print(f"  Total time: {total_time:.1f}s ({self._format_time(total_time)})")
        print(f"  Rate: {rate:.0f} px/s ({1000/rate:.1f} ms/px)")
        print(f"{'='*72}")

    def _check_resume(self) -> Optional[dict]:
        """Check if output file has a valid checkpoint for resume.

        Returns:
            Dict with checkpoint info, or None if no valid checkpoint.
        """
        import h5py as _h5py
        output = Path(self.output_path)
        if not output.exists():
            return None

        try:
            with _h5py.File(self.output_path, 'r') as f:
                if '_deppro_incomplete' not in f.attrs:
                    print(f"[Resume] Output file exists but is already complete.")
                    return None

                info = {}
                if '_deppro_last_chunk_phase1' in f.attrs:
                    info['last_chunk_phase1'] = int(f.attrs['_deppro_last_chunk_phase1'])
                else:
                    info['last_chunk_phase1'] = -1  # No chunks done

                if '_deppro_last_chunk_phase2' in f.attrs:
                    info['last_chunk_phase2'] = int(f.attrs['_deppro_last_chunk_phase2'])

                n_pixels = int(f.attrs.get('_deppro_n_pixels', 0))
                info['n_pixels'] = n_pixels

                print(f"[Resume] Found checkpoint: "
                      f"Phase 1 chunk {info['last_chunk_phase1']}, "
                      f"{n_pixels:,} pixels")
                return info
        except Exception as e:
            print(f"[Resume] Error reading checkpoint: {e}")
            return None

    def run_phase1(
        self,
        reader: StreamingDepthReader,
        writer: StreamingDepthWriter,
        start_chunk: int = 0,
    ):
        """Phase 1: Iterative LLS/L1 solve for all pixels, chunk by chunk.

        Args:
            reader: Input H5 reader
            writer: Output H5 writer
            start_chunk: First chunk to process (for resume support)
        """
        meta = self._meta
        cfg = self.config
        n_total = meta.n_pixels
        chunk_size = self._chunk_size

        n_chunks_total = (n_total + chunk_size - 1) // chunk_size
        # max_chunks: limit for testing (0 = all)
        if cfg.max_chunks > 0:
            end_chunk = min(start_chunk + cfg.max_chunks, n_chunks_total)
        else:
            end_chunk = n_chunks_total
        n_chunks = end_chunk - start_chunk

        print(f"\n{'='*72}")
        print(f"Phase 1: {cfg.method}×{cfg.n_loops}+TM+Xreg "
              f"({n_total:,} pixels, {self._n_workers} workers)")
        if start_chunk > 0:
            print(f"  Resuming from chunk {start_chunk}/{n_chunks_total}")
        if cfg.max_chunks > 0:
            print(f"  ⚠️  Limited to {n_chunks} chunks (--max-chunks {cfg.max_chunks})")
        print(f"  Chunks: {n_chunks} × {chunk_size:,} pixels "
              f"(of {n_chunks_total} total)")
        print(f"{'='*72}")

        phase1_start = time.time()
        total_success = 0
        total_pixels_done = start_chunk * chunk_size

        # Create multiprocessing pool (reused across chunks)
        pool = mp.Pool(self._n_workers, initializer=worker_initializer)

        try:
            # Start prefetching first chunk
            first_offset = start_chunk * chunk_size
            reader.prefetch_xstk_chunk(first_offset, chunk_size)

            for chunk_idx in range(start_chunk, end_chunk):
                offset = chunk_idx * chunk_size
                count = min(chunk_size, n_total - offset)
                chunk_start = time.time()

                # Get current chunk (prefetched or direct read)
                Xstk = reader.get_prefetched_xstk()
                if Xstk is None:
                    Xstk = reader.read_xstk_chunk(offset, count)

                # Start prefetching next chunk
                next_offset = offset + count
                if next_offset < n_total:
                    reader.prefetch_xstk_chunk(next_offset,
                                                min(chunk_size, n_total - next_offset))

                # Build Phase 1 args for this chunk
                pixel_indices = np.arange(offset, offset + count)
                args_list = self._build_phase1_args(Xstk, pixel_indices)

                # Parallel solve
                D_opt_chunk = np.zeros((count, meta.n_depth, meta.n_elements))
                Q_chunk = np.zeros((count, 3))  # chi2, thickness, abruptness
                success_chunk = 0

                for res in pool.imap_unordered(worker_phase1, args_list):
                    # worker_phase1 returns 9 values (chi2 and objective are
                    # separate); the 8-name unpacking used until 2026-11-07
                    # raised ValueError on the first pixel.
                    idx, D_opt, thk, abrp, chi2, obj, suc, warm, nit = res
                    if D_opt is not None:
                        D_opt_chunk[idx] = D_opt
                        Q_chunk[idx] = [chi2, thk, abrp]
                        success_chunk += 1

                # Write results
                writer.write_dtrn_chunk(offset, D_opt_chunk, Q_chunk)

                # GUI compat: copy source datasets + compute derived metrics
                if cfg.full_gui_compat and self._derived_config is not None:
                    writer.copy_source_chunk(reader, offset, count)

                    thicknesses = Q_chunk[:, 1]
                    abruptnesses = Q_chunk[:, 2]
                    derived = compute_all_derived_batch(
                        D_opt_chunk, thicknesses, abruptnesses,
                        self._derived_config,
                    )
                    writer.write_derived_trn(
                        offset, derived,
                        alpha_val=cfg.alpha,
                        degree_val=cfg.degree_decay ** max(0, cfg.n_loops - 1),
                    )

                writer.mark_chunk_complete(1, chunk_idx)

                total_success += success_chunk
                total_pixels_done += count
                chunk_time = time.time() - chunk_start
                elapsed = time.time() - phase1_start
                pixels_in_scope = total_pixels_done - start_chunk * chunk_size
                rate = pixels_in_scope / elapsed if elapsed > 0 else 0
                target_pixels = end_chunk * chunk_size
                remaining = target_pixels - total_pixels_done
                eta = remaining / rate if rate > 0 else 0

                print(f"  Phase 1 [{chunk_idx+1}/{n_chunks_total}] "
                      f"{total_pixels_done:,}/{n_total:,} "
                      f"({total_pixels_done*100/n_total:.1f}%) "
                      f"rate={rate:.0f} px/s "
                      f"ETA={self._format_time(eta)} "
                      f"chunk={chunk_time:.1f}s "
                      f"success={success_chunk}/{count}")

        finally:
            pool.terminate()
            pool.join()

        phase1_time = time.time() - phase1_start
        print(f"\nPhase 1 complete: {total_pixels_done:,} pixels, "
              f"{total_success:,} success ({total_success*100/total_pixels_done:.1f}%), "
              f"{phase1_time:.1f}s ({total_pixels_done/phase1_time:.0f} px/s)")

    def run_phase1_nn(
        self,
        reader: StreamingDepthReader,
        writer: StreamingDepthWriter,
    ):
        """Phase 1 (NN mode): Batch neural network inference for all pixels.

        Replaces the per-pixel SLSQP optimization of Phase 1 with batched
        MLX NN inference.

        NN predicts D(z) directly from normalized I_obs.
        Post-processing enforces sum=1 constraint.
        Thickness/abruptness are estimated from the predicted profile.
        """
        import mlx.core as mx
        from deppro.inference.train import load_model
        from deppro.inference.postprocess import enforce_constraints_mlx

        meta = self._meta
        cfg = self.config
        n_total = meta.n_pixels
        chunk_size = self._chunk_size
        nn_batch = cfg.nn_batch_size

        # Load NN model
        input_dim = meta.n_angles * meta.n_elements
        output_dim = meta.n_depth * meta.n_elements
        print(f"\n  Loading NN model: {cfg.nn_model_path}")
        model = load_model(cfg.nn_model_path,
                           input_dim=input_dim, output_dim=output_dim)

        n_chunks_total = (n_total + chunk_size - 1) // chunk_size
        if cfg.max_chunks > 0:
            end_chunk = min(cfg.max_chunks, n_chunks_total)
        else:
            end_chunk = n_chunks_total

        print(f"\n{'='*72}")
        print(f"Phase 1: NN inference ({n_total:,} pixels)")
        print(f"  Model: {cfg.nn_model_path}")
        print(f"  Chunks: {end_chunk} × {chunk_size:,} pixels")
        print(f"{'='*72}")

        phase1_start = time.time()
        total_pixels_done = 0
        # Per-stage wall-clock accounting — the NN matmul is a tiny fraction of
        # streaming cost; this exposes the real bottleneck (H5 read/write/gzip).
        stage_t = {'read': 0.0, 'norm': 0.0, 'infer': 0.0, 'thick': 0.0, 'write': 0.0}
        # Throughput path: write Drlt only (Dtrn unused), and overlap that write
        # with the NEXT chunk's read+compute by skipping the per-chunk checkpoint
        # sync (these runs are not resumed). The single async write future is
        # drained at the next chunk's write_drlt_chunk and at finalize, so the
        # writer thread keeps exclusive file access — no h5py thread-safety race.
        drlt_inline = (cfg.mem_loops == 0 and not cfg.full_gui_compat)
        # Double-buffer reads on the throughput path: prefetch chunk N+1 in the
        # reader's background thread while chunk N computes (and writes). Reader
        # and writer use different files + own threads, so read(next)/compute/
        # write(current) form a 3-way overlap bounded by disk bandwidth.
        prefetch_on = drlt_inline and getattr(reader, 'prefetch', False)
        if prefetch_on:
            reader.prefetch_xstk_chunk(0, min(chunk_size, n_total), dtype=np.float32)
        if drlt_inline:
            # Non-resumable: skip the per-chunk disk sync (finalize still flushes).
            writer.flush_per_chunk = False

        for chunk_idx in range(end_chunk):
            offset = chunk_idx * chunk_size
            count = min(chunk_size, n_total - offset)
            chunk_start = time.time()

            # Read I_obs for this chunk (float32 — NN path, avoids f64 round-trip).
            # On the throughput path, collect the prefetched chunk and immediately
            # kick off the next chunk's read so it overlaps this chunk's compute.
            _ts = time.perf_counter()
            if prefetch_on:
                Xstk = reader.get_prefetched_xstk()
                if Xstk is None:
                    Xstk = reader.read_xstk_chunk(offset, count, dtype=np.float32)
                next_off = offset + chunk_size
                if next_off < n_total:
                    reader.prefetch_xstk_chunk(
                        next_off, min(chunk_size, n_total - next_off), dtype=np.float32)
            else:
                Xstk = reader.read_xstk_chunk(offset, count, dtype=np.float32)
            # Xstk: (count, n_angles, n_elements)
            stage_t['read'] += time.perf_counter() - _ts

            # Normalize: I / sum(I)
            _ts = time.perf_counter()
            I_flat = np.ascontiguousarray(Xstk.reshape(count, -1))
            I_sum = I_flat.sum(axis=1, keepdims=True)
            I_sum[I_sum == 0] = 1
            I_flat /= I_sum
            stage_t['norm'] += time.perf_counter() - _ts

            # Batch NN inference + on-device post-processing (sum=1 enforced
            # in MLX before the GPU->host transfer, which avoids a separate
            # numpy/CPU pass)
            _ts = time.perf_counter()
            D_flat = np.zeros((count, output_dim), dtype=np.float32)
            for b_start in range(0, count, nn_batch):
                b_end = min(b_start + nn_batch, count)
                I_batch = mx.array(I_flat[b_start:b_end])
                D_batch = model(I_batch)
                D_batch = enforce_constraints_mlx(D_batch, meta.n_depth, meta.n_elements)
                mx.eval(D_batch)
                D_flat[b_start:b_end] = np.array(D_batch)

            # Reshape to (count, n_depth, n_elements). Keep float32 — the writer
            # casts to float32 anyway and thickness est works in float32; the
            # GUI-compat derived path casts to float64 locally where it needs it.
            D_opt_chunk = D_flat.reshape(count, meta.n_depth, meta.n_elements)
            stage_t['infer'] += time.perf_counter() - _ts

            # Estimate thickness from profile (find substrate crossover)
            _ts = time.perf_counter()
            Q_chunk = np.zeros((count, 3))  # chi2, thickness, abruptness
            Q_chunk[:, 1] = cfg.thickness_init  # default thickness
            Q_chunk[:, 2] = cfg.abruptness_init  # default abruptness

            if len(meta.substrate_indices) > 0:
                sub_idx = meta.substrate_indices[0]
                sub_profile = D_opt_chunk[:, :, sub_idx]  # (count, n_depth)
                depth = meta.depth
                # First 50% crossover depth per pixel — vectorized over the
                # whole chunk (argmax returns the first True index; any() gates
                # pixels with no crossover, which keep thickness_init). Replaces
                # a per-pixel Python for-loop that dominated post-inference cost.
                above = sub_profile > 0.5                 # (count, n_depth)
                has_cross = above.any(axis=1)             # (count,)
                first_idx = above.argmax(axis=1)          # (count,) first True or 0
                Q_chunk[has_cross, 1] = depth[first_idx[has_cross]]
            stage_t['thick'] += time.perf_counter() - _ts

            # Write results.
            # Throughput path (no Phase 2, no GUI-compat): Drlt is the final
            # result and Dtrn is an unused intermediate — write Drlt ONLY,
            # inline from the in-memory chunk. This both (a) avoids the full
            # read-back+rewrite pass at finalize (copy_dtrn_to_drlt, which
            # dominated wall clock — gzip double-compression) and (b) halves
            # the per-chunk compressed-write work by skipping Dtrn entirely.
            # GUI-compat keeps Dtrn + the copy so derived *trn→*rlt stay consistent.
            _ts = time.perf_counter()
            if drlt_inline:
                writer.write_drlt_chunk(offset, D_opt_chunk, Q_chunk)
            else:
                writer.write_dtrn_chunk(offset, D_opt_chunk, Q_chunk)
            stage_t['write'] += time.perf_counter() - _ts

            # GUI compat
            if cfg.full_gui_compat and self._derived_config is not None:
                writer.copy_source_chunk(reader, offset, count)

                thicknesses = Q_chunk[:, 1]
                abruptnesses = Q_chunk[:, 2]
                derived = compute_all_derived_batch(
                    D_opt_chunk.astype(np.float64), thicknesses, abruptnesses,
                    self._derived_config,
                )
                writer.write_derived_trn(
                    offset, derived,
                    alpha_val=0.0,
                    degree_val=1.0,
                )

            # Skip the per-chunk checkpoint sync on the throughput path so the
            # async write overlaps the next chunk's compute (see drlt_inline note).
            if not drlt_inline:
                writer.mark_chunk_complete(1, chunk_idx)

            total_pixels_done += count
            chunk_time = time.time() - chunk_start
            elapsed = time.time() - phase1_start
            rate = total_pixels_done / elapsed if elapsed > 0 else 0

            remaining = n_total - total_pixels_done
            eta = remaining / rate if rate > 0 else 0

            print(f"  Phase 1 NN [{chunk_idx+1}/{n_chunks_total}] "
                  f"{total_pixels_done:,}/{n_total:,} "
                  f"({total_pixels_done*100/n_total:.1f}%) "
                  f"rate={rate:,.0f} px/s "
                  f"ETA={self._format_time(eta)} "
                  f"chunk={chunk_time:.2f}s")

        phase1_time = time.time() - phase1_start
        print(f"\nPhase 1 (NN) complete: {total_pixels_done:,} pixels, "
              f"{phase1_time:.1f}s ({total_pixels_done/phase1_time:,.0f} px/s)")
        tracked = sum(stage_t.values()) or 1e-9
        print("  Stage breakdown (px/s | %% of tracked):")
        for k in ('read', 'norm', 'infer', 'thick', 'write'):
            t = stage_t[k]
            pps = total_pixels_done / t if t > 0 else float('inf')
            print(f"    {k:<6}: {t:6.2f}s  {pps:>13,.0f} px/s  {100*t/tracked:5.1f}%")

    def run_phase2(
        self,
        reader: StreamingDepthReader,
        writer: StreamingDepthWriter,
    ):
        """Phase 2: Neighbor-MEM refinement, chunk by chunk.

        Reads Dtrn (Phase 1 results) from writer's output file,
        computes neighbor averages per time step, and runs MEM.
        """
        meta = self._meta
        cfg = self.config
        n_total = meta.n_pixels
        nxy = meta.nxy

        if nxy == 0:
            print("[Phase 2] ERROR: Cannot run neighbor-MEM without spatial grid (nx/ny)")
            writer.copy_dtrn_to_drlt()
            return

        n_time_steps = meta.nt
        print(f"\n{'='*72}")
        print(f"Phase 2: {cfg.n_ring}-ring Neighbor-MEM×{cfg.mem_loops} "
              f"(α={cfg.mem_alpha}, {self._n_workers} workers)")
        print(f"  Time steps: {n_time_steps}, {nxy:,} pixels/step")
        print(f"{'='*72}")

        phase2_start = time.time()

        # We need to read Dtrn from the output file
        # Flush any pending writes first
        writer._wait_pending_write()

        # Read Dtrn back and process per time step
        # Open output file for reading Dtrn
        import h5py
        out_f = writer._f

        # Build neighbor graph (same for all time steps)
        pixel_template = np.arange(nxy)
        neighbors = build_first_neighbor_indices(
            pixel_template, nx=meta.nx, ny=meta.ny,
            n_ring=cfg.n_ring,
        )

        pool = mp.Pool(self._n_workers, initializer=worker_initializer)

        try:
            for mem_loop in range(cfg.mem_loops):
                print(f"\n  MEM loop {mem_loop+1}/{cfg.mem_loops}...")
                loop_start = time.time()

                for t in range(n_time_steps):
                    offset = t * nxy
                    count = nxy

                    # Read current D_opt for this time step
                    # On first MEM loop: read from Dtrn
                    # On subsequent loops: read from Drlt
                    source_key = 'Dtrn' if mem_loop == 0 else 'Drlt'
                    raw = out_f[source_key][offset:offset + count].astype(np.float64)
                    D_opt_step = np.ascontiguousarray(np.transpose(raw, (0, 2, 1)))

                    # Read Xstk for this time step
                    Xstk_step = reader.read_xstk_chunk(offset, count)

                    # Read thickness/abruptness from Q
                    # Q layout: (3, n_pixels) — row 0=chi2, row 1=thickness, row 2=abruptness
                    Q_source = 'Qtrn' if mem_loop == 0 else 'Qrlt'
                    Q_step_raw = out_f[Q_source][:, offset:offset + count]  # (3, count)
                    Q_step = Q_step_raw.T.astype(np.float64)  # (count, 3) for local use
                    thickness_step = Q_step[:, 1]
                    abruptness_step = Q_step[:, 2]

                    # Compute neighbor-averaged D_prior
                    D_prior_step = neighbor_average_prior(D_opt_step, neighbors)

                    # Build Phase 2 args
                    args_list = self._build_phase2_args(
                        Xstk_step, D_opt_step, D_prior_step,
                        thickness_step, abruptness_step,
                        np.arange(count),
                    )

                    # Parallel MEM solve
                    D_opt_result = D_opt_step.copy()
                    Q_result = Q_step.copy()  # (count, 3) — local Python layout

                    for res in pool.imap_unordered(worker_phase2, args_list):
                        idx, D_opt, thk, abrp, chi2, obj, suc, warm = res
                        if D_opt is not None:
                            D_opt_result[idx] = D_opt
                            Q_result[idx, 0] = chi2
                            Q_result[idx, 1] = thk
                            Q_result[idx, 2] = abrp

                    # Write Drlt for this time step
                    # Q_result is (count, 3) Python layout → _do_write transposes to (3, count)
                    writer.write_drlt_chunk(offset, D_opt_result, Q_result)

                    if (t + 1) % max(1, n_time_steps // 10) == 0:
                        elapsed = time.time() - loop_start
                        print(f"    [{t+1}/{n_time_steps}] ({elapsed:.1f}s)")

                writer.mark_chunk_complete(2, mem_loop)

        finally:
            pool.terminate()
            pool.join()

        phase2_time = time.time() - phase2_start
        print(f"\nPhase 2 complete: {n_total:,} pixels, {phase2_time:.1f}s")

    # ==========================================================================
    # Setup
    # ==========================================================================

    def _setup(self, reader: StreamingDepthReader):
        """Initialize metadata, TransferMatrix, solver config."""
        cfg = self.config

        # Read metadata
        self._meta = reader.read_metadata(nx=cfg.nx, ny=cfg.ny)
        meta = self._meta

        # Auto-detect workers
        self._n_workers = cfg.n_workers if cfg.n_workers > 0 else mp.cpu_count()

        # Auto-detect chunk size
        if cfg.chunk_pixels > 0:
            self._chunk_size = cfg.chunk_pixels
        else:
            self._chunk_size = auto_chunk_size(meta, cfg.memory_budget_gb)

        # Build TransferMatrix template
        tm = TransferMatrix(
            depth=meta.depth,
            step_size=meta.step_size,
            angles=meta.angles,
            photon_energy=meta.photon_energy,
            work_function=meta.work_function,
            gamma_eff=meta.gamma_eff,
        )
        tm.trans_mat = meta.trans_mat
        tm.n_elements = meta.n_elements

        # Compute lambda_basic (once)
        compound_db = CompoundDatabase()
        self._lambda_basic = tm.calculate_lambda_basic(meta.elements, compound_db)

        # Serialize TransferMatrix for workers
        self._tm_data = serialize_transfer_matrix(tm)

        # Build SemiFreeSolverConfig base dict
        self._semi_config_base = dict(
            abruptness_init=cfg.abruptness_init,
            free_indices=meta.free_indices,
            linked_indices=meta.linked_indices,
            substrate_indices=meta.substrate_indices,
            optimize_thickness=True,
            optimize_abruptness=False,
            thickness_bounds=(50.0, 500.0),
            abruptness_bounds=(1.0, 100.0),
            free_linked_const_override=meta.linkedconst.tolist(),
        )

        # GUI compat: setup derived metrics config
        if cfg.full_gui_compat:
            self._setup_derived_config(reader, meta)

    def _setup_derived_config(
        self,
        reader: StreamingDepthReader,
        meta: DepthProfileMetadata,
    ):
        """Setup DerivedMetricsConfig for GUI-compatible output.

        Reads TransMatOpt from /ezdeprof/ and computes FV/SV dimensions.
        """
        f = reader._f
        ez = f['ezdeprof']

        # TransMatOpt for Xrlt computation
        # H5: (n_elements, n_depth, n_angles_opt) → Python: (n_angles_opt, n_depth, n_elements)
        trans_mat_opt = None
        n_angles_opt = 0
        if 'TransMatOpt' in ez:
            tm_opt_raw = ez['TransMatOpt'][:].astype(np.float64)
            trans_mat_opt = np.transpose(tm_opt_raw, (2, 1, 0))
            n_angles_opt = trans_mat_opt.shape[0]

        # const_indices: elements that are neither free, substrate, nor linked
        all_indices = set(range(meta.n_elements))
        used = set(meta.free_indices) | set(meta.substrate_indices) | set(meta.linked_indices)
        const_indices = sorted(all_indices - used)

        # FV length: 2 + 3 * len(free + const)
        fv_indices = sorted(meta.free_indices + const_indices)
        n_fv = 2 + 3 * len(fv_indices)

        # SV length: 3 * n_subst + 3 * n_linked
        n_sv = 3 * len(meta.substrate_indices) + 3 * len(meta.linked_indices)

        self._derived_config = DerivedMetricsConfig(
            depth=meta.depth,
            n_elements=meta.n_elements,
            free_indices=meta.free_indices,
            subst_indices=meta.substrate_indices,
            linked_indices=meta.linked_indices,
            const_indices=const_indices,
            trans_mat_opt=trans_mat_opt,
            n_angles_opt=n_angles_opt,
        )
        self._n_angles_opt = n_angles_opt
        self._n_fv = n_fv
        self._n_sv = n_sv

    def _build_phase1_args(
        self,
        Xstk: np.ndarray,
        pixel_indices: np.ndarray,
    ) -> list:
        """Build worker_phase1 argument tuples for a chunk.

        Args:
            Xstk: (count, n_angles, n_elements) — this chunk's intensities
            pixel_indices: Global pixel indices for this chunk

        Returns:
            List of arg tuples for worker_phase1
        """
        cfg = self.config
        meta = self._meta
        args = []
        for i in range(len(pixel_indices)):
            args.append((
                i,                           # local_idx
                int(pixel_indices[i]),       # px (global)
                Xstk[i],                     # I_obs
                self._tm_data,               # trans_mat_data
                meta.elements,               # elements
                meta.linkedconst,            # linkedconst
                self._lambda_basic,          # lambda_basic
                cfg.method,                  # method
                cfg.alpha,                   # alpha
                cfg.n_loops,                 # n_loops
                cfg.degree_decay,            # degree_decay
                cfg.use_xreg,               # use_xreg
                cfg.thickness_init,          # thickness_init
                cfg.abruptness_init,         # abruptness_init
                cfg.solver_max_iter,         # solver_max_iter
                cfg.solver_tol,              # solver_tol
                self._semi_config_base,      # semi_config_base_dict
            ))
        return args

    def _build_phase2_args(
        self,
        Xstk: np.ndarray,
        D_opt: np.ndarray,
        D_prior: np.ndarray,
        thickness: np.ndarray,
        abruptness: np.ndarray,
        local_indices: np.ndarray,
    ) -> list:
        """Build worker_phase2 argument tuples for a time step.

        Args:
            Xstk: (count, n_angles, n_elements)
            D_opt: (count, n_depth, n_elements) — current solution
            D_prior: (count, n_depth, n_elements) — neighbor-averaged
            thickness: (count,)
            abruptness: (count,)
            local_indices: (count,) — local indices within this time step

        Returns:
            List of arg tuples for worker_phase2
        """
        cfg = self.config
        meta = self._meta
        args = []
        for i in range(len(local_indices)):
            args.append((
                int(local_indices[i]),       # local_idx
                Xstk[i],                     # I_obs
                D_opt[i],                    # D_opt_current
                D_prior[i],                  # D_prior
                float(thickness[i]),         # thickness_current
                float(abruptness[i]),        # abruptness_current
                self._tm_data,               # trans_mat_data
                meta.elements,               # elements
                meta.linkedconst,            # linkedconst
                self._lambda_basic,          # lambda_basic
                cfg.mem_alpha,               # mem_alpha
                cfg.use_xreg,               # use_xreg
                cfg.solver_max_iter,         # solver_max_iter
                cfg.solver_tol,              # solver_tol
                cfg.mem_scipy_method,        # mem_scipy_method
                self._semi_config_base,      # semi_config_base_dict
            ))
        return args

    def _print_header(self):
        """Print pipeline configuration summary."""
        meta = self._meta
        cfg = self.config
        budget = estimate_memory_budget(meta, self._n_workers)

        print(f"\n{'='*72}")
        print(f"Streaming Depth Profiling Pipeline")
        print(f"{'='*72}")
        print(f"  Input:     {self.h5_path}")
        print(f"  Output:    {self.output_path}")
        print(f"  Grid:      {meta.nx}×{meta.ny} spatial × {meta.nt} temporal "
              f"= {meta.n_pixels:,} pixels")
        print(f"  Elements:  {meta.n_elements} ({', '.join(meta.element_names)})")
        print(f"  Depth:     {meta.n_depth} points, {meta.step_size}Å step")
        if cfg.method == 'NN':
            print(f"  Phase 1:   NN inference ({cfg.nn_model_path})")
        else:
            print(f"  Phase 1:   {cfg.method}×{cfg.n_loops}+TM+Xreg "
                  f"(α={cfg.alpha})")
        if cfg.mem_loops > 0:
            print(f"  Phase 2:   Neighbor-MEM×{cfg.mem_loops} (α={cfg.mem_alpha})")
        else:
            print(f"  Phase 2:   Disabled")
        print(f"  Workers:   {self._n_workers}")
        print(f"  Chunk:     {self._chunk_size:,} pixels "
              f"({self._chunk_size / meta.nxy:.0f} time steps)" if meta.nxy > 0 else "")
        print(f"  Memory:    ~{budget['total_gb']:.1f} GB working set")
        print(f"  Output:    ~{budget['output_compressed_gb']:.1f} GB "
              f"({cfg.compression or 'no'} compression)")

    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format seconds as human-readable string."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            m, s = divmod(seconds, 60)
            return f"{int(m)}m{int(s)}s"
        else:
            h, remainder = divmod(seconds, 3600)
            m, s = divmod(remainder, 60)
            return f"{int(h)}h{int(m)}m"

    def dry_run(self):
        """Print estimated time and resources without processing."""
        with StreamingDepthReader(self.h5_path) as reader:
            self._setup(reader)
            self._print_header()

            meta = self._meta
            cfg = self.config

            # Rough per-pixel costs, used only for this dry-run estimate.
            # Actual speed depends on the hardware, the data and the settings.
            if cfg.method == 'NN':
                ms_per_px_p1 = 0.003
            elif cfg.method == 'LLS':
                ms_per_px_p1 = 39  # approximate
            else:
                ms_per_px_p1 = 123

            ms_per_px_mem = 36  # MEM warm-start

            p1_seconds = meta.n_pixels * ms_per_px_p1 / 1000 / self._n_workers
            if cfg.mem_loops > 0:
                p2_seconds = meta.n_pixels * cfg.mem_loops * ms_per_px_mem / 1000 / self._n_workers
            else:
                p2_seconds = 0

            total = p1_seconds + p2_seconds

            print(f"\n--- Dry Run Estimates ({self._n_workers} workers) ---")
            print(f"  Phase 1: ~{self._format_time(p1_seconds)} "
                  f"({ms_per_px_p1}ms/px effective)")
            if cfg.mem_loops > 0:
                print(f"  Phase 2: ~{self._format_time(p2_seconds)} "
                      f"({ms_per_px_mem}ms/px × {cfg.mem_loops} loops)")
            print(f"  Total:   ~{self._format_time(total)}")
            print(f"\n  ⚠️  These are rough estimates.")
            print(f"       Actual speed depends on CPU, memory, and I/O.")


# ==============================================================================
# CLI
# ==============================================================================

def main():
    """Command-line interface for streaming pipeline."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Streaming Depth Profiling Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic LLS pipeline
  python -m deppro.streaming_pipeline --h5 input.h5 --output output.h5 \\
      --method LLS --loops 10 --xreg

  # L1 pipeline with neighbor-MEM
  python -m deppro.streaming_pipeline --h5 input.h5 --output output.h5 \\
      --method L1 --loops 10 --alpha 2.0 --xreg \\
      --mem-alpha 0.01 --mem-loops 5

  # Dry run (estimate time & memory)
  python -m deppro.streaming_pipeline --h5 input.h5 --dry-run

  # Resume interrupted run
  python -m deppro.streaming_pipeline --h5 input.h5 --output output.h5 --resume

  # NN inference mode (batched inference instead of per-pixel SLSQP)
  python -m deppro.streaming_pipeline --h5 input.h5 --output output.h5 \\
      --method NN --nn-model depth_mlp.npz
        """,
    )

    # I/O
    parser.add_argument("--h5", required=True, help="Input HDF5 file path")
    parser.add_argument("--output", "-o", required=False,
                        help="Output HDF5 file path (default: input_streaming.h5)")

    # Phase 1
    parser.add_argument("--method", default="LLS", choices=["LLS", "L1", "NN"],
                        help="Phase 1 method (default: LLS)")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Regularization strength (default: 1.0)")
    parser.add_argument("--loops", type=int, default=10,
                        help="Iterative loops per pixel (default: 10)")
    parser.add_argument("--degree-decay", type=float, default=0.8,
                        help="Degree decay per loop (default: 0.8)")
    parser.add_argument("--xreg", action="store_true",
                        help="Use Xreg weighting")

    # Phase 2
    parser.add_argument("--mem-alpha", type=float, default=0.01,
                        help="MEM regularization (default: 0.01)")
    parser.add_argument("--mem-loops", type=int, default=0,
                        help="MEM loops (0=no MEM, default: 0)")
    parser.add_argument("--mem-scipy-method", default="SLSQP",
                        help="SciPy method for MEM (default: SLSQP)")
    parser.add_argument("--n-ring", type=int, default=1,
                        help="Neighbor ring radius for MEM (Manhattan L1 distance, default: 1)")

    # Grid
    parser.add_argument("--nx", type=int, default=0,
                        help="Spatial grid width (0=auto)")
    parser.add_argument("--ny", type=int, default=0,
                        help="Spatial grid height (0=auto)")

    # Solver
    parser.add_argument("--thickness-init", type=float, default=210.0,
                        help="Initial thickness (default: 210)")
    parser.add_argument("--abruptness-init", type=float, default=20.0,
                        help="Initial abruptness (default: 20)")
    parser.add_argument("--max-iter", type=int, default=200,
                        help="Max solver iterations (default: 200)")
    parser.add_argument("--tol", type=float, default=1e-6,
                        help="Solver tolerance (default: 1e-6)")

    # NN mode
    parser.add_argument("--nn-model", default="",
                        help="Path to NN model weights (.npz), required for --method NN")
    parser.add_argument("--nn-batch-size", type=int, default=65536,
                        help="NN inference batch size (default: 65536)")

    # Resources
    parser.add_argument("--workers", type=int, default=0,
                        help="Parallel workers (0=auto)")
    parser.add_argument("--memory-budget", type=float, default=4.0,
                        help="Memory budget in GB (default: 4.0)")
    parser.add_argument("--chunk-pixels", type=int, default=0,
                        help="Pixels per chunk (0=auto)")
    parser.add_argument("--max-chunks", type=int, default=0,
                        help="Max chunks to process (0=all, for testing)")

    # Compression
    parser.add_argument("--no-compress", action="store_true",
                        help="Disable output compression")
    parser.add_argument("--compress-level", type=int, default=1,
                        help="gzip compression level 1-9 (default: 1)")

    # GUI compatibility
    parser.add_argument("--full-compat", action="store_true", default=True,
                        help="Create all 24 MATLAB GUI datasets (default: on)")
    parser.add_argument("--no-full-compat", action="store_true",
                        help="Only output Dtrn/Drlt/Q (minimal mode)")

    # Modes
    parser.add_argument("--dry-run", action="store_true",
                        help="Estimate time and memory without processing")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint (not yet implemented)")

    args = parser.parse_args()

    # Default output path
    if args.output is None:
        p = Path(args.h5)
        args.output = str(p.parent / f"{p.stem}_streaming{p.suffix}")

    # Validate NN args
    if args.method == 'NN' and not args.nn_model:
        parser.error("--nn-model is required when --method NN is used")

    # Build config
    full_compat = not args.no_full_compat  # default True
    config = StreamingPipelineConfig(
        method=args.method,
        alpha=args.alpha,
        n_loops=args.loops,
        degree_decay=args.degree_decay,
        use_xreg=args.xreg,
        n_workers=args.workers,
        nx=args.nx,
        ny=args.ny,
        mem_alpha=args.mem_alpha,
        mem_loops=args.mem_loops,
        mem_scipy_method=args.mem_scipy_method,
        n_ring=args.n_ring,
        thickness_init=args.thickness_init,
        abruptness_init=args.abruptness_init,
        solver_max_iter=args.max_iter,
        solver_tol=args.tol,
        memory_budget_gb=args.memory_budget,
        chunk_pixels=args.chunk_pixels,
        max_chunks=args.max_chunks,
        full_gui_compat=full_compat,
        compression=None if args.no_compress else 'gzip',
        compression_opts=args.compress_level,
        nn_model_path=args.nn_model,
        nn_batch_size=args.nn_batch_size,
    )

    # Run
    pipeline = StreamingDepthPipeline(args.h5, args.output, config)

    if args.dry_run:
        pipeline.dry_run()
    elif args.resume:
        pipeline.run(resume=True)
    else:
        pipeline.run()


# ==============================================================================
# MPI Extension Stubs
# ==============================================================================

def _mpi_available() -> bool:
    """Check if MPI (mpi4py) is available."""
    try:
        from mpi4py import MPI
        return True
    except ImportError:
        return False


def _assign_chunks_to_ranks(
    n_chunks: int,
    n_ranks: int,
    rank: int,
) -> list:
    """Assign chunks to MPI ranks (round-robin or block).

    Args:
        n_chunks: Total number of chunks
        n_ranks: Number of MPI ranks
        rank: This rank's ID

    Returns:
        List of chunk indices assigned to this rank

    Example:
        4 chunks, 2 ranks → rank 0: [0, 2], rank 1: [1, 3]
    """
    return list(range(rank, n_chunks, n_ranks))


def _open_h5_parallel(path: str, mode: str = 'r'):
    """Open H5 file with MPI parallel I/O (stub).

    Requires h5py built with MPI support and mpi4py.

    Args:
        path: H5 file path
        mode: 'r' for read, 'w' for write, 'a' for append

    Returns:
        h5py.File with MPI driver

    Note:
        This is a stub. Actual implementation requires:
        - h5py compiled with --enable-parallel
        - MPI-aware filesystem (Lustre, GPFS)
        - Collective/independent I/O selection
    """
    raise NotImplementedError(
        "MPI parallel I/O is not yet implemented. "
        "Requires h5py built with MPI support (driver='mpio'). "
        "Install: HDF5_MPI=ON pip install --no-binary h5py h5py"
    )


def _phase2_boundary_exchange(
    D_opt_local: np.ndarray,
    local_chunk_offset: int,
    chunk_size: int,
    nx: int,
    ny: int,
) -> dict:
    """Exchange boundary pixels between MPI ranks for Phase 2 neighbor averaging.

    In Phase 2, neighbor-averaging requires adjacent time steps' data.
    When chunks are distributed across MPI ranks, boundary pixels must
    be exchanged via MPI_Sendrecv.

    Args:
        D_opt_local: This rank's D_opt data
        local_chunk_offset: Global pixel offset of this chunk
        chunk_size: Pixels in this chunk
        nx: Spatial grid width
        ny: Spatial grid height

    Returns:
        Dict with 'prev_boundary' and 'next_boundary' arrays

    Note:
        This is a stub. Actual implementation requires MPI comm.
        For Phase 2, only spatial neighbors within the same time step
        are used, so NO boundary exchange is needed when chunks are
        temporal slices (which is our default chunking strategy).
        This stub is here for future extensions (e.g., spatial chunking).
    """
    raise NotImplementedError(
        "MPI boundary exchange is not yet implemented. "
        "Note: With temporal chunking (default), boundary exchange is NOT needed "
        "because spatial neighbors are always within the same time step."
    )


if __name__ == '__main__':
    main()
