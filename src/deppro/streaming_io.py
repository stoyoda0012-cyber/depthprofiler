"""
Streaming HDF5 I/O for Large-Scale Depth Profiling
====================================================

Chunked reading and writing of depth profiling data for datasets
too large to fit in memory (e.g., 500×500×200 = 50M pixels, 157GB).

Design principles:
    - Never load all pixels at once — read/write in temporal chunks
    - Double-buffer I/O: prefetch next chunk while computing current one
    - gzip-1 compression for output (4x compression on near-zero-heavy Dtrn/Drlt)
    - MATLAB-compatible H5 layout: (n_pixels, n_elements, n_depth)
    - Internal Python layout: (n_pixels, n_depth, n_elements) — depth-major

Inspired by:
    - toyomacro StreamingFitparaWriter (ring buffer, ThreadPoolExecutor)
    - voigtfit h5io_fast (rdcc cache, PrefetchReader, read_direct)

Usage:
    with StreamingDepthReader(input_h5) as reader:
        meta = reader.read_metadata()
        for offset in range(0, meta['n_pixels'], chunk_size):
            Xstk_chunk = reader.read_xstk_chunk(offset, chunk_size)
            # ... compute ...

    with StreamingDepthWriter(output_h5, meta) as writer:
        writer.write_dtrn_chunk(offset, D_opt, Q)
        writer.finalize()

Date: 2026-02-14
"""

import numpy as np
import h5py
import json
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, Future

from deppro import ElementConfig


# ==============================================================================
# Constants
# ==============================================================================

# H5 chunk cache: 64MB (default 1MB is too small for streaming)
RDCC_NBYTES = 64 * 1024 * 1024
RDCC_NSLOTS = 10007   # Prime number for hash table
RDCC_W0 = 0.75        # Preemption bias


# ==============================================================================
# Metadata container
# ==============================================================================

@dataclass
class DepthProfileMetadata:
    """Metadata read from ezdeprof group in H5 file.

    All index arrays are 1-based (MATLAB convention).
    """
    # Dimensions
    n_pixels: int
    n_elements: int
    n_depth: int
    n_angles: int

    # Spatial grid (may be unknown for raw H5)
    nx: int = 0
    ny: int = 0
    nt: int = 0

    # Physics
    depth: np.ndarray = field(default_factory=lambda: np.array([]))
    angles: np.ndarray = field(default_factory=lambda: np.array([]))
    step_size: float = 0.0
    gamma_eff: float = 0.8
    photon_energy: float = 8000.0
    work_function: float = 5.0

    # Elements
    element_names: List[str] = field(default_factory=list)
    symbols: List[str] = field(default_factory=list)
    orbitals: List[str] = field(default_factory=list)
    compounds: List[str] = field(default_factory=list)
    binding_energies: np.ndarray = field(default_factory=lambda: np.array([]))
    charges: np.ndarray = field(default_factory=lambda: np.array([]))

    # Index arrays (MATLAB 1-based)
    free_matlab: np.ndarray = field(default_factory=lambda: np.array([]))
    subst_matlab: np.ndarray = field(default_factory=lambda: np.array([]))
    linked_matlab: np.ndarray = field(default_factory=lambda: np.array([]))
    linkedconst: np.ndarray = field(default_factory=lambda: np.array([]))

    # Transfer matrix template
    trans_mat: np.ndarray = field(default_factory=lambda: np.array([]))

    # Elements config list (for solver)
    elements: List[ElementConfig] = field(default_factory=list)

    @property
    def free_indices(self) -> List[int]:
        """0-based free element indices."""
        return [int(f) - 1 for f in self.free_matlab]

    @property
    def linked_indices(self) -> List[int]:
        """0-based linked element indices."""
        return [int(l) - 1 for l in self.linked_matlab]

    @property
    def substrate_indices(self) -> List[int]:
        """0-based substrate element indices."""
        return [int(s) - 1 for s in self.subst_matlab]

    @property
    def nxy(self) -> int:
        """Pixels per time step."""
        return self.nx * self.ny if self.nx > 0 and self.ny > 0 else 0

    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict (for checkpoint attrs)."""
        return {
            'n_pixels': self.n_pixels,
            'n_elements': self.n_elements,
            'n_depth': self.n_depth,
            'n_angles': self.n_angles,
            'nx': self.nx, 'ny': self.ny, 'nt': self.nt,
            'step_size': self.step_size,
            'gamma_eff': self.gamma_eff,
            'photon_energy': self.photon_energy,
            'work_function': self.work_function,
            'element_names': self.element_names,
        }


# ==============================================================================
# Streaming Reader
# ==============================================================================

class StreamingDepthReader:
    """Streaming HDF5 reader for large depth profiling datasets.

    Reads Xstk (and optionally Dtrn for Phase 2) in chunks,
    with optional double-buffer prefetching.

    Data layout in H5:
        Xstk: (n_pixels, n_elements, n_angles) float32  — MATLAB layout
        Dtrn: (n_pixels, n_elements, n_depth) float32    — MATLAB layout
        Dstk: (n_pixels, n_elements, n_depth) float32    — ground truth

    Returns arrays in Python layout:
        Xstk: (chunk, n_angles, n_elements) float64
        Dtrn: (chunk, n_depth, n_elements) float64
    """

    def __init__(self, h5_path: str, prefetch: bool = True):
        """Open H5 file with optimized cache settings.

        Args:
            h5_path: Path to input H5 file
            prefetch: Enable double-buffer prefetching (default: True)
        """
        self.h5_path = str(h5_path)
        self.prefetch = prefetch
        self._f: Optional[h5py.File] = None
        self._executor: Optional[ThreadPoolExecutor] = None
        self._prefetch_future: Optional[Future] = None
        self._prefetch_buffer: Optional[np.ndarray] = None

    def __enter__(self):
        self._f = h5py.File(
            self.h5_path, 'r',
            rdcc_nbytes=RDCC_NBYTES,
            rdcc_nslots=RDCC_NSLOTS,
            rdcc_w0=RDCC_W0,
        )
        if self.prefetch:
            self._executor = ThreadPoolExecutor(max_workers=1)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._prefetch_future is not None:
            self._prefetch_future.result()  # Wait for pending prefetch
        if self._executor is not None:
            self._executor.shutdown(wait=True)
        if self._f is not None:
            self._f.close()
        return False

    def read_metadata(self, nx: int = 0, ny: int = 0) -> DepthProfileMetadata:
        """Read all metadata from /ezdeprof/ group.

        Args:
            nx: Spatial grid width (0 = auto-detect)
            ny: Spatial grid height (0 = auto-detect)

        Returns:
            DepthProfileMetadata with all solver-relevant fields
        """
        f = self._f
        ez = f['ezdeprof']

        # Dimensions from Xstk shape
        xstk = f['Xstk']
        n_pixels = xstk.shape[0]
        n_elements = xstk.shape[1]
        n_angles = xstk.shape[2]

        # Depth
        depth = ez['depth'][:].flatten().astype(np.float64)
        n_depth = len(depth)

        # Angles
        angles = ez['thetaobs'][:].flatten().astype(np.float64)

        # Scalars
        step_size = float(ez['stepsize'][:].flatten()[0])
        gamma_eff = float(ez['gammaeff'][:].flatten()[0])
        photon_energy = float(ez['photonenergy'][:].flatten()[0])
        work_function = float(ez['workfunction'][:].flatten()[0])

        # Element info
        elem_names_raw = ez['element'][:].flatten()
        element_names = [x.decode() if isinstance(x, bytes) else str(x)
                        for x in elem_names_raw]

        symbols_raw = ez['elementsymbol'][:].flatten()
        symbols = [x.decode() if isinstance(x, bytes) else str(x)
                  for x in symbols_raw]

        orbitals_raw = ez['orbital'][:].flatten()
        orbitals = [x.decode() if isinstance(x, bytes) else str(x)
                   for x in orbitals_raw]

        compounds_raw = ez['compound'][:].flatten()
        compounds = [x.decode() if isinstance(x, bytes) else str(x)
                    for x in compounds_raw]

        binding_energies = ez['bindingenergy'][:].flatten().astype(np.float64)
        charges = ez['charge'][:].flatten().astype(np.float64)

        # Index arrays (MATLAB 1-based)
        free_matlab = ez['free'][:].flatten().astype(int)
        subst_matlab = ez['subst'][:].flatten().astype(int)
        linked_matlab = ez['linked'][:].flatten().astype(int)
        linkedconst = ez['linkedconst'][:].flatten().astype(np.float64)

        # Transfer matrix template
        # H5: (n_elements, n_depth, n_angles) → Python: (n_angles, n_depth, n_elements)
        tm_raw = ez['TransMatObs'][:].astype(np.float64)
        trans_mat = np.transpose(tm_raw, (2, 1, 0))

        # Build ElementConfig list
        elements = []
        for i in range(n_elements):
            elements.append(ElementConfig(
                symbol=symbols[i],
                orbital=orbitals[i],
                compound=compounds[i],
                binding_energy=binding_energies[i],
                oxidation_state=charges[i],
            ))

        # Auto-detect spatial grid
        if nx == 0 or ny == 0:
            nx, ny = self._auto_detect_grid(n_pixels, nx, ny)
        nt = n_pixels // (nx * ny) if nx > 0 and ny > 0 else 0

        return DepthProfileMetadata(
            n_pixels=n_pixels, n_elements=n_elements,
            n_depth=n_depth, n_angles=n_angles,
            nx=nx, ny=ny, nt=nt,
            depth=depth, angles=angles,
            step_size=step_size, gamma_eff=gamma_eff,
            photon_energy=photon_energy, work_function=work_function,
            element_names=element_names, symbols=symbols,
            orbitals=orbitals, compounds=compounds,
            binding_energies=binding_energies, charges=charges,
            free_matlab=free_matlab, subst_matlab=subst_matlab,
            linked_matlab=linked_matlab, linkedconst=linkedconst,
            trans_mat=trans_mat, elements=elements,
        )

    def read_xstk_chunk(self, offset: int, count: int,
                        buffer: Optional[np.ndarray] = None,
                        dtype=np.float64) -> np.ndarray:
        """Read a chunk of Xstk data.

        H5 layout: (n_pixels, n_elements, n_angles) float32
        Returns: (count, n_angles, n_elements) — Python layout, dtype as given

        Args:
            offset: Start pixel index
            count: Number of pixels to read
            buffer: Optional pre-allocated buffer (count, n_angles, n_elements)
            dtype: Output dtype. Default float64 (classic solver path). The NN
                path passes float32 to avoid a float32→float64→float32 round-trip
                (on-disk Xstk is already float32, so copy=False makes this free).

        Returns:
            Xstk chunk in Python layout
        """
        dset = self._f['Xstk']
        actual_count = min(count, dset.shape[0] - offset)

        # Read directly from H5: (actual_count, n_elements, n_angles)
        raw = dset[offset:offset + actual_count].astype(dtype, copy=False)

        # Transpose to Python layout: (count, n_angles, n_elements)
        result = np.transpose(raw, (0, 2, 1))

        if buffer is not None:
            buffer[:actual_count] = result
            return buffer[:actual_count]
        return np.ascontiguousarray(result)

    def read_dtrn_chunk(self, offset: int, count: int) -> np.ndarray:
        """Read Phase 1 results (Dtrn) for Phase 2 processing.

        H5 layout: (n_pixels, n_elements, n_depth) float32
        Returns: (count, n_depth, n_elements) float64

        Args:
            offset: Start pixel index
            count: Number of pixels to read

        Returns:
            Dtrn chunk in Python layout
        """
        # Try Py_Dtrn first (Python results), fall back to Dtrn (MATLAB)
        for key in ['Py_Dtrn', 'Dtrn']:
            if key in self._f:
                dset = self._f[key]
                actual_count = min(count, dset.shape[0] - offset)
                raw = dset[offset:offset + actual_count].astype(np.float64)
                return np.ascontiguousarray(np.transpose(raw, (0, 2, 1)))
        raise KeyError("Neither 'Py_Dtrn' nor 'Dtrn' found in H5 file")

    def read_dstk_chunk(self, offset: int, count: int) -> np.ndarray:
        """Read ground truth (Dstk) for metrics.

        Returns: (count, n_depth, n_elements) float64
        """
        dset = self._f['Dstk']
        actual_count = min(count, dset.shape[0] - offset)
        raw = dset[offset:offset + actual_count].astype(np.float64)
        return np.ascontiguousarray(np.transpose(raw, (0, 2, 1)))

    def prefetch_xstk_chunk(self, offset: int, count: int, dtype=np.float64):
        """Start prefetching the next Xstk chunk in background.

        Call this while processing the current chunk. Then call
        get_prefetched_xstk() to get the result (blocking if needed).
        dtype is forwarded to read_xstk_chunk (NN path passes float32).
        """
        if not self.prefetch or self._executor is None:
            return

        # Wait for any pending prefetch
        if self._prefetch_future is not None:
            self._prefetch_future.result()

        self._prefetch_future = self._executor.submit(
            self._do_prefetch_xstk, offset, count, dtype
        )

    def get_prefetched_xstk(self) -> Optional[np.ndarray]:
        """Get the result of a previous prefetch call.

        Returns:
            Prefetched Xstk chunk, or None if no prefetch was started
        """
        if self._prefetch_future is None:
            return None
        result = self._prefetch_future.result()
        self._prefetch_future = None
        return result

    def _do_prefetch_xstk(self, offset: int, count: int, dtype=np.float64) -> np.ndarray:
        """Background prefetch worker."""
        return self.read_xstk_chunk(offset, count, dtype=dtype)

    @staticmethod
    def _auto_detect_grid(n_pixels: int, nx: int, ny: int) -> Tuple[int, int]:
        """Auto-detect spatial grid dimensions from total pixel count."""
        if nx > 0 and ny > 0:
            return nx, ny
        # Try common grid sizes
        for try_nx in [5, 10, 20, 25, 50, 100, 200, 250, 500, 1000]:
            for try_ny in [5, 10, 20, 25, 50, 100, 200, 250, 500, 1000]:
                if n_pixels % (try_nx * try_ny) == 0:
                    nt = n_pixels // (try_nx * try_ny)
                    if nt >= 1:
                        return (nx if nx > 0 else try_nx,
                                ny if ny > 0 else try_ny)
        return (1, 1)  # Fallback: treat as 1D


# ==============================================================================
# Streaming Writer
# ==============================================================================

class StreamingDepthWriter:
    """Streaming HDF5 writer for depth profiling results.

    Creates a new H5 file with chunked, compressed datasets.
    Two modes:
        - full_gui_compat=False: Dtrn/Drlt/Qtrn/Qrlt only (fast, minimal)
        - full_gui_compat=True: All 24 MATLAB GUI datasets + ezdeprof (GUI-compatible)

    Output layout (MATLAB-compatible):
        3D datasets: (n_pixels, n_elements, n_depth/n_angles) float32
        2D datasets: (dim, n_pixels) float32  — MATLAB reads as (n_pixels, dim)
        /ezdeprof/: metadata (copied from input)

    Internal Python layout:
        D_opt: (chunk, n_depth, n_elements) — transposed on write
    """

    # Category A: datasets to copy from input (simulation/ground truth)
    # 3D datasets: (n_pixels, n_elem, dim3) in H5
    COPY_3D = ['Dstk', 'Xstk']
    # 2D datasets: (dim, n_pixels) in H5
    COPY_2D = ['FV0stk', 'FVstk', 'SVstk', 'SPSstk', 'Cstk', 'NSstk']
    # Q-like: (dim, n_pixels) in H5  — note Q/alpha/degree shapes differ
    COPY_Q = ['Qstk']

    # Top-level metadata datasets to copy from input (GUI requires these)
    GUI_META_DATASETS = [
        'thetaobs', 'tsmp', 'Col', 'StrData',
        'FermiEnergy', 'BindingEnergy', 'SubIndex', 'UnlockIndex',
    ]

    def __init__(
        self,
        output_path: str,
        meta: DepthProfileMetadata,
        async_write: bool = True,
        compression: str = 'gzip',
        compression_opts: int = 1,
        full_gui_compat: bool = False,
        n_angles_opt: int = 0,
        n_fv: int = 0,
        n_sv: int = 0,
        output_dtype='float32',
    ):
        """Create output H5 file with pre-allocated datasets.

        Args:
            output_path: Output H5 file path
            meta: Metadata from StreamingDepthReader
            async_write: Use ThreadPoolExecutor for async writes
            compression: Compression filter ('gzip' or None)
            compression_opts: Compression level (1=fast, 9=best)
            full_gui_compat: Create all 24 GUI datasets (not just 4)
            n_angles_opt: Number of optimization angles (for Xrlt), 0=skip
            n_fv: Feature vector length (2 + 3*n_free_const), 0=auto
            n_sv: Support vector length (3*n_subst + 3*n_linked), 0=auto
        """
        self.output_path = str(output_path)
        self.meta = meta
        self.async_write = async_write
        self.compression = compression
        self.compression_opts = compression_opts
        # Flush after every chunk write (resume safety). Disabled on throughput
        # runs to avoid forcing a disk sync per chunk; finalize() still flushes.
        self.flush_per_chunk = True
        self.full_gui_compat = full_gui_compat
        self.n_angles_opt = n_angles_opt
        self.n_fv = n_fv
        self.n_sv = n_sv
        # Storage dtype for the D (depth-profile) datasets. float16 halves the
        # write volume at the cost of rounding the stored concentrations.
        # Thickness Q stays float32 (it's computed from float32 D
        # before storage; recomputing from f16 D can jitter near-0.5 crossovers).
        self.output_dtype = np.dtype(output_dtype)

        self._f: Optional[h5py.File] = None
        self._executor: Optional[ThreadPoolExecutor] = None
        self._write_future: Optional[Future] = None

        # Chunk size for H5 datasets: 1 time step
        nxy = meta.nxy if meta.nxy > 0 else min(meta.n_pixels, 250000)
        self._h5_chunk_d = (nxy, meta.n_elements, meta.n_depth)
        self._h5_chunk_q = (3, nxy)  # (dim, n_pixels) — MATLAB convention
        self._nxy = nxy

    def __enter__(self):
        self._f = h5py.File(self.output_path, 'w')
        if self.async_write:
            self._executor = ThreadPoolExecutor(max_workers=1)

        n = self.meta.n_pixels
        ne = self.meta.n_elements
        nd = self.meta.n_depth

        # Compression kwargs. Only gzip takes a numeric level; lzf (h5py
        # built-in, much faster than gzip, no extra dependency) takes none.
        comp_kw = {}
        if self.compression:
            comp_kw = dict(compression=self.compression)
            if self.compression == 'gzip':
                comp_kw['compression_opts'] = self.compression_opts

        # --- Category B: core result datasets (always created) ---
        for name in ['Dtrn', 'Drlt']:
            self._f.create_dataset(
                name,
                shape=(n, ne, nd),
                dtype=self.output_dtype,
                chunks=self._h5_chunk_d,
                **comp_kw,
            )

        for name in ['Qtrn', 'Qrlt']:
            self._f.create_dataset(
                name,
                shape=(3, n),
                dtype='float32',
                chunks=self._h5_chunk_q,
            )

        # --- Full GUI compat: Category A (copy targets) + Category C (derived) ---
        if self.full_gui_compat:
            self._create_gui_datasets(n, ne, nd, comp_kw)

        # Mark as incomplete
        self._f.attrs['_deppro_incomplete'] = True
        self._f.attrs['_deppro_n_pixels'] = n
        self._f.attrs['_deppro_created'] = time.strftime('%Y-%m-%d %H:%M:%S')

        return self

    def _create_gui_datasets(self, n: int, ne: int, nd: int, comp_kw: dict):
        """Create all GUI-required datasets (Category A + C).

        Category A: Copy targets (Dstk, Xstk, FV0stk, etc.)
        Category C: Derived from D_opt (Xrlt, FVtrn, etc.)

        Args:
            n: total pixels
            ne: number of elements
            nd: number of depth points
            comp_kw: compression kwargs
        """
        na = self.meta.n_angles     # thetaobs angles (14)
        na_opt = self.n_angles_opt  # thetaopt angles (16)
        n_fv = self.n_fv            # feature vector length
        n_sv = self.n_sv            # support vector length
        nxy = self._nxy

        # --- Category A: 3D copy targets ---
        # Dstk: (n, ne, nd) same as Dtrn
        self._f.create_dataset(
            'Dstk', shape=(n, ne, nd), dtype='float32',
            chunks=(nxy, ne, nd), **comp_kw,
        )
        # Xstk: (n, ne, na) observation angles
        self._f.create_dataset(
            'Xstk', shape=(n, ne, na), dtype='float32',
            chunks=(nxy, ne, na), **comp_kw,
        )
        # Ismp: same as Xstk (may not exist in input)
        # Created only if source has it — handled in copy_source_chunk

        # --- Category A: 2D copy targets (dim, n_pixels) ---
        # MATLAB stores as (n_pixels, dim) → h5py reads as (dim, n_pixels)
        _2d_specs = {
            'FV0stk': (n_fv, n) if n_fv > 0 else None,
            'FVstk': (n_fv, n) if n_fv > 0 else None,
            'SVstk': (n_sv, n) if n_sv > 0 else None,
            'SPSstk': (2, n),
            'Cstk': (ne, n),
            'NSstk': (1, n),
            'Qstk': (3, n),
        }
        for name, shape in _2d_specs.items():
            if shape is not None:
                chunk_2d = (shape[0], nxy)
                self._f.create_dataset(
                    name, shape=shape, dtype='float32',
                    chunks=chunk_2d, **comp_kw,
                )

        # --- Category C: Derived datasets ---
        # Xrlt: (n, ne, na_opt)
        if na_opt > 0:
            self._f.create_dataset(
                'Xrlt', shape=(n, ne, na_opt), dtype='float32',
                chunks=(nxy, ne, na_opt), **comp_kw,
            )

        # Concentration: (ne, n) for both trn/rlt
        for name in ['Ctrn', 'Crlt']:
            self._f.create_dataset(
                name, shape=(ne, n), dtype='float32',
                chunks=(ne, nxy), **comp_kw,
            )

        # Feature vectors: (n_fv, n)
        if n_fv > 0:
            for name in ['FVtrn', 'FVrlt']:
                self._f.create_dataset(
                    name, shape=(n_fv, n), dtype='float32',
                    chunks=(n_fv, nxy), **comp_kw,
                )

        # Support vectors: (n_sv, n)
        if n_sv > 0:
            for name in ['SVtrn', 'SVrlt']:
                self._f.create_dataset(
                    name, shape=(n_sv, n), dtype='float32',
                    chunks=(n_sv, nxy), **comp_kw,
                )

        # Sparsity: (2, n)
        for name in ['SPStrn', 'SPSrlt']:
            self._f.create_dataset(
                name, shape=(2, n), dtype='float32',
                chunks=(2, nxy), **comp_kw,
            )

        # Alpha and degree: (3, n)
        for name in ['alpha', 'degree']:
            self._f.create_dataset(
                name, shape=(3, n), dtype='float32',
                chunks=(3, nxy), **comp_kw,
            )

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Wait for any pending async write
        if self._write_future is not None:
            self._write_future.result()
        if self._executor is not None:
            self._executor.shutdown(wait=True)
        if self._f is not None:
            self._f.close()
        return False

    def write_dtrn_chunk(
        self,
        offset: int,
        D_opt: np.ndarray,
        Q: Optional[np.ndarray] = None,
    ):
        """Write Phase 1 results to Dtrn dataset.

        Args:
            offset: Start pixel index
            D_opt: (chunk, n_depth, n_elements) — Python layout
            Q: (chunk, 3) — [chi2, thickness, abruptness] per pixel
        """
        self._submit_write('Dtrn', 'Qtrn', offset, D_opt, Q)

    def write_drlt_chunk(
        self,
        offset: int,
        D_opt: np.ndarray,
        Q: Optional[np.ndarray] = None,
    ):
        """Write Phase 2 results to Drlt dataset.

        Args:
            offset: Start pixel index
            D_opt: (chunk, n_depth, n_elements) — Python layout
            Q: (chunk, 3) — [chi2, thickness, abruptness] per pixel
        """
        self._submit_write('Drlt', 'Qrlt', offset, D_opt, Q)

    def _submit_write(self, d_key, q_key, offset, D_opt, Q):
        """Two-stage write (ring-buffer pipeline).

        Stage 1 (THIS calling thread): transpose Python → H5 layout + cast to
        the storage dtype. Done BEFORE waiting on the previous write, so this
        chunk's CPU prep overlaps the previous chunk's disk I/O. The cast also
        produces a fresh array, so no extra .copy() is needed for async safety.

        Stage 2 (writer thread): a pure h5py assign (which triggers the HDF5
        compression filter + disk write). Keeping ONLY the assign on the writer
        thread shrinks its per-chunk time, so the throughput floor moves closer
        to raw disk/compress bandwidth instead of CPU-prep + disk serialized.
        """
        count = D_opt.shape[0]
        # Stage 1 — calling thread (overlaps previous chunk's disk write)
        D_h5 = np.transpose(D_opt, (0, 2, 1)).astype(self.output_dtype)
        Q_h5 = Q.T.astype(np.float32) if Q is not None else None

        self._wait_pending_write()
        if self.async_write and self._executor is not None:
            self._write_future = self._executor.submit(
                self._assign, d_key, q_key, offset, count, D_h5, Q_h5)
        else:
            self._assign(d_key, q_key, offset, count, D_h5, Q_h5)

    def _assign(self, d_key, q_key, offset, count, D_h5, Q_h5):
        """Stage 2: pure h5py dataset assign (writer thread). Inputs are already
        in H5 layout and storage dtype — the assign triggers compress + disk."""
        self._f[d_key][offset:offset + count] = D_h5

        if Q_h5 is not None:
            # Q already transposed to H5 (3, chunk) MATLAB convention
            self._f[q_key][:, offset:offset + count] = Q_h5

        # Per-chunk flush forces a disk sync every chunk — needed for resume
        # checkpointing, but pure overhead on non-resumable throughput runs
        # (finalize() + file close flush anyway). Skip it when disabled.
        if self.flush_per_chunk:
            self._f.flush()

    def _wait_pending_write(self):
        """Block until any pending async write completes."""
        if self._write_future is not None:
            self._write_future.result()
            self._write_future = None

    def mark_chunk_complete(self, phase: int, chunk_id: int):
        """Record checkpoint for resume capability.

        Args:
            phase: 1 or 2
            chunk_id: Index of the completed chunk
        """
        self._wait_pending_write()
        key = f'_deppro_last_chunk_phase{phase}'
        self._f.attrs[key] = chunk_id

    def copy_metadata_from(self, reader: StreamingDepthReader):
        """Copy /ezdeprof/ group and top-level metadata from input H5.

        Copies:
            - /ezdeprof/ group (99 datasets)
            - Top-level GUI metadata: thetaobs, tsmp, Col, StrData, etc.

        Args:
            reader: StreamingDepthReader with open H5 file
        """
        self._wait_pending_write()
        src = reader._f

        # Copy ezdeprof group
        if 'ezdeprof' in src:
            src.copy('ezdeprof', self._f)

        # Copy top-level GUI metadata datasets
        if self.full_gui_compat:
            for name in self.GUI_META_DATASETS:
                if name in src and name not in self._f:
                    src.copy(name, self._f)

    def copy_source_chunk(
        self,
        reader: StreamingDepthReader,
        offset: int,
        count: int,
    ):
        """Copy Category A datasets from input for a pixel range.

        Copies Dstk, Xstk, FV0stk, FVstk, SVstk, SPSstk, Cstk, Qstk, NSstk
        for pixels [offset, offset+count).

        For 3D datasets (Dstk, Xstk): reads (count, ne, dim) directly.
        For 2D datasets: reads column slice (:, offset:offset+count).

        Args:
            reader: Open StreamingDepthReader
            offset: Start pixel index
            count: Number of pixels
        """
        self._wait_pending_write()
        src = reader._f
        actual_count = min(count, self.meta.n_pixels - offset)

        # 3D datasets: (n_pixels, n_elements, dim3)
        for name in self.COPY_3D:
            if name in src and name in self._f:
                self._f[name][offset:offset + actual_count] = \
                    src[name][offset:offset + actual_count]

        # Also copy Ismp if it exists in source
        if 'Ismp' in src:
            if 'Ismp' not in self._f:
                # Create on first access (shape from source)
                ds = src['Ismp']
                comp_kw = {}
                if self.compression:
                    comp_kw = dict(
                        compression=self.compression,
                        compression_opts=self.compression_opts,
                    )
                nxy = self._nxy
                self._f.create_dataset(
                    'Ismp', shape=ds.shape, dtype=ds.dtype,
                    chunks=(nxy, ds.shape[1], ds.shape[2]),
                    **comp_kw,
                )
            self._f['Ismp'][offset:offset + actual_count] = \
                src['Ismp'][offset:offset + actual_count]

        # 2D datasets: (dim, n_pixels) — column-slice copy
        for name in self.COPY_2D + self.COPY_Q:
            if name in src and name in self._f:
                self._f[name][:, offset:offset + actual_count] = \
                    src[name][:, offset:offset + actual_count]

        self._f.flush()

    def write_derived_trn(
        self,
        offset: int,
        derived: 'DerivedResults',
        alpha_val: float = 1.0,
        degree_val: float = 1.0,
    ):
        """Write Category C derived metrics for Phase 1 (training).

        Writes: Xrlt, Ctrn, FVtrn, SVtrn, SPStrn, alpha, degree

        Args:
            offset: Start pixel index
            derived: DerivedResults from compute_all_derived_batch()
            alpha_val: Regularization alpha used
            degree_val: Final degree value
        """
        self._wait_pending_write()
        count = derived.C.shape[0] if derived.C is not None else 0
        if count == 0:
            return

        # Xrlt: (batch, n_angles_opt, n_elem) → H5 (batch, n_elem, n_angles_opt)
        if derived.Xrlt is not None and 'Xrlt' in self._f:
            self._f['Xrlt'][offset:offset + count] = \
                np.transpose(derived.Xrlt, (0, 2, 1)).astype(np.float32)

        # Ctrn: (batch, n_elem) → H5 (n_elem, batch)
        if derived.C is not None and 'Ctrn' in self._f:
            self._f['Ctrn'][:, offset:offset + count] = \
                derived.C.T.astype(np.float32)

        # FVtrn: (batch, n_fv) → H5 (n_fv, batch)
        if derived.FV is not None and 'FVtrn' in self._f:
            self._f['FVtrn'][:, offset:offset + count] = \
                derived.FV.T.astype(np.float32)

        # SVtrn: (batch, n_sv) → H5 (n_sv, batch)
        if derived.SV is not None and 'SVtrn' in self._f:
            self._f['SVtrn'][:, offset:offset + count] = \
                derived.SV.T.astype(np.float32)

        # SPStrn: (batch, 2) → H5 (2, batch)
        if derived.SPS is not None and 'SPStrn' in self._f:
            self._f['SPStrn'][:, offset:offset + count] = \
                derived.SPS.T.astype(np.float32)

        # Alpha: (3, n) — row 0 = alpha, rows 1-2 = 0
        if 'alpha' in self._f:
            alpha_col = np.zeros((3, count), dtype=np.float32)
            alpha_col[0, :] = alpha_val
            self._f['alpha'][:, offset:offset + count] = alpha_col

        # Degree: (3, n) — row 0 = degree, rows 1-2 = 0
        if 'degree' in self._f:
            degree_col = np.zeros((3, count), dtype=np.float32)
            degree_col[0, :] = degree_val
            self._f['degree'][:, offset:offset + count] = degree_col

        self._f.flush()

    def write_derived_rlt(
        self,
        offset: int,
        derived: 'DerivedResults',
    ):
        """Write Category C derived metrics for Phase 2 (refined).

        Writes: Crlt, FVrlt, SVrlt, SPSrlt
        (Xrlt is shared between phases; alpha/degree already written by trn)

        Args:
            offset: Start pixel index
            derived: DerivedResults from compute_all_derived_batch()
        """
        self._wait_pending_write()
        count = derived.C.shape[0] if derived.C is not None else 0
        if count == 0:
            return

        if derived.C is not None and 'Crlt' in self._f:
            self._f['Crlt'][:, offset:offset + count] = \
                derived.C.T.astype(np.float32)

        if derived.FV is not None and 'FVrlt' in self._f:
            self._f['FVrlt'][:, offset:offset + count] = \
                derived.FV.T.astype(np.float32)

        if derived.SV is not None and 'SVrlt' in self._f:
            self._f['SVrlt'][:, offset:offset + count] = \
                derived.SV.T.astype(np.float32)

        if derived.SPS is not None and 'SPSrlt' in self._f:
            self._f['SPSrlt'][:, offset:offset + count] = \
                derived.SPS.T.astype(np.float32)

        self._f.flush()

    def write_config(self, config_dict: dict):
        """Store pipeline configuration as JSON attr for reproducibility.

        Args:
            config_dict: Pipeline configuration dict
        """
        self._f.attrs['_deppro_config_json'] = json.dumps(config_dict)

    def finalize(self):
        """Mark the output file as complete.

        Removes the _deppro_incomplete flag.
        Call this after all chunks have been written.
        """
        self._wait_pending_write()
        if '_deppro_incomplete' in self._f.attrs:
            del self._f.attrs['_deppro_incomplete']
        self._f.attrs['_deppro_finalized'] = time.strftime('%Y-%m-%d %H:%M:%S')
        self._f.flush()

    def copy_dtrn_to_drlt(self):
        """Copy Dtrn → Drlt for Phase-1-only runs (no MEM).

        Also copies trn → rlt for derived 2D datasets (C, FV, SV, SPS).
        """
        self._wait_pending_write()
        # Read Dtrn and write to Drlt in chunks
        dtrn = self._f['Dtrn']
        drlt = self._f['Drlt']
        n = dtrn.shape[0]
        chunk_size = self._h5_chunk_d[0]

        for offset in range(0, n, chunk_size):
            end = min(offset + chunk_size, n)
            drlt[offset:end] = dtrn[offset:end]

        # Copy Q as well: (3, n_pixels) — column-slice copy
        if 'Qtrn' in self._f and 'Qrlt' in self._f:
            qtrn = self._f['Qtrn']
            qrlt = self._f['Qrlt']
            n_cols = qtrn.shape[1]
            for offset in range(0, n_cols, chunk_size):
                end = min(offset + chunk_size, n_cols)
                qrlt[:, offset:end] = qtrn[:, offset:end]

        # Copy derived 2D datasets: trn → rlt
        if self.full_gui_compat:
            pairs_2d = [
                ('Ctrn', 'Crlt'), ('FVtrn', 'FVrlt'),
                ('SVtrn', 'SVrlt'), ('SPStrn', 'SPSrlt'),
            ]
            for src_name, dst_name in pairs_2d:
                if src_name in self._f and dst_name in self._f:
                    src_ds = self._f[src_name]
                    dst_ds = self._f[dst_name]
                    # 2D: (dim, n_pixels) — copy in column chunks
                    n_cols = src_ds.shape[1]
                    for off in range(0, n_cols, chunk_size):
                        end = min(off + chunk_size, n_cols)
                        dst_ds[:, off:end] = src_ds[:, off:end]

        self._f.flush()


# ==============================================================================
# Utility functions
# ==============================================================================

def estimate_memory_budget(
    meta: DepthProfileMetadata,
    n_workers: int = 16,
) -> Dict[str, float]:
    """Estimate memory usage for different chunk sizes.

    Args:
        meta: Dataset metadata
        n_workers: Number of parallel workers

    Returns:
        Dict with memory estimates in GB
    """
    ne = meta.n_elements
    nd = meta.n_depth
    na = meta.n_angles
    nxy = meta.nxy if meta.nxy > 0 else 250000

    # Per-pixel memory
    xstk_per_px = na * ne * 8   # float64
    dopt_per_px = nd * ne * 8   # float64
    solver_overhead = 50000     # ~50KB per solver instance (conservative)

    # Per chunk (1 time step = nxy pixels)
    xstk_chunk = nxy * xstk_per_px
    dopt_chunk = nxy * dopt_per_px
    io_buffers = 2 * (xstk_chunk + dopt_chunk)  # double buffer

    # Worker pool memory
    worker_mem = n_workers * solver_overhead

    total_bytes = xstk_chunk + dopt_chunk + io_buffers + worker_mem

    return {
        'chunk_pixels': nxy,
        'xstk_chunk_gb': xstk_chunk / 1e9,
        'dopt_chunk_gb': dopt_chunk / 1e9,
        'io_buffers_gb': io_buffers / 1e9,
        'worker_overhead_gb': worker_mem / 1e9,
        'total_gb': total_bytes / 1e9,
        'output_file_gb': meta.n_pixels * ne * nd * 4 / 1e9,  # float32
        'output_compressed_gb': meta.n_pixels * ne * nd * 4 / 1e9 / 4.0,  # ~4x gzip-1
    }


def auto_chunk_size(
    meta: DepthProfileMetadata,
    memory_budget_gb: float = 4.0,
) -> int:
    """Calculate optimal chunk size (number of pixels) given memory budget.

    Args:
        meta: Dataset metadata
        memory_budget_gb: Memory budget in GB (default: 4GB)

    Returns:
        Number of pixels per chunk (aligned to nxy if possible)
    """
    ne = meta.n_elements
    nd = meta.n_depth
    na = meta.n_angles

    # Memory per pixel (read buffer + output buffer + double buffer overhead)
    bytes_per_px = (na * ne * 8) + (nd * ne * 8) * 2  # Xstk + D_opt * 2
    bytes_per_px *= 2  # Double buffer

    budget_bytes = memory_budget_gb * 1e9
    max_pixels = int(budget_bytes / bytes_per_px)

    # Align to time step boundary
    nxy = meta.nxy
    if nxy > 0 and max_pixels >= nxy:
        # Round down to multiple of nxy
        n_time_steps = max_pixels // nxy
        return n_time_steps * nxy
    elif nxy > 0 and max_pixels < nxy:
        # Can't fit even 1 time step — use nxy anyway (warn?)
        return nxy
    else:
        # No grid info — use raw pixel count
        return max(1000, max_pixels)
