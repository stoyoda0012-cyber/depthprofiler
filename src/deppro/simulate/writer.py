"""
Dataset Writers
===============

Parquet (NN training) and HDF5 (MATLAB compatible) output writers
for simulated depth profile data.

Parquet:
    - FixedSizeList columns for zero-copy PyTorch tensor loading
    - Snappy compression, batch-streaming write
    - Schema: I_obs, D_true, thickness, abruptness, n_layers

HDF5:
    - MATLAB DepthProfiler GUI compatible format
    - Dstk/Xstk in MATLAB layout (pixels, elements, depth/angles)
    - ezdeprof metadata group with all required fields
    - Compatible with deppro.io.dstk.load_dstk_data()
"""

import numpy as np
import h5py
from pathlib import Path
from typing import Dict, List, Optional

from .presets import ElementSystemPreset


# =============================================================================
# Parquet Writer (NN Training)
# =============================================================================

class ParquetWriter:
    """Streaming Parquet writer for NN training data.

    Writes in row groups of batch_size for efficient streaming reads.
    Uses fixed-size list columns for zero-copy tensor loading.

    Parameters
    ----------
    path : str
        Output file path.
    n_angles : int
        Number of observation angles.
    n_depth : int
        Number of depth points.
    n_elements : int
        Number of elements.
    batch_size : int
        Samples per row group.
    compression : str
        Parquet compression ('snappy', 'gzip', 'zstd').

    Schema
    ------
    I_obs : FixedSizeList[float32, n_angles * n_elements]
        Flattened angle-resolved intensities (NN input).
        Reshape to (n_angles, n_elements) in dataloader.
    D_true : FixedSizeList[float32, n_depth * n_elements]
        Flattened depth profile (NN target).
        Reshape to (n_depth, n_elements) in dataloader.
    thickness : float32
        Primary interface position (Angstroms).
    abruptness : float32
        Primary interface width FWHM (Angstroms).
    n_layers : int8
        Number of overlay layers.
    """

    def __init__(
        self,
        path: str,
        n_angles: int,
        n_depth: int,
        n_elements: int,
        batch_size: int = 10_000,
        compression: str = 'snappy',
    ):
        import pyarrow as pa
        import pyarrow.parquet as pq

        self.path = Path(path)
        self.n_angles = n_angles
        self.n_depth = n_depth
        self.n_elements = n_elements
        self.batch_size = batch_size

        self._i_obs_size = n_angles * n_elements
        self._d_true_size = n_depth * n_elements

        self._schema = pa.schema([
            ('I_obs', pa.list_(pa.float32(), self._i_obs_size)),
            ('D_true', pa.list_(pa.float32(), self._d_true_size)),
            ('thickness', pa.float32()),
            ('abruptness', pa.float32()),
            ('n_layers', pa.int8()),
        ])

        self._writer = pq.ParquetWriter(
            str(self.path), self._schema,
            compression=compression,
        )
        self._buffer_I: List[np.ndarray] = []
        self._buffer_D: List[np.ndarray] = []
        self._buffer_meta: List[Dict] = []
        self._total_written = 0

    def write_sample(
        self,
        I_obs: np.ndarray,
        D_true: np.ndarray,
        thickness: float,
        abruptness: float,
        n_layers: int,
    ) -> None:
        """Buffer a single sample."""
        self._buffer_I.append(I_obs.flatten().astype(np.float32))
        self._buffer_D.append(D_true.flatten().astype(np.float32))
        self._buffer_meta.append({
            'thickness': float(thickness),
            'abruptness': float(abruptness),
            'n_layers': int(n_layers),
        })
        if len(self._buffer_I) >= self.batch_size:
            self._flush()

    def write_batch(
        self,
        I_obs_batch: np.ndarray,
        D_true_batch: np.ndarray,
        thickness_batch: np.ndarray,
        abruptness_batch: np.ndarray,
        n_layers_batch: np.ndarray,
    ) -> None:
        """Write a pre-collected batch directly.

        Parameters
        ----------
        I_obs_batch : ndarray, shape (N, n_angles, n_elements)
        D_true_batch : ndarray, shape (N, n_depth, n_elements)
        thickness_batch : ndarray, shape (N,)
        abruptness_batch : ndarray, shape (N,)
        n_layers_batch : ndarray, shape (N,)
        """
        import pyarrow as pa

        N = len(I_obs_batch)
        I_flat = I_obs_batch.reshape(N, -1).astype(np.float32)
        D_flat = D_true_batch.reshape(N, -1).astype(np.float32)

        table = pa.table({
            'I_obs': pa.FixedSizeListArray.from_arrays(
                I_flat.flatten(), self._i_obs_size,
            ),
            'D_true': pa.FixedSizeListArray.from_arrays(
                D_flat.flatten(), self._d_true_size,
            ),
            'thickness': pa.array(thickness_batch.astype(np.float32)),
            'abruptness': pa.array(abruptness_batch.astype(np.float32)),
            'n_layers': pa.array(n_layers_batch.astype(np.int8)),
        }, schema=self._schema)

        self._writer.write_table(table)
        self._total_written += N

    def _flush(self) -> None:
        """Flush buffer to Parquet row group."""
        if not self._buffer_I:
            return

        import pyarrow as pa

        N = len(self._buffer_I)
        I_arr = np.array(self._buffer_I)
        D_arr = np.array(self._buffer_D)

        table = pa.table({
            'I_obs': pa.FixedSizeListArray.from_arrays(
                I_arr.flatten(), self._i_obs_size,
            ),
            'D_true': pa.FixedSizeListArray.from_arrays(
                D_arr.flatten(), self._d_true_size,
            ),
            'thickness': pa.array(
                [m['thickness'] for m in self._buffer_meta], pa.float32(),
            ),
            'abruptness': pa.array(
                [m['abruptness'] for m in self._buffer_meta], pa.float32(),
            ),
            'n_layers': pa.array(
                [m['n_layers'] for m in self._buffer_meta], pa.int8(),
            ),
        }, schema=self._schema)

        self._writer.write_table(table)
        self._total_written += N
        self._buffer_I.clear()
        self._buffer_D.clear()
        self._buffer_meta.clear()

    @property
    def total_written(self) -> int:
        return self._total_written

    def close(self) -> None:
        self._flush()
        self._writer.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# =============================================================================
# HDF5 Writer (MATLAB Compatible)
# =============================================================================

def _element_label(symbol: str, charge: float) -> str:
    """MATLAB ``ezdeprof/element`` label: ``('Si', 4) -> 'Si(+4)'``.

    Same convention as ``deppro.io.h5_writer.format_element_with_charge``
    (``+4`` / ``-2`` / ``0``); kept local so this module does not depend on
    the legacy io writer.
    """
    c = int(charge)
    sign = "+" if c > 0 else ""
    return f"{symbol}({sign}{c})"


class H5Writer:
    """HDF5 writer compatible with MATLAB DepthProfiler GUI.

    Writes Dstk + Xstk + ezdeprof metadata in a format that
    deppro.io.dstk.load_dstk_data() can read directly.

    MATLAB layout:
        Dstk: (n_pixels, n_elements, n_depth) float32
        Xstk: (n_pixels, n_elements, n_angles) float32
        TransMatObs: (n_elements, n_depth, n_angles) float64

    Besides the element tables (``elementsymbol`` / ``orbital`` /
    ``compound``), ``finalize`` writes the two datasets that
    ``deppro.visualize_4d.load_gt_4d`` reads without an existence check:
    ``ezdeprof/element`` (charge-labeled names, ``'Si(+4)'``) and
    ``ezdeprof/dxdydt`` (``[[nx, ny, nt]]``, shape (1, 3) float64, the
    layout of ``deppro.app.depth_result_writer``). Samples that are not on
    a spatial grid are recorded as one row, ``[[n_pixels, 1, 1]]``.

    Parameters
    ----------
    path : str
        Output H5 file path.
    preset : ElementSystemPreset
        Element system configuration.
    """

    def __init__(self, path: str, preset: ElementSystemPreset):
        self.path = Path(path)
        self.preset = preset
        self._D_batches: List[np.ndarray] = []
        self._X_batches: List[np.ndarray] = []

    def add_batch(
        self,
        D_batch: np.ndarray,
        X_batch: np.ndarray,
    ) -> None:
        """Accumulate samples.

        Parameters
        ----------
        D_batch : ndarray, shape (N, n_depth, n_elements)
            Python layout depth profiles.
        X_batch : ndarray, shape (N, n_angles, n_elements)
            Python layout intensities.
        """
        self._D_batches.append(D_batch)
        self._X_batches.append(X_batch)

    def finalize(
        self,
        depth: np.ndarray,
        angles: np.ndarray,
        step_size: float,
        photon_energy: float,
        work_function: float = 4.5,
        gamma_eff: float = 1.0,
        trans_mat: Optional[np.ndarray] = None,
        nx: Optional[int] = None,
        ny: Optional[int] = None,
    ) -> str:
        """Write accumulated data to H5 file.

        Parameters
        ----------
        depth : ndarray
            Depth array in Angstroms.
        angles : ndarray
            Observation angles in degrees.
        step_size : float
            Depth step size in Angstroms.
        photon_energy : float
            X-ray photon energy in eV.
        trans_mat : ndarray, optional
            Transfer matrix [n_angles, n_depth, n_elements].
            If provided, saved as /ezdeprof/TransMatObs in MATLAB layout.
        nx, ny : int, optional
            Spatial grid of the accumulated samples, written to
            ``ezdeprof/dxdydt`` as ``[[nx, ny, nt]]`` with
            ``nt = n_pixels // (nx * ny)``. Both or neither: with neither,
            the samples are recorded as one row (``nx = n_pixels``,
            ``ny = 1``). ``n_pixels`` must be a multiple of ``nx * ny``.

        Returns
        -------
        str
            Path to written file.
        """
        # Concatenate all batches
        Dstk = np.concatenate(self._D_batches, axis=0)   # (N, n_depth, n_elem)
        Xstk = np.concatenate(self._X_batches, axis=0)   # (N, n_angles, n_elem)
        n_pixels = Dstk.shape[0]

        # Spatial grid -> dxdydt. Resolved before anything is written so a
        # bad grid leaves no partial file behind.
        if (nx is None) != (ny is None):
            raise ValueError("nx and ny must be given together (or neither)")
        if nx is None:
            nx, ny = n_pixels, 1
        nx, ny = int(nx), int(ny)
        if nx < 1 or ny < 1 or n_pixels % (nx * ny) != 0:
            raise ValueError(
                f"n_pixels={n_pixels} is not a multiple of nx*ny={nx}*{ny}"
            )
        nt = n_pixels // (nx * ny)

        # Transpose to MATLAB layout: (pixels, elements, depth/angles)
        Dstk_ml = np.transpose(Dstk, (0, 2, 1)).astype(np.float32)
        Xstk_ml = np.transpose(Xstk, (0, 2, 1)).astype(np.float32)

        preset = self.preset

        with h5py.File(self.path, 'w') as f:
            # Core datasets
            f.create_dataset('Dstk', data=Dstk_ml, compression='gzip',
                             compression_opts=1)
            f.create_dataset('Xstk', data=Xstk_ml, compression='gzip',
                             compression_opts=1)

            # Ezdeprof metadata group
            ez = f.create_group('ezdeprof')

            # Depth and angle arrays (column vectors, MATLAB convention)
            ez.create_dataset('depth', data=depth.reshape(-1, 1).astype(np.float64))
            ez.create_dataset('thetaobs', data=angles.reshape(1, -1).astype(np.float64))

            # Scalar parameters
            ez.create_dataset('stepsize', data=np.array([[step_size]], dtype=np.float64))
            ez.create_dataset('gammaeff', data=np.array([[gamma_eff]], dtype=np.float64))
            ez.create_dataset('photonenergy', data=np.array([[photon_energy]], dtype=np.float64))
            ez.create_dataset('workfunction', data=np.array([[work_function]], dtype=np.float64))

            # Element metadata
            n_elem = preset.n_elements
            symbols = [e.symbol for e in preset.elements]
            orbitals = [e.orbital for e in preset.elements]
            compounds = [e.compound for e in preset.elements]

            # String arrays — store as bytes for MATLAB compatibility
            dt_str = h5py.special_dtype(vlen=str)
            ez.create_dataset('elementsymbol', data=np.array(symbols, dtype='S10'))
            ez.create_dataset('orbital', data=np.array(orbitals, dtype='S10'))
            ez.create_dataset('compound', data=np.array(compounds, dtype='S20'))

            # Charge-labeled element names ('Si(+4)'): what the loaders key
            # colors and element order on (ezdeprof/element, MATLAB).
            labels = [_element_label(sym, c) for sym, c in zip(symbols, preset.charges)]
            ez.create_dataset('element', data=labels, dtype=dt_str)

            # Spatial grid: (1, 3) float64 like app/depth_result_writer.
            ez.create_dataset('dxdydt', data=np.array([[nx, ny, nt]], dtype=np.float64))

            # Charges and linkedconst
            ez.create_dataset('charge', data=np.array(
                preset.charges, dtype=np.float64).reshape(1, -1))
            ez.create_dataset('linkedconst', data=np.array(
                preset.linkedconst, dtype=np.float64).reshape(1, -1))

            # Constraint indices (1-based for MATLAB)
            ez.create_dataset('free', data=preset.free_matlab.reshape(1, -1))
            ez.create_dataset('subst', data=preset.subst_matlab.reshape(1, -1))
            ez.create_dataset('linked', data=preset.linked_matlab.reshape(1, -1))

            # Transfer matrix (MATLAB layout: elements, depth, angles)
            if trans_mat is not None:
                # Python: (angles, depth, elements) -> MATLAB: (elements, depth, angles)
                tm_ml = np.transpose(trans_mat, (2, 1, 0)).astype(np.float64)
                ez.create_dataset('TransMatObs', data=tm_ml)

            # Simulation metadata
            ez.create_dataset('stack', data=np.array([[n_pixels]], dtype=np.float64))
            ez.create_dataset('loop', data=np.array([[0]], dtype=np.float64))
            ez.create_dataset('currentloop', data=np.array([[0]], dtype=np.float64))

            # Binding energies
            be_list = []
            for e in preset.elements:
                be_list.append(e.binding_energy if e.binding_energy else 0.0)
            ez.create_dataset('BindingEnergy', data=np.array(
                be_list, dtype=np.float64).reshape(1, -1))
            # MATLAB-dialect name read by streaming_io.StreamingDepthReader
            # (the capitalized key above is kept for files that already use it).
            ez.create_dataset('bindingenergy', data=ez['BindingEnergy'][()])

        self._D_batches.clear()
        self._X_batches.clear()

        return str(self.path)
