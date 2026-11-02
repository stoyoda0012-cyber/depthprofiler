"""
4D Profile Data Structure for Sputter Depth Profiling
======================================================

Handles time-series depth profiling data with spatial mapping.

Data dimensions:
    - depth (z): Depth axis in Angstroms
    - elements: Chemical species
    - spatial (x, y): Lateral position on sample
    - time (t): Sputter time / cycle index

This corresponds to the MATLAB data structure:
    - D: (depth, elements, spatial_points, time)
    - V: (x, y, z, 1, elements) for 3D visualization
    - Dstk/Drlt/Dtrn: Initial/Result/Training profiles

Usage:
    from deppro.profile_4d import Profile4D, TimeSeriesAnalyzer

    # Create 4D profile
    profile = Profile4D(
        depth=np.linspace(0, 50, 51),
        elements=['C', 'O', 'Si', 'Al'],
        spatial_shape=(10, 10),
        n_times=20,
    )

    # Add data for each time step
    profile.set_profile(t=0, data=initial_profile)

    # Analyze time series
    analyzer = TimeSeriesAnalyzer(profile)
    thickness_vs_time = analyzer.track_layer_thickness(layer_idx=0)

Date: 2026-01-23
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Union, Callable, Any
from pathlib import Path
import h5py
import warnings

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


@dataclass
class Profile4DConfig:
    """Configuration for 4D profile storage.

    Attributes:
        depth: Depth axis array (Angstroms)
        elements: List of element labels
        spatial_shape: (nx, ny) spatial grid shape
        n_times: Number of time steps
        store_mode: 'memory' or 'hdf5'
        hdf5_path: Path for HDF5 storage (if store_mode='hdf5')
        chunk_size: Chunk size for HDF5 storage
    """
    depth: np.ndarray
    elements: List[str]
    spatial_shape: Tuple[int, int] = (1, 1)
    n_times: int = 1
    store_mode: str = 'memory'
    hdf5_path: Optional[str] = None
    chunk_size: int = 100


class Profile4D:
    """4D depth profile container for time-series spatial mapping.

    Stores concentration profiles as:
        data[z, element, x, y, t] = concentration (0-1)

    Supports both in-memory and HDF5-backed storage.
    """

    def __init__(self, config: Profile4DConfig):
        """Initialize 4D profile container.

        Args:
            config: Profile configuration
        """
        self.config = config
        self.depth = config.depth
        self.elements = config.elements
        self.n_depth = len(config.depth)
        self.n_elements = len(config.elements)
        self.nx, self.ny = config.spatial_shape
        self.n_times = config.n_times

        # Shape: (depth, elements, x, y, time)
        self.shape = (self.n_depth, self.n_elements, self.nx, self.ny, self.n_times)

        # Initialize storage
        if config.store_mode == 'memory':
            self._data = np.zeros(self.shape, dtype=np.float32)
            self._hdf5_file = None
        elif config.store_mode == 'hdf5':
            if config.hdf5_path is None:
                raise ValueError("hdf5_path required for HDF5 storage")
            self._init_hdf5(config.hdf5_path, config.chunk_size)
            self._data = None
        else:
            raise ValueError(f"Unknown store_mode: {config.store_mode}")

        # Metadata
        self.metadata: Dict[str, Any] = {}
        self._modified_times: List[int] = []

    def _init_hdf5(self, path: str, chunk_size: int):
        """Initialize HDF5 storage."""
        self._hdf5_file = h5py.File(path, 'w')

        # Main data array
        chunks = (
            min(self.n_depth, 10),
            min(self.n_elements, 4),
            min(self.nx, chunk_size),
            min(self.ny, chunk_size),
            1,
        )
        self._hdf5_file.create_dataset(
            'Drlt',  # Result profiles (matches MATLAB naming)
            shape=self.shape,
            dtype='float32',
            chunks=chunks,
            compression='gzip',
        )

        # Optional: Initial/Training profiles
        self._hdf5_file.create_dataset(
            'Dstk', shape=self.shape, dtype='float32',
            chunks=chunks, compression='gzip',
        )
        self._hdf5_file.create_dataset(
            'Dtrn', shape=self.shape, dtype='float32',
            chunks=chunks, compression='gzip',
        )

        # Film parameters (thickness, interface width per spatial point per time)
        self._hdf5_file.create_dataset(
            'FVrlt',
            shape=(self.nx * self.ny * self.n_times, 2),
            dtype='float32',
        )

        # Objective function history
        self._hdf5_file.create_dataset(
            'Qrlt',
            shape=(self.nx * self.ny * self.n_times, 3),
            dtype='float32',
        )

        # Store metadata
        self._hdf5_file.attrs['depth'] = self.depth
        self._hdf5_file.attrs['elements'] = np.array(self.elements, dtype='S')
        self._hdf5_file.attrs['spatial_shape'] = np.array([self.nx, self.ny])
        self._hdf5_file.attrs['n_times'] = self.n_times

    def __del__(self):
        """Close HDF5 file on deletion."""
        if self._hdf5_file is not None:
            self._hdf5_file.close()

    @property
    def data(self) -> np.ndarray:
        """Get data array (loads from HDF5 if needed)."""
        if self._data is not None:
            return self._data
        else:
            return self._hdf5_file['Drlt'][:]

    def set_profile(
        self,
        t: int,
        x: int = 0,
        y: int = 0,
        data: np.ndarray = None,
        profile_type: str = 'result',
    ):
        """Set profile at specific spatial/time coordinates.

        Args:
            t: Time index
            x: X coordinate
            y: Y coordinate
            data: Profile array (depth, elements)
            profile_type: 'result', 'initial', or 'training'
        """
        if data.shape != (self.n_depth, self.n_elements):
            raise ValueError(f"Expected shape {(self.n_depth, self.n_elements)}, got {data.shape}")

        if self._data is not None:
            self._data[:, :, x, y, t] = data
        else:
            key = {'result': 'Drlt', 'initial': 'Dstk', 'training': 'Dtrn'}[profile_type]
            self._hdf5_file[key][:, :, x, y, t] = data

        if t not in self._modified_times:
            self._modified_times.append(t)

    def get_profile(
        self,
        t: int,
        x: int = 0,
        y: int = 0,
        profile_type: str = 'result',
    ) -> np.ndarray:
        """Get profile at specific coordinates.

        Args:
            t: Time index
            x: X coordinate
            y: Y coordinate
            profile_type: 'result', 'initial', or 'training'

        Returns:
            Profile array (depth, elements)
        """
        if self._data is not None:
            return self._data[:, :, x, y, t].copy()
        else:
            key = {'result': 'Drlt', 'initial': 'Dstk', 'training': 'Dtrn'}[profile_type]
            return self._hdf5_file[key][:, :, x, y, t]

    def set_time_slice(
        self,
        t: int,
        data: np.ndarray,
        profile_type: str = 'result',
    ):
        """Set all spatial profiles for a time step.

        Args:
            t: Time index
            data: Profile array (depth, elements, x, y)
            profile_type: 'result', 'initial', or 'training'
        """
        expected = (self.n_depth, self.n_elements, self.nx, self.ny)
        if data.shape != expected:
            raise ValueError(f"Expected shape {expected}, got {data.shape}")

        if self._data is not None:
            self._data[:, :, :, :, t] = data
        else:
            key = {'result': 'Drlt', 'initial': 'Dstk', 'training': 'Dtrn'}[profile_type]
            self._hdf5_file[key][:, :, :, :, t] = data

        if t not in self._modified_times:
            self._modified_times.append(t)

    def get_time_slice(
        self,
        t: int,
        profile_type: str = 'result',
    ) -> np.ndarray:
        """Get all spatial profiles for a time step.

        Args:
            t: Time index
            profile_type: 'result', 'initial', or 'training'

        Returns:
            Profile array (depth, elements, x, y)
        """
        if self._data is not None:
            return self._data[:, :, :, :, t].copy()
        else:
            key = {'result': 'Drlt', 'initial': 'Dstk', 'training': 'Dtrn'}[profile_type]
            return self._hdf5_file[key][:, :, :, :, t]

    def get_spatial_average(
        self,
        t: int,
        profile_type: str = 'result',
    ) -> np.ndarray:
        """Get spatially averaged profile for a time step.

        Args:
            t: Time index
            profile_type: Profile type

        Returns:
            Averaged profile (depth, elements)
        """
        slice_data = self.get_time_slice(t, profile_type)
        return np.mean(slice_data, axis=(2, 3))

    def get_element_profile(
        self,
        element: Union[str, int],
        t: int,
        profile_type: str = 'result',
    ) -> np.ndarray:
        """Get 3D profile for a single element.

        Args:
            element: Element name or index
            t: Time index
            profile_type: Profile type

        Returns:
            Profile (depth, x, y)
        """
        if isinstance(element, str):
            elem_idx = self.elements.index(element)
        else:
            elem_idx = element

        slice_data = self.get_time_slice(t, profile_type)
        return slice_data[:, elem_idx, :, :]

    def set_film_parameters(
        self,
        t: int,
        x: int,
        y: int,
        thickness: float,
        width: float,
    ):
        """Set film parameters for a spatial point.

        Args:
            t: Time index
            x: X coordinate
            y: Y coordinate
            thickness: Film thickness
            width: Interface width
        """
        idx = t * self.nx * self.ny + y * self.nx + x
        if self._hdf5_file is not None:
            self._hdf5_file['FVrlt'][idx, :] = [thickness, width]

    def get_film_parameters(
        self,
        t: int,
        x: int = 0,
        y: int = 0,
    ) -> Tuple[float, float]:
        """Get film parameters for a spatial point.

        Returns:
            (thickness, width)
        """
        idx = t * self.nx * self.ny + y * self.nx + x
        if self._hdf5_file is not None:
            params = self._hdf5_file['FVrlt'][idx, :]
            return float(params[0]), float(params[1])
        return 0.0, 0.0


class TimeSeriesAnalyzer:
    """Analyzer for time-series 4D profiles.

    Tracks layer evolution, thickness changes, and composition trends.
    """

    def __init__(self, profile: Profile4D):
        """Initialize analyzer.

        Args:
            profile: Profile4D container
        """
        self.profile = profile

    def track_layer_thickness(
        self,
        layer_element: Union[str, int],
        threshold: float = 0.5,
        spatial_average: bool = True,
    ) -> np.ndarray:
        """Track layer thickness over time.

        Args:
            layer_element: Element defining the layer
            threshold: Concentration threshold for layer boundary
            spatial_average: Average over spatial coordinates

        Returns:
            Thickness array (n_times,) or (n_times, nx, ny)
        """
        if isinstance(layer_element, str):
            elem_idx = self.profile.elements.index(layer_element)
        else:
            elem_idx = layer_element

        thicknesses = np.zeros((self.profile.n_times, self.profile.nx, self.profile.ny))
        depth = self.profile.depth

        for t in range(self.profile.n_times):
            slice_data = self.profile.get_time_slice(t)
            elem_profile = slice_data[:, elem_idx, :, :]  # (depth, x, y)

            for x in range(self.profile.nx):
                for y in range(self.profile.ny):
                    profile = elem_profile[:, x, y]
                    # Find where concentration drops below threshold
                    above_threshold = profile >= threshold
                    if np.any(above_threshold):
                        # Find last depth above threshold
                        last_idx = np.where(above_threshold)[0][-1]
                        thicknesses[t, x, y] = depth[last_idx]

        if spatial_average:
            return np.mean(thicknesses, axis=(1, 2))
        return thicknesses

    def track_interface_position(
        self,
        upper_element: Union[str, int],
        lower_element: Union[str, int],
        spatial_average: bool = True,
    ) -> np.ndarray:
        """Track interface position between two layers.

        Interface defined as where concentrations are equal.

        Args:
            upper_element: Element in upper layer
            lower_element: Element in lower layer
            spatial_average: Average over spatial coordinates

        Returns:
            Interface position array
        """
        if isinstance(upper_element, str):
            upper_idx = self.profile.elements.index(upper_element)
        else:
            upper_idx = upper_element

        if isinstance(lower_element, str):
            lower_idx = self.profile.elements.index(lower_element)
        else:
            lower_idx = lower_element

        positions = np.zeros((self.profile.n_times, self.profile.nx, self.profile.ny))
        depth = self.profile.depth

        for t in range(self.profile.n_times):
            slice_data = self.profile.get_time_slice(t)

            for x in range(self.profile.nx):
                for y in range(self.profile.ny):
                    upper_profile = slice_data[:, upper_idx, x, y]
                    lower_profile = slice_data[:, lower_idx, x, y]

                    # Find crossover point
                    diff = upper_profile - lower_profile
                    crossings = np.where(np.diff(np.sign(diff)))[0]

                    if len(crossings) > 0:
                        # Interpolate for precise position
                        idx = crossings[0]
                        d0, d1 = depth[idx], depth[idx + 1]
                        v0, v1 = diff[idx], diff[idx + 1]
                        positions[t, x, y] = d0 - v0 * (d1 - d0) / (v1 - v0)

        if spatial_average:
            return np.mean(positions, axis=(1, 2))
        return positions

    def calculate_sputter_rate(
        self,
        element: Union[str, int],
        time_axis: np.ndarray,
    ) -> float:
        """Estimate sputter rate from thickness evolution.

        Args:
            element: Reference element for thickness tracking
            time_axis: Time values for each time index

        Returns:
            Sputter rate (Angstroms/unit time)
        """
        thicknesses = self.track_layer_thickness(element, spatial_average=True)

        # Linear fit
        valid = ~np.isnan(thicknesses) & (thicknesses > 0)
        if np.sum(valid) < 2:
            return 0.0

        coeffs = np.polyfit(time_axis[valid], thicknesses[valid], 1)
        return -coeffs[0]  # Negative because thickness decreases

    def get_composition_trend(
        self,
        depth_range: Tuple[float, float] = None,
        spatial_average: bool = True,
    ) -> Dict[str, np.ndarray]:
        """Get composition trends over time.

        Args:
            depth_range: (min_depth, max_depth) to average over
            spatial_average: Average over spatial coordinates

        Returns:
            Dict of element -> concentration vs time
        """
        trends = {elem: [] for elem in self.profile.elements}
        depth = self.profile.depth

        if depth_range is None:
            depth_mask = np.ones(len(depth), dtype=bool)
        else:
            depth_mask = (depth >= depth_range[0]) & (depth <= depth_range[1])

        for t in range(self.profile.n_times):
            if spatial_average:
                profile = self.profile.get_spatial_average(t)
            else:
                profile = self.profile.get_time_slice(t)

            for i, elem in enumerate(self.profile.elements):
                if spatial_average:
                    avg = np.mean(profile[depth_mask, i])
                else:
                    avg = np.mean(profile[depth_mask, i, :, :], axis=0)
                trends[elem].append(avg)

        return {elem: np.array(vals) for elem, vals in trends.items()}


# ============================================================================
# Visualization
# ============================================================================

def plot_3d_profile(
    profile: Profile4D,
    t: int,
    element: Union[str, int],
    ax: Optional['Axes3D'] = None,
    **kwargs,
) -> 'Axes3D':
    """Plot 3D isosurface of element distribution.

    Args:
        profile: Profile4D container
        t: Time index
        element: Element to visualize
        ax: Matplotlib 3D axes
        **kwargs: Additional plot arguments

    Returns:
        Matplotlib axes
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("Matplotlib required for plotting")

    from mpl_toolkits.mplot3d import Axes3D

    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

    elem_data = profile.get_element_profile(element, t)  # (depth, x, y)
    depth = profile.depth
    nx, ny = profile.nx, profile.ny

    # Create meshgrid
    X, Y = np.meshgrid(np.arange(ny), np.arange(nx))

    # Plot surface at different depths
    n_surfaces = min(5, len(depth))
    depth_indices = np.linspace(0, len(depth) - 1, n_surfaces, dtype=int)
    cmap = kwargs.get('cmap', 'viridis')
    alpha = kwargs.get('alpha', 0.7)

    for d_idx in depth_indices:
        Z = np.full_like(X, depth[d_idx], dtype=float)
        colors = plt.cm.get_cmap(cmap)(elem_data[d_idx, :, :])
        ax.plot_surface(X, Y, Z, facecolors=colors, alpha=alpha, shade=False)

    ax.set_xlabel('Y position')
    ax.set_ylabel('X position')
    ax.set_zlabel('Depth (Å)')
    ax.invert_zaxis()

    if isinstance(element, int):
        element = profile.elements[element]
    ax.set_title(f'{element} distribution @ t={t}')

    return ax


def plot_time_evolution(
    analyzer: TimeSeriesAnalyzer,
    time_axis: np.ndarray,
    elements: Optional[List[str]] = None,
    depth_range: Tuple[float, float] = None,
    ax: Optional['plt.Axes'] = None,
) -> 'plt.Axes':
    """Plot composition evolution over time.

    Args:
        analyzer: TimeSeriesAnalyzer instance
        time_axis: Time values
        elements: Elements to plot (None = all)
        depth_range: Depth range to average
        ax: Matplotlib axes

    Returns:
        Matplotlib axes
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("Matplotlib required for plotting")

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    trends = analyzer.get_composition_trend(depth_range=depth_range)

    if elements is None:
        elements = list(trends.keys())

    for elem in elements:
        if elem in trends:
            ax.plot(time_axis[:len(trends[elem])], trends[elem] * 100, label=elem, marker='o')

    ax.set_xlabel('Sputter time')
    ax.set_ylabel('Concentration (%)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    if depth_range:
        ax.set_title(f'Composition vs time (depth {depth_range[0]:.1f}-{depth_range[1]:.1f} Å)')
    else:
        ax.set_title('Composition vs time (full depth average)')

    return ax


def plot_thickness_evolution(
    analyzer: TimeSeriesAnalyzer,
    time_axis: np.ndarray,
    element: Union[str, int],
    threshold: float = 0.5,
    ax: Optional['plt.Axes'] = None,
) -> 'plt.Axes':
    """Plot layer thickness evolution.

    Args:
        analyzer: TimeSeriesAnalyzer instance
        time_axis: Time values
        element: Layer element
        threshold: Concentration threshold
        ax: Matplotlib axes

    Returns:
        Matplotlib axes
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("Matplotlib required for plotting")

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    thicknesses = analyzer.track_layer_thickness(element, threshold)
    ax.plot(time_axis[:len(thicknesses)], thicknesses, 'o-', label=f'{element} layer')

    # Fit linear trend
    valid = ~np.isnan(thicknesses) & (thicknesses > 0)
    if np.sum(valid) > 2:
        coeffs = np.polyfit(time_axis[valid], thicknesses[valid], 1)
        trend_line = np.polyval(coeffs, time_axis)
        ax.plot(time_axis, trend_line, '--', alpha=0.7, label=f'Trend ({coeffs[0]:.2f} Å/unit)')

    ax.set_xlabel('Sputter time')
    ax.set_ylabel('Layer thickness (Å)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title(f'{element} layer thickness evolution')

    return ax


# ============================================================================
# Factory functions
# ============================================================================

def create_profile_4d(
    depth: np.ndarray,
    elements: List[str],
    spatial_shape: Tuple[int, int] = (1, 1),
    n_times: int = 1,
    store_mode: str = 'memory',
    hdf5_path: Optional[str] = None,
) -> Profile4D:
    """Create 4D profile container.

    Args:
        depth: Depth axis (Angstroms)
        elements: Element labels
        spatial_shape: (nx, ny) spatial grid
        n_times: Number of time steps
        store_mode: 'memory' or 'hdf5'
        hdf5_path: Path for HDF5 storage

    Returns:
        Profile4D instance
    """
    config = Profile4DConfig(
        depth=depth,
        elements=elements,
        spatial_shape=spatial_shape,
        n_times=n_times,
        store_mode=store_mode,
        hdf5_path=hdf5_path,
    )
    return Profile4D(config)


def load_profile_4d(hdf5_path: str) -> Profile4D:
    """Load 4D profile from HDF5 file.

    Args:
        hdf5_path: Path to HDF5 file

    Returns:
        Profile4D instance
    """
    with h5py.File(hdf5_path, 'r') as f:
        depth = f.attrs['depth'][:]
        elements = [s.decode() for s in f.attrs['elements'][:]]
        spatial_shape = tuple(f.attrs['spatial_shape'][:])
        n_times = int(f.attrs['n_times'])

    config = Profile4DConfig(
        depth=depth,
        elements=elements,
        spatial_shape=spatial_shape,
        n_times=n_times,
        store_mode='hdf5',
        hdf5_path=hdf5_path,
    )
    return Profile4D(config)


if __name__ == '__main__':
    print("4D Profile Data Structure Module")
    print("=" * 40)
    print("\nData dimensions:")
    print("  - depth (z): Depth in Angstroms")
    print("  - elements: Chemical species")
    print("  - spatial (x, y): Sample position")
    print("  - time (t): Sputter cycle")
    print("\nUsage:")
    print("  from deppro.profile_4d import create_profile_4d, TimeSeriesAnalyzer")
    print("  profile = create_profile_4d(depth, elements, (10, 10), 20)")
    print("  analyzer = TimeSeriesAnalyzer(profile)")
