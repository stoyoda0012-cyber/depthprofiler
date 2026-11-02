"""
Depth Profile to Spectra Generator (depthprofile2spectra)
=========================================================

Generates synthetic XPS spectra from depth profiles at multiple
observation angles. This is the forward model for depth profiling.

Workflow:
    Depth Profile D(z) → Transfer Matrix → I(θ) → Voigt Spectra → H5 Files

Usage:
    from deppro.depth_generator import DepthSpectraGenerator

    generator = DepthSpectraGenerator(
        profile=profile,
        angles=[0, 30, 45, 60, 75],
        photon_energy=1486.6,
    )

    # Generate spectra
    generator.generate(output_dir='./spectra/', noise_level=100)

CLI:
    python -m deppro.cli_depth generate \
        --profile profile.json \
        --angles 0,30,45,60,75 \
        --output spectra/ \
        --noise-level 100
"""

import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Literal
import h5py
import json
import time

from .imfp import CompoundDatabase
from .depth_profile import DepthProfile, ElementConfig, create_five_layer_profile
from .transfer_matrix import TransferMatrix, AngleResolvedData, compute_angle_profile

# voigtfit lives in toyomacro (optional extra ``fit``). It is imported inside
# the code that needs it, from the installed package only (no sibling checkout,
# no sys.path change), so that ``import deppro.depth_generator`` succeeds
# without toyomacro and a missing toyomacro raises an ImportError that names
# the extra at the point of use (constructing a generator), as for every
# optional dependency.
_FIT_HINT = "pip install 'deppro[fit]' (toyomacro, the optional extra `fit`)"


def _voigtfit():
    """``toyomacro.voigtfit.spectra_generator`` or a named ImportError."""
    try:
        from toyomacro.voigtfit import spectra_generator as sg
    except ImportError as exc:
        raise ImportError(
            f"deppro.depth_generator needs toyomacro.voigtfit for Voigt spectra "
            f"and Poisson noise: {_FIT_HINT}. Underlying import error: {exc}"
        ) from exc
    return sg


@dataclass
class DepthGeneratorConfig:
    """Configuration for depth profile to spectra generation."""
    # Photon source
    photon_energy: float = 1486.6   # Al Kα (eV)
    work_function: float = 4.5      # eV

    # Spectral parameters
    energy_edge: float = 2.5        # Edge width around center (eV)
    energy_step: float = 0.05       # Energy step (eV)
    amplitude_scale: float = 1e6    # Base amplitude scaling

    # Background
    bg_base: float = 0.001          # Background base level
    bg_level_factor: float = 10.0   # Background level multiplier

    # Noise
    noise_level: float = 0.0        # 0 = no noise
    noise_type: Literal['poisson', 'gaussian', 'none'] = 'poisson'
    use_mlx: bool = True            # Use MLX for noise if available

    # Output format
    output_format: Literal['h5', 'npz'] = 'h5'
    max_components: int = 9

    @property
    def bg_level(self) -> float:
        return self.bg_base * self.bg_level_factor


class DepthSpectraGenerator:
    """
    Generates XPS spectra from depth profiles at multiple angles.

    This implements the forward model:
        D(z, element) → TransMat → I(θ, element) → Voigt spectra
    """

    def __init__(
        self,
        profile: DepthProfile,
        angles: Union[List[float], np.ndarray],
        config: Optional[DepthGeneratorConfig] = None,
    ):
        """
        Initialize the generator.

        Args:
            profile: DepthProfile object
            angles: Observation angles in degrees
            config: Generator configuration (optional)
        """
        self._sg = _voigtfit()   # ImportError naming `deppro[fit]` if toyomacro is absent
        self.profile = profile
        self.angles = np.asarray(angles)
        self.config = config or DepthGeneratorConfig()

        # Initialize transfer matrix
        self.transfer_matrix = TransferMatrix(
            depth=profile.depth,
            step_size=profile.step_size,
            angles=self.angles,
            photon_energy=self.config.photon_energy,
            work_function=self.config.work_function,
        )

        # Compute angle-resolved intensities
        self._compute_intensities()

    def _compute_intensities(self) -> None:
        """Compute angle-resolved intensities from depth profile."""
        self.transfer_matrix.compute(
            self.profile.concentrations,
            self.profile.elements,
        )
        self.intensities = self.transfer_matrix.forward(
            self.profile.concentrations,
            normalize=True,
        )

    def generate_spectrum(
        self,
        element_idx: int,
        angle_idx: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a single spectrum for one element at one angle.

        Args:
            element_idx: Index of element in profile
            angle_idx: Index of observation angle

        Returns:
            Tuple of (energy_axis, spectrum)
        """
        elem = self.profile.elements[element_idx]
        intensity = self.intensities[angle_idx, element_idx]

        # Energy axis centered on binding energy
        be = elem.binding_energy
        energy = np.arange(
            be - self.config.energy_edge,
            be + self.config.energy_edge + self.config.energy_step,
            self.config.energy_step
        )

        # Generate Voigt profile
        sigma = elem.gaussian_fwhm / (2 * np.sqrt(2 * np.log(2)))
        gamma = elem.lorentzian_fwhm / 2

        spectrum = self._sg.voigt_profile(energy, be, sigma, gamma)

        # Scale by intensity and amplitude
        spectrum *= intensity * self.config.amplitude_scale

        # Add background
        bg = self.config.bg_level * self.config.amplitude_scale
        spectrum += bg

        return energy, spectrum

    def generate_all_spectra(
        self,
        noise_level: Optional[float] = None,
    ) -> Dict[str, Dict[int, Tuple[np.ndarray, np.ndarray]]]:
        """
        Generate spectra for all elements at all angles.

        Args:
            noise_level: Override config noise level

        Returns:
            Dict[element_label][angle_idx] = (energy, spectrum)
        """
        noise = noise_level if noise_level is not None else self.config.noise_level

        result = {}
        for elem_idx, elem in enumerate(self.profile.elements):
            label = elem.label
            result[label] = {}

            for angle_idx, angle in enumerate(self.angles):
                energy, spectrum = self.generate_spectrum(elem_idx, angle_idx)

                # Add noise if requested
                if noise > 0 and self.config.noise_type != 'none':
                    if self.config.use_mlx and getattr(self._sg, "HAS_MLX", False) \
                            and getattr(self._sg, "add_poisson_noise_mlx", None):
                        spectrum = self._sg.add_poisson_noise_mlx(spectrum, noise)
                    else:
                        spectrum = self._sg.add_poisson_noise(spectrum, noise)

                result[label][angle_idx] = (energy, spectrum)

        return result

    def generate(
        self,
        output_dir: Union[str, Path],
        noise_level: Optional[float] = None,
        project_name: str = "depth_profile",
    ) -> Dict[str, Path]:
        """
        Generate and save spectra to files.

        Output structure:
            output_dir/
            ├── angle_00deg/
            │   ├── C1s_{project}.h5
            │   ├── O1s_{project}.h5
            │   └── ...
            ├── angle_30deg/
            │   └── ...
            └── project.h5 (metadata)

        Args:
            output_dir: Output directory path
            noise_level: Override config noise level
            project_name: Project name for filenames

        Returns:
            Dict of output file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        noise = noise_level if noise_level is not None else self.config.noise_level

        output_files = {}

        # Generate spectra for each angle
        for angle_idx, angle in enumerate(self.angles):
            angle_dir = output_dir / f"angle_{int(angle):02d}deg"
            angle_dir.mkdir(exist_ok=True)

            for elem_idx, elem in enumerate(self.profile.elements):
                energy, spectrum = self.generate_spectrum(elem_idx, angle_idx)

                # Add noise
                if noise > 0 and self.config.noise_type != 'none':
                    if self.config.use_mlx and getattr(self._sg, "HAS_MLX", False) \
                            and getattr(self._sg, "add_poisson_noise_mlx", None):
                        spectrum = self._sg.add_poisson_noise_mlx(spectrum, noise)
                    else:
                        spectrum = self._sg.add_poisson_noise(spectrum, noise)

                # Save to H5
                filename = f"{elem.label}_{project_name}.h5"
                filepath = angle_dir / filename

                self._save_h5(filepath, energy, spectrum, elem, angle)
                output_files[f"{elem.label}_angle{int(angle)}"] = filepath

        # Save project metadata
        project_path = output_dir / f"{project_name}.h5"
        self._save_project_metadata(project_path, noise_level=noise)
        output_files['project'] = project_path

        # Save depth profile
        profile_path = output_dir / f"{project_name}_profile.json"
        self.profile.save_json(profile_path)
        output_files['profile'] = profile_path

        return output_files

    def _save_h5(
        self,
        filepath: Path,
        energy: np.ndarray,
        spectrum: np.ndarray,
        elem: ElementConfig,
        angle: float,
    ) -> None:
        """Save spectrum to H5 file (MATLAB DepthProfiler compatible format)."""
        n_energy = len(energy)

        # Create specdata array (first row is energy axis)
        # Format: [n_pixels+1, n_energy] where row 0 is energy
        specdata = np.zeros((2, n_energy), dtype=np.float64)
        specdata[0, :] = energy
        specdata[1, :] = spectrum

        # Create fitpara array [n_pixels, max_comp, 9]
        # Columns: [Int0, E0, GW, LW, α, BR, SO, ft, Area]
        fitpara = np.zeros((1, self.config.max_components, 9), dtype=np.float64)
        fitpara[0, 0, :] = [
            spectrum.max(),  # Int0
            elem.binding_energy,  # E0
            elem.gaussian_fwhm,  # GW
            elem.lorentzian_fwhm,  # LW
            0,  # α (asymmetry)
            0,  # BR (branching ratio)
            0,  # SO (spin-orbit)
            1,  # ft (function type)
            spectrum.sum() * (energy[1] - energy[0]),  # Area
        ]

        # Create xytdata array [3, n_pixels]
        xytdata = np.zeros((3, 1), dtype=np.float64)
        xytdata[0, 0] = angle  # x = angle
        xytdata[1, 0] = 0      # y = 0
        xytdata[2, 0] = 0      # t = 0

        # Create otherpara array [10, n_pixels]
        otherpara = np.zeros((10, 1), dtype=np.float64)
        otherpara[0, 0] = 1  # numComp
        otherpara[1, 0] = energy.min()  # xMin
        otherpara[2, 0] = energy.max()  # xMax
        otherpara[3, 0] = 1  # bgType (Shirley)
        otherpara[4, 0] = self.config.bg_level  # bgParamA
        otherpara[5, 0] = 0  # bgParamB
        otherpara[6, 0] = 0  # bgParamC
        otherpara[7, 0] = spectrum.max()  # maxH
        otherpara[8, 0] = spectrum.sum()  # totalArea
        otherpara[9, 0] = 0  # χ²

        with h5py.File(filepath, 'w') as f:
            f.create_dataset('specdata', data=specdata)
            f.create_dataset('fitpara', data=fitpara)
            f.create_dataset('xytdata', data=xytdata)
            f.create_dataset('otherpara', data=otherpara)

            # Misc group
            misc = f.create_group('misc')
            misc.create_dataset('xdata_str', data=np.array(['Angle'], dtype='S'))
            misc.create_dataset('ydata_str', data=np.array(['Y'], dtype='S'))
            misc.create_dataset('tdata_str', data=np.array(['T'], dtype='S'))
            misc.attrs['maxcomp'] = self.config.max_components
            misc.attrs['bindingenergysign'] = 1
            misc.attrs['fermienergy'] = 0

    def _save_project_metadata(
        self,
        filepath: Path,
        noise_level: float = 0.0,
    ) -> None:
        """Save project metadata to H5 file."""
        with h5py.File(filepath, 'w') as f:
            # ezdeprof group (MATLAB DepthProfiler compatible)
            ezdeprof = f.create_group('ezdeprof')

            # Color mapping (for visualization)
            n_elements = self.profile.n_elements
            colors = np.array([
                [1, 0, 0],   # Red
                [0, 1, 0],   # Green
                [0, 0, 1],   # Blue
                [1, 1, 0],   # Yellow
                [1, 0, 1],   # Magenta
                [0, 1, 1],   # Cyan
                [1, 0.5, 0], # Orange
                [0.5, 0, 1], # Purple
            ])[:n_elements].T  # [3, n_elements]
            ezdeprof.create_dataset('Col', data=colors)

            # Element names
            elements = [e.symbol for e in self.profile.elements]
            ezdeprof.create_dataset('element',
                                    data=np.array(elements, dtype='S'))

            # Element symbols
            ezdeprof.create_dataset('elementsymbol',
                                    data=np.array(elements, dtype='S'))

            # Orbitals
            orbitals = [e.orbital for e in self.profile.elements]
            ezdeprof.create_dataset('orbital',
                                    data=np.array(orbitals, dtype='S'))

            # Additional metadata
            f.attrs['photon_energy'] = self.config.photon_energy
            f.attrs['work_function'] = self.config.work_function
            f.attrs['noise_level'] = noise_level
            f.attrs['angles'] = self.angles
            f.attrs['n_angles'] = len(self.angles)
            f.attrs['n_elements'] = n_elements

            # Depth profile summary
            f.create_dataset('depth', data=self.profile.depth)
            f.create_dataset('concentrations', data=self.profile.concentrations)
            f.create_dataset('intensities', data=self.intensities)


def generate_from_profile(
    profile: Union[DepthProfile, str, Path],
    angles: Union[List[float], np.ndarray],
    output_dir: Union[str, Path],
    noise_level: float = 0.0,
    photon_energy: float = 1486.6,
    project_name: str = "depth_profile",
) -> Dict[str, Path]:
    """
    Convenience function to generate spectra from a depth profile.

    Args:
        profile: DepthProfile object or path to profile file
        angles: Observation angles in degrees
        output_dir: Output directory
        noise_level: Noise level (0 = no noise)
        photon_energy: X-ray photon energy (eV)
        project_name: Project name for filenames

    Returns:
        Dict of output file paths
    """
    # Load profile if path
    if isinstance(profile, (str, Path)):
        profile = DepthProfile.load(profile)

    config = DepthGeneratorConfig(
        photon_energy=photon_energy,
        noise_level=noise_level,
    )

    generator = DepthSpectraGenerator(profile, angles, config)
    return generator.generate(output_dir, project_name=project_name)


if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    print("Depth Profile to Spectra Generator Test")
    print("=" * 50)

    # Create test profile
    profile = create_five_layer_profile()
    print(f"Profile: {profile.element_labels}")

    # Define angles
    angles = np.array([0, 30, 45, 60, 75])
    print(f"Angles: {angles}")

    # Create generator
    config = DepthGeneratorConfig(noise_level=100)
    generator = DepthSpectraGenerator(profile, angles, config)

    print(f"\nAngle-resolved intensities:")
    print(f"{'Angle':>6} | " + " | ".join(f"{l:>6}" for l in profile.element_labels))
    print("-" * (8 + 9 * len(profile.element_labels)))
    for i, theta in enumerate(angles):
        vals = " | ".join(f"{v:>6.3f}" for v in generator.intensities[i])
        print(f"{theta:>6.0f} | {vals}")

    # Generate spectra
    output_dir = Path('/tmp/depth_spectra_test')
    print(f"\nGenerating spectra to {output_dir}...")

    files = generator.generate(
        output_dir,
        noise_level=100,
        project_name="five_layer_test",
    )

    print(f"Generated {len(files)} files:")
    for key, path in list(files.items())[:5]:
        print(f"  {key}: {path.name}")
    if len(files) > 5:
        print(f"  ... and {len(files) - 5} more")

    # Plot example spectra
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Top row: Depth profile and angle intensities
    ax1 = axes[0, 0]
    for i, label in enumerate(profile.element_labels):
        ax1.plot(profile.depth, profile.concentrations[:, i], label=label, linewidth=2)
    ax1.set_xlabel('Depth (Å)')
    ax1.set_ylabel('Concentration')
    ax1.set_title('Depth Profile')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2 = axes[0, 1]
    for i, label in enumerate(profile.element_labels):
        ax2.plot(angles, generator.intensities[:, i], 'o-', label=label, linewidth=2)
    ax2.set_xlabel('Angle (degrees)')
    ax2.set_ylabel('Normalized Intensity')
    ax2.set_title('Angle-Resolved Intensities')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Plot spectra at different angles for Si2p
    ax3 = axes[0, 2]
    for angle_idx, angle in enumerate([0, 45, 75]):
        idx = list(angles).index(angle)
        energy, spectrum = generator.generate_spectrum(4, idx)  # Si2p
        ax3.plot(energy, spectrum, label=f'{angle}°', linewidth=1.5)
    ax3.set_xlabel('Binding Energy (eV)')
    ax3.set_ylabel('Intensity')
    ax3.set_title('Si 2p at Different Angles')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.invert_xaxis()

    # Bottom row: All elements at 0°, 45°, 75°
    for plot_idx, angle in enumerate([0, 45, 75]):
        ax = axes[1, plot_idx]
        angle_idx = list(angles).index(angle)

        for elem_idx, elem in enumerate(profile.elements):
            energy, spectrum = generator.generate_spectrum(elem_idx, angle_idx)
            # Add noise for visualization
            spectrum_noisy = _voigtfit().add_poisson_noise(spectrum.copy(), 100)
            ax.plot(energy, spectrum_noisy, label=elem.label, linewidth=1)

        ax.set_xlabel('Binding Energy (eV)')
        ax.set_ylabel('Intensity')
        ax.set_title(f'All Elements at {angle}°')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/tmp/depth_generator_test.png', dpi=150)
    print(f"\nSaved plot to /tmp/depth_generator_test.png")
