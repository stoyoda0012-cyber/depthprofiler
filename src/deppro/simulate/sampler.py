"""
Random Profile Sampler
======================

Generates random but physically valid depth profiles for NN training
and benchmarking. All profiles follow the MATLAB CalcDistrib forward
model (erf-only interfaces) to ensure they are exactly representable
by the SemiFreeSolver.

Usage:
    from deppro.simulate.sampler import RandomProfileSampler
    from deppro.simulate.presets import get_preset
    import numpy as np

    preset = get_preset('layered_6elem')
    depth = np.arange(0, 105, 5.0)
    sampler = RandomProfileSampler(preset, depth=depth, seed=42)

    profile = sampler.sample()
    print(profile.D.shape)          # (21, 6)
    print(profile.thickness)        # e.g. 87.3
    print(profile.D.sum(axis=1))    # all 1.0
"""

import numpy as np
from dataclasses import dataclass
from math import (
    erf as _math_erf, exp as _math_exp,
    sqrt as _math_sqrt, cos as _math_cos,
)
from typing import List, Optional, Tuple

from .presets import ElementSystemPreset
from .configs import SimulationConfig

try:
    from numba import jit
    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator


# =========================================================================
# Numba-accelerated kernels
# =========================================================================

_LN2 = 0.6931471805599453  # ln(2)


@jit(nopython=True, cache=True, fastmath=True)
def _build_profile_calcdistrib(
    depth: np.ndarray,
    n_elem: int,
    min_conc: float,
    sub_indices: np.ndarray,
    sub_ratios: np.ndarray,
    sub_thickness: float,
    sub_abruptness: float,
    layer_params: np.ndarray,
    n_layers: int,
    free_indices: np.ndarray,
    n_free: int,
    contam_params: np.ndarray,
    n_contam_params: int,
    linkedconst: np.ndarray,
    linked_indices: np.ndarray,
) -> np.ndarray:
    """Build depth profile D(n_depth, n_elem) following MATLAB CalcDistrib.

    Numba kernel: builds x_free from erf layer params, then applies
    CalcDistrib model (D_subst, D_free, D_linked, normalize).

    Parameters
    ----------
    depth : (n_depth,) float64
    n_elem : int
    min_conc : float
    sub_indices : (n_sub,) int64
    sub_ratios : (n_sub,) float64
    sub_thickness : float — outermost interface position
    sub_abruptness : float — outermost interface FWHM
    layer_params : (n_layers, 6) float64
        Each row: (local_free_idx, conc, t_layer, sigma, t_prev, sigma_prev).
        t_prev < 0 means first layer (no lower erf boundary).
    n_layers : int
    free_indices : (n_free,) int64
    n_free : int
    contam_params : (n_contam, 3) float64
        Each row: (global_idx, amplitude, decay_length).
    n_contam_params : int
    linkedconst : (n_free,) float64
    linked_indices : (n_linked,) int64

    Returns
    -------
    D : (n_depth, n_elem) float64
    """
    n_depth = len(depth)
    D = np.full((n_depth, n_elem), min_conc)
    sigma_sub = sub_abruptness / (2.0 * _math_sqrt(_LN2))

    # Build x_free from erf layer boundaries
    x_free = np.zeros((n_depth, n_free))
    for li in range(n_layers):
        local_idx = int(layer_params[li, 0])
        conc = layer_params[li, 1]
        t_layer = layer_params[li, 2]
        sigma_layer = layer_params[li, 3]
        t_prev = layer_params[li, 4]
        sigma_prev = layer_params[li, 5]
        for d in range(n_depth):
            erf_upper = 0.5 * abs(1.0 + _math_erf(-(depth[d] - t_layer) / sigma_layer))
            if t_prev >= 0.0:
                erf_lower = 0.5 * abs(1.0 + _math_erf((depth[d] - t_prev) / sigma_prev))
            else:
                erf_lower = 1.0
            x_free[d, local_idx] += conc * erf_upper * erf_lower

    # Clamp x_free
    for d in range(n_depth):
        total = 0.0
        for fi in range(n_free):
            total += x_free[d, fi]
        if total > 1.0:
            for fi in range(n_free):
                x_free[d, fi] /= total

    # Substrate: D_subst = sub_ratio * 0.5*(1 + erf(rising))
    for d in range(n_depth):
        erf_val = 0.5 * abs(1.0 + _math_erf((depth[d] - sub_thickness) / sigma_sub))
        for si in range(len(sub_indices)):
            D[d, sub_indices[si]] = sub_ratios[si] * erf_val

    # Free elements: D_free = x_free * (1 - sum(D_subst))
    for d in range(n_depth):
        sub_total = 0.0
        for si in range(len(sub_indices)):
            sub_total += D[d, sub_indices[si]]
        film_fraction = 1.0 - sub_total
        if film_fraction < 0.0:
            film_fraction = 0.0
        for fi in range(n_free):
            D[d, free_indices[fi]] = x_free[d, fi] * film_fraction

    # Surface contamination (additive)
    for ci in range(n_contam_params):
        global_idx = int(contam_params[ci, 0])
        amp = contam_params[ci, 1]
        decay = contam_params[ci, 2]
        for d in range(n_depth):
            D[d, global_idx] += amp * _math_exp(-depth[d] / decay)

    # Linked element (O) from stoichiometry
    for li_idx in range(len(linked_indices)):
        linked_idx = linked_indices[li_idx]
        for d in range(n_depth):
            D[d, linked_idx] = 0.0
        for fi in range(n_free):
            coeff = linkedconst[fi] - 1.0
            if coeff > 1e-10:
                free_global = free_indices[fi]
                for d in range(n_depth):
                    D[d, linked_idx] += D[d, free_global] * coeff

    # Normalize to sum=1
    for d in range(n_depth):
        total = 0.0
        for e in range(n_elem):
            if D[d, e] < min_conc:
                D[d, e] = min_conc
            total += D[d, e]
        if total > 1e-10:
            for e in range(n_elem):
                D[d, e] /= total

    return D


@jit(nopython=True, cache=True, fastmath=True)
def _generate_one_fused(
    depth: np.ndarray,
    n_elem: int,
    min_conc: float,
    sub_indices: np.ndarray,
    sub_ratios: np.ndarray,
    sub_thickness: float,
    sub_abruptness: float,
    layer_params: np.ndarray,
    n_layers: int,
    free_indices: np.ndarray,
    n_free: int,
    contam_params: np.ndarray,
    n_contam_params: int,
    linkedconst: np.ndarray,
    linked_indices: np.ndarray,
    # Forward model params
    lambda_basic: np.ndarray,   # (n_elem, n_elem)
    gamma_eff: float,
    angles_rad: np.ndarray,     # (n_angles,)
    step_size: float,
) -> Tuple:
    """Fused kernel: build D + compute lambda_obs + trans_mat + forward.

    Returns (D, I_obs) where D is (n_depth, n_elem) and I_obs is
    (n_angles, n_elem). Both float64.
    """
    n_depth = len(depth)
    n_angles = len(angles_rad)

    # Build profile
    D = _build_profile_calcdistrib(
        depth, n_elem, min_conc,
        sub_indices, sub_ratios, sub_thickness, sub_abruptness,
        layer_params, n_layers,
        free_indices, n_free,
        contam_params, n_contam_params,
        linkedconst, linked_indices,
    )

    # lambda_obs
    lambda_obs = np.empty((n_depth, n_elem))
    for e in range(n_elem):
        for d in range(n_depth):
            s = 0.0
            for i in range(n_elem):
                s += D[d, i] / lambda_basic[e, i]
            if s == 0.0:
                s = 1e-10
            lambda_obs[d, e] = gamma_eff / s

    # trans_mat + forward (fused to avoid allocating trans_mat)
    I_obs = np.zeros((n_angles, n_elem))
    for a in range(n_angles):
        cos_theta = _math_cos(angles_rad[a])
        for e in range(n_elem):
            cumsum_val = 0.0
            s = 0.0
            for d in range(n_depth):
                t1 = -step_size / lambda_obs[d, e] / cos_theta
                tm_val = _math_exp(cumsum_val)
                s += tm_val * D[d, e]
                cumsum_val += t1
            I_obs[a, e] = s

    # Normalize I_obs
    for a in range(n_angles):
        total = 0.0
        for e in range(n_elem):
            total += I_obs[a, e]
        if total > 0.0:
            for e in range(n_elem):
                I_obs[a, e] /= total

    return D, I_obs


# =========================================================================
# Data classes
# =========================================================================

@dataclass
class SampledProfile:
    """A single sampled ground truth profile with metadata.

    Attributes
    ----------
    D : ndarray, shape (n_depth, n_elements)
        Concentration profile. Normalized to sum=1 at each depth.
    depth : ndarray, shape (n_depth,)
        Depth array in Angstroms.
    thickness : float
        Primary (outermost) interface position in Angstroms.
    abruptness : float
        Primary interface width FWHM in Angstroms.
    n_layers : int
        Number of overlay layers.
    layer_thicknesses : ndarray
        Per-layer interface positions (sorted).
    layer_abruptnesses : ndarray
        Per-layer interface widths.
    """

    D: np.ndarray
    depth: np.ndarray
    thickness: float
    abruptness: float
    n_layers: int
    layer_thicknesses: np.ndarray
    layer_abruptnesses: np.ndarray


# =========================================================================
# RandomProfileSampler
# =========================================================================

class RandomProfileSampler:
    """Generate random but physically valid depth profiles.

    All profiles follow the MATLAB CalcDistrib model with erf-only
    interfaces, ensuring they are exactly representable by the
    SemiFreeSolver.

    Strategy:
        1. Sample the number of overlay layers (1 to n_layers_max)
        2. For each layer, sample thickness/abruptness parameters
        3. Build substrate with erf function at the outermost interface
        4. Assign oxide cation elements (Al, Ti, Si(+4)) to layers
        5. Add optional surface contamination (C, B) as exponential decay
        6. Compute linked element (O) from linkedconst stoichiometry
        7. Normalize to sum=1 at each depth

    Parameters
    ----------
    preset : ElementSystemPreset
        Element system configuration.
    depth : ndarray
        Depth array in Angstroms.
    seed : int or None
        Random seed for reproducibility.
    n_layers_range : (int, int)
        Min/max number of overlay layers.
    thickness_range : (float, float)
        Min/max interface position in Angstroms.
    abruptness_range : (float, float)
        Min/max interface FWHM in Angstroms.
    max_abruptness_ratio : float
        Cumpson limit: FWHM <= layer_spacing * ratio.
    min_conc : float
        Minimum concentration floor.
    surface_contamination_prob : float
        Probability of adding surface C/B contamination.
    contamination_decay_range : (float, float)
        Min/max decay length for surface contamination.
    """

    def __init__(
        self,
        preset: ElementSystemPreset,
        depth: np.ndarray,
        seed: Optional[int] = None,
        n_layers_range: Tuple[int, int] = (1, 4),
        thickness_range: Tuple[float, float] = (10, 500),
        abruptness_range: Tuple[float, float] = (3, 15),
        max_abruptness_ratio: float = 0.3,
        min_conc: float = 1e-6,
        surface_contamination_prob: float = 0.7,
        contamination_decay_range: Tuple[float, float] = (2, 10),
    ):
        self.preset = preset
        self.depth = np.asarray(depth, dtype=np.float64)
        self.rng = np.random.default_rng(seed)

        self.n_layers_range = n_layers_range
        self.thickness_range = thickness_range
        self.abruptness_range = abruptness_range
        self.max_abruptness_ratio = max_abruptness_ratio
        self.min_conc = min_conc
        self.surface_contamination_prob = surface_contamination_prob
        self.contamination_decay_range = contamination_decay_range

        # Classify free elements into cations (contribute to O) vs inert (C, B)
        self._cation_local_idx: List[int] = []  # index into free_indices
        self._inert_local_idx: List[int] = []   # index into free_indices
        for i, flc in enumerate(preset.linkedconst):
            if flc > 1.0 + 1e-6:
                self._cation_local_idx.append(i)
            else:
                self._inert_local_idx.append(i)

        # Pre-compute constant arrays (reused across sample/generate_batch)
        self._sub_indices = np.array(preset.substrate_indices, dtype=np.int64)
        self._sub_ratios = np.ones(len(self._sub_indices), dtype=np.float64)
        self._free_indices = np.array(preset.free_indices, dtype=np.int64)
        self._linkedconst_arr = np.array(preset.linkedconst, dtype=np.float64)
        self._linked_indices = np.array(preset.linked_indices, dtype=np.int64)
        self._n_free = len(preset.free_indices)
        self._n_elem = preset.n_elements
        self._sqrt_ln2 = np.sqrt(np.log(2.0))

        # Inert element global indices for contamination
        self._inert_global_indices: List[int] = []
        for inert_i in self._inert_local_idx:
            global_idx = preset.free_indices[inert_i]
            elem = preset.elements[global_idx]
            if elem.symbol in ('C', 'B'):
                self._inert_global_indices.append(global_idx)

    @classmethod
    def from_config(
        cls,
        preset: ElementSystemPreset,
        config: SimulationConfig,
    ) -> 'RandomProfileSampler':
        """Create sampler from SimulationConfig."""
        depth = np.arange(0, config.depth_end + config.step_size, config.step_size)
        return cls(
            preset=preset,
            depth=depth,
            seed=config.seed,
            n_layers_range=config.n_layers_range,
            thickness_range=config.thickness_range,
            abruptness_range=config.abruptness_range,
            max_abruptness_ratio=config.max_abruptness_ratio,
            min_conc=config.min_conc,
            surface_contamination_prob=config.surface_contamination_prob,
            contamination_decay_range=config.contamination_decay_range,
        )

    def _sample_params(self) -> Tuple[int, np.ndarray, np.ndarray,
                                       np.ndarray, int, np.ndarray, int]:
        """Sample random layer/contamination parameters.

        Returns
        -------
        n_layers : int
        layer_thicknesses : (n_layers,) float64
        layer_abruptnesses : (n_layers,) float64
        layer_params : (n_layers, 6) float64
        n_layer_params : int
        contam_params : (n_contam, 3) float64
        n_contam : int
        """
        rng = self.rng

        # Number of layers
        n_layers = int(rng.integers(
            self.n_layers_range[0], self.n_layers_range[1] + 1))

        # Interface positions (sorted)
        max_thickness = min(self.thickness_range[1], self.depth[-1] * 0.8)
        min_thickness = min(self.thickness_range[0], max_thickness * 0.5)
        layer_thicknesses = np.sort(rng.uniform(
            min_thickness, max_thickness, size=n_layers))

        # Abruptness clamped by Cumpson limit
        layer_abruptnesses = np.empty(n_layers)
        for li in range(n_layers):
            spacing = (layer_thicknesses[0] if li == 0
                       else layer_thicknesses[li] - layer_thicknesses[li - 1])
            abr_max = min(self.abruptness_range[1],
                          spacing * self.max_abruptness_ratio)
            abr_min = min(self.abruptness_range[0], abr_max * 0.9)
            layer_abruptnesses[li] = float(rng.uniform(
                abr_min, max(abr_min, abr_max)))

        # Layer params for Numba
        n_cations = len(self._cation_local_idx)
        if n_cations > 0 and n_layers > 0:
            layer_params = np.empty((n_layers, 6), dtype=np.float64)
            for li in range(n_layers):
                local_idx = self._cation_local_idx[li % n_cations]
                sigma = layer_abruptnesses[li] / (2.0 * self._sqrt_ln2)
                layer_params[li, 0] = local_idx
                layer_params[li, 1] = float(rng.uniform(0.15, 0.6))
                layer_params[li, 2] = layer_thicknesses[li]
                layer_params[li, 3] = sigma
                if li > 0:
                    layer_params[li, 4] = layer_thicknesses[li - 1]
                    layer_params[li, 5] = layer_abruptnesses[li - 1] / (2.0 * self._sqrt_ln2)
                else:
                    layer_params[li, 4] = -1.0
                    layer_params[li, 5] = 1.0
            n_lp = n_layers
        else:
            layer_params = np.empty((0, 6), dtype=np.float64)
            n_lp = 0

        # Contamination params
        contam_list = []
        for global_idx in self._inert_global_indices:
            if rng.random() < self.surface_contamination_prob:
                decay = float(rng.uniform(*self.contamination_decay_range))
                amp = float(rng.uniform(0.03, 0.25))
                contam_list.append([global_idx, amp, decay])
        contam_params = (np.array(contam_list, dtype=np.float64)
                         if contam_list
                         else np.empty((0, 3), dtype=np.float64))

        return (n_layers, layer_thicknesses, layer_abruptnesses,
                layer_params, n_lp, contam_params, len(contam_list))

    def sample(self) -> SampledProfile:
        """Generate one random depth profile following MATLAB CalcDistrib model.

        Returns
        -------
        SampledProfile
            A physically valid depth profile with all constraints satisfied.
        """
        (n_layers, layer_thicknesses, layer_abruptnesses,
         layer_params, n_lp, contam_params, n_contam) = self._sample_params()

        sub_thickness = float(layer_thicknesses[-1])
        sub_abruptness = float(layer_abruptnesses[-1])

        D = _build_profile_calcdistrib(
            self.depth, self._n_elem, self.min_conc,
            self._sub_indices, self._sub_ratios,
            sub_thickness, sub_abruptness,
            layer_params, n_lp,
            self._free_indices, self._n_free,
            contam_params, n_contam,
            self._linkedconst_arr, self._linked_indices,
        )

        return SampledProfile(
            D=D.astype(np.float32),
            depth=self.depth.copy(),
            thickness=sub_thickness,
            abruptness=sub_abruptness,
            n_layers=n_layers,
            layer_thicknesses=layer_thicknesses,
            layer_abruptnesses=layer_abruptnesses,
        )

    def sample_batch(self, n: int) -> List[SampledProfile]:
        """Generate n random profiles.

        Parameters
        ----------
        n : int
            Number of profiles to generate.

        Returns
        -------
        list of SampledProfile
        """
        return [self.sample() for _ in range(n)]

    def generate_batch(
        self,
        n: int,
        lambda_basic: np.ndarray,
        angles: np.ndarray,
        step_size: float,
        gamma_eff: float = 1.0,
        noise_level: float = 0.0,
        noise_type: str = 'relative',
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate a batch of D + I_obs using fused Numba kernel.

        This is the high-performance path for training data generation.
        Pre-generates all random numbers in vectorized NumPy, then calls
        the fused kernel (_generate_one_fused) for each sample with minimal
        Python overhead.

        Parameters
        ----------
        n : int
            Number of samples to generate.
        lambda_basic : ndarray, shape (n_elem, n_elem)
            Pre-computed IMFP matrix (from TransferMatrix.calculate_lambda_basic).
        angles : ndarray
            Observation angles in degrees.
        step_size : float
            Depth step size in Angstroms.
        gamma_eff : float
            Effective gamma factor (default: 1.0).
        noise_level : float
            Noise level (0 = no noise).
        noise_type : str
            'relative' or 'poisson'.

        Returns
        -------
        D_batch : (n, n_depth, n_elem) float32
        I_batch : (n, n_angles, n_elem) float32
        thickness : (n,) float32
        abruptness : (n,) float32
        n_layers_arr : (n,) int8
        """
        n_depth = len(self.depth)
        n_elem = self._n_elem
        n_free = self._n_free
        n_angles = len(angles)
        n_cations = len(self._cation_local_idx)
        max_layers = self.n_layers_range[1]
        n_inert = len(self._inert_global_indices)

        # Pre-allocate output
        D_batch = np.empty((n, n_depth, n_elem), dtype=np.float32)
        I_batch = np.empty((n, n_angles, n_elem), dtype=np.float32)
        thickness_arr = np.empty(n, dtype=np.float32)
        abruptness_arr = np.empty(n, dtype=np.float32)
        n_layers_arr = np.empty(n, dtype=np.int8)

        # Pre-compute constant arrays
        lb_c = np.ascontiguousarray(lambda_basic, dtype=np.float64)
        # The compiled kernel divides by these IMFPs. NaN (an orbital whose
        # binding energy exceeds the photon energy) or a non-positive value
        # does not propagate as NaN there but ends in a bare ZeroDivisionError
        # with no hint of the cause, so refuse the matrix here by name.
        if lb_c.shape != (n_elem, n_elem):
            raise ValueError(
                f"lambda_basic must have shape ({n_elem}, {n_elem}); got {lb_c.shape}"
            )
        bad = ~(np.isfinite(lb_c) & (lb_c > 0.0))
        if bad.any():
            elems = self.preset.elements
            rows = sorted({int(i) for i in np.nonzero(bad)[0]})
            names = ", ".join(f"{elems[i].symbol} {elems[i].orbital}" for i in rows)
            raise ValueError(
                f"lambda_basic has non-finite or non-positive IMFPs for the emitting "
                f"element(s) {names} ({int(bad.sum())} of {bad.size} entries). This is "
                f"what TransferMatrix.calculate_lambda_basic returns when the photon "
                f"energy is below an orbital's binding energy; use a photon energy of "
                f"at least preset.min_photon_energy "
                f"({self.preset.min_photon_energy:g} eV for {self.preset.name!r})."
            )
        angles_rad = np.deg2rad(np.asarray(angles, dtype=np.float64))
        _sqrt_ln2 = self._sqrt_ln2

        # Pre-generate random numbers for the entire batch
        rng = self.rng
        max_thickness = min(self.thickness_range[1], self.depth[-1] * 0.8)
        min_thickness = min(self.thickness_range[0], max_thickness * 0.5)

        layer_counts = rng.integers(
            self.n_layers_range[0], self.n_layers_range[1] + 1, size=n,
        ).astype(np.int32)
        thickness_raw = rng.uniform(min_thickness, max_thickness,
                                    size=(n, max_layers))
        abruptness_raw = rng.uniform(0.0, 1.0, size=(n, max_layers))
        cation_concs = rng.uniform(0.15, 0.6, size=(n, max_layers))

        contam_coins = rng.random(size=(n, max(n_inert, 1)))
        contam_decays = rng.uniform(
            self.contamination_decay_range[0],
            self.contamination_decay_range[1],
            size=(n, max(n_inert, 1)),
        )
        contam_amps = rng.uniform(0.03, 0.25, size=(n, max(n_inert, 1)))

        # Noise
        if noise_level > 0:
            noise_rng = np.random.default_rng(rng.integers(0, 2**31))
            noise_batch = noise_rng.standard_normal(
                (n, n_angles, n_elem), dtype=np.float64)
        else:
            noise_batch = None

        # Reusable empty arrays
        empty_lp = np.empty((0, 6), dtype=np.float64)
        empty_cp = np.empty((0, 3), dtype=np.float64)

        # Main loop
        for i in range(n):
            nl = int(layer_counts[i])

            layer_thicknesses = np.sort(thickness_raw[i, :nl])

            # Cumpson-constrained abruptness
            layer_abruptnesses = np.empty(nl, dtype=np.float64)
            for li in range(nl):
                spacing = (layer_thicknesses[0] if li == 0
                           else layer_thicknesses[li] - layer_thicknesses[li - 1])
                abr_max = min(self.abruptness_range[1],
                              spacing * self.max_abruptness_ratio)
                abr_min = min(self.abruptness_range[0], abr_max * 0.9)
                layer_abruptnesses[li] = abr_min + abruptness_raw[i, li] * max(0, abr_max - abr_min)

            sub_thickness = layer_thicknesses[-1]
            sub_abruptness = layer_abruptnesses[-1]

            # Build layer_params
            if n_cations > 0 and nl > 0:
                layer_params = np.empty((nl, 6), dtype=np.float64)
                for li in range(nl):
                    local_idx = self._cation_local_idx[li % n_cations]
                    sigma = layer_abruptnesses[li] / (2.0 * _sqrt_ln2)
                    layer_params[li, 0] = local_idx
                    layer_params[li, 1] = cation_concs[i, li]
                    layer_params[li, 2] = layer_thicknesses[li]
                    layer_params[li, 3] = sigma
                    if li > 0:
                        layer_params[li, 4] = layer_thicknesses[li - 1]
                        layer_params[li, 5] = layer_abruptnesses[li - 1] / (2.0 * _sqrt_ln2)
                    else:
                        layer_params[li, 4] = -1.0
                        layer_params[li, 5] = 1.0
            else:
                layer_params = empty_lp
                nl = 0

            # Build contamination params
            contam_list_n = 0
            contam_buf = np.empty((n_inert, 3), dtype=np.float64) if n_inert > 0 else empty_cp
            for ci in range(n_inert):
                if contam_coins[i, ci] < self.surface_contamination_prob:
                    contam_buf[contam_list_n, 0] = self._inert_global_indices[ci]
                    contam_buf[contam_list_n, 1] = contam_amps[i, ci]
                    contam_buf[contam_list_n, 2] = contam_decays[i, ci]
                    contam_list_n += 1
            contam_params = contam_buf[:contam_list_n] if contam_list_n > 0 else empty_cp

            # Fused kernel: D + forward model
            D, I_obs = _generate_one_fused(
                self.depth, n_elem, self.min_conc,
                self._sub_indices, self._sub_ratios,
                sub_thickness, sub_abruptness,
                layer_params, nl,
                self._free_indices, n_free,
                contam_params, contam_list_n,
                self._linkedconst_arr, self._linked_indices,
                lb_c, gamma_eff, angles_rad, step_size,
            )

            # Noise injection
            if noise_level > 0 and noise_batch is not None:
                if noise_type == 'relative':
                    I_obs = I_obs * (1.0 + noise_level * noise_batch[i])
                elif noise_type == 'poisson':
                    I_obs = I_obs + noise_level * noise_batch[i] * np.sqrt(np.abs(I_obs))
                total = I_obs.sum(axis=1, keepdims=True)
                total[total == 0] = 1
                I_obs = np.clip(I_obs / total, 0, None)

            D_batch[i] = D.astype(np.float32)
            I_batch[i] = I_obs.astype(np.float32)
            thickness_arr[i] = sub_thickness
            abruptness_arr[i] = sub_abruptness
            n_layers_arr[i] = nl

        return D_batch, I_batch, thickness_arr, abruptness_arr, n_layers_arr
