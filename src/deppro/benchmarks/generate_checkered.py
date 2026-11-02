#!/usr/bin/env python
"""
Checkered Ground Truth Generator
=================================

Generates Dstk (ground truth depth profiles) and Xstk (angle-resolved
intensities) for a Checkered benchmark, replicating the MATLAB
DistribTopLayerCheckered + CalculateDistribution pipeline in pure Python.

Layer model: Error function interfaces
  - Substrate Si(0): erf rising from 0 at surface to 1 at bulk
  - Free surface elements: erf falling from surface to zero
  - Linked O(-2): charge-balanced from cation concentrations
  - Checkered pattern: alternates which surface elements are present

Output H5 format is compatible with benchmark_dstk.py / load_dstk_data().

Usage:
    python -m deppro.benchmarks.generate_checkered
    python -m deppro.benchmarks.generate_checkered --nx 10 --ny 10 --output /path/to/output.h5
    python -m deppro.benchmarks.generate_checkered --no-checkered  # uniform profiles

Date: 2026-02-20
"""

import numpy as np
from .._compat import trapezoid
import h5py
import argparse
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from deppro import TransferMatrix, ElementConfig, CompoundDatabase


# ==============================================================================
# Configuration
# ==============================================================================

@dataclass
class CheckeredConfig:
    """Configuration for Checkered ground truth generation."""
    # Spatial grid
    nx: int = 5          # X pixels
    ny: int = 5          # Y pixels

    # Depth grid
    n_depth: int = 20
    step_size: float = 30.0  # Å
    total_depth: float = 570.0  # Å (= (n_depth-1) * step_size)

    # Observation angles
    n_angles: int = 14
    angle_min: float = 0.0    # degrees
    angle_max: float = 60.0   # degrees

    # Physical parameters
    photon_energy: float = 8000.0  # eV (HAXPES)
    work_function: float = 5.0     # eV
    gamma_eff: float = 0.80

    # Layer structure: FeatureVector_Source
    # [thck, abrp, pos1, width1, height1, pos2, w2, h2, ...]
    thickness: float = 210.0    # Å — substrate interface position
    abruptness: float = 20.0    # Å FWHM — interface width

    # Free elements (Error fcn profiles)
    # Each: (position_Å, width_FWHM_Å, height_%)
    # MATLAB order: Si(+4), Ti(+4), Al(+3), C(0), B(0)
    free_profiles: list = field(default_factory=lambda: [
        {"pos": 210.0, "width": 30.0, "height": 33.333},   # Si(+4)
        {"pos": 150.0, "width": 80.0, "height": 33.333},   # Ti(+4)
        {"pos":  80.0, "width": 40.0, "height": 40.0},      # Al(+3)
        {"pos":   0.0, "width": 70.0, "height": 100.0},     # C(0)
        {"pos":   0.0, "width": 70.0, "height": 100.0},     # B(0)
    ])

    # Element definitions
    # Order: Si(0), Si(+4), Ti(+4), Al(+3), C(0), B(0), O(-2)
    # 7-element OBCAlTiSi system (Checkered standard)
    elements: list = field(default_factory=lambda: [
        {"symbol": "Si", "orbital": "1s", "compound": "Si",
         "BE": 1839.1, "charge": 0, "density": 2.329, "Aw": 28.086, "Nv": 4, "Eg": 1.12},
        {"symbol": "Si", "orbital": "1s", "compound": "SiO2",
         "BE": 1843.8, "charge": 4, "density": 2.2, "Aw": 60.08, "Nv": 16, "Eg": 8.9},
        {"symbol": "Ti", "orbital": "1s", "compound": "TiO2",
         "BE": 4968.7, "charge": 4, "density": 4.23, "Aw": 79.866, "Nv": 16, "Eg": 3.1},
        {"symbol": "Al", "orbital": "1s", "compound": "Al2O3",
         "BE": 1561.8, "charge": 3, "density": 3.95, "Aw": 101.96, "Nv": 24, "Eg": 7.6},
        {"symbol": "C",  "orbital": "1s", "compound": "C",
         "BE": 284.4,  "charge": 0, "density": 2.26, "Aw": 12.011, "Nv": 4, "Eg": 0.0},
        {"symbol": "B",  "orbital": "1s", "compound": "B",
         "BE": 188.0,  "charge": 0, "density": 2.08, "Aw": 10.812, "Nv": 3, "Eg": 0.0},
        {"symbol": "O",  "orbital": "1s", "compound": "Oxide",
         "BE": 531.0,  "charge": -2, "density": 2.2, "Aw": 16.0, "Nv": 6, "Eg": 6.0},
    ])

    # Substrate / Free / Linked indices (0-based)
    subst_idx: list = field(default_factory=lambda: [0])         # Si(0)
    free_idx: list = field(default_factory=lambda: [1,2,3,4,5])  # Si(+4), Ti(+4), Al(+3), C(0), B(0)
    linked_idx: list = field(default_factory=lambda: [6])        # O(-2)

    # Linked element stoichiometry constants
    # O(-2) balances: Si(+4)*3 + Ti(+4)*3 + Al(+3)*2.5 + C(0)*1 + B(0)*1
    linked_const: list = field(default_factory=lambda: [3.0, 3.0, 2.5, 1.0, 1.0])

    # Linked element profiles (from SupportVector_Set, substrate columns removed)
    linked_profiles: list = field(default_factory=lambda: [
        {"pos": 98.025, "width": 197.279, "height": 28.954},  # O(-2)
    ])

    # Checkered pattern: which two free element heights to swap
    # Indices into free_profiles — last two free elements
    checkered_idx1: int = 3   # C(0)  = free_profiles index 3
    checkered_idx2: int = 4   # B(0)  = free_profiles index 4

    # Noise
    noise_level: float = 0.0  # 0 = no noise


# ==============================================================================
# Depth profile generation (CalculateDistribution equivalent)
# ==============================================================================


# 場面生成の共有カーネルは配布対象の deppro.simulate.scene_kernel へ移した。
# 以前ここに定義があり、deppro/simulate/d2s2d.py が module 直下で import していたが、
# pyproject が deppro.benchmarks* を配布から除外しているため、インストールした
# wheel では deppro 全体が import できなかった。依存の向きを benchmarks -> simulate
# に直し、ここからは再 export する（既存の import 経路はそのまま動く）。
from deppro.simulate.scene_kernel import (  # noqa: F401
    CHECKERED_7ELEM,
    PRESET_6ELEM_STANDARD,
    PRESET_7ELEM_CHECKERED,
    STANDARD_6ELEM,
    SimulationConfig,
    _etchgrow_fvid1,
    _intdifsep_indices,
    _phenom_phase_progress,
    calculate_distribution,
    calculate_distribution_gaussian,
    distrib_bottom_layer_exponential,
    distrib_bottom_layer_linear,
    distrib_bottom_layer_trigonometric,
    distrib_top_layer_checkered,
    erf_safe,
    fv_to_profiles,
    phenom_etch_grow_exponential,
    phenom_etch_grow_linear,
    phenom_etch_grow_trigonometric,
    phenom_int_dif_sep_exponential,
    phenom_int_dif_sep_linear,
    phenom_int_dif_sep_trigonometric,
    phenom_interf_redox_exponential,
    phenom_interf_redox_linear,
    phenom_interf_redox_trigonometric,
    update_feature_vector,
)






# ==============================================================================
# Checkered pattern
# ==============================================================================

def apply_checkered(
    height_base: list[dict],
    sk: int,
    dx: int,
    dy: int,
    idx1: int,
    idx2: int,
) -> list[dict]:
    """Apply Checkered pattern to free element heights.

    Replicates MATLAB DistribTopLayerCheckered.m

    Args:
        height_base: Base free_profiles (will be copied)
        sk: Stack index (1-based, as in MATLAB)
        dx: X dimension
        dy: Y dimension
        idx1: Index into free_profiles for element 1 (Al)
        idx2: Index into free_profiles for element 2 (C)

    Returns:
        Modified free_profiles
    """
    import copy
    profiles = copy.deepcopy(height_base)

    id1 = not (int(np.ceil(6.0 * sk / dx)) % 2)
    id2 = not (int(np.ceil(6.0 * sk / (dx * dy))) % 2)

    if (id1 and id2) or (not id1 and not id2):
        profiles[idx1]["height"] = 0.0
        profiles[idx2]["height"] = 100.0
    else:
        profiles[idx1]["height"] = 100.0
        profiles[idx2]["height"] = 0.0

    return profiles


# ==============================================================================
# FeatureVector modulation functions (MATLAB SimulationUpdater port)
# ==============================================================================
#
# Each function takes FV (1D array), sk (1-based stack index), and grid params.
# Returns modified FV (copy).  Exact replication of MATLAB .m functions.
#
# FV layout: [thck, abrp, pos1, w1, h1, pos2, w2, h2, ..., posN, wN, hN]
#   Index 0: thickness, 1: abruptness, then 3 values per free element.
#   SizeofFree = (len(FV) - 2) // 3

# --- A. DistribBottomLayer (spatial, 3 variants) ---







# --- B. DistribTopLayerCheckered (spatial, FV-based version) ---



# --- C. PhenomInterfRedox (temporal, 3 variants) ---









# --- D. PhenomEtchGrow (temporal, 3 variants) ---









# --- E. PhenomIntDifSep (temporal, 3 variants) ---









# ==============================================================================
# SimulationUpdater: FeatureVector update orchestrator
# ==============================================================================





# ==============================================================================
# SimulationConfig and generate_simulation
# ==============================================================================











def generate_simulation(
    config: Optional[SimulationConfig] = None,
    verbose: bool = True,
) -> dict:
    """Generate simulation benchmark with spatial + temporal FV modulation.

    Full pipeline: base FV → update_feature_vector → calculate_distribution → forward model.

    Args:
        config: SimulationConfig (default: PRESET_6ELEM_STANDARD)
        verbose: Print progress

    Returns:
        Dictionary with Dstk, Xstk, FV0stk, FVstk, depth, angles, etc.
    """
    if config is None:
        config = PRESET_6ELEM_STANDARD

    n_pixels = config.n_pixels
    depth = np.arange(config.n_depth) * config.step_size
    angles = np.linspace(config.angle_min, config.angle_max, config.n_angles)
    n_elements = len(config.elements)
    n_free = len(config.free_idx)
    charges = np.array([e["charge"] for e in config.elements], dtype=np.float64)

    element_configs = [
        ElementConfig(
            symbol=e["symbol"], orbital=e["orbital"], compound=e["compound"],
            binding_energy=e["BE"], oxidation_state=e["charge"],
        )
        for e in config.elements
    ]
    compound_db = CompoundDatabase()

    base_fv = config.base_fv()
    n_fv = len(base_fv)

    Dstk = np.zeros((n_pixels, config.n_depth, n_elements))
    Xstk = np.zeros((n_pixels, config.n_angles, n_elements))
    FV0stk = np.zeros((n_fv, n_pixels))  # MATLAB layout: (n_fv, n_pixels)
    FVstk = np.zeros((n_fv, n_pixels))

    # Select distribution function
    dist_func = calculate_distribution
    if getattr(config, 'dist_type', 'Error fcn.') == 'Gaussian':
        dist_func = calculate_distribution_gaussian

    if verbose:
        print(f"Generating simulation: {config.nx}x{config.ny}x{config.dt}"
              f" (phases={config.n_phases}) = {n_pixels} pixels")
        print(f"  depth: {config.n_depth} pts, step={config.step_size} A")
        print(f"  angles: {config.n_angles} pts, {config.angle_min}-{config.angle_max} deg")
        print(f"  elements: {[e['symbol'] for e in config.elements]} ({n_elements})")
        print(f"  dist_type: {getattr(config, 'dist_type', 'Error fcn.')}")
        print(f"  flags: {[k for k,v in config.flags.items() if v]}")
        print()

    import time as _time
    t0 = _time.perf_counter()

    for px in range(n_pixels):
        sk = px + 1  # MATLAB 1-based

        # Apply FV modulation
        FV = update_feature_vector(
            base_fv, sk, config.nx, config.ny, config.dt, config.flags)

        FV0stk[:, px] = FV  # Store modulated FV (before CalculateDistribution)
        FVstk[:, px] = FV   # Will be updated later if needed

        # Convert FV to profiles
        thickness, abruptness, free_profiles = fv_to_profiles(FV, n_free)

        # Generate depth profile
        D = dist_func(
            depth=depth,
            thickness=thickness,
            abruptness_fwhm=abruptness,
            free_profiles=free_profiles,
            linked_profiles=config.linked_profiles,
            charges=charges,
            subst_idx=config.subst_idx,
            free_idx=config.free_idx,
            linked_idx=config.linked_idx,
        )
        Dstk[px] = D

        # Forward model: D → Xstk
        tm = TransferMatrix(
            depth=depth, step_size=config.step_size, angles=angles,
            photon_energy=config.photon_energy, work_function=config.work_function,
            gamma_eff=config.gamma_eff,
        )
        tm.compute(D, element_configs, compound_db)
        X = tm.forward(D, normalize=True)
        Xstk[px] = X

        if verbose and (px % max(1, n_pixels // 20) == 0 or px == n_pixels - 1):
            elapsed = _time.perf_counter() - t0
            rate = (px + 1) / elapsed if elapsed > 0 else 0
            print(f"  pixel {px}/{n_pixels-1} ({rate:.0f} px/s)")

    elapsed = _time.perf_counter() - t0
    if verbose:
        print(f"\nGeneration complete: {n_pixels} pixels in {elapsed:.1f}s "
              f"({n_pixels/elapsed:.0f} px/s)")
        # Show thickness range
        thicknesses = FV0stk[0, :]
        print(f"  Thickness range: {thicknesses.min():.0f}-{thicknesses.max():.0f} A")

    return {
        "Dstk": Dstk,
        "Xstk": Xstk,
        "FV0stk": FV0stk,
        "FVstk": FVstk,
        "depth": depth,
        "angles": angles,
        "element_configs": element_configs,
        "config": config,
        "charges": charges,
    }


# ==============================================================================
# Full pipeline
# ==============================================================================

def generate_checkered(
    config: Optional[CheckeredConfig] = None,
    verbose: bool = True,
) -> dict:
    """Generate Checkered benchmark dataset.

    Args:
        config: Configuration (default: standard 5x5 setup)
        verbose: Print progress

    Returns:
        Dictionary with Dstk, Xstk, depth, angles, elements, metadata
    """
    if config is None:
        config = CheckeredConfig()

    n_pixels = config.nx * config.ny
    depth = np.arange(config.n_depth) * config.step_size
    angles = np.linspace(config.angle_min, config.angle_max, config.n_angles)
    n_elements = len(config.elements)

    charges = np.array([e["charge"] for e in config.elements], dtype=np.float64)

    # Build ElementConfig list for TransferMatrix
    element_configs = [
        ElementConfig(
            symbol=e["symbol"],
            orbital=e["orbital"],
            compound=e["compound"],
            binding_energy=e["BE"],
            oxidation_state=e["charge"],
        )
        for e in config.elements
    ]

    compound_db = CompoundDatabase()

    Dstk = np.zeros((n_pixels, config.n_depth, n_elements))
    Xstk = np.zeros((n_pixels, config.n_angles, n_elements))

    if verbose:
        print(f"Generating Checkered benchmark: {config.nx}×{config.ny} = {n_pixels} pixels")
        print(f"  depth: {config.n_depth} points, step={config.step_size} Å")
        print(f"  angles: {config.n_angles} points, {config.angle_min}°–{config.angle_max}°")
        print(f"  elements: {[e['symbol']+e['orbital'] for e in config.elements]}")
        print(f"  photon_energy={config.photon_energy} eV, gamma_eff={config.gamma_eff}")
        print()

    for px in range(n_pixels):
        sk = px + 1  # MATLAB 1-based

        # Apply Checkered pattern to free element heights
        if config.nx > 1 or config.ny > 1:
            profiles = apply_checkered(
                config.free_profiles,
                sk=sk,
                dx=config.nx,
                dy=config.ny,
                idx1=config.checkered_idx1,
                idx2=config.checkered_idx2,
            )
        else:
            profiles = config.free_profiles

        # Generate depth profile
        D = calculate_distribution(
            depth=depth,
            thickness=config.thickness,
            abruptness_fwhm=config.abruptness,
            free_profiles=profiles,
            linked_profiles=config.linked_profiles,
            charges=charges,
            subst_idx=config.subst_idx,
            free_idx=config.free_idx,
            linked_idx=config.linked_idx,
        )
        Dstk[px] = D

        # Forward model: D → Xstk
        tm = TransferMatrix(
            depth=depth,
            step_size=config.step_size,
            angles=angles,
            photon_energy=config.photon_energy,
            work_function=config.work_function,
            gamma_eff=config.gamma_eff,
        )
        tm.compute(D, element_configs, compound_db)
        X = tm.forward(D, normalize=True)
        Xstk[px] = X

        if verbose and (px % max(1, n_pixels // 10) == 0 or px == n_pixels - 1):
            print(f"  pixel {px}/{n_pixels-1}")

    if verbose:
        print(f"\nGeneration complete.")
        # Show pattern statistics
        elem1_idx = config.free_idx[config.checkered_idx1]
        elem2_idx = config.free_idx[config.checkered_idx2]
        elem1_name = config.elements[elem1_idx]["symbol"]
        elem2_name = config.elements[elem2_idx]["symbol"]
        e1_surface = Dstk[:, 0, elem1_idx]
        e2_surface = Dstk[:, 0, elem2_idx]
        n_e1_dominant = np.sum(e1_surface > e2_surface)
        n_e2_dominant = np.sum(e2_surface > e1_surface)
        print(f"  Checkered pattern ({elem1_name}↔{elem2_name}): "
              f"{n_e1_dominant} px {elem1_name}-dominant, "
              f"{n_e2_dominant} px {elem2_name}-dominant")

    result = {
        "Dstk": Dstk,
        "Xstk": Xstk,
        "depth": depth,
        "angles": angles,
        "element_configs": element_configs,
        "config": config,
        "charges": charges,
    }
    return result


# ==============================================================================
# H5 output (compatible with load_dstk_data)
# ==============================================================================

def save_to_h5(result: dict, h5_path: str, verbose: bool = True):
    """Save generated data to H5 in benchmark_dstk-compatible format.

    Args:
        result: Output from generate_checkered()
        h5_path: Output file path
        verbose: Print progress
    """
    config = result["config"]
    Dstk = result["Dstk"]
    Xstk = result["Xstk"]
    depth = result["depth"]
    angles = result["angles"]
    n_pixels, n_depth, n_elements = Dstk.shape
    n_angles = Xstk.shape[1]

    # Transpose to MATLAB layout for H5 storage
    # Python (pixels, depth, elements) → MATLAB (pixels, elements, depth)
    Dstk_h5 = np.transpose(Dstk, (0, 2, 1)).astype(np.float32)
    Xstk_h5 = np.transpose(Xstk, (0, 2, 1)).astype(np.float32)

    # Compute template TransMatObs from first pixel
    tm = TransferMatrix(
        depth=depth,
        step_size=config.step_size,
        angles=angles,
        photon_energy=config.photon_energy,
        work_function=config.work_function,
        gamma_eff=config.gamma_eff,
    )
    compound_db = CompoundDatabase()
    element_configs = result["element_configs"]
    tm.compute(Dstk[0], element_configs, compound_db)

    # TransMatObs: Python (angles, depth, elements) → MATLAB (elements, depth, angles)
    TransMatObs_h5 = np.transpose(tm.trans_mat, (2, 1, 0)).astype(np.float32)

    with h5py.File(h5_path, "w") as f:
        # Core datasets
        f.create_dataset("Dstk", data=Dstk_h5)
        f.create_dataset("Xstk", data=Xstk_h5)
        # Drlt = Dstk as placeholder (no solver result yet)
        f.create_dataset("Drlt", data=Dstk_h5)

        # ezdeprof group
        ez = f.create_group("ezdeprof")
        ez.create_dataset("depth", data=depth.reshape(1, -1).astype(np.float64))
        ez.create_dataset("thetaobs", data=angles.reshape(1, -1).astype(np.float64))
        ez.create_dataset("stepsize", data=np.array([config.step_size]))
        ez.create_dataset("gammaeff", data=np.array([config.gamma_eff]))
        ez.create_dataset("photonenergy",
                          data=np.full((n_elements, 1), config.photon_energy))
        ez.create_dataset("workfunction", data=np.array([config.work_function]))

        # Element info
        elem_labels = []
        symbols = []
        orbitals = []
        compounds = []
        binding_energies = []
        for e in config.elements:
            charge_str = f"({'+' if e['charge']>0 else ''}{e['charge']})" if e['charge'] != 0 else "(0)"
            elem_labels.append(f"{e['symbol']}{charge_str}")
            symbols.append(e["symbol"])
            orbitals.append(e["orbital"])
            compounds.append(e["compound"])
            binding_energies.append(e["BE"])

        dt_str = h5py.string_dtype()
        ez.create_dataset("element",
                          data=np.array(elem_labels, dtype=object).reshape(-1, 1),
                          dtype=dt_str)
        ez.create_dataset("elementsymbol",
                          data=np.array(symbols, dtype=object).reshape(-1, 1),
                          dtype=dt_str)
        ez.create_dataset("orbital",
                          data=np.array(orbitals, dtype=object).reshape(-1, 1),
                          dtype=dt_str)
        ez.create_dataset("compound",
                          data=np.array(compounds, dtype=object).reshape(-1, 1),
                          dtype=dt_str)
        ez.create_dataset("bindingenergy",
                          data=np.array(binding_energies).reshape(-1, 1))
        ez.create_dataset("charge",
                          data=result["charges"].reshape(-1, 1))

        # Index arrays (MATLAB 1-based)
        ez.create_dataset("free",
                          data=np.array([i+1 for i in config.free_idx]).reshape(-1, 1))
        ez.create_dataset("subst",
                          data=np.array([i+1 for i in config.subst_idx]).reshape(-1, 1))
        ez.create_dataset("linked",
                          data=np.array([i+1 for i in config.linked_idx]).reshape(-1, 1))
        ez.create_dataset("linkedconst",
                          data=np.array(config.linked_const).reshape(-1, 1))

        # TransMatObs template
        ez.create_dataset("TransMatObs", data=TransMatObs_h5)

        # Metadata
        ez.create_dataset("stack", data=np.array([n_pixels]))
        ez.create_dataset("totaldepth", data=np.array([config.total_depth]))
        ez.create_dataset("m", data=np.array([n_depth]))
        ez.create_dataset("noise", data=np.array([config.noise_level]))

        # Additional metadata
        ez.attrs["generator"] = "deppro.benchmarks.generate_checkered"
        ez.attrs["nx"] = config.nx
        ez.attrs["ny"] = config.ny

    if verbose:
        print(f"Saved to: {h5_path}")
        print(f"  Dstk: {Dstk_h5.shape}, Xstk: {Xstk_h5.shape}")


# ==============================================================================
# Full MATLAB-compatible H5 (for CalculationRunner.fromDataFile)
# ==============================================================================

# MATLAB thetaopt: fixed 16-point grid from -10° to 90°
THETAOPT_16 = np.array([-10, 0, 10, 20, 30, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90],
                        dtype=np.float64)

# Element number (atomic number) lookup
ELEMENT_Z = {"Si": 14, "Ti": 22, "Al": 13, "C": 6, "B": 5, "O": 8}


def save_to_matlab_h5(result: dict, h5_path: str, verbose: bool = True):
    """Save generated data to full MATLAB CalculationRunner-compatible H5.

    Replicates the exact structure of a MATLAB-generated ezdeprof H5 file
    so that CalculationRunner.fromDataFile() can load it directly.

    Args:
        result: Output from generate_checkered()
        h5_path: Output file path
        verbose: Print progress
    """
    config = result["config"]
    Dstk = result["Dstk"]  # (n_pixels, n_depth, n_elements)
    Xstk = result["Xstk"]  # (n_pixels, n_angles, n_elements)
    depth = result["depth"]
    angles = result["angles"]
    charges = result["charges"]
    element_configs = result["element_configs"]
    n_pixels, n_depth, n_elements = Dstk.shape
    n_angles = Xstk.shape[1]

    compound_db = CompoundDatabase()

    # Transpose to MATLAB layout: Python (px, depth, elem) → H5 (px, elem, depth)
    Dstk_h5 = np.transpose(Dstk, (0, 2, 1)).astype(np.float32)
    Xstk_h5 = np.transpose(Xstk, (0, 2, 1)).astype(np.float32)

    # Compute TransferMatrix from first pixel for template values
    tm_obs = TransferMatrix(
        depth=depth, step_size=config.step_size, angles=angles,
        photon_energy=config.photon_energy, work_function=config.work_function,
        gamma_eff=config.gamma_eff,
    )
    tm_obs.compute(Dstk[0], element_configs, compound_db)

    # Also compute at thetaopt (16 angles)
    tm_opt = TransferMatrix(
        depth=depth, step_size=config.step_size, angles=THETAOPT_16,
        photon_energy=config.photon_energy, work_function=config.work_function,
        gamma_eff=config.gamma_eff,
    )
    tm_opt.compute(Dstk[0], element_configs, compound_db)

    # TransMatObs/Opt: Python (angles, depth, elem) → H5 (elem, depth, angles)
    # (In MATLAB column-major, H5 (a,b,c) reads as MATLAB (c,b,a))
    TransMatObs_h5 = np.transpose(tm_obs.trans_mat, (2, 1, 0)).astype(np.float32)
    TransMatOpt_h5 = np.transpose(tm_opt.trans_mat, (2, 1, 0)).astype(np.float32)

    # lambda_basic: (n_elem, n_elem) → H5 (1, n_elem, n_elem)
    lambda_basic_h5 = tm_obs.lambda_basic[np.newaxis, :, :].astype(np.float32)

    # lambda_obs: (n_depth, n_elem) → H5 (1, n_elem, n_depth)
    lambda_obs_h5 = tm_obs.lambda_obs.T[np.newaxis, :, :].astype(np.float32)
    lambda_opt_h5 = tm_opt.lambda_obs.T[np.newaxis, :, :].astype(np.float32)

    # lambda*cos(theta): (n_elem, n_angles)
    lambda_costheta_obs = np.zeros((n_elements, n_angles), dtype=np.float32)
    for e in range(n_elements):
        for a in range(n_angles):
            lambda_costheta_obs[e, a] = (
                tm_obs.lambda_obs[0, e] * np.cos(np.radians(angles[a]))
            )
    n_opt = len(THETAOPT_16)
    lambda_costheta_opt = np.zeros((n_elements, n_opt), dtype=np.float32)
    for e in range(n_elements):
        for a in range(n_opt):
            lambda_costheta_opt[e, a] = (
                tm_opt.lambda_obs[0, e] * np.cos(np.radians(THETAOPT_16[a]))
            )

    # Forward model at thetaopt for Xopt/Xini/Xmod
    Xopt_template = tm_opt.forward(Dstk[0], normalize=True)  # (16, 7)
    Xopt_h5 = Xopt_template.T.astype(np.float32)  # (7, 16)
    Xobs_template = Xstk[0].T.astype(np.float32)   # (7, 14) first pixel

    # Initial guess profile (Dini): use the average Dstk as initial
    # For CalculationRunner, Dini = Dopt = first pixel's Drlt
    Dini_h5 = Dstk_h5[0]  # (n_elem, n_depth) first pixel

    # Build FeatureVector from config
    n_free = len(config.free_idx)
    # FV = [thck, abrp, pos1, w1, h1, pos2, w2, h2, ...]
    fv_list = [config.thickness, config.abruptness]
    for prof in config.free_profiles:
        fv_list.extend([prof["pos"], prof["width"], prof["height"]])
    FeatureVector_Source = np.array(fv_list, dtype=np.float32).reshape(-1, 1)
    n_fv = len(fv_list)

    # Build SupportVector from linked profiles
    # SV = [pos_sub1, w_sub1, h_sub1, pos_linked1, w_linked1, h_linked1]
    sv_list = []
    # Substrate peak params (from Dstk)
    for si in config.subst_idx:
        D_sub = Dstk[0, :, si]
        p_sub = np.sum(D_sub * depth) / max(np.sum(D_sub), 1e-10)
        h_sub = 100.0 * np.max(D_sub)
        w_sub = 2 * np.sqrt(np.log(2) / np.pi) * trapezoid(100 * D_sub, depth) / max(h_sub, 1e-10)
        sv_list.extend([p_sub, w_sub, h_sub])
    # Linked element peak params
    for li in config.linked_idx:
        D_lnk = Dstk[0, :, li]
        p_lnk = np.sum(D_lnk * depth) / max(np.sum(D_lnk), 1e-10)
        h_lnk = 100.0 * np.max(D_lnk)
        w_lnk = 2 * np.sqrt(np.log(2) / np.pi) * trapezoid(100 * D_lnk, depth) / max(h_lnk, 1e-10)
        sv_list.extend([p_lnk, w_lnk, h_lnk])
    SupportVector_Set = np.array(sv_list, dtype=np.float32).reshape(-1, 1)
    n_sv = len(sv_list)

    # Concentration: 100 * sum(D_elem) / sum(D_antisubst)
    antisubst = sorted(config.free_idx + config.linked_idx)
    D0 = Dstk[0]
    Conc_vals = 100.0 * np.sum(D0, axis=0) / max(np.sum(D0[:, antisubst]), 1e-10)
    Conc_Set = Conc_vals.astype(np.float32).reshape(-1, 1)

    # Buffer size for per-stack arrays
    BUFFER = 10000

    # Color palette for elements (from reference file, extended to 7)
    if n_elements <= 6:
        Col = np.array([
            [0.0, 0.765, 0.836, 0.445, 0.419, 0.271],
            [0.402, 0.293, 0.625, 0.166, 0.607, 0.071],
            [0.667, 0.088, 0.113, 0.500, 0.169, 0.166],
        ], dtype=np.float32)[:, :n_elements]
    else:
        Col = np.array([
            [0.0, 0.765, 0.836, 0.445, 0.419, 0.271, 0.572],
            [0.402, 0.293, 0.625, 0.166, 0.607, 0.671, 0.070],
            [0.667, 0.088, 0.113, 0.500, 0.169, 0.840, 0.166],
        ], dtype=np.float32)[:, :n_elements]

    # Element metadata arrays
    elem_labels, symbols, orbitals, compounds = [], [], [], []
    binding_energies, densities, mol_weights, bandgaps, val_charges = [], [], [], [], []
    elem_numbers, orbital_numbers = [], []
    string_charges, prpties = [], []

    for i, e in enumerate(config.elements):
        ch = e["charge"]
        if ch > 0:
            charge_str = f"(+{ch})"
            scharge = f"+{ch}"
        elif ch < 0:
            charge_str = f"({ch})"
            scharge = str(ch)
        else:
            charge_str = "(0)"
            scharge = "0"
        elem_labels.append(f"{e['symbol']}{charge_str}")
        symbols.append(e["symbol"])
        orbitals.append(e["orbital"])
        compounds.append(e["compound"])
        binding_energies.append(e["BE"])
        densities.append(e["density"])
        mol_weights.append(e["Aw"])
        val_charges.append(e["Nv"])
        bandgaps.append(e["Eg"])
        elem_numbers.append(ELEMENT_Z.get(e["symbol"], 0))
        orbital_numbers.append(1)  # 1s = 1
        string_charges.append(scharge)

        if i in config.subst_idx:
            prpties.append("Subst.")
        elif i in config.free_idx:
            prpties.append("Free")
        elif i in config.linked_idx:
            prpties.append("Linked")
        else:
            prpties.append("Const.")

    dt_str = h5py.string_dtype()

    with h5py.File(h5_path, "w") as f:
        # =====================================================
        # Top-level 3D datasets (px, elem, depth)/(px, elem, angles)
        # =====================================================
        f.create_dataset("Dstk", data=Dstk_h5)
        f.create_dataset("Drlt", data=Dstk_h5.copy())  # Initial = GT
        f.create_dataset("Dtrn", data=Dstk_h5.copy())  # Initial = GT

        f.create_dataset("Xstk", data=Xstk_h5)
        # Xrlt: (px, elem, n_opt=16) — initialize zeros
        f.create_dataset("Xrlt", data=np.zeros(
            (n_pixels, n_elements, n_opt), dtype=np.float32))

        # =====================================================
        # Top-level 2D per-stack arrays (features, BUFFER)
        # =====================================================
        # Initialize FVrlt/FVtrn with FeatureVector_Source for each stack
        FVrlt_init = np.zeros((n_fv, BUFFER), dtype=np.float32)
        for sk in range(n_pixels):
            FVrlt_init[:, sk] = FeatureVector_Source.flatten()
        f.create_dataset("FVrlt", data=FVrlt_init)
        f.create_dataset("FVtrn", data=FVrlt_init.copy())
        f.create_dataset("FVstk", data=FVrlt_init.copy())
        f.create_dataset("FV0stk", data=FVrlt_init.copy())

        SVrlt_init = np.zeros((n_sv, BUFFER), dtype=np.float32)
        for sk in range(n_pixels):
            SVrlt_init[:, sk] = SupportVector_Set.flatten()
        f.create_dataset("SVrlt", data=SVrlt_init)
        f.create_dataset("SVtrn", data=SVrlt_init.copy())
        f.create_dataset("SVstk", data=SVrlt_init.copy())

        Crlt_init = np.zeros((n_elements, BUFFER), dtype=np.float32)
        for sk in range(n_pixels):
            Crlt_init[:, sk] = Conc_Set.flatten()
        f.create_dataset("Crlt", data=Crlt_init)
        f.create_dataset("Ctrn", data=Crlt_init.copy())
        f.create_dataset("Cstk", data=Crlt_init.copy())

        f.create_dataset("SPSrlt", data=np.zeros((2, BUFFER), dtype=np.float32))
        f.create_dataset("SPStrn", data=np.zeros((2, BUFFER), dtype=np.float32))
        f.create_dataset("SPSstk", data=np.zeros((2, BUFFER), dtype=np.float32))

        # Q arrays: (3, BUFFER), init NaN
        q_nan = np.full((3, BUFFER), np.nan, dtype=np.float32)
        f.create_dataset("Qrlt", data=q_nan.copy())
        f.create_dataset("Qtrn", data=q_nan.copy())
        f.create_dataset("Qstk", data=q_nan.copy())

        # NSstk, degree, alpha
        f.create_dataset("NSstk", data=np.zeros((1, BUFFER), dtype=np.float32))
        f.create_dataset("degree", data=np.full((3, BUFFER), 1e5, dtype=np.float32))
        f.create_dataset("alpha", data=np.full((3, BUFFER), 1e5, dtype=np.float32))

        # Col (top-level, for coloring)
        f.create_dataset("Col", data=Col.astype(np.float64))

        # =====================================================
        # /ezdeprof/ group — full metadata
        # =====================================================
        ez = f.create_group("ezdeprof")

        # --- Element info (string arrays) ---
        ez.create_dataset("element",
                          data=np.array(elem_labels, dtype=object).reshape(-1, 1), dtype=dt_str)
        ez.create_dataset("elementsymbol",
                          data=np.array(symbols, dtype=object).reshape(-1, 1), dtype=dt_str)
        ez.create_dataset("orbital",
                          data=np.array(orbitals, dtype=object).reshape(-1, 1), dtype=dt_str)
        ez.create_dataset("compound",
                          data=np.array(compounds, dtype=object).reshape(-1, 1), dtype=dt_str)
        ez.create_dataset("stringcharge",
                          data=np.array(string_charges, dtype=object).reshape(-1, 1), dtype=dt_str)
        ez.create_dataset("prpties",
                          data=np.array(prpties, dtype=object).reshape(-1, 1), dtype=dt_str)

        # --- Element numeric properties ---
        ez.create_dataset("charge",
                          data=charges.reshape(-1, 1).astype(np.float32))
        ez.create_dataset("bindingenergy",
                          data=np.array(binding_energies, dtype=np.float32).reshape(-1, 1))
        ez.create_dataset("density",
                          data=np.array(densities, dtype=np.float32).reshape(-1, 1))
        ez.create_dataset("molecularweight",
                          data=np.array(mol_weights, dtype=np.float32).reshape(-1, 1))
        ez.create_dataset("valencecharge",
                          data=np.array(val_charges, dtype=np.float32).reshape(-1, 1))
        ez.create_dataset("bandgap",
                          data=np.array(bandgaps, dtype=np.float32).reshape(-1, 1))
        ez.create_dataset("elementnumber",
                          data=np.array(elem_numbers, dtype=np.uint8).reshape(-1, 1))
        ez.create_dataset("orbitalnumber",
                          data=np.array(orbital_numbers, dtype=np.uint8).reshape(-1, 1))

        # --- Index arrays (MATLAB 1-based, uint8) ---
        ez.create_dataset("free",
                          data=np.array([i+1 for i in config.free_idx], dtype=np.uint8).reshape(-1, 1))
        ez.create_dataset("subst",
                          data=np.array([i+1 for i in config.subst_idx], dtype=np.uint8).reshape(-1, 1))
        ez.create_dataset("linked",
                          data=np.array([i+1 for i in config.linked_idx], dtype=np.uint8).reshape(-1, 1))
        ez.create_dataset("const", data=np.array([0], dtype=np.uint8).reshape(-1, 1))
        ez.create_dataset("antisubst",
                          data=np.array([i+1 for i in antisubst], dtype=np.uint8).reshape(-1, 1))
        ez.create_dataset("linkedconst",
                          data=np.array(config.linked_const, dtype=np.float32).reshape(-1, 1))

        # --- Physical parameters ---
        ez.create_dataset("photonenergy",
                          data=np.full((n_elements, 1), config.photon_energy, dtype=np.float32))
        ez.create_dataset("photon",
                          data=np.array([[config.photon_energy]], dtype=np.float32))
        ez.create_dataset("deltaphoton", data=np.array([1000.0], dtype=np.float32))
        ez.create_dataset("workfunction", data=np.array([config.work_function], dtype=np.float32))
        ez.create_dataset("gammaeff", data=np.array([config.gamma_eff], dtype=np.float32))

        # --- Depth grid ---
        # MATLAB totaldepth = m * stepsize (not (m-1)*stepsize)
        matlab_totaldepth = n_depth * config.step_size
        ez.create_dataset("depth", data=depth.reshape(1, -1).astype(np.float32))
        ez.create_dataset("totaldepth", data=np.array([matlab_totaldepth], dtype=np.float32))
        ez.create_dataset("m", data=np.array([n_depth], dtype=np.uint16))
        ez.create_dataset("stepsize", data=np.array([config.step_size], dtype=np.float32))
        ez.create_dataset("slice", data=np.array([config.step_size], dtype=np.float32))

        # --- Angle grids ---
        ez.create_dataset("thetaobs", data=angles.reshape(1, -1).astype(np.float32))
        ez.create_dataset("thetaopt", data=THETAOPT_16.reshape(1, -1).astype(np.float32))
        ez.create_dataset("angle", data=np.array([[angles[-1]]], dtype=np.float32))
        ez.create_dataset("datapoints",
                          data=np.array([[n_angles], [1], [n_opt]], dtype=np.uint16))

        # --- Transfer matrices ---
        ez.create_dataset("TransMatObs", data=TransMatObs_h5)
        ez.create_dataset("TransMatOpt", data=TransMatOpt_h5)

        # --- IMFP data ---
        ez.create_dataset("lambdabasic", data=lambda_basic_h5)
        ez.create_dataset("lambdaobs", data=lambda_obs_h5)
        ez.create_dataset("lambdaopt", data=lambda_opt_h5)
        ez.create_dataset("lambdacosinethetaobs", data=lambda_costheta_obs)
        ez.create_dataset("lambdacosinethetaopt", data=lambda_costheta_opt)

        # --- Depth profiles (2D: elem × depth) ---
        ez.create_dataset("Dini", data=Dini_h5.astype(np.float32))
        ez.create_dataset("Dmod", data=Dini_h5.astype(np.float32))
        ez.create_dataset("Dopt", data=Dini_h5.astype(np.float32))
        ez.create_dataset("Dobs", data=Dini_h5.astype(np.float32))

        # --- Angle profiles (2D: elem × angles) ---
        ez.create_dataset("Xobs", data=Xobs_template)
        ez.create_dataset("Xini", data=Xopt_h5.copy())
        ez.create_dataset("Xmod", data=Xopt_h5.copy())
        ez.create_dataset("Xopt", data=Xopt_h5.copy())

        # --- Feature/Support vectors ---
        ez.create_dataset("FeatureVector_Source", data=FeatureVector_Source)
        ez.create_dataset("FeatureVector_Set", data=FeatureVector_Source.copy())
        ez.create_dataset("FeatureVector_Calc", data=FeatureVector_Source.copy())
        ez.create_dataset("FeatureVector_Lower",
                          data=np.empty((0, 1), dtype=np.float32))
        ez.create_dataset("FeatureVector_Upper",
                          data=np.empty((0, 1), dtype=np.float32))
        ez.create_dataset("SupportVector_Set", data=SupportVector_Set)
        ez.create_dataset("SupportVector_Calc", data=SupportVector_Set.copy())

        # --- Concentration ---
        ez.create_dataset("Conc_Set", data=Conc_Set)
        ez.create_dataset("Conc_Calc", data=Conc_Set.copy())

        # --- Sparsity ---
        ez.create_dataset("Sprs_Set",
                          data=np.array([[n_free * n_depth], [n_free * n_depth]], dtype=np.float32))
        ez.create_dataset("Sprs_Calc",
                          data=np.array([[n_free * n_depth], [n_free * n_depth]], dtype=np.float32))

        # --- Thickness/Abruptness/Substrate bounds [value, lower, upper] ---
        thck_val = config.thickness
        abrp_val = config.abruptness
        ez.create_dataset("thckini",
                          data=np.array([[thck_val], [0.0], [matlab_totaldepth]], dtype=np.float32))
        ez.create_dataset("thckmod",
                          data=np.array([[thck_val], [0.0], [matlab_totaldepth]], dtype=np.float32))
        ez.create_dataset("abrpini",
                          data=np.array([[abrp_val], [abrp_val], [abrp_val]], dtype=np.float32))
        ez.create_dataset("abrpmod",
                          data=np.array([[abrp_val], [abrp_val], [abrp_val]], dtype=np.float32))
        n_subst = len(config.subst_idx)
        ez.create_dataset("subsini",
                          data=np.ones((3, n_subst), dtype=np.float32))
        ez.create_dataset("subsmod",
                          data=np.ones((3, n_subst), dtype=np.float32))

        # --- Quality metrics ---
        ez.create_dataset("Qini", data=np.full((3, 1), np.nan, dtype=np.float32))
        ez.create_dataset("Qopt", data=np.full((3, 1), np.nan, dtype=np.float32))

        # --- Loop/stack control ---
        ez.create_dataset("stack", data=np.array([n_pixels], dtype=np.uint64))
        ez.create_dataset("loop", data=np.array([10], dtype=np.uint16))
        ez.create_dataset("currentloop", data=np.array([0], dtype=np.uint16))
        ez.create_dataset("currentstack", data=np.array([0], dtype=np.uint64))

        # --- Spatial grid ---
        ez.create_dataset("dxdydt",
                          data=np.array([[config.nx], [config.ny], [1]], dtype=np.uint16))

        # --- Optimization method/mode flags ---
        ez.create_dataset("optimizationmethodvalue", data=np.array([3], dtype=np.uint8))  # L1
        ez.create_dataset("initialguessvalue", data=np.array([3], dtype=np.uint8))  # SameAsInit
        ez.create_dataset("priormodelvalue", data=np.array([4], dtype=np.uint8))    # None
        ez.create_dataset("transfermatrixupdatevalue", data=np.array([1], dtype=np.uint8))  # Default
        ez.create_dataset("SimulationOrExperiment", data=np.array([1], dtype=np.uint8))
        ez.create_dataset("verificationmethod", data=np.array([2], dtype=np.uint8))
        ez.create_dataset("verificationschemevalue", data=np.array([4], dtype=np.uint8))
        ez.create_dataset("verificationstatus", data=np.array([0], dtype=np.uint8))
        ez.create_dataset("degreecontrol", data=np.array([0], dtype=np.uint8))
        ez.create_dataset("alphacontrol", data=np.array([0], dtype=np.uint8))
        ez.create_dataset("distnfnctnvalue", data=np.array([0], dtype=np.uint8))
        ez.create_dataset("photoionizationcrosssection", data=np.array([1], dtype=np.uint8))
        ez.create_dataset("crossecapproximord", data=np.array([3], dtype=np.uint8))
        ez.create_dataset("crossecapproximview", data=np.array([0], dtype=np.uint8))
        ez.create_dataset("TPPver", data=np.array([3], dtype=np.uint8))

        # --- Misc ---
        ez.create_dataset("noise", data=np.array([config.noise_level], dtype=np.float32))
        ez.create_dataset("NP_Set", data=np.array([2], dtype=np.uint8))
        ez.create_dataset("NP_Calc", data=np.array([2], dtype=np.uint8))
        ez.create_dataset("NP_Max", data=np.array([n_elements], dtype=np.uint8))
        ez.create_dataset("Kfold", data=np.array([1], dtype=np.uint16))
        ez.create_dataset("holdout", data=np.array([1], dtype=np.uint64))
        ez.create_dataset("iterationinterval", data=np.array([1], dtype=np.uint64))
        ez.create_dataset("iterationtime", data=np.array([0.0], dtype=np.float32))
        ez.create_dataset("iterationtotal", data=np.array([0], dtype=np.uint64))
        ez.create_dataset("numberofmodeling", data=np.array([1], dtype=np.uint64))
        ez.create_dataset("NumberofNearestNeighbor", data=np.array([1], dtype=np.uint16))

        # calculationinitialized: [status, 1, 0] — not yet calculated
        ez.create_dataset("calculationinitialized",
                          data=np.array([[0], [0], [0]], dtype=np.uint64))

        # Color
        ez.create_dataset("Col", data=Col)

        # Generator metadata
        ez.attrs["generator"] = "deppro.benchmarks.generate_checkered"
        ez.attrs["nx"] = config.nx
        ez.attrs["ny"] = config.ny

    if verbose:
        print(f"Saved MATLAB-compatible H5 to: {h5_path}")
        print(f"  Dstk: {Dstk_h5.shape} (px={n_pixels}, elem={n_elements}, depth={n_depth})")
        print(f"  Xstk: {Xstk_h5.shape} (angles={n_angles})")
        print(f"  FVrlt: ({n_fv}, {BUFFER})")
        print(f"  elements: {elem_labels}")
        print(f"  method: L1 (optimizationmethodvalue=3)")
        print(f"  initialguessvalue=3 (SameAsInit)")


# ==============================================================================
# Validation: compare with existing MATLAB data
# ==============================================================================

def validate_against_matlab(
    result: dict,
    matlab_h5_path: str,
    verbose: bool = True,
) -> dict:
    """Compare generated Dstk with MATLAB's Dstk from existing H5.

    Args:
        result: Output from generate_checkered()
        matlab_h5_path: Path to existing MATLAB H5 file
        verbose: Print comparison

    Returns:
        Comparison metrics
    """
    from deppro.benchmarks.benchmark_dstk import load_dstk_data

    matlab = load_dstk_data(matlab_h5_path)

    n_compare = min(result["Dstk"].shape[0], matlab.n_pixels)

    dstk_diff = np.abs(result["Dstk"][:n_compare] - matlab.Dstk[:n_compare])
    xstk_diff = np.abs(result["Xstk"][:n_compare] - matlab.Xstk[:n_compare])

    metrics = {
        "dstk_max_diff": float(dstk_diff.max()),
        "dstk_mean_diff": float(dstk_diff.mean()),
        "xstk_max_diff": float(xstk_diff.max()),
        "xstk_mean_diff": float(xstk_diff.mean()),
    }

    if verbose:
        print(f"\n=== Validation against MATLAB: {matlab_h5_path} ===")
        print(f"  Dstk max diff: {metrics['dstk_max_diff']:.6e}")
        print(f"  Dstk mean diff: {metrics['dstk_mean_diff']:.6e}")
        print(f"  Xstk max diff: {metrics['xstk_max_diff']:.6e}")
        print(f"  Xstk mean diff: {metrics['xstk_mean_diff']:.6e}")

        # Per-element breakdown
        elem_names = [e.label for e in result["element_configs"]]
        print(f"\n  Per-element Dstk max diff:")
        for e, name in enumerate(elem_names):
            e_diff = dstk_diff[:, :, e].max()
            print(f"    {name:10s}: {e_diff:.6e}")

    return metrics


# ==============================================================================
# CLI
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate Checkered benchmark ground truth"
    )
    parser.add_argument("--nx", type=int, default=5, help="X pixels")
    parser.add_argument("--ny", type=int, default=5, help="Y pixels")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output H5 path")
    parser.add_argument("--no-checkered", action="store_true",
                        help="Uniform profiles (no checkered pattern)")
    parser.add_argument("--matlab-h5", type=str, default=None,
                        help="Output full MATLAB-compatible H5 for CalculationRunner")
    parser.add_argument("--validate", type=str, default=None,
                        help="MATLAB H5 to validate against")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    config = CheckeredConfig(nx=args.nx, ny=args.ny)
    if args.no_checkered:
        config.nx = 1
        config.ny = 1

    verbose = not args.quiet
    result = generate_checkered(config, verbose=verbose)

    if args.output:
        save_to_h5(result, args.output, verbose=verbose)

    if args.matlab_h5:
        save_to_matlab_h5(result, args.matlab_h5, verbose=verbose)

    if args.validate:
        validate_against_matlab(result, args.validate, verbose=verbose)

    return result


if __name__ == "__main__":
    main()
