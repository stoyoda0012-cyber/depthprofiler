#!/usr/bin/env python
"""
Example that builds from scratch an H5 file for an HXPS (8 keV) multilayer sample

Produces an H5 file with the same structure as the 6-element H5 written by the MATLAB DepthProfiler.
- 6 elements: Si (substrate), SiO2, TiO2, Al2O3, C (contamination), O (oxide)
- assumes an 8 keV HXPS measurement
- 4-layer structure: C surface contamination / Al2O3 / TiO2 / Si substrate

Uses the new initializer module to perform the same initialization
as MATLAB's Set button.
"""

import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from deppro.io import H5Writer
from deppro.io.initializer import DepthProfileInitializer, create_initialized_h5


def create_multilayer_profile(depth: np.ndarray) -> np.ndarray:
    """
    Build the depth profile of the 4-layer structure

    Structure (from the top):
    - C contamination layer: 0-20 Å
    - Al2O3 layer: 20-80 Å
    - TiO2 layer: 80-160 Å
    - Si substrate: below 160 Å

    Elements: [Si(0), Si(+4), Ti(+4), Al(+3), C(0), O(-2)]

    Returns:
        profile: (n_depth, 6) depth profile
    """
    from scipy.special import erf

    n_depth = len(depth)
    profile = np.zeros((n_depth, 6))

    # FWHM → σ conversion factor
    fwhm_to_sigma = 1.0 / (2.0 * np.sqrt(np.log(2)))

    # Interface positions and abruptness (FWHM)
    z1 = 20.0   # C/Al2O3 interface
    z2 = 80.0   # Al2O3/TiO2 interface
    z3 = 160.0  # TiO2/Si interface
    w = 20.0    # interface width (FWHM)
    sigma = w * fwhm_to_sigma

    for i, z in enumerate(depth):
        # Presence function of each layer (sigmoid transitions)
        f_C = 0.5 * (1 - erf((z - z1) / sigma))
        f_Al2O3 = 0.5 * (1 + erf((z - z1) / sigma)) * 0.5 * (1 - erf((z - z2) / sigma))
        f_TiO2 = 0.5 * (1 + erf((z - z2) / sigma)) * 0.5 * (1 - erf((z - z3) / sigma))
        f_Si = 0.5 * (1 + erf((z - z3) / sigma))

        # Element concentrations
        # [Si(0), Si(+4), Ti(+4), Al(+3), C(0), O(-2)]
        profile[i, 0] = f_Si * 1.0                          # Si(0): substrate
        profile[i, 1] = 0.0                                  # Si(+4): none
        profile[i, 2] = f_TiO2 * 0.33                        # Ti(+4): TiO2 layer
        profile[i, 3] = f_Al2O3 * 0.40                       # Al(+3): Al2O3 layer
        profile[i, 4] = f_C * 0.90                           # C(0): surface contamination only
        profile[i, 5] = (f_TiO2 * 0.67 + f_Al2O3 * 0.60 +    # O(-2): oxides
                        f_C * 0.08)

    # Normalize
    total = profile.sum(axis=1, keepdims=True)
    profile = np.where(total > 0, profile / total, profile)

    return profile


def main():
    print("=" * 70)
    print("Sample: generating an H5 file for an HXPS multilayer")
    print("=" * 70)

    # =========================================================================
    # 1. Basic parameters
    # =========================================================================
    photon_energy = 8000.0  # 8keV HXPS
    total_depth = 600.0     # analysis depth [Å]
    n_depth = 20            # number of depth points
    slice_size = 20.0       # film slice size [Å]

    # Depth axis
    depth = np.linspace(0, total_depth, n_depth)
    step_size = depth[1] - depth[0]

    # Measurement angles (14 points, evenly spaced 0-60°)
    thetaobs = np.linspace(0, 60, 14)

    # Optimization / display angles (the standard 16 points)
    thetaopt = np.array([-10, 0, 10, 20, 30, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90],
                       dtype=np.float32)

    # =========================================================================
    # 2. Element settings
    # =========================================================================
    # Element symbols (one per chemical species)
    elements = ['Si', 'Si', 'Ti', 'Al', 'C', 'O']

    # Orbitals (1s for 8 keV HXPS)
    orbitals = ['1s', '1s', '1s', '1s', '1s', '1s']

    # Compound names (for looking up material properties)
    compounds = ['Si', 'SiO2', 'TiO2', 'Al2O3', 'C', 'Oxide']

    # Charge states
    charges = [0.0, 4.0, 4.0, 3.0, 0.0, -2.0]

    # Substrate / free / linked settings (1-based index)
    subst = [1]           # Si(0) is the substrate
    free = [2, 3, 4, 5]   # Si(+4), Ti, Al, C are free
    linked = [6]          # O is linked (follows the other elements as oxides)

    # Property strings
    prpties = ['Subst.', 'Free', 'Free', 'Free', 'Free', 'Linked']

    # =========================================================================
    # 3. Depth profile
    # =========================================================================
    print("\nBuilding the depth profile...")
    profile_true = create_multilayer_profile(depth)

    # Initial estimate = a copy of the true profile (recomputed by MATLAB's Set)
    profile_init = profile_true.copy()

    # Result (the truth with a little noise)
    noise_level = 0.02
    profile_result = profile_true + np.random.normal(0, noise_level, profile_true.shape)
    profile_result = np.clip(profile_result, 0, 1)
    profile_result = profile_result / profile_result.sum(axis=1, keepdims=True)

    print(f"  depth points: {n_depth}")
    print(f"  elements: {len(elements)}")

    # =========================================================================
    # 4. Initialization (equivalent of MATLAB's Set button)
    # =========================================================================
    print("\nInitializing (equivalent of MATLAB Set)...")

    initializer = DepthProfileInitializer(
        depth=depth,
        angles_obs=thetaobs,
        elements=elements,
        compounds=compounds,
        angles_opt=thetaopt,
        photon_energy=photon_energy,
        charges=charges,
    )

    # Transfer matrix and angle profiles from the true profile
    init_results = initializer.initialize_from_profile(profile_true)

    print(f"  lambdabasic (diagonal): {np.diag(init_results['lambdabasic'])}")
    print(f"  Xobs[0] (angle=0): {init_results['Xobs'][0]}")

    # =========================================================================
    # 5. Write the file with H5Writer
    # =========================================================================
    from deppro.paths import ensure_outputs_dir
    output_dir = ensure_outputs_dir()
    output_path = output_dir / 'example_hxps_multilayer.h5'

    print(f"\nWriting the H5 file...")
    print(f"  output: {output_path}")

    writer = H5Writer(output_path)

    # Basic parameters
    writer.set_basic_params(
        depth=depth,
        angles=thetaobs,
        angles_opt=thetaopt,
        elements=elements,
        orbitals=orbitals,
        compounds=compounds,
        photon_energy=photon_energy,
        simulation=True,
        slice_size=slice_size,
        subst=subst,
        free=free,
        linked=linked,
        prpties=prpties,
        charges=charges,
    )

    # Apply the initialization results to the writer (transfer matrices, IMFP, ...)
    initializer.apply_to_writer(writer)

    # Profiles
    writer.set_profiles(
        Drlt=profile_result,
        Dtrn=profile_true,
        Dstk=profile_init,
        Xobs=init_results['Xobs'],
        Xopt=init_results['Xopt'],
    )

    # ★ Re-apply the initialization results after set_profiles (so they are not overwritten)
    initializer.apply_to_writer(writer)

    # Objective function (dummy values)
    writer.set_objectives(
        Qrlt=np.array([[1000.0, 0.0, 1000.0]]),
    )

    # Additional ezdeprof settings
    ez = writer._ezdeprof

    # Thickness parameters (4-layer structure)
    thck_init = 160.0  # position of the interface with the Si substrate
    abrp_init = 20.0   # interface abruptness
    ez['thckini'] = np.array([thck_init, 0.0, total_depth], dtype=np.float32)
    ez['abrpini'] = np.array([abrp_init, abrp_init, abrp_init], dtype=np.float32)
    ez['thckmod'] = ez['thckini'].copy()
    ez['abrpmod'] = ez['abrpini'].copy()

    # FeatureVector_Source (freeconst = 4 elements)
    # [thck, abrp, pos1, width1, height1, pos2, width2, height2, ...]
    n_freeconst = 4
    fv = np.zeros(2 + 3 * n_freeconst, dtype=np.float32)
    fv[0] = thck_init
    fv[1] = abrp_init
    # Parameters of each free element (reference values)
    free_params = [
        (thck_init, 60.0, 33.3),   # Si(+4): near the interface
        (120.0, 80.0, 33.3),       # Ti: TiO2 layer
        (50.0, 80.0, 33.3),        # Al: Al2O3 layer
        (10.0, 40.0, 80.0),        # C: surface contamination
    ]
    for i, (pos, width, height) in enumerate(free_params):
        fv[2 + 3*i + 0] = pos
        fv[2 + 3*i + 1] = width
        fv[2 + 3*i + 2] = height
    ez['FeatureVector_Source'] = fv
    ez['FeatureVector_Set'] = fv.copy()
    ez['FeatureVector_Calc'] = fv.copy()

    # SupportVector (substrate + linked)
    # [pos_Si, width_Si, height_Si, pos_O, width_O, height_O]
    sv = np.array([
        total_depth / 2, total_depth / 2, 100.0,  # Si substrate
        thck_init / 2, thck_init, 60.0,           # O (oxides)
    ], dtype=np.float32)
    ez['SupportVector_Set'] = sv
    ez['SupportVector_Calc'] = sv.copy()

    # Write
    writer.write()

    # =========================================================================
    # 6. Verification
    # =========================================================================
    print("\n" + "=" * 70)
    print("Verification")
    print("=" * 70)

    import h5py
    with h5py.File(output_path, 'r') as f:
        print("\n[Root datasets]")
        for key in ['Drlt', 'Dtrn', 'Dstk', 'Xrlt', 'Xstk', 'Qrlt']:
            if key in f:
                print(f"  {key}: {f[key].shape}")

        print("\n[Main ezdeprof fields]")
        ez = f['ezdeprof']
        for key in ['depth', 'thetaobs', 'elementsymbol', 'Dini', 'lambdabasic']:
            if key in ez:
                print(f"  {key}: {ez[key].shape}")

        print("\n[Material parameters]")
        print(f"  photon: {ez['photon'][()][0]} eV")
        print(f"  bindingenergy: {ez['bindingenergy'][:].flatten()}")
        print(f"  FeatureVector_Source: {ez['FeatureVector_Source'][:].flatten()}")

    print("\n" + "=" * 70)
    print("Done")
    print("=" * 70)
    print(f"""
Reading in MATLAB:
  cd '{output_dir}'

  % read in the GUI
  DepthProfiler

  % File > Read Result, choose example_hxps_multilayer.h5
  % Distribution function: choose 'Gaussian'
  % press Set and check the initial profile

  % or read the data directly
  store = deppro.io.DataStore('example_hxps_multilayer.h5');
  ez = store.getEzdeprof('full', true);
  D = store.getDrlt(1);
  plot(ez.depth, D');
  legend(ez.element);
""")


if __name__ == '__main__':
    main()
