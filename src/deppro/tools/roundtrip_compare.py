#!/usr/bin/env python3
"""
MATLAB vs Python depth profiling round-trip comparison.

This script:
1. Loads map_*.h5 (MATLAB intermediate) → extracts angle-resolved intensities
2. Runs Python SemiFreeSolver (L1, α=1.0, D_prior=None)
3. Loads MATLAB result (YYMMDD*.h5) → extracts Dopt
4. Compares Python vs MATLAB depth profiles

Usage:
    python -m deppro.tools.roundtrip_compare [--map MAP_H5] [--result RESULT_H5]

Without arguments, both files are taken from DEPPRO_TESTDATA_ROOT: the only
``map_*.h5`` there, and the only MATLAB result named ``YYMMDD<elements>.h5``.
"""

import argparse
import re
import sys
from pathlib import Path

import h5py
import numpy as np

# Add parent to path if running as script
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from deppro.depth_profile import ElementConfig
from deppro.inverse_solver import SemiFreeSolver, SemiFreeSolverConfig, SolverConfig
from deppro.transfer_matrix import TransferMatrix
from deppro.paths import env_path_or_placeholder


# ============================================================
# Configuration
# ============================================================
# MATLAB DepthProfiler test data dir (DEPPRO_TESTDATA_ROOT). Unset: import
# succeeds; loading fails with a path naming the variable.
TEST_DATA = env_path_or_placeholder("DEPPRO_TESTDATA_ROOT")
RESULT_NAME_RE = re.compile(r"^\d{6}[A-Za-z]+\.h5$")   # MATLAB result: YYMMDD + element symbols


def _only(folder: Path, match, what: str) -> Path:
    """The single file in ``folder`` accepted by ``match``, else a placeholder path naming ``what``."""
    found = sorted(p for p in folder.glob("*.h5") if match(p.name)) if folder.is_dir() else []
    return found[0] if len(found) == 1 else folder / f"<{what}: {len(found)} candidates>"


MAP_FILE = _only(TEST_DATA, lambda n: n.startswith("map_"), "map_*.h5")
MATLAB_RESULT = _only(TEST_DATA, RESULT_NAME_RE.match, "YYMMDD*.h5")
from deppro.paths import outputs_dir as _outputs_dir  # noqa: E402
OUTPUT_DIR = _outputs_dir() / "roundtrip_test"


def load_map_h5(h5_path: Path) -> dict:
    """Load map_*.h5 and return structured data."""
    with h5py.File(h5_path, "r") as f:
        data = {}
        data["Ismp"] = f["Ismp"][()]             # (stacks, elements, angles)
        data["thetaobs"] = f["thetaobs"][0]      # (angles,)
        data["BindingEnergy"] = f["BindingEnergy"][:, 0]  # (elements,)
        data["FermiEnergy"] = f["FermiEnergy"][0, 0]
        data["SubIndex"] = f["SubIndex"][:, 0]

        raw_labels = f["StrData"][:].flatten()
        data["labels_raw"] = [
            s.decode() if isinstance(s, bytes) else str(s) for s in raw_labels
        ]
    return data


def load_matlab_result(h5_path: Path) -> dict:
    """Load MATLAB depth profiling result from YYMMDD*.h5."""
    with h5py.File(h5_path, "r") as f:
        ezd = f["ezdeprof"]
        data = {}
        data["Dopt"] = ezd["Dopt"][()]           # (elements, depth) in MATLAB format
        data["depth"] = ezd["depth"][()].flatten()
        data["element"] = [
            s.decode() if isinstance(s, bytes) else str(s)
            for s in ezd["element"][:].flatten()
        ]
        data["thetaobs"] = ezd["thetaobs"][()].flatten()
        data["Xobs"] = ezd["Xobs"][()]           # (elements, angles)
        data["TransMatObs"] = ezd["TransMatObs"][()]  # (elements, depth, angles)
        data["lambdabasic"] = ezd["lambdabasic"][()]
        data["charge"] = ezd["charge"][:].flatten()
        data["totaldepth"] = float(ezd["totaldepth"][()])
        data["stepsize"] = float(ezd["stepsize"][()])
        data["photon"] = float(ezd["photon"][()])

        # Element classification
        data["free"] = ezd["free"][:].flatten()
        data["subst"] = ezd["subst"][:].flatten()
        data["linked"] = ezd["linked"][:].flatten()

        # Binding energies
        data["bindingenergy"] = ezd["bindingenergy"][:].flatten()

        # Compounds
        data["compound"] = [
            s.decode() if isinstance(s, bytes) else str(s)
            for s in ezd["compound"][:].flatten()
        ]
    return data


def run_python_solver(
    angles: np.ndarray,
    I_obs: np.ndarray,
    elements: list[ElementConfig],
    depth: np.ndarray,
    step_size: float,
    photon_energy: float,
    free_idx: list[int],
    linked_idx: list[int],
    sub_idx: list[int],
) -> dict:
    """Run Python SemiFreeSolver with L1, α=1.0, D_prior=None."""

    n_elements = len(elements)
    n_depth = len(depth)

    # Build initial depth profile (uniform)
    D_init = np.full((n_depth, n_elements), 1.0 / n_elements)

    # Create transfer matrix
    tm = TransferMatrix(
        depth=depth,
        step_size=step_size,
        angles=angles,
        photon_energy=photon_energy,
        work_function=4.5,
        gamma_eff=1.0,
    )
    tm.compute(D_init, elements)

    print(f"\n  Transfer matrix shape: {tm.trans_mat.shape}")
    print(f"  Lambda basic (diagonal):")
    for i, el in enumerate(elements):
        print(f"    {el.symbol}({el.compound}): {tm.lambda_basic[i, i]:.2f} Å")

    # Configure solver: L1, α=1.0, D_prior=None (→ zero prior)
    solver_config = SolverConfig(
        method="L1",
        alpha=1.0,
        max_iter=100,
        tol=1e-8,
        scipy_method="SLSQP",
    )

    semi_free_config = SemiFreeSolverConfig(
        free_indices=free_idx,
        linked_indices=linked_idx,
        substrate_indices=sub_idx,
        thickness_init=200.0,       # Closer to expected ~200 Å
        abruptness_init=30.0,       # ~30 Å interface width
        optimize_thickness=True,
        optimize_abruptness=False,
        thickness_bounds=(10.0, 500.0),  # Wide bounds for 600 Å grid
        use_charge_constraint=True,
        use_sum_constraint=True,
    )

    solver = SemiFreeSolver(
        transfer_matrix=tm,
        config=solver_config,
        semi_free_config=semi_free_config,
        D_prior=None,  # Zero prior for L1 sparsity
        elements=elements,
        use_numba=True,
    )

    # Normalize intensities (angle-wise)
    I_norm = I_obs.copy()
    for a in range(I_obs.shape[0]):
        row_sum = I_obs[a].sum()
        if row_sum > 0:
            I_norm[a] = I_obs[a] / row_sum

    print(f"\n  Normalized intensities shape: {I_norm.shape}")
    print(f"  Running SemiFreeSolver (L1, α=1.0, D_prior=None)...")

    result = solver.solve(I_norm, verbose=True)

    print(f"  Success: {result.success}")
    print(f"  Thickness: {result.thickness:.2f} Å")
    print(f"  Chi²: {result.chi2:.6f}")

    return {
        "D_opt": result.D_opt,
        "thickness": result.thickness,
        "chi2": result.chi2,
        "result": result,
        "tm": tm,
    }


def compare_and_plot(
    matlab_data: dict,
    python_result: dict,
    output_dir: Path,
):
    """Compare MATLAB and Python results and generate plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    # MATLAB data
    D_matlab = matlab_data["Dopt"]  # (elements, depth)
    depth_matlab = matlab_data["depth"]
    elements_matlab = matlab_data["element"]

    # Python data
    D_python = python_result["D_opt"]  # (depth, elements)

    # Use MATLAB depth grid for comparison
    depth_py = np.arange(0, D_python.shape[0]) * (depth_matlab[-1] - depth_matlab[0]) / (D_python.shape[0] - 1) if D_python.shape[0] != len(depth_matlab) else depth_matlab

    print("\n" + "=" * 60)
    print("Comparison: MATLAB vs Python")
    print("=" * 60)
    print(f"  MATLAB: {D_matlab.shape} (elements, depth)")
    print(f"  Python: {D_python.shape} (depth, elements)")
    print(f"  MATLAB depth: {depth_matlab[0]:.1f} - {depth_matlab[-1]:.1f} Å ({len(depth_matlab)} pts)")
    print(f"  MATLAB elements: {elements_matlab}")

    # Element mapping
    # MATLAB: ['Si(0)', 'Si(+4)', 'Ti(+4)', 'Al(+3)', 'C(0)', 'O(-2)'] or similar
    # Python: depends on ElementConfig order

    # Plot each element
    n_elem_matlab = D_matlab.shape[0]
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i in range(min(n_elem_matlab, 6)):
        ax = axes[i]

        # MATLAB profile
        ax.plot(depth_matlab, D_matlab[i, :], "b-o", markersize=3,
                label=f"MATLAB: {elements_matlab[i]}", linewidth=2)

        # Python profile (if element count matches)
        if i < D_python.shape[1]:
            depth_py_i = np.linspace(depth_matlab[0], depth_matlab[-1], D_python.shape[0])
            ax.plot(depth_py_i, D_python[:, i], "r-", linewidth=2,
                    label=f"Python: elem[{i}]")

        ax.set_xlabel("Depth (Å)")
        ax.set_ylabel("Concentration")
        ax.set_title(elements_matlab[i])
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Hide unused axes
    for i in range(n_elem_matlab, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle("MATLAB vs Python Depth Profile Comparison\n"
                 f"MATLAB: {MATLAB_RESULT.name}, Python: SemiFreeSolver (L1, α=1.0)",
                 fontsize=12)
    plt.tight_layout()

    plot_path = output_dir / "matlab_vs_python_comparison.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Plot saved: {plot_path}")

    # Print numerical comparison
    print("\n  Element-wise RMSE (where comparable):")
    for i in range(min(n_elem_matlab, D_python.shape[1])):
        # Interpolate Python to MATLAB depth grid
        from scipy.interpolate import interp1d
        depth_py_i = np.linspace(depth_matlab[0], depth_matlab[-1], D_python.shape[0])
        interp_fn = interp1d(depth_py_i, D_python[:, i], kind="linear", fill_value="extrapolate")
        D_py_interp = interp_fn(depth_matlab)

        rmse = np.sqrt(np.mean((D_matlab[i, :] - D_py_interp) ** 2))
        corr = np.corrcoef(D_matlab[i, :], D_py_interp)[0, 1] if np.std(D_matlab[i, :]) > 1e-10 else 0.0
        print(f"    {elements_matlab[i]:12s}: RMSE={rmse:.6f}, corr={corr:.4f}")


def main(argv=None):
    global MAP_FILE, MATLAB_RESULT
    parser = argparse.ArgumentParser(description="MATLAB vs Python depth profiling round-trip comparison")
    parser.add_argument("--map", type=Path, default=MAP_FILE, help="MATLAB intermediate map_*.h5")
    parser.add_argument("--result", type=Path, default=MATLAB_RESULT, help="MATLAB result YYMMDD*.h5")
    args = parser.parse_args(argv)
    MAP_FILE, MATLAB_RESULT = args.map, args.result

    print("=" * 60)
    print("MATLAB vs Python Depth Profiling Round-Trip Comparison")
    print("=" * 60)

    # ========================================
    # Step 1: Load map_*.h5
    # ========================================
    print(f"\n[Step 1] Loading map file: {MAP_FILE.name}")
    map_data = load_map_h5(MAP_FILE)

    angles = map_data["thetaobs"]
    I_mean = map_data["Ismp"].mean(axis=0)  # (elements, angles) averaged over stacks
    fermi = map_data["FermiEnergy"]
    binding = map_data["BindingEnergy"]

    print(f"  Angles: {len(angles)} points ({angles[0]:.1f} - {angles[-1]:.1f} deg)")
    print(f"  Elements: {map_data['labels_raw']}")
    print(f"  SubIndex: {map_data['SubIndex']}")
    print(f"  Intensity shape: {I_mean.shape}")

    # ========================================
    # Step 2: Load MATLAB result for reference
    # ========================================
    print(f"\n[Step 2] Loading MATLAB result: {MATLAB_RESULT.name}")
    matlab_data = load_matlab_result(MATLAB_RESULT)

    print(f"  Elements: {matlab_data['element']}")
    print(f"  Compounds: {matlab_data['compound']}")
    print(f"  Charges: {matlab_data['charge']}")
    print(f"  Depth: {matlab_data['depth'][0]:.0f} - {matlab_data['depth'][-1]:.0f} Å")
    print(f"  Step size: {matlab_data['stepsize']:.0f} Å")
    print(f"  Photon energy: {matlab_data['photon']:.0f} eV")
    print(f"  Free: {matlab_data['free']}")
    print(f"  Subst: {matlab_data['subst']}")
    print(f"  Linked: {matlab_data['linked']}")

    # MATLAB lambda_basic
    print(f"  Lambda basic shape: {matlab_data['lambdabasic'].shape}")
    lb = matlab_data["lambdabasic"]
    if lb.ndim == 3:
        lb = lb[0]  # (1, nElem, nElem) → (nElem, nElem)
    print(f"  Lambda basic (diagonal):")
    for i, el in enumerate(matlab_data["element"]):
        print(f"    {el}: {lb[i, i]:.2f} Å")

    # ========================================
    # Step 3: Configure Python solver
    # ========================================
    print(f"\n[Step 3] Configuring Python solver")

    # Element configuration matching MATLAB
    # MATLAB order: Si(0), Si(+4), Ti(+4), Al(+3), C(0), O(-2)
    # with compounds: Si, SiO2, TiO2, Al2O3, C, Oxide
    elements = [
        ElementConfig(symbol="Si", orbital="1s", compound="Si",
                      binding_energy=binding[0], oxidation_state=0),
        ElementConfig(symbol="Si", orbital="1s", compound="SiO2",
                      binding_energy=binding[1], oxidation_state=4),
        ElementConfig(symbol="Ti", orbital="1s", compound="TiO2",
                      binding_energy=binding[2], oxidation_state=4),
        ElementConfig(symbol="Al", orbital="1s", compound="Al2O3",
                      binding_energy=binding[3], oxidation_state=3),
        ElementConfig(symbol="C", orbital="1s", compound="C",
                      binding_energy=binding[4], oxidation_state=0),
        ElementConfig(symbol="O", orbital="1s", compound="SiO2",
                      binding_energy=binding[5], oxidation_state=-2),
    ]

    # MATLAB classification uses 1-based indices stored as arrays of element numbers
    # e.g., free=[2,3,4,5] means elements 2,3,4,5 are free (1-based)
    # Convert to 0-based indices
    free_vals = matlab_data["free"]    # e.g., [2, 3, 4, 5] (1-based)
    subst_vals = matlab_data["subst"]  # e.g., [1] (1-based)
    linked_vals = matlab_data["linked"]  # e.g., [6] (1-based)

    free_idx = [int(v) - 1 for v in free_vals]    # 1-based → 0-based
    sub_idx = [int(v) - 1 for v in subst_vals]
    linked_idx = [int(v) - 1 for v in linked_vals]

    print(f"  Free indices: {free_idx}")
    print(f"  Substrate indices: {sub_idx}")
    print(f"  Linked indices: {linked_idx}")

    # Use MATLAB depth grid
    depth = matlab_data["depth"]
    step_size = matlab_data["stepsize"]
    photon_energy = matlab_data["photon"]

    # Transpose I_mean to (angles, elements) for solver
    I_obs = I_mean.T  # (angles, elements)

    # ========================================
    # Step 4: Run Python solver
    # ========================================
    print(f"\n[Step 4] Running Python solver")
    python_result = run_python_solver(
        angles=angles,
        I_obs=I_obs,
        elements=elements,
        depth=depth,
        step_size=step_size,
        photon_energy=photon_energy,
        free_idx=free_idx,
        linked_idx=linked_idx,
        sub_idx=sub_idx,
    )

    # ========================================
    # Step 5: Compare results
    # ========================================
    print(f"\n[Step 5] Comparing results")
    compare_and_plot(matlab_data, python_result, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
