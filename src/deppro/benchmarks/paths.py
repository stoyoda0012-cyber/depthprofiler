"""
Canonical benchmark data paths for DepthProfiler.

All benchmark scripts should import paths from here rather than
hardcoding them. The folder convention is:

    Simulation/DepthProfiler/{Standard,Checkered}/{Nx}x{Ny}x{Nt}_{N}elem[_variant]/

Standard = 6 elements (OCAlTiSi), Checkered = 7 elements (OBCAlTiSi, B added).
Nt = time frames (stack / (Nx*Ny)), depth info in H5 metadata.

Each dated ``GT_*`` path is the main H5 of a dataset folder, found at import
with the same rule as the dataset preflight
(``deppro.app.contract.dataset_files.discover_files``: the H5 whose stem
matches the .prj, else the only non-spectrum H5, else the only file named by
the producer pattern). If the folder is missing or the main H5 is not unique,
the value is a placeholder path ``<folder>/<main-h5-not-found>`` so that the
error appears when the file is opened, not on import. ``GT_50x50*`` are
named files, not dated main H5s.
"""

from pathlib import Path

from deppro.paths import legacy_simulation_root

# Root of all benchmark data: DEPPRO_SIMULATION_ROOT if set, else the
# home-relative location these scripts have always used.
BENCHMARK_ROOT = legacy_simulation_root()

MAIN_H5_NOT_FOUND = "<main-h5-not-found>"


def main_h5(folder: Path) -> str:
    """Main H5 of a dataset folder, or the placeholder path if it cannot be found uniquely."""
    from deppro.app.contract.dataset_files import discover_files

    folder = Path(folder)
    if folder.is_dir():
        main = discover_files(folder)["main"]
        if len(main) == 1:
            return str(main[0])
    return str(folder / MAIN_H5_NOT_FOUND)


# --- Standard (6-element) ---------------------------------------------------
GT_6ELEM = main_h5(BENCHMARK_ROOT / "Standard/5x5x100_6elem")
GT_6ELEM_NOISY = main_h5(BENCHMARK_ROOT / "Standard/5x5x100_6elem_noisy")
GT_6ELEM_100x100 = main_h5(BENCHMARK_ROOT / "Standard/100x100x200_6elem")

# 10x10x200 variants (jitter, noisy)
GT_6ELEM_10x10 = main_h5(BENCHMARK_ROOT / "Standard/10x10x200_6elem")
GT_6ELEM_JITTER = main_h5(BENCHMARK_ROOT / "Standard/10x10x200_6elem_jitter")
GT_6ELEM_JITTER_NOISY = main_h5(BENCHMARK_ROOT / "Standard/10x10x200_6elem_jitter_noisy")

# --- Checkered (7-element) ---------------------------------------------------
GT_7ELEM = main_h5(BENCHMARK_ROOT / "Checkered/10x10x200_7elem")
GT_50x50 = str(
    BENCHMARK_ROOT / "Checkered/50x50x1_7elem/50x50_python_checkered.h5"
)
GT_50x50_MATLAB = str(
    BENCHMARK_ROOT / "Checkered/50x50x1_7elem/50x50_matlab_v2.h5"
)

# --- Legacy Checkered path (500x500, blocked/conference) ---------------------
# Lives beside BENCHMARK_ROOT (Simulation/Checkered), not under it.
GT_500x500 = main_h5(BENCHMARK_ROOT.parent / "Checkered/500x500x200x20x6")
