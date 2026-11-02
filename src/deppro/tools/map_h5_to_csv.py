#!/usr/bin/env python3
"""
Convert MATLAB map_*.h5 intermediate files to CSV format for Python CLI.

Usage:
    python -m deppro.tools.map_h5_to_csv map_dataset.h5 -o intensity.csv
    python -m deppro.tools.map_h5_to_csv map_dataset.h5 --stack 0
    python -m deppro.tools.map_h5_to_csv map_dataset.h5 --all-stacks
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np


def load_map_h5(h5_path: str) -> dict:
    """Load map_*.h5 and return structured data."""
    with h5py.File(h5_path, "r") as f:
        data = {}
        data["Ismp"] = f["Ismp"][()]           # (stacks, elements, angles)
        data["thetaobs"] = f["thetaobs"][0]    # (angles,)
        data["BindingEnergy"] = f["BindingEnergy"][:, 0]  # (elements,)
        data["FermiEnergy"] = f["FermiEnergy"][0, 0]      # scalar

        # Element labels from StrData
        raw_labels = f["StrData"][:].flatten()
        data["labels_raw"] = [
            s.decode() if isinstance(s, bytes) else str(s) for s in raw_labels
        ]

        # Substrate index
        data["SubIndex"] = f["SubIndex"][:, 0]  # (elements,)

        # Stack count
        data["n_stacks"] = data["Ismp"].shape[0]
        data["n_elements"] = data["Ismp"].shape[1]
        data["n_angles"] = data["Ismp"].shape[2]

    return data


def make_element_labels(data: dict) -> list[str]:
    """Create clean element labels, disambiguating duplicate entries."""
    raw = data["labels_raw"]
    sub_idx = data["SubIndex"]

    labels = []
    seen = {}
    for i, raw_label in enumerate(raw):
        # Extract element symbol: "Si1s_dataset" -> "Si"
        parts = raw_label.split("1s")
        if parts:
            symbol = parts[0].split("2p")[0].split("2s")[0].split("3d")[0]
        else:
            symbol = raw_label

        # Clean up
        symbol = symbol.strip("_").strip()
        if not symbol:
            symbol = f"Elem{i}"

        # Track duplicates
        if symbol in seen:
            # Disambiguate: substrate vs oxide
            prev_i = seen[symbol]
            if sub_idx[prev_i] == 1:
                labels[prev_i] = f"{symbol}_subst"
                labels.append(f"{symbol}_ox")
            elif sub_idx[i] == 1:
                labels[prev_i] = f"{symbol}_ox"
                labels.append(f"{symbol}_subst")
            else:
                labels[prev_i] = f"{symbol}_1"
                labels.append(f"{symbol}_2")
        else:
            seen[symbol] = i
            labels.append(symbol)

    return labels


def convert_to_csv(
    h5_path: str,
    csv_path: str,
    stack_index: int | str = "mean",
    include_ke: bool = True,
) -> None:
    """Convert map_*.h5 to intensity CSV.

    Args:
        h5_path: Path to map_*.h5 file
        csv_path: Output CSV path
        stack_index: Stack to use - int for specific, "mean" for average
        include_ke: Whether to include kinetic energy row
    """
    data = load_map_h5(h5_path)
    labels = make_element_labels(data)
    angles = data["thetaobs"]

    # Select intensity data
    if stack_index == "mean":
        intensities = data["Ismp"].mean(axis=0)  # (elements, angles)
    else:
        intensities = data["Ismp"][int(stack_index)]  # (elements, angles)

    # Compute kinetic energies
    kinetic_energies = data["FermiEnergy"] - data["BindingEnergy"]

    # Write CSV
    with open(csv_path, "w") as f:
        # Header
        f.write("angle," + ",".join(labels) + "\n")

        # KE row (optional)
        if include_ke:
            ke_strs = [f"{ke:.1f}" for ke in kinetic_energies]
            f.write("KE," + ",".join(ke_strs) + "\n")

        # Data rows: one per angle
        for ai in range(len(angles)):
            row = [f"{angles[ai]:.3f}"]
            for ei in range(data["n_elements"]):
                row.append(f"{intensities[ei, ai]:.6f}")
            f.write(",".join(row) + "\n")

    print(f"Converted: {h5_path} -> {csv_path}")
    print(f"  Elements: {labels}")
    print(f"  Angles: {len(angles)} points ({angles[0]:.1f} - {angles[-1]:.1f} deg)")
    print(f"  Stack: {'mean of all' if stack_index == 'mean' else stack_index}")
    if include_ke:
        print(f"  KE range: {kinetic_energies.min():.0f} - {kinetic_energies.max():.0f} eV")


def convert_all_stacks(h5_path: str, output_dir: str, include_ke: bool = True) -> None:
    """Convert each stack to a separate CSV file."""
    data = load_map_h5(h5_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(h5_path).stem  # "map_dataset"

    # Mean
    csv_mean = out_dir / f"{stem}_mean.csv"
    convert_to_csv(h5_path, str(csv_mean), stack_index="mean", include_ke=include_ke)

    # Individual stacks
    for si in range(data["n_stacks"]):
        csv_si = out_dir / f"{stem}_stack{si}.csv"
        convert_to_csv(h5_path, str(csv_si), stack_index=si, include_ke=include_ke)


def main():
    parser = argparse.ArgumentParser(
        description="Convert MATLAB map_*.h5 to intensity CSV for Python CLI"
    )
    parser.add_argument("input", help="Input map_*.h5 file")
    parser.add_argument("-o", "--output", help="Output CSV file (default: auto-named)")
    parser.add_argument(
        "--stack",
        default="mean",
        help='Stack index (0-based) or "mean" for average (default: mean)',
    )
    parser.add_argument("--no-ke", action="store_true", help="Omit kinetic energy row")
    parser.add_argument(
        "--all-stacks",
        action="store_true",
        help="Export each stack as separate CSV",
    )

    args = parser.parse_args()

    if args.all_stacks:
        out_dir = args.output or str(Path(args.input).parent)
        convert_all_stacks(args.input, out_dir, include_ke=not args.no_ke)
    else:
        if args.output:
            csv_path = args.output
        else:
            stem = Path(args.input).stem.replace("map_", "intensity_")
            csv_path = str(Path(args.input).parent / f"{stem}.csv")

        stack = args.stack if args.stack == "mean" else int(args.stack)
        convert_to_csv(args.input, csv_path, stack_index=stack, include_ke=not args.no_ke)


if __name__ == "__main__":
    main()
