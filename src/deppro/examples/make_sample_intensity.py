"""Regenerate ``sample_intensity.csv``, the synthetic intensity example.

The CSV is a noise-free forward calculation, not a measurement. Every input is
fixed here, so running this script reproduces the file:

- depth profile: :func:`deppro.create_five_layer_profile` with its defaults
  (C / Al2O3 / TiO2 / SiO2 / Si, 0-100 Å in 1 Å steps);
- channels: the profile's five elements, C 1s, O 1s, Al 2p, Ti 2p, Si 2p
  (Si 2p is the sum over the SiO2 layer and the substrate);
- photon energy 1486.6 eV (Al Kα), work function 4.5 eV;
- angles 0, 30, 45, 55, 65, 75 degrees;
- noise: none, so there is no random seed;
- intensities normalised to sum to 1 at each angle
  (``TransferMatrix.forward(..., normalize=True)``).

Compound and binding-energy values come from whichever tables deppro resolves:
toyomacro's bundled tables by default (``pip install 'deppro[fit]'``), or the
CSVs named by ``DEPPRO_COMPOUND_TABLE_PATH`` / ``DEPPRO_BINDING_ENERGY_TABLE_PATH``.
The committed file was made with toyomacro 0.3.1 (the release the ``fit`` extra
pins) and no variable set; the sources are written into the CSV's comment lines.
The version is read from the installed package metadata, which an editable
development checkout of toyomacro may not keep up to date.

Run::

    python -m deppro.examples.make_sample_intensity            # in a clone: rewrites the committed file
    python -m deppro.examples.make_sample_intensity out.csv    # anywhere: writes out.csv

In an installed package (not a clone) the CSV is not shipped, so an output
path is required.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PHOTON_ENERGY = 1486.6      # eV, Al Kα
WORK_FUNCTION = 4.5         # eV, the TransferMatrix default
ANGLES = np.array([0.0, 30.0, 45.0, 55.0, 65.0, 75.0])


def _toyomacro() -> str:
    try:
        from importlib.metadata import version
        return f"toyomacro {version('toyomacro')} bundled table"
    except Exception:  # noqa: BLE001
        return "toyomacro bundled table (version unknown)"


def table_source() -> str:
    """Name the compound and binding-energy table sources the run used, for the CSV header."""
    import os
    parts = []
    for what, var in (("compounds", "DEPPRO_COMPOUND_TABLE_PATH"),
                      ("binding energies", "DEPPRO_BINDING_ENERGY_TABLE_PATH")):
        path = os.environ.get(var)
        parts.append(f"{what}: CSV named by {var} ({Path(path).name})" if path
                     else f"{what}: {_toyomacro()}")
    return "; ".join(parts)


def make(path: Path) -> Path:
    from deppro import TransferMatrix, create_five_layer_profile
    from deppro.data_io import IntensityData, save_intensity_csv

    profile = create_five_layer_profile()
    tm = TransferMatrix(depth=profile.depth, step_size=profile.step_size, angles=ANGLES,
                        photon_energy=PHOTON_ENERGY, work_function=WORK_FUNCTION)
    tm.compute(profile.concentrations, profile.elements)
    intensities = tm.forward(profile.concentrations, normalize=True)
    ke = np.array([PHOTON_ENERGY - e.binding_energy - WORK_FUNCTION for e in profile.elements])

    data = IntensityData(angles=ANGLES, element_labels=list(profile.element_labels),
                         intensities=np.asarray(intensities), kinetic_energies=ke)
    save_intensity_csv(data, path)

    header = [
        "# Synthetic angle-resolved XPS intensities (noise-free forward calculation, not a measurement).",
        "# Regenerate with: python -m deppro.examples.make_sample_intensity",
        "# Profile: deppro.create_five_layer_profile() defaults (C / Al2O3 / TiO2 / SiO2 / Si, 0-100 A, 1 A steps).",
        f"# Photon energy {PHOTON_ENERGY} eV, work function {WORK_FUNCTION} eV, no noise, "
        "intensities normalised to 1 per angle.",
        f"# Tables: {table_source()}.",
        "# Columns: angle (degrees), then one column per element; the KE row gives kinetic energies (eV).",
    ]
    body = path.read_text()
    path.write_text("\n".join(header) + "\n" + body)
    return path


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv:
        out = Path(argv[0])
    else:
        from deppro.paths import in_checkout
        if not in_checkout():
            print("usage: python -m deppro.examples.make_sample_intensity OUT.csv "
                  "(an output path is required outside a clone of the repository)", file=sys.stderr)
            return 2
        out = Path(__file__).with_name("sample_intensity.csv")
    print(make(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
