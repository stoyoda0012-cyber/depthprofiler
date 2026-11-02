"""``examples/sample_intensity.csv`` is reproduced by its generator script.

The CSV is a synthetic example whose inputs are fixed in
``examples/make_sample_intensity.py``. Regenerating it must give the committed
numbers: the same columns, angles, kinetic energies (to the 0.1 eV the file
keeps) and intensities (to the 1e-6 the file keeps). The comment line that
names the table source is not compared, because it records the toyomacro
version of the run.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import deppro
from deppro.data_io import load_intensity_csv

COMMITTED = Path(deppro.__file__).parent / "examples" / "sample_intensity.csv"


def test_committed_file_says_it_is_synthetic_and_how_to_regenerate():
    head = [l for l in COMMITTED.read_text().splitlines() if l.startswith("#")]
    text = "\n".join(head)
    assert "Synthetic" in text and "not a measurement" in text
    assert "python -m deppro.examples.make_sample_intensity" in text
    assert "no noise" in text


def test_regenerating_reproduces_the_committed_numbers(tmp_path, monkeypatch):
    pytest.importorskip("toyomacro.data", reason="toyomacro not installed (the generator reads its tables)")
    monkeypatch.delenv("DEPPRO_COMPOUND_TABLE_PATH", raising=False)
    monkeypatch.delenv("DEPPRO_BINDING_ENERGY_TABLE_PATH", raising=False)
    from deppro.examples.make_sample_intensity import make

    fresh = load_intensity_csv(make(tmp_path / "s.csv"))
    ref = load_intensity_csv(COMMITTED)
    assert fresh.element_labels == ref.element_labels == ["C1s", "O1s", "Al2p", "Ti2p", "Si2p"]
    np.testing.assert_array_equal(fresh.angles, ref.angles)
    np.testing.assert_allclose(fresh.kinetic_energies, ref.kinetic_energies, atol=0.05 + 1e-9)
    np.testing.assert_allclose(fresh.intensities, ref.intensities, atol=5e-7 + 1e-12)
    np.testing.assert_allclose(ref.intensities.sum(axis=1), 1.0, atol=1e-5)
