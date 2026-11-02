"""Conformance of `deppro.imfp.IMFP.tpp2m` to the published TPP-2M.

This file **changes nothing**. It records, as executable assertions,
where deppro's TPP-2M sits relative to (a) the primary source and
(b) the sibling implementation in `toyomacro.data.imfp`, which has been
audited against that source and is the intended canonical version.

Primary source
    S. Tanuma, C. J. Powell & D. R. Penn, *Surf. Interface Anal.* **21**,
    165 (1994), DOI 10.1002/sia.740210302. Equations (3), (4b), (4c),
    (4d), (4e) and (8):

        lambda = E / (Ep^2 [beta ln(gamma E) - C/E + D/E^2])   (3)
        gamma  = 0.191 rho^-0.5                                (4b)
        C      = 1.97 - 0.91 U                                 (4c)
        D      = 53.4 - 20.8 U                                 (4d)
        U      = Nv rho / M = Ep^2 / 829.4                     (4e)
        beta   = -0.10 + 0.944 (Ep^2+Eg^2)^-1/2 + 0.069 rho^0.1 (8)

    with Ep = 28.8 (Nv rho / M)^1/2. Paper p. 170-172.

Two deppro coefficients do not match that source:

    ==================  ============  ============  =================
    quantity            paper         deppro        effect on lambda
    ==================  ============  ============  =================
    Ep prefactor        28.8          28.816        -0.11%, all E
    Eqn (8) rho term    0.069         0.0691        -0.24 to -1.1%
    ==================  ============  ============  =================

Both push lambda *down*; combined they are -0.35% (Si) to -1.24%
(SiO2 at 50 eV). This is small, systematic, and below the formula's own
~10-19% RMS against optically-derived IMFPs — so it is a fidelity
question, not an accuracy one, and no result computed with deppro is
invalidated by it. The tests below pin the deviation so that it stays
known and bounded rather than drifting.

These are *pins*, not aspirations. Adopting the paper's 28.8 and 0.069
will make several of them fail — that is the intended handshake: change
the coefficient, tighten `MAX_DEVIATION_PCT` to ~0.0, and invert the two
"is not the paper's value" tests. Do not widen a bound to make a failure
go away.

No `toyomacro` import: deppro must not gain a hard dependency for this.
The toyomacro values are frozen here as literals instead, which is what
makes this a cross-repository conformance check.
"""

import math

import numpy as np
import pytest

from deppro.imfp import IMFP

# ------------------------------------------------------------------ #
# The shared golden grid. Identical materials and energies to
# toyomacro's tests/test_imfp_tpp2m.py, so the two repositories are
# demonstrably talking about the same numbers.
# ------------------------------------------------------------------ #
GOLDEN_MATERIALS = {
    "Si": (4, 2.33, 28.09, 1.12),
    "SiO2": (16, 2.2, 60.08, 8.9),
    "Au": (11, 19.3, 196.97, 0.0),
}
GOLDEN_ENERGIES = (50.0, 100.0, 500.0, 1000.0, 1386.6, 7413.0)

#: IMFP in Angstroms from `toyomacro.data.imfp`, at GOLDEN_ENERGIES.
#: Verified there against a transcription of Eqns (3)/(4)/(8) made
#: independently of either implementation.
TOYOMACRO_ANGSTROM = {
    "Si": (4.160667, 5.30576, 14.12216, 23.86375, 30.88695, 123.8463),
    "SiO2": (7.091791, 7.773316, 18.00757, 30.0803, 38.82754, 155.029),
    "Au": (4.859177, 3.859865, 7.651501, 12.31047, 15.65406, 59.03839),
}

#: Worst |deviation| of deppro from the source-faithful values, per
#: material, over the grid above. Tight enough that changing either
#: non-conforming coefficient trips it.
MAX_DEVIATION_PCT = {"Si": 0.45, "SiO2": 1.30, "Au": 0.95}


def _paper_imfp_angstrom(E, Nv, rho, M, Eg):
    """TPP-2M exactly as published, in Angstroms. Not imported from anywhere."""
    Ep = 28.8 * math.sqrt(Nv * rho / M)
    U = Nv * rho / M
    beta = -0.10 + 0.944 / math.sqrt(Ep**2 + Eg**2) + 0.069 * rho**0.1
    gamma = 0.191 * rho**-0.5
    C, D = 1.97 - 0.91 * U, 53.4 - 20.8 * U
    return E / (Ep**2 * (beta * math.log(gamma * E) - C / E + D / E**2))


# =============== 1. agreement with the canonical values ================

@pytest.mark.parametrize("material", sorted(GOLDEN_MATERIALS))
def test_toyomacro_literals_are_the_paper(material):
    """Guard the frozen literals themselves.

    If these ever stop matching a from-the-paper evaluation, the frozen
    table was mis-copied and every comparison below is meaningless.
    """
    Nv, rho, M, Eg = GOLDEN_MATERIALS[material]

    for energy_ev, expected in zip(GOLDEN_ENERGIES, TOYOMACRO_ANGSTROM[material]):
        assert _paper_imfp_angstrom(energy_ev, Nv, rho, M, Eg) == pytest.approx(
            expected, rel=1e-6
        )


@pytest.mark.parametrize("material", sorted(GOLDEN_MATERIALS))
def test_deppro_tracks_the_canonical_imfp_within_a_known_bound(material):
    """deppro agrees with toyomacro to within the coefficient deviation.

    A failure here means something beyond the two known coefficients has
    diverged — a different equation, a unit slip, or a material-parameter
    change — and needs diagnosis, not a wider tolerance.
    """
    Nv, rho, M, Eg = GOLDEN_MATERIALS[material]
    bound = MAX_DEVIATION_PCT[material]

    for energy_ev, canonical in zip(GOLDEN_ENERGIES, TOYOMACRO_ANGSTROM[material]):
        got = float(IMFP.tpp2m(energy_ev, Nv=Nv, density=rho, Aw=M, Eg=Eg))
        deviation_pct = 100.0 * (got - canonical) / canonical

        assert -bound < deviation_pct <= 0.0, (
            f"{material} at {energy_ev} eV: deppro {got:.4f} A vs canonical "
            f"{canonical:.4f} A ({deviation_pct:+.3f}%), outside the "
            f"documented [-{bound}%, 0%] window"
        )


# ================= 2. the two non-conforming coefficients ==============

def test_plasmon_prefactor_is_28_816_not_the_papers_28_8():
    """The 1994 paper gives Ep = 28.8 (Nv rho / M)^1/2 in the Eqn (3) text.

    This is a *defensible* divergence, unlike the 0.0691 below. The same
    authors later restate TPP-2M's parameters as U = Nv rho / M =
    (Ep/28.816)^2 — Shinotsuka, Tanuma, Powell & Penn, Surf. Interface
    Anal. 47, 871 (2015), Eqn (29e). 28.816 is also the physically
    correct prefactor (28.8159 from CODATA); 28.8 is that rounded and
    0.055% low. The 1994 fits used 28.8 (Eqn (4e)'s 829.4 is 28.8^2
    rounded, not 28.816^2 = 830.36), so following either is citable.

    Using 28.816 lowers lambda by 0.111% at every energy relative to the
    1994 convention that toyomacro follows. Pinned so the choice stays
    deliberate rather than accidental.
    """
    Nv, rho, M = GOLDEN_MATERIALS["Si"][:3]

    _, params = IMFP.tpp2m_with_params(1000.0, Nv=Nv, density=rho, Aw=M, Eg=1.12)

    assert params["Ep"] == pytest.approx(28.816 * math.sqrt(Nv * rho / M), rel=1e-12)
    assert params["Ep"] != pytest.approx(28.8 * math.sqrt(Nv * rho / M), rel=1e-6)
    assert (28.816 / 28.8) ** 2 - 1 == pytest.approx(0.001111, abs=1e-6)


def test_eqn8_density_term_is_0_0691_not_the_papers_0_069():
    """Eqn (8): beta_M = -0.10 + 0.944 (Ep^2+Eg^2)^-1/2 + 0.069 rho^0.1.

    Unlike the 28.816 above, this one has no source. The 1994 paper
    prints 0.069 in Eqns (7) and (8); the authors' 2015 restatement
    gives the same coefficient as 0.69 in nm^-1 units, i.e. 0.069 in
    A^-1 (Shinotsuka et al., SIA 47, 871, Eqn (29a)). Both agree.
    deppro's 0.0691 matches neither.

    beta is small for wide-gap oxides, so the same absolute offset is a
    larger relative one there — hence SiO2's wider bound than Si's.
    """
    for material in GOLDEN_MATERIALS:
        Nv, rho, M, Eg = GOLDEN_MATERIALS[material]
        _, params = IMFP.tpp2m_with_params(1000.0, Nv=Nv, density=rho, Aw=M, Eg=Eg)

        Ep = params["Ep"]
        paper_beta = -0.10 + 0.944 / math.sqrt(Ep**2 + Eg**2) + 0.069 * rho**0.1

        assert params["beta"] - paper_beta == pytest.approx(0.0001 * rho**0.1, rel=1e-9)
        assert params["beta"] > paper_beta  # so lambda comes out low


def test_U_is_nv_rho_over_m_despite_being_spelled_as_ep_squared():
    """Eqn (4e). deppro writes U = (Ep/28.816)^2, which is the same thing.

    The identity is exact only because the *same* 28.816 appears in both
    places. It is worth pinning: substituting anything else for U here is
    the transcription error that was found and fixed in toyomacro on
    2026-07-29 (it had used Ep/1000), where it inflated Au's IMFP at
    50 eV by ~900%.
    """
    for material in GOLDEN_MATERIALS:
        Nv, rho, M, Eg = GOLDEN_MATERIALS[material]
        _, params = IMFP.tpp2m_with_params(1000.0, Nv=Nv, density=rho, Aw=M, Eg=Eg)

        assert params["U"] == pytest.approx(Nv * rho / M, rel=1e-12)
        assert params["C"] == pytest.approx(1.97 - 0.91 * Nv * rho / M, rel=1e-12)
        assert params["D"] == pytest.approx(53.4 - 20.8 * Nv * rho / M, rel=1e-12)


def test_gamma_is_eqn4b_exactly():
    """gamma = 0.191 rho^-0.5 — this one does match the paper."""
    for material in GOLDEN_MATERIALS:
        Nv, rho, M, Eg = GOLDEN_MATERIALS[material]
        _, params = IMFP.tpp2m_with_params(1000.0, Nv=Nv, density=rho, Aw=M, Eg=Eg)

        assert params["gamma"] == pytest.approx(0.191 * rho**-0.5, rel=1e-12)


# ========================= 3. interface contract =======================

def test_the_return_unit_is_angstroms():
    """deppro returns Angstroms; toyomacro returns nanometres.

    Both are defensible — deppro's transfer matrix works in Angstroms
    throughout — but the factor of 10 is the single most likely thing to
    go wrong if these are ever unified, so pin it against a magnitude
    that could not be mistaken for the other unit.
    """
    Nv, rho, M, Eg = GOLDEN_MATERIALS["Si"]

    got = float(IMFP.tpp2m(1000.0, Nv=Nv, density=rho, Aw=M, Eg=Eg))

    assert 20.0 < got < 30.0  # ~23.8 A, i.e. ~2.4 nm
    assert got == pytest.approx(TOYOMACRO_ANGSTROM["Si"][3], rel=0.005)


def test_array_input_is_elementwise_and_matches_scalar():
    """deppro is vectorised over energy; toyomacro is scalar-only.

    Any future wrapper has to reintroduce this, so pin that it is plain
    elementwise evaluation with no broadcasting surprises.
    """
    Nv, rho, M, Eg = GOLDEN_MATERIALS["Au"]
    energies = np.array(GOLDEN_ENERGIES)

    vectorised = IMFP.tpp2m(energies, Nv=Nv, density=rho, Aw=M, Eg=Eg)

    assert isinstance(vectorised, np.ndarray)
    assert vectorised.shape == energies.shape
    for i, energy_ev in enumerate(GOLDEN_ENERGIES):
        assert vectorised[i] == pytest.approx(
            float(IMFP.tpp2m(energy_ev, Nv=Nv, density=rho, Aw=M, Eg=Eg)), rel=1e-12
        )


def test_version_3_is_tpp2m():
    """`calculate(..., version=3)` and the default dispatch to TPP-2M."""
    Nv, rho, M, Eg = GOLDEN_MATERIALS["SiO2"]
    kw = dict(Nv=Nv, density=rho, Aw=M, Eg=Eg)

    direct = float(IMFP.tpp2m(1000.0, **kw))

    assert float(IMFP.calculate(1000.0, Nv, rho, M, Eg)) == pytest.approx(direct)
    assert float(IMFP.calculate(1000.0, Nv, rho, M, Eg, version=3)) == pytest.approx(
        direct
    )
    # TPP-2 uses Eqn (4a) for beta and must differ appreciably.
    assert float(IMFP.calculate(1000.0, Nv, rho, M, Eg, version=2)) != pytest.approx(
        direct, rel=0.01
    )


# =================== 4. material-parameter divergence ==================

# Entries where CompoundTable.csv (deppro) and compounds.json
# (toyomacro) disagree, with the resulting spread in lambda at 1000 eV.
# Independent of the formula: same equations, different input constants.
KNOWN_DB_DIVERGENCE = {
    #          (Nv, rho, M, Eg) as deppro has them        lambda spread
    "SiO2": ((16.0, 2.19, 60.008, 9.1), 1.03),
    "Al2O3": ((24.0, 3.97, 101.961, 8.63), 2.71),
    "SiC": ((8.0, 3.22, 40.096, 2.31), -1.11),
    "GaAs": ((8.0, 5.32, 144.645, 1.47), 0.01),
}


@pytest.mark.parametrize("name", sorted(KNOWN_DB_DIVERGENCE))
def test_compound_table_values_are_as_recorded(name):
    """Pin deppro's own material constants for the divergent entries.

    Not a claim that either table is right — Al2O3's band gap and SiC's
    (polytype-dependent) gap are genuinely reported over a range in the
    literature. The point is that a 1-3% IMFP difference between the two
    repositories can come from the *inputs* alone, so it must never be
    read as evidence about the formula.
    """
    import csv
    from pathlib import Path

    from deppro.imfp import CompoundDatabase

    # The CSV is selected only through DEPPRO_COMPOUND_TABLE_PATH (the default
    # compound source has been toyomacro since 2026-08-03). Without it there
    # is no CSV to pin, so skip explicitly -- a skip here is NOT a pass.
    csv_path = CompoundDatabase._get_override_csv_path()
    if csv_path is None:
        pytest.skip("DEPPRO_COMPOUND_TABLE_PATH unset: CSV table not selected")
    if not Path(csv_path).exists():
        pytest.skip(f"CompoundTable.csv not present at {csv_path}")

    expected, _ = KNOWN_DB_DIVERGENCE[name]
    with open(csv_path, encoding="utf-8") as f:
        rows = {r["CompoundName"]: r for r in csv.DictReader(f)}

    row = rows[name]
    got = (
        float(row["ValenceCharge"]),
        float(row["Density [g/cm3]"]),
        float(row["MolecularWeight [g/mol]"]),
        float(row["BandGap [eV]"]),
    )

    assert got == pytest.approx(expected, rel=1e-9)
