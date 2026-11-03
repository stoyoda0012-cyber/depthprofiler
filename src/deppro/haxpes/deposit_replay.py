"""Re-computation of published metrics from the papers' Zenodo deposit.

This module closes **one** of the three kinds of reproduction that the
JVST A / SIA papers admit, and only that one:

    stored-result check  — recompute the published tables and annotations
                           from the *deposited* reconstructed depth profiles
                           ``D_map`` and pseudo-ground-truth ``D_ref``.

It does **not** re-run the denoiser, the peak fit or the depth inversion.
A pass here says: *given the arrays the authors deposited, the metric
definitions below reproduce the numbers printed in the papers*. It says
nothing about whether those arrays follow from the raw measurements, and
nothing about re-training.

Inputs are the two archives of the deposit, version 1.0.0:

    concept DOI  10.5281/zenodo.22092108   (resolves to the latest version)
    version DOI  10.5281/zenodo.22092109   (v1.0.0, what this module expects)

extracted anywhere. Every deposited file this module reads is pinned by the
SHA-256 the deposit itself publishes in ``SHA256SUMS.txt``; a file that does
not match **voids the comparison** (an error is raised) instead of being
compared anyway.

The aggregation formulas are transcribed from the figure scripts shipped
verbatim in ``sia_figure_data/scripts/`` (``fig4_hax26_equiv_exposure.py``,
``figS9_seed_distribution.py``, ``figS10_seed_ordering.py``) and, for the JVST
metric table, from the authors' analysis script that produced the ``metrics``
JSON stored in ``figure_data.h5``. Where the deposit does not define a stored
quantity, the recomputation is labeled *inferred* in the output rather than
presented as the published definition.

The published reference values, and the rule for calling a cell "matched",
are fixed in :data:`PUBLISHED_SIA` **before** any run: a K_eq matches when it
rounds to the printed integer, a depth RMSE / W1 matches when it rounds to the
printed one-decimal value. Nothing here is tuned to a result.

Run::

    python -m deppro.haxpes.deposit_replay --sia-dir <…/sia_figure_data> \\
        --jvst-h5 <…/jvst_figure_data/figure_data.h5> --out replay.json

``deppro.haxpes.config`` is deliberately not imported: it reads
``DEPPRO_FIX_IBW_DIR`` / ``DEPPRO_FIX_PXT_DIR`` / ``DEPPRO_MAP_H5`` /
``DEPPRO_OUTPUTS``, and this replay must run from the deposit alone.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from deppro.depth_profile import wasserstein_distance_1d

__all__ = [
    "DEPOSIT_CONCEPT_DOI",
    "DEPOSIT_VERSION_DOI",
    "DEPTH",
    "ELEMENTS",
    "SIA_DATASETS",
    "SIA_FILES",
    "JVST_FILES",
    "PUBLISHED_SIA",
    "DepositMismatch",
    "sha256_of",
    "verify_files",
    "depth_rmse_percent",
    "per_pixel_rmse_percent",
    "across_seed_sd",
    "w1_median_mean",
    "fit_frame_average",
    "equivalent_k",
    "replay_sia_fig4",
    "replay_sia_seed_pilot",
    "replay_jvst_metrics",
    "environment_record",
    "main",
]

# ----------------------------------------------------------------------------
# The deposit, pinned
# ----------------------------------------------------------------------------

DEPOSIT_CONCEPT_DOI = "10.5281/zenodo.22092108"
DEPOSIT_VERSION_DOI = "10.5281/zenodo.22092109"
DEPOSIT_VERSION = "1.0.0"

#: Archive-level MD5 as served by the Zenodo record API for v1.0.0, for the
#: two archives this module reads from (the third archive of the record,
#: ``software_arhaxpes_denoise.zip``, md5 f8ac7f5ca4abafb30239ebe1d3d0b217, is
#: not read here). Recorded for the manifest; the per-file SHA-256 below is
#: what is actually enforced.
ARCHIVE_MD5 = {
    "sia_figure_data.zip": "09180ea25f0ea9fa0b42569dc321717a",
    "jvst_figure_data.zip": "fe345e7bcd6697a74e89ccc7eb81ec72",
}

#: Axis conventions stated in both DATA_READMEs of the deposit.
DEPTH = np.arange(1, 21) * 30.0          # Å, 30–600 Å, 20 points
ELEMENTS = ("Si", "Ti", "Al", "C", "O")  # last axis of D_map / D_ref

#: Fig. 4 of the SIA paper: what the frame-average ladder and the DNN points
#: are read from, transcribed from ``fig4_hax26_equiv_exposure.py``.
#: ``k_values`` is the ladder the script asks for (k = 1 included; the fit
#: form absorbs photon starvation through its offset, and ``fit_k_min`` of
#: the script is declared but never applied).
SIA_DATASETS = {
    "4.8sec": {"gt": "outputs/hax26_5sec_pseudo_gt.npz",
               "k_values": (1, 5, 10, 25, 50, 125),
               "best_cross": "24sec"},
    "24sec": {"gt": "outputs/hax26_24sec_pseudo_gt.npz",
              "k_values": (1, 5, 10, 25, 50, 125),
              "best_cross": "120sec"},
    "120sec": {"gt": "outputs/hax26_120sec_pseudo_gt.npz",
               "k_values": (1, 5, 10, 25, 50, 125),
               "best_cross": None},
}

#: SHA-256 of every file under ``sia_figure_data/`` that this module reads,
#: copied from the deposit's ``SHA256SUMS.txt`` (record root, v1.0.0).
SIA_FILES: Dict[str, str] = {
    "outputs/hax26_120sec_pseudo_gt.npz":
        "c89a51ef443169ecc7c01f0e64b1aa8e4f623e4fa9be346d412f62ffc8a203f3",
    "outputs/hax26_24sec_pseudo_gt.npz":
        "6a10cb8dd2341fc42138970426441a795ea4fb419d9d16a452c7488f91aec58a",
    "outputs/hax26_5sec_pseudo_gt.npz":
        "9d571f207799dc71a755bff1857408a75a96d92fd0dac21fb82914916cdc5da2",
    "outputs/hax26_selfdn_cache/120sec_cross_from_24sec.npz":
        "8e6e8a28f4d688a5eaf7f1d7a8f928d2f4e70d3445d6c984ba4f3ca792f230a0",
    "outputs/hax26_selfdn_cache/120sec_cross_from_4.8sec.npz":
        "485d71bbd9f1f13bcbd64bad5493f6182a1b14c09835e57ba70fa85024cdce42",
    "outputs/hax26_selfdn_cache/120sec_dn_W1.npz":
        "21e2bf64eea809d4925e1faae58e7f254d67788d728eea88b3a3737fb55e8b45",
    "outputs/hax26_selfdn_cache/24sec_cross_from_120sec.npz":
        "1aaff367fd59e96160a6fad19d8ef80e5946c811b7783bb71a59aa7d6d593f16",
    "outputs/hax26_selfdn_cache/24sec_cross_from_4.8sec.npz":
        "95716d4738e02fc15299fb36be5471d674593b04fbdd19b395c2735af98312e1",
    "outputs/hax26_selfdn_cache/24sec_dn_W1.npz":
        "f73dea1583e09613e6b241e6896c14e1b09002b6309801b6fae57b5507d9471b",
    "outputs/hax26_selfdn_cache/4.8sec_cross_from_120sec.npz":
        "ebd27c66d3b37acda1d0ab27d511b7a8e33fefa9e72d52bf713b6b8ee46815b9",
    "outputs/hax26_selfdn_cache/4.8sec_cross_from_24sec.npz":
        "1bc9ea7ef0787b533aea4dc400b137f083c17f2f07071d4fa5389f52e6843fa6",
    "outputs/hax26_selfdn_cache/4.8sec_dn_W1.npz":
        "2436448e266b32dbf6ada5e38af42c2defc72a4575af3ea6c4a1f2727b75e753",
    "outputs/hax26_sysavg_cache/120sec_k1.npz":
        "55ea50fbffb5d7b6558f825abcb150a1b6a67debb17086291c9e83e3c5d0e68e",
    "outputs/hax26_sysavg_cache/120sec_k5.npz":
        "4007171e9597cf9652fbb9f01550504e8c04e0f8c17f1f2770ac3911a69fe100",
    "outputs/hax26_sysavg_cache/120sec_k10.npz":
        "fbaf0c5375ab633c98fe36fb423d9dc02dc0b63a36fc4a1fd19c399481bb79ae",
    "outputs/hax26_sysavg_cache/120sec_k25.npz":
        "fe819b5a299ae6ce1395074b2e0b0ccbabb15585d95e90d1b4d66efc460ab787",
    "outputs/hax26_sysavg_cache/120sec_k50.npz":
        "f9bde8397adc7401e183eec51dad63024ffcb938056d5b5c4e49c672db9b0998",
    "outputs/hax26_sysavg_cache/120sec_k125.npz":
        "9ae2ba3ff6718a2522f954fa0686125704e94d30126631ca19a65bd8279d7678",
    "outputs/hax26_sysavg_cache/24sec_k1.npz":
        "fa458b18318bab9069e1eb8773da8772e6a860db90a4b55d260f0a6a2f995dfd",
    "outputs/hax26_sysavg_cache/24sec_k5.npz":
        "a0bf8ef242395442dc533919fc47f492592edbdd0e3bda2b368acd912e696c0f",
    "outputs/hax26_sysavg_cache/24sec_k10.npz":
        "b6752b78af964c004ac071deccf224d820579bb3ade6a500bcc106c7f7044ba9",
    "outputs/hax26_sysavg_cache/24sec_k25.npz":
        "08b28ee884417509e8217cc7b934a462be371da2be63bd11457b5acf145b6215",
    "outputs/hax26_sysavg_cache/24sec_k50.npz":
        "aacf48bd92f3cf816cb2e5353e117f96c16655e9611bef6307484abacb134d96",
    "outputs/hax26_sysavg_cache/24sec_k125.npz":
        "31f12e4e6840f9140f51b67171837bc657b9e9f2d83385baeb918fb2b485eadd",
    "outputs/hax26_sysavg_cache/4.8sec_k1.npz":
        "e58c8119fa33b3a1ef4dee5d9eae178e3f6fc47885350e6e315023bb1862bd40",
    "outputs/hax26_sysavg_cache/4.8sec_k5.npz":
        "60a002f6921c5b52de75bfed53fb09e923578e9a1cc2e8fea1c0b3e346a427c6",
    "outputs/hax26_sysavg_cache/4.8sec_k10.npz":
        "f50fa77f098cfc1a76f1de8c739895966b4e4a6729c32c9dd5e86afcde96281c",
    "outputs/hax26_sysavg_cache/4.8sec_k25.npz":
        "073c6b8106f4160a16d132aa58b0b25f72f7a2be037e2015a53453868ae8de66",
    "outputs/hax26_sysavg_cache/4.8sec_k50.npz":
        "2a6cebf8cf7fb93f8c06cbbfb2f38096859d76c73db061f8c6fda33c4ae22860",
    "outputs/hax26_sysavg_cache/4.8sec_k125.npz":
        "1211e68c38f0264ac8f1dbf61565a1a61a7b6b3c3be414a2b2b4ab8e59d8d2e7",
    # seed pilot (Figs. S9, S10)
    "scripts/seed_pilot/cache/seed_0/inference/120sec_dn_W1_seed0.npz":
        "65766c218d7d8a0fb439fe48c06a3a755d71ee67bd944edf4b85bdb17509d5fb",
    "scripts/seed_pilot/cache/seed_0/inference/4.8sec_cross_from_120sec_seed0.npz":
        "dbb1305fb50be68f9c20180db0efb5dcd3203ad6a6cd5a3c72f4e952a730b005",
    "scripts/seed_pilot/cache/seed_0/inference/4.8sec_cross_from_24sec_seed0.npz":
        "cd217d874b187a20e066591943791b1eddda6b1135ad941a9e22a781f99c4add",
    "scripts/seed_pilot/cache/seed_0/inference/4.8sec_dn_W1_seed0.npz":
        "8a01c0f290425cca35ac38b4c7ea9ff100c21ccadab0097d092b7e4130508884",
    "scripts/seed_pilot/cache/seed_1/inference/4.8sec_cross_from_120sec_seed1.npz":
        "391c832149aadb7b7e6b83e4b148b96a11b564b86a15fe9958815b99d3317867",
    "scripts/seed_pilot/cache/seed_1/inference/4.8sec_cross_from_24sec_seed1.npz":
        "c605b3a2b5d724aac6248c0831a6d09a3fe9d5d21147d51779e9a41aa1e40833",
    "scripts/seed_pilot/cache/seed_1/inference/4.8sec_dn_W1_seed1.npz":
        "957dac7888c9ea1b6d11846529bba252031a4048972798a17fece2aa358b935e",
    "scripts/seed_pilot/cache/seed_2/inference/4.8sec_cross_from_120sec_seed2.npz":
        "85e0cbd7a3a4449bde327c138cdf304152a31012f8c83f9979ea3fd835a87620",
    "scripts/seed_pilot/cache/seed_2/inference/4.8sec_cross_from_24sec_seed2.npz":
        "e3ec3cb1b69d820c3febf96679845f750e8d820591048053d56d85e7b50610a0",
    "scripts/seed_pilot/cache/seed_2/inference/4.8sec_dn_W1_seed2.npz":
        "9046f80fe7f9ab2d44e3c948d1403dbdddb5cab7f178a67fec5f2967190c8a68",
    # the scripts the formulas below are transcribed from
    "scripts/fig4_hax26_equiv_exposure.py":
        "b445af609ebf1f8d9e1293c273c2b64ca757fb7accf9ab40cf9bb9e2d696145e",
    "scripts/figS9_seed_distribution.py":
        "4c5d19fa8e30085ab6be4e806e8da10d06e2571bf5ed6eabbdb28fa104ebe862",
    "scripts/figS10_seed_ordering.py":
        "a8a7833d903e68d81c633d488ed70d10eccfcb540f698110f27921ba469e1b9f",
}

#: SHA-256 of the one JVST file this module reads (same manifest).
JVST_FILES: Dict[str, str] = {
    "figure_data.h5":
        "51b4ce6a3d2bfea0d563796d84e2761198e0da3bbf23e158baa2e2aeea7e7254",
}

#: Published values, as printed in the SIA paper (Surf. Interface Anal.,
#: doi:10.1002/sia.70123). ``table1`` = Table 1 (K_eq read from the Fig. 4
#: annotations; integers). ``text`` = depth RMSE [%] and W1 [Å] quoted in
#: Sections 3.3–3.4 to one decimal. The match rule is ``round()`` to that
#: printed precision. Cells not printed in the paper carry ``None`` and are
#: reported without a verdict.
PUBLISHED_SIA = {
    "table1": {  # (apply, train) -> (K_eq RMSE, K_eq W1)
        ("4.8sec", "self"): (17, 7),
        ("4.8sec", "24sec"): (48, 14),
        ("24sec", "self"): (15, 8),
        ("24sec", "120sec"): (17, 9),
        ("120sec", "self"): (5, 3),
    },
    "text": {  # (apply, train) -> (depth RMSE %, W1 Å)
        ("4.8sec", "self"): (7.8, 13.5),
        ("4.8sec", "24sec"): (4.4, 6.1),
        ("4.8sec", "120sec"): (6.0, 6.2),
        ("24sec", "self"): (3.5, 3.0),
        ("24sec", "4.8sec"): (8.2, None),
        ("120sec", "self"): (3.0, 2.2),
        ("120sec", "4.8sec"): (8.5, 14.1),
    },
    # Section 4.5 / Figure S10: across-seed standard deviation of the per-cell
    # mean depth RMSE at the 4.8 s target, "approximately 1.4 %" (120 s-trained)
    # and "approximately 1.9 %" (24 s-trained). Sample standard deviation
    # (ddof=1) over the three seeds; same one-decimal match rule.
    "seed_sd": {("4.8sec", "120sec"): 1.4, ("4.8sec", "24sec"): 1.9},
    # Figure S9 caption: per-seed mean depth RMSE (%) at the 4.8 s target, two
    # decimals, and the Fig. 5 single-seed reference for the same cells.
    "fig_s9_caption": {
        ("4.8sec", "self"): {"fig5_reference": 7.81, "seeds": {0: 9.62, 1: 6.51, 2: 8.35}},
        ("4.8sec", "24sec"): {"fig5_reference": 4.39, "seeds": {0: 6.17, 1: 3.59, 2: 7.29}},
        ("4.8sec", "120sec"): {"fig5_reference": 6.03, "seeds": {0: 6.80, 1: 7.69, 2: 4.89}},
    },
}


class DepositMismatch(RuntimeError):
    """A deposited file is missing or does not match its published SHA-256."""


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_files(root: Path, expected: Dict[str, str]) -> Dict[str, str]:
    """Check every file in ``expected`` under ``root``; raise on any problem.

    Returns ``{relative path: sha256}`` for the manifest. Missing and
    mismatching files are all listed in the error, not only the first.
    """
    root = Path(root)
    seen: Dict[str, str] = {}
    problems: List[str] = []
    for rel, want in expected.items():
        p = root / rel
        if not p.is_file():
            problems.append(f"missing: {rel}")
            continue
        got = sha256_of(p)
        seen[rel] = got
        if got != want:
            problems.append(f"sha256 mismatch: {rel}\n    expected {want}\n    got      {got}")
    if problems:
        raise DepositMismatch(
            f"{len(problems)} deposited file(s) under {root} cannot be used "
            f"(deposit {DEPOSIT_VERSION_DOI}, v{DEPOSIT_VERSION}):\n  "
            + "\n  ".join(problems)
        )
    return seen


# ----------------------------------------------------------------------------
# Metric aggregation — transcribed from the deposited figure scripts
# ----------------------------------------------------------------------------

def depth_rmse_percent(D_map: np.ndarray, D_ref: np.ndarray) -> float:
    """Fig. 4 / Fig. 6 ``depth_rmse``: mean over pixels of the per-pixel RMSE
    over (depth, element), times 100.

    Note this is **not** the pooled RMSE over all elements of ``D_map``
    (``sqrt(mean((D - R)**2))``) that ``arhaxpes_denoise.metrics.depth_rmse``
    computes when handed a whole map; by Jensen's inequality the per-pixel
    mean is the smaller of the two. The paper's Eq. (2) and figure scripts use
    this per-pixel-mean form.
    """
    D_map = np.asarray(D_map, dtype=float)
    D_ref = np.asarray(D_ref, dtype=float)
    per_px = np.sqrt(np.nanmean((D_map - D_ref[np.newaxis, :, :]) ** 2, axis=(1, 2)))
    return float(np.nanmean(per_px)) * 100.0


def per_pixel_rmse_percent(D_map: np.ndarray, D_ref: np.ndarray) -> np.ndarray:
    """Figs. S9 / S10 ``per_pixel_rmse``: the per-pixel RMSE vector, in %."""
    D_map = np.asarray(D_map, dtype=float)
    D_ref = np.asarray(D_ref, dtype=float)
    sq = (D_map - D_ref[np.newaxis, :, :]) ** 2
    return np.sqrt(np.nanmean(sq, axis=(1, 2))) * 100.0


def across_seed_sd(cell_means) -> Optional[float]:
    """Sample standard deviation (ddof=1) of per-seed cell means; ``None`` with
    fewer than two seeds. This is the "across-seed standard deviation" the SIA
    paper quotes in Section 4.5; the population form (ddof=0) gives different
    numbers and is not what is printed."""
    means = np.asarray(list(cell_means), dtype=float)
    if means.size < 2:
        return None
    return float(np.std(means, ddof=1))


def w1_median_mean(D_map: np.ndarray, D_ref: np.ndarray,
                   depth: np.ndarray = DEPTH) -> float:
    """Fig. 4 / Fig. 6 ``compute_w1``: for each element the median over pixels
    of W1(reference, reconstruction), then the mean over elements.

    Uses ``deppro.depth_profile.wasserstein_distance_1d``, the function the
    deposited scripts import (the deposit's ``arhaxpes_denoise.metrics
    .wasserstein_w1`` is a copy of it). A pixel whose W1 is NaN (reference
    mass ≈ 0) is dropped by the NaN-aware median, as in the script.
    """
    D_map = np.asarray(D_map, dtype=float)
    D_ref = np.asarray(D_ref, dtype=float)
    per_elem = []
    for ie in range(D_map.shape[2]):
        vals = [wasserstein_distance_1d(depth, D_ref[:, ie], D_map[px, :, ie])
                for px in range(D_map.shape[0])]
        per_elem.append(float(np.nanmedian(vals)))
    return float(np.mean(per_elem))


# --- the frame-average fit and its inverse (Fig. 4) -------------------------

#: Soft-floor anchors from the cross-pseudo-GT repeatability (Fig. S2), fixed
#: in the script per metric: 0.5 % for RMSE, 0.5 Å for W1.
SOFT_FLOOR = {"rmse": 0.5, "w1": 0.5}


def _rmse_offset_floor(N, A, gamma, N0, C):
    """RMSE ladder model: y = sqrt(A·(N+N0)^(-2γ) + C²)."""
    return np.sqrt(A * np.power(N + N0, -2 * gamma) + C * C)


def _w1_offset_floor(N, B, gamma, N0, C):
    """W1 ladder model: y = B·(N+N0)^(-γ) + C."""
    return B * np.power(N + N0, -gamma) + C


def fit_frame_average(k: np.ndarray, y: np.ndarray, metric: str) -> Optional[dict]:
    """``fit_frame_avg`` of the Fig. 4 script, verbatim in its numerics.

    Offset-origin power law with the floor ``C`` fixed per metric; fitted with
    ``scipy.optimize.curve_fit`` and the script's ``p0`` / ``bounds`` /
    ``maxfev``. Points that are non-finite, non-positive, or at ``k <= 0`` are
    excluded. Returns ``None`` when fewer than three points remain or the fit
    fails, exactly as the script does (the script then draws no curve).
    """
    from scipy.optimize import curve_fit

    k = np.asarray(k, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(y) & (y > 0) & (k > 0)
    if valid.sum() < 3:
        return None
    Nv, yv = k[valid], y[valid]
    C = SOFT_FLOOR[metric]
    try:
        if metric == "rmse":
            A0 = max(float(yv[0]) ** 2 * float(Nv[0]), 1.0)

            def _f(N, A, gamma, N0):
                return _rmse_offset_floor(N, A, gamma, N0, C)

            popt, _ = curve_fit(_f, Nv, yv, p0=[A0, 0.5, 1.0],
                                bounds=([0, 0.1, 0], [np.inf, 3, 1e4]), maxfev=20000)
            return {"mode": "rmse", "A": float(popt[0]), "gamma": float(popt[1]),
                    "N0": float(popt[2]), "C": C}
        B0 = max(float(yv[0]), 1e-3)

        def _g(N, B, gamma, N0):
            return _w1_offset_floor(N, B, gamma, N0, C)

        popt, _ = curve_fit(_g, Nv, yv, p0=[B0, 0.5, 1.0],
                            bounds=([0, 0.01, 0], [np.inf, 5, 1e4]), maxfev=20000)
        return {"mode": "w1", "B": float(popt[0]), "gamma": float(popt[1]),
                "N0": float(popt[2]), "C": C}
    except Exception:  # the script swallows every fit failure the same way
        return None


def equivalent_k(popt: Optional[dict], target_y: float, n_max: float = 500) -> Optional[float]:
    """``find_equiv_N`` of the Fig. 4 script: invert the fitted ladder at
    ``target_y``. ``None`` when there is no fit, the target is at or below the
    floor, or the inverse falls outside ``[0.5, n_max]`` (the script then
    prints no annotation, and Table 1 shows "—"). The script additionally
    hides annotations beyond its plot range ``X_MAX = 150``; no deposited
    cell exceeds that, so it is not reproduced here."""
    if popt is None or target_y <= 0:
        return None
    C = popt["C"]
    if target_y <= C:
        return None
    try:
        if popt["mode"] == "rmse":
            inside = popt["A"] / (target_y ** 2 - C ** 2)
            n_eq = inside ** (1.0 / (2 * popt["gamma"])) - popt["N0"]
        else:
            n_eq = (popt["B"] / (target_y - C)) ** (1.0 / popt["gamma"]) - popt["N0"]
    except Exception:
        return None
    if 0.5 <= n_eq <= n_max:
        return float(n_eq)
    return None


# ----------------------------------------------------------------------------
# SIA replay
# ----------------------------------------------------------------------------

@dataclass
class SiaCell:
    """One denoiser operating point of Fig. 4 / Fig. 5 and its comparison."""
    apply: str                     # inference exposure, e.g. '4.8sec'
    train: str                     # 'self' or the training exposure
    source_file: str
    n_pixels: int
    depth_rmse_percent: float
    w1_angstrom: float
    k_eq_rmse: Optional[float]     # None = undefined (Table 1 "—")
    k_eq_w1: Optional[float]
    in_fig4: bool                  # the cell the Fig. 4 script actually plots
    published: dict = field(default_factory=dict)
    verdict: dict = field(default_factory=dict)


def _compare(recomputed: Optional[float], published, *, ndigits: int) -> Optional[bool]:
    if published is None:
        return None
    if recomputed is None:
        return False
    return round(recomputed, ndigits) == published


def _np_load(path: Path):
    return np.load(path, allow_pickle=False)


def replay_sia_fig4(sia_dir: Path, *, verify: bool = True) -> dict:
    """Recompute the Fig. 4 ladder fits, the DNN operating points, and the
    Table 1 K_eq values from ``sia_figure_data/``.

    Besides the two points the Fig. 4 script plots per exposure (Self-DNN and
    the designated best Cross-DNN), every other deposited Cross-DNN file is
    evaluated too, so that the Fig. 5 numbers quoted in the text can be
    checked. Those extra cells are marked ``in_fig4=False``.
    """
    sia_dir = Path(sia_dir)
    hashes = verify_files(sia_dir, SIA_FILES) if verify else {}

    ladders = {}
    cells: List[SiaCell] = []
    for apply_ds, spec in SIA_DATASETS.items():
        D_ref = _np_load(sia_dir / spec["gt"])["D_ref"]

        ks, rmse, w1 = [], [], []
        for k in spec["k_values"]:
            f = sia_dir / f"outputs/hax26_sysavg_cache/{apply_ds}_k{k}.npz"
            d = _np_load(f)
            ks.append(float(k))
            rmse.append(depth_rmse_percent(d["D_map"], D_ref))
            w1.append(w1_median_mean(d["D_map"], D_ref))
        ks_a, rmse_a, w1_a = np.array(ks), np.array(rmse), np.array(w1)
        fit_r = fit_frame_average(ks_a, rmse_a, "rmse")
        fit_w = fit_frame_average(ks_a, w1_a, "w1")
        ladders[apply_ds] = {
            "k": ks, "depth_rmse_percent": rmse, "w1_angstrom": w1,
            "fit_rmse": fit_r, "fit_w1": fit_w,
        }

        points = [("self", f"outputs/hax26_selfdn_cache/{apply_ds}_dn_W1.npz", True)]
        for src in SIA_DATASETS:
            if src == apply_ds:
                continue
            rel = f"outputs/hax26_selfdn_cache/{apply_ds}_cross_from_{src}.npz"
            points.append((src, rel, spec["best_cross"] == src))
        for train, rel, in_fig4 in points:
            d = _np_load(sia_dir / rel)
            r = depth_rmse_percent(d["D_map"], D_ref)
            w = w1_median_mean(d["D_map"], D_ref)
            kr = equivalent_k(fit_r, r)
            kw = equivalent_k(fit_w, w)
            pub_t1 = PUBLISHED_SIA["table1"].get((apply_ds, train))
            pub_tx = PUBLISHED_SIA["text"].get((apply_ds, train))
            published = {
                "k_eq_rmse": pub_t1[0] if pub_t1 else None,
                "k_eq_w1": pub_t1[1] if pub_t1 else None,
                "depth_rmse_percent": pub_tx[0] if pub_tx else None,
                "w1_angstrom": pub_tx[1] if pub_tx else None,
            }
            verdict = {
                "k_eq_rmse": _compare(kr, published["k_eq_rmse"], ndigits=0),
                "k_eq_w1": _compare(kw, published["k_eq_w1"], ndigits=0),
                "depth_rmse_percent": _compare(r, published["depth_rmse_percent"], ndigits=1),
                "w1_angstrom": _compare(w, published["w1_angstrom"], ndigits=1),
            }
            cells.append(SiaCell(apply_ds, train, rel, int(d["D_map"].shape[0]),
                                 r, w, kr, kw, in_fig4, published, verdict))

    verdicts = [v for c in cells for v in c.verdict.values() if v is not None]
    return {
        "kind": "stored-result check",
        "paper": "SIA, doi:10.1002/sia.70123, Fig. 4 / Table 1 and the Fig. 5 values quoted in the text",
        "deposit": {"version_doi": DEPOSIT_VERSION_DOI, "version": DEPOSIT_VERSION,
                    "verified": bool(verify), "files": hashes},
        "match_rule": "K_eq: round to integer equals Table 1; RMSE/W1: round to one decimal equals the text",
        "ladders": ladders,
        "cells": [asdict(c) for c in cells],
        "n_compared": len(verdicts),
        "n_matched": int(sum(verdicts)),
    }


def replay_sia_seed_pilot(sia_dir: Path, *, verify: bool = True) -> dict:
    """Recompute the Figs. S9 / S10 seed-pilot statistics: for each
    (apply, train, seed) the mean and standard deviation of the per-pixel
    depth RMSE (%), and the Fig. S10 ordering test ``mean(24 s) < mean(120 s)``
    per seed for the 4.8 s target, the across-seed standard deviation of the
    cell means that the text quotes (``PUBLISHED_SIA["seed_sd"]``), and the
    per-seed means and Fig. 5 reference values printed in the Fig. S9 caption
    (``PUBLISHED_SIA["fig_s9_caption"]``, two decimals)."""
    import re

    sia_dir = Path(sia_dir)
    hashes = verify_files(sia_dir, SIA_FILES) if verify else {}
    gt = {ds: _np_load(sia_dir / spec["gt"])["D_ref"] for ds, spec in SIA_DATASETS.items()}

    rows = []
    for rel in sorted(SIA_FILES):
        if not rel.startswith("scripts/seed_pilot/"):
            continue
        name = Path(rel).stem
        m_self = re.match(r"^([\d.]+sec)_dn_W\d+_seed(\d+)$", name)
        m_cross = re.match(r"^([\d.]+sec)_cross_from_([\d.]+sec)_seed(\d+)$", name)
        if m_self:
            apply_ds, train, seed = m_self.group(1), "self", int(m_self.group(2))
        elif m_cross:
            apply_ds, train, seed = m_cross.group(1), m_cross.group(2), int(m_cross.group(3))
        else:
            continue
        d = _np_load(sia_dir / rel)
        rmse = per_pixel_rmse_percent(d["D_map"], gt[apply_ds])
        caption = PUBLISHED_SIA["fig_s9_caption"].get((apply_ds, train), {})
        published = caption.get("seeds", {}).get(seed)
        rows.append({
            "apply": apply_ds, "train": train, "seed": seed, "source_file": rel,
            "n_pixels": int(rmse.size),
            "mean_percent": float(np.mean(rmse)),
            "std_percent": float(np.std(rmse)),
            "sem_percent": float(np.std(rmse) / np.sqrt(rmse.size)),
            "collapsed": bool(np.std(rmse) < 1e-6),
            "published_mean_percent": published,
            "verdict": _compare(float(np.mean(rmse)), published, ndigits=2),
        })

    # the Fig. 5 single-seed reference values the caption quotes beside the seeds
    references = []
    for (apply_ds, train), caption in PUBLISHED_SIA["fig_s9_caption"].items():
        rel = (f"outputs/hax26_selfdn_cache/{apply_ds}_dn_W1.npz" if train == "self"
               else f"outputs/hax26_selfdn_cache/{apply_ds}_cross_from_{train}.npz")
        value = depth_rmse_percent(_np_load(sia_dir / rel)["D_map"], gt[apply_ds])
        references.append({"apply": apply_ds, "train": train, "source_file": rel,
                           "mean_percent": value, "published": caption["fig5_reference"],
                           "verdict": _compare(value, caption["fig5_reference"], ndigits=2)})

    across_seed = []
    for (apply_ds, train), published in PUBLISHED_SIA["seed_sd"].items():
        means = [r["mean_percent"] for r in rows if r["apply"] == apply_ds and r["train"] == train]
        sd = across_seed_sd(means)
        across_seed.append({"apply": apply_ds, "train": train, "n_seeds": len(means),
                            "cell_means_percent": means, "sd_percent": sd,
                            "published": published,
                            "verdict": _compare(sd, published, ndigits=1)})

    ordering = []
    for seed in sorted({r["seed"] for r in rows}):
        m24 = [r for r in rows if r["apply"] == "4.8sec" and r["train"] == "24sec" and r["seed"] == seed]
        m120 = [r for r in rows if r["apply"] == "4.8sec" and r["train"] == "120sec" and r["seed"] == seed]
        if m24 and m120:
            ordering.append({"seed": seed, "mean_24sec": m24[0]["mean_percent"],
                             "mean_120sec": m120[0]["mean_percent"],
                             "holds_24_below_120": m24[0]["mean_percent"] < m120[0]["mean_percent"]})
    return {
        "kind": "stored-result check",
        "paper": "SIA, Figs. S9 / S10 (seed pilot, 4.8 s target)",
        "deposit": {"version_doi": DEPOSIT_VERSION_DOI, "version": DEPOSIT_VERSION,
                    "verified": bool(verify), "files": {k: v for k, v in hashes.items()
                                                       if k.startswith("scripts/seed_pilot/")}},
        "match_rule": ("per-seed means and Fig. 5 references: round to two decimals equals the Fig. S9 caption; "
                       "across-seed sample sd (ddof=1) of cell means: round to one decimal equals the text"),
        "cells": rows,
        "fig5_references": references,
        "across_seed_sd": across_seed,
        "n_compared": sum(r["verdict"] is not None for r in rows) + len(references) + len(across_seed),
        "n_matched": int(sum(bool(r["verdict"]) for r in rows) + sum(bool(r["verdict"]) for r in references)
                         + sum(bool(a["verdict"]) for a in across_seed)),
        "ordering_fig_s10": ordering,
    }


# ----------------------------------------------------------------------------
# JVST replay
# ----------------------------------------------------------------------------

def _jvst_w1_stats(D_map: np.ndarray, D_ref: np.ndarray) -> Dict[str, float]:
    """``compute_w1_stats`` of the authors' JVST analysis script: per element
    the median over pixels of the finite W1 values; ``mean`` = mean of the
    finite per-element medians."""
    out: Dict[str, float] = {}
    for e, name in enumerate(ELEMENTS):
        vals = [wasserstein_distance_1d(DEPTH, D_ref[:, e], D_map[px, :, e])
                for px in range(D_map.shape[0])]
        vals = [v for v in vals if not np.isnan(v)]
        out[name] = float(np.median(vals)) if vals else float("nan")
    finite = [v for v in out.values() if not np.isnan(v)]
    out["mean"] = float(np.mean(finite)) if finite else float("nan")
    return out


def _jvst_group_for(label: str, session: str, dataset: str) -> str:
    """Map a ``metrics`` JSON key to the reconstruction group that holds its
    arrays.

    ``frame_N{k}`` → ``ladder_k{k}``; the other labels keep their name. The
    prefix is ``session102u``/``session103u`` (fix25) or ``…f`` (fix23), with
    one exception taken from the authors' session-103 analysis script: its
    ``frame_N*``, ``perbin_B*`` and ``frame_raw`` entries were computed from
    the **session-102** arrays (the script reuses them), and the deposit's
    converter names groups after the source file, so those entries of the
    session-103 JSON are compared with the session-102 groups.
    """
    suffix = "u" if dataset == "fix25" else "f"
    if label.startswith("frame_N"):
        stem, src = "ladder_k" + label[len("frame_N"):], "session102"
    elif label.startswith("perbin_B") or label == "frame_raw":
        stem, src = label, "session102"
    else:
        stem, src = label, session
    return f"{src}{suffix}_{stem}"


def replay_jvst_metrics(h5_path: Path, *, verify: bool = True) -> dict:
    """Recompute, from the ``D_map`` / ``thk_map`` / ``rrmse_map`` arrays in
    ``figure_data.h5``, the per-condition metrics stored as JSON in the same
    file (``metrics/{fix25,fix23}_{session102,session103}``), and compare.

    ``rrmse``, ``sigma_thk``, ``n_valid`` and ``w1`` follow the authors'
    analysis script (pixels with ``thk_map > 0`` are valid; ``rrmse`` is the
    median over valid finite pixels ×100; ``sigma_thk`` the population
    standard deviation of valid thicknesses). The stored ``depth_rmse`` field
    has no definition in the deposit; two candidate forms are evaluated and
    the result is labeled *inferred*. A stored field whose recomputation
    agrees to 1e-6 relative is ``matched``; the ladder entries whose group is
    missing from the file are reported as ``no_arrays``.
    """
    import h5py

    h5_path = Path(h5_path)
    hashes = verify_files(h5_path.parent, {h5_path.name: JVST_FILES["figure_data.h5"]}) if verify else {}

    rows = []
    with h5py.File(h5_path, "r") as f:
        for dataset in ("fix25", "fix23"):
            D_ref = f[f"ground_truth/{dataset}/D_ref"][:]
            for session in ("session102", "session103"):
                key = f"metrics/{dataset}_{session}"
                if key not in f:
                    continue
                stored_all = json.loads(f[key][()])
                for label, stored in stored_all.items():
                    group = f"reconstructions/{dataset}/{_jvst_group_for(label, session, dataset)}"
                    row = {"dataset": dataset, "metrics_key": key, "label": label,
                           "group": group, "stored": stored}
                    if group not in f:
                        row["status"] = "no_arrays"
                        rows.append(row)
                        continue
                    g = f[group]
                    D, thk, rr = g["D_map"][:], g["thk_map"][:], g["rrmse_map"][:]
                    valid = thk > 0
                    if valid.sum() == 0:
                        row["status"] = "no_valid_pixels"
                        rows.append(row)
                        continue
                    rr_v = rr[valid & ~np.isnan(rr)]
                    recomputed = {
                        "rrmse": float(np.nanmedian(rr_v) * 100),
                        "sigma_thk": float(thk[valid].std()),
                        "n_valid": int(valid.sum()),
                        "w1": _jvst_w1_stats(D[valid], D_ref),
                    }
                    inferred = {}
                    if "depth_rmse" in stored:
                        inferred["depth_rmse_per_pixel_mean_percent"] = depth_rmse_percent(D[valid], D_ref)
                        inferred["depth_rmse_pooled_percent"] = float(
                            np.sqrt(np.mean((D[valid] - D_ref[np.newaxis]) ** 2))) * 100
                    row["recomputed"] = recomputed
                    row["inferred"] = inferred
                    row["match"] = {
                        "rrmse": _close(recomputed["rrmse"], stored.get("rrmse")),
                        "sigma_thk": _close(recomputed["sigma_thk"], stored.get("sigma_thk")),
                        "n_valid": recomputed["n_valid"] == stored.get("n_valid"),
                        "w1": all(_close(recomputed["w1"][k], stored.get("w1", {}).get(k))
                                  for k in list(ELEMENTS) + ["mean"]),
                    }
                    if inferred:
                        row["match"]["depth_rmse_inferred_form"] = next(
                            (k for k, v in inferred.items() if _close(v, stored["depth_rmse"], rel=1e-4)),
                            None)
                    row["status"] = "compared"
                    rows.append(row)

    compared = [r for r in rows if r["status"] == "compared"]

    def _all_defined(r):
        return all(v for k, v in r["match"].items() if k != "depth_rmse_inferred_form")

    by_dataset = {}
    for ds in ("fix25", "fix23"):
        sub = [r for r in compared if r["dataset"] == ds]
        by_dataset[ds] = {
            "n_compared": len(sub),
            "n_all_defined_fields_matched": sum(_all_defined(r) for r in sub),
            "fields_matched": {k: sum(bool(r["match"].get(k)) for r in sub)
                               for k in ("rrmse", "sigma_thk", "n_valid", "w1")},
            "depth_rmse_inferred_forms": sorted({str(r["match"].get("depth_rmse_inferred_form"))
                                                 for r in sub if "depth_rmse" in r["stored"]}),
        }
    return {
        "kind": "stored-result check (internal consistency of the deposit)",
        "paper": "JVST A, doi:10.1116/6.0005585, metrics JSON vs arrays in figure_data.h5",
        "deposit": {"version_doi": DEPOSIT_VERSION_DOI, "version": DEPOSIT_VERSION,
                    "verified": bool(verify), "files": hashes},
        "match_rule": "recomputed value within 1e-6 relative of the stored JSON value (depth_rmse: 1e-4, inferred form)",
        "reference_used": "ground_truth/<dataset>/D_ref of the same file (the canonical pseudo-GT)",
        "rows": rows,
        "n_rows": len(rows),
        "n_compared": len(compared),
        "n_no_arrays": sum(r["status"] == "no_arrays" for r in rows),
        "n_all_defined_fields_matched": sum(_all_defined(r) for r in compared),
        "by_dataset": by_dataset,
    }


def _close(a, b, rel: float = 1e-6) -> Optional[bool]:
    if b is None:
        return None
    a = float(a)
    b = float(b)
    if np.isnan(a) and np.isnan(b):
        return True
    return bool(abs(a - b) <= rel * max(1.0, abs(b)))


# ----------------------------------------------------------------------------
# Manifest and CLI
# ----------------------------------------------------------------------------

def environment_record() -> dict:
    """What produced a replay: interpreter, libraries, platform, and the
    ``deppro`` commit when run from a checkout."""
    import scipy

    rec = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }
    try:
        import h5py
        rec["h5py"] = h5py.__version__
    except ImportError:
        rec["h5py"] = None
    try:
        from importlib.metadata import version
        rec["deppro"] = version("deppro")
    except Exception:
        rec["deppro"] = None
    here = Path(__file__).resolve()
    try:
        r = subprocess.run(["git", "-C", str(here.parent), "rev-parse", "HEAD"],
                           capture_output=True, text=True, timeout=30)
        rec["deppro_git_commit"] = r.stdout.strip() if r.returncode == 0 else None
        r = subprocess.run(["git", "-C", str(here.parent), "status", "--porcelain", "--", str(here)],
                           capture_output=True, text=True, timeout=30)
        rec["deppro_replay_module_modified"] = bool(r.stdout.strip()) if r.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        rec["deppro_git_commit"] = None
        rec["deppro_replay_module_modified"] = None
    return rec


def _print_sia_table(res: dict) -> None:
    print(f"\nSIA Fig. 4 / Table 1 — {res['kind']}; match rule: {res['match_rule']}")
    print(f"{'apply':>7} {'train':>7} {'fig4':>4} {'RMSE %':>8} {'pub':>5} {'W1 Å':>8} {'pub':>5} "
          f"{'Keq_R':>7} {'pub':>4} {'Keq_W':>7} {'pub':>4}")
    for c in res["cells"]:
        p, v = c["published"], c["verdict"]

        def fmt(x, nd):
            return "—" if x is None else f"{x:.{nd}f}"

        def mark(ok):
            return "" if ok is None else (" ✓" if ok else " ✗")

        print(f"{c['apply']:>7} {c['train']:>7} {'yes' if c['in_fig4'] else 'no':>4} "
              f"{c['depth_rmse_percent']:8.3f} {fmt(p['depth_rmse_percent'],1):>5}{mark(v['depth_rmse_percent'])} "
              f"{c['w1_angstrom']:8.3f} {fmt(p['w1_angstrom'],1):>5}{mark(v['w1_angstrom'])} "
              f"{fmt(c['k_eq_rmse'],2):>7} {fmt(p['k_eq_rmse'],0):>4}{mark(v['k_eq_rmse'])} "
              f"{fmt(c['k_eq_w1'],2):>7} {fmt(p['k_eq_w1'],0):>4}{mark(v['k_eq_w1'])}")
    print(f"matched {res['n_matched']} of {res['n_compared']} published values")
    for ds, lad in res["ladders"].items():
        fr, fw = lad["fit_rmse"], lad["fit_w1"]
        print(f"  ladder {ds}: RMSE fit γ={fr['gamma']:.3f} K0={fr['N0']:.2f}; "
              f"W1 fit γ={fw['gamma']:.3f} K0={fw['N0']:.2f}")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sia-dir", type=Path, help="extracted sia_figure_data/ directory")
    ap.add_argument("--jvst-h5", type=Path, help="extracted jvst_figure_data/figure_data.h5")
    ap.add_argument("--out", type=Path, help="write the full results and manifest as JSON")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip the SHA-256 check (the result is then NOT a check of the deposit)")
    args = ap.parse_args(argv)
    if args.sia_dir is None and args.jvst_h5 is None:
        ap.error("give --sia-dir and/or --jvst-h5")

    verify = not args.no_verify
    out = {"environment": environment_record(),
           "deposit": {"concept_doi": DEPOSIT_CONCEPT_DOI, "version_doi": DEPOSIT_VERSION_DOI,
                       "version": DEPOSIT_VERSION, "archive_md5": ARCHIVE_MD5},
           "verified": verify}
    if args.sia_dir is not None:
        out["sia_fig4"] = replay_sia_fig4(args.sia_dir, verify=verify)
        out["sia_seed_pilot"] = replay_sia_seed_pilot(args.sia_dir, verify=verify)
        _print_sia_table(out["sia_fig4"])
        print("\nSIA seed pilot (per-pixel depth RMSE %, Figs. S9/S10):")
        for r in out["sia_seed_pilot"]["cells"]:
            mark = "" if r["verdict"] is None else (" ✓" if r["verdict"] else " ✗")
            pub = "" if r["published_mean_percent"] is None else f" (caption {r['published_mean_percent']})"
            print(f"  {r['apply']:>7} train={r['train']:>7} seed={r['seed']}  "
                  f"mean={r['mean_percent']:.3f}{pub}{mark}  std={r['std_percent']:.3f}"
                  f"{'  (collapsed)' if r['collapsed'] else ''}")
        for r in out["sia_seed_pilot"]["fig5_references"]:
            mark = " ✓" if r["verdict"] else " ✗"
            print(f"  Fig. 5 reference {r['apply']} train={r['train']}: {r['mean_percent']:.3f} "
                  f"(caption {r['published']}){mark}")
        sp = out["sia_seed_pilot"]
        print(f"  seed pilot: matched {sp['n_matched']} of {sp['n_compared']} caption / text values")
        for a in out["sia_seed_pilot"]["across_seed_sd"]:
            mark = "" if a["verdict"] is None else (" ✓" if a["verdict"] else " ✗")
            print(f"  across-seed sd {a['apply']} train={a['train']}: {a['sd_percent']:.3f} % "
                  f"(printed ≈{a['published']}){mark}")
        for o in out["sia_seed_pilot"]["ordering_fig_s10"]:
            print(f"  seed {o['seed']}: 24 s-trained {o['mean_24sec']:.3f} vs 120 s-trained "
                  f"{o['mean_120sec']:.3f} -> 24<120 {'holds' if o['holds_24_below_120'] else 'fails'}")
    if args.jvst_h5 is not None:
        out["jvst_metrics"] = replay_jvst_metrics(args.jvst_h5, verify=verify)
        j = out["jvst_metrics"]
        print(f"\nJVST metrics JSON vs arrays: {j['n_rows']} rows, {j['n_compared']} compared, "
              f"{j['n_no_arrays']} without arrays, "
              f"{j['n_all_defined_fields_matched']} with every defined field matched")
        for ds, b in j["by_dataset"].items():
            print(f"  {ds}: {b['n_all_defined_fields_matched']}/{b['n_compared']} rows with every defined "
                  f"field matched; per field {b['fields_matched']}; "
                  f"depth_rmse (undefined in the deposit) inferred forms {b['depth_rmse_inferred_forms']}")
    if not verify:
        print("\nWARNING: --no-verify given; the files were not checked against the deposit's SHA-256.")
    if args.out is not None:
        args.out.write_text(json.dumps(out, indent=2, default=_json_default), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


if __name__ == "__main__":
    sys.exit(main())
