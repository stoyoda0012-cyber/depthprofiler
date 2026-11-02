"""deppro.haxpes.geometry: emission angle under polar / tilt / azimuth and the photon–electron angle."""
import numpy as np
import pytest

from deppro.haxpes import geometry as g
from deppro.haxpes.config import FIX_PXT_INSTRUMENT, SAMPLE_POLAR_DEG

BETA = np.flip(np.linspace(FIX_PXT_INSTRUMENT.detector_bin_lo,
                           FIX_PXT_INSTRUMENT.detector_bin_hi,
                           FIX_PXT_INSTRUMENT.n_prebinned_angles))


def test_polar_only_reproduces_the_existing_emission_angles():
    """tilt = 0, gamma = 0: |theta| equals flip(detector) + SAMPLE_POLAR_DEG, channel by channel."""
    np.testing.assert_allclose(
        g.emission_angle_deg(BETA, SAMPLE_POLAR_DEG),
        np.abs(FIX_PXT_INSTRUMENT.emission_angles_prebinned()), atol=1e-9)


def test_polar_shifts_every_channel_by_the_same_amount():
    a = g.emission_angle_deg(BETA[:9], 30.0)
    b = g.emission_angle_deg(BETA[:9], 31.0)
    np.testing.assert_allclose(b - a, 1.0, atol=1e-9)


@pytest.mark.parametrize("azimuth", [0.0, 17.0, 90.0, -135.0])
def test_azimuth_does_not_change_the_emission_angle(azimuth):
    base = g.emission_cos(BETA, 30.0, 4.0, 0.0, gamma_deg=1.5)
    np.testing.assert_allclose(g.emission_cos(BETA, 30.0, 4.0, azimuth, gamma_deg=1.5), base, atol=1e-14)


@pytest.mark.parametrize("tilt,gamma", [(0.0, 0.0), (5.0, 0.0), (10.0, 0.0), (5.0, 2.0), (-3.0, -1.0)])
def test_closed_form_for_polar_bottom_tilt_top(tilt, gamma):
    p, t, gm = np.deg2rad(30.0), np.deg2rad(tilt), np.deg2rad(gamma)
    want = np.cos(t) * np.cos(gm) * np.cos(p + np.deg2rad(BETA)) + np.sin(t) * np.sin(gm)
    np.testing.assert_allclose(g.emission_cos(BETA, 30.0, tilt, gamma_deg=gamma), want, atol=1e-14)


def test_tilt_does_not_add_to_polar():
    """With tilt T, theta is not polar + beta + T (the naive sum)."""
    theta = g.emission_angle_deg(BETA[1:9], 30.0, 5.0)
    naive = np.abs(30.0 + BETA[1:9] + 5.0)
    assert np.all(np.abs(theta - naive) > 0.1)
    # and 1/cos(theta) scales uniformly by 1/cos(T) at gamma = 0
    ratio = g.emission_cos(BETA, 30.0, 0.0) / g.emission_cos(BETA, 30.0, 5.0)
    np.testing.assert_allclose(ratio, 1 / np.cos(np.deg2rad(5.0)), rtol=1e-12)


def test_photon_electron_angle_is_a_lab_frame_quantity():
    """alpha = 90 - beta for the default X-ray direction, independent of any sample angle."""
    np.testing.assert_allclose(g.photon_electron_angle_deg(BETA), 90.0 - BETA, atol=1e-9)
    import inspect
    assert set(inspect.signature(g.photon_electron_angle_deg).parameters) == {
        "beta_deg", "gamma_deg", "xray_direction"}


def test_alpha_is_120_minus_theta_at_polar_30():
    """Derived identity used by the SIA non-dipole check: at polar = 30, alpha = 120 - theta."""
    theta_signed = 30.0 + BETA
    np.testing.assert_allclose(g.photon_electron_angle_deg(BETA), 120.0 - theta_signed, atol=1e-9)


@pytest.mark.parametrize("bad", [(0.0, 0.0, 0.0), (float("nan"), 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), (1.0, 0.0)])
def test_invalid_xray_direction_is_rejected(bad):
    with pytest.raises(ValueError, match="xray_direction"):
        g.photon_electron_angle_deg(BETA, xray_direction=bad)


def test_vectors_are_unit_and_broadcast():
    k = g.electron_direction(BETA[:, None], np.array([-2.0, 0.0, 2.0])[None, :])
    assert k.shape == (12, 3, 3)
    np.testing.assert_allclose(np.linalg.norm(k, axis=-1), 1.0, atol=1e-15)
    n = g.sample_normal(30.0, 7.0, 33.0)
    assert np.linalg.norm(n) == pytest.approx(1.0, abs=1e-15)


def test_no_tilt_is_zero():
    assert g.NO_TILT == 0.0


def test_passing_the_polar_constant_as_tilt_is_wrong():
    """Negative control: SAMPLE_POLAR_DEG is the polar angle, not a tilt."""
    wrong = g.emission_angle_deg(BETA, 0.0, SAMPLE_POLAR_DEG)
    right = g.emission_angle_deg(BETA, SAMPLE_POLAR_DEG)
    assert np.max(np.abs(wrong - right)) > 10.0


@pytest.mark.parametrize("polar", [0.0, 30.0, -20.0, 75.0])
def test_tilt_90_points_the_normal_along_y_for_any_polar(polar):
    """Stack order check independent of the closed form: tilt sits on top of polar."""
    np.testing.assert_allclose(g.sample_normal(polar, 90.0), [0.0, 1.0, 0.0], atol=1e-15)


@pytest.mark.parametrize("azimuth", [17.0, -60.0, 135.0])
def test_azimuth_rotates_the_in_plane_axes_about_the_normal(azimuth):
    f0 = g.sample_frame(30.0, 4.0, 0.0)
    f = g.sample_frame(30.0, 4.0, azimuth)
    a = np.deg2rad(azimuth)
    np.testing.assert_allclose(f[:, 2], f0[:, 2], atol=1e-15)
    np.testing.assert_allclose(f[:, 0], np.cos(a) * f0[:, 0] + np.sin(a) * f0[:, 1], atol=1e-15)
    np.testing.assert_allclose(f[:, 1], -np.sin(a) * f0[:, 0] + np.cos(a) * f0[:, 1], atol=1e-15)


def test_beta_is_the_angle_projected_into_the_dispersion_plane():
    k = g.electron_direction(np.array([10.0, -25.0]), np.array([3.0, -4.0]))
    np.testing.assert_allclose(k[:, 0] / k[:, 2], np.tan(np.deg2rad([10.0, -25.0])), atol=1e-14)
