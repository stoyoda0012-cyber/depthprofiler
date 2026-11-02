"""Sample and detector geometry: emission angle and photon–electron angle.

Lab frame
---------
- ``z``: the analyzer's central (lens) axis, pointing from the sample to the
  analyzer.
- ``x``: the analyzer's angular-dispersion direction (the detector-angle axis in
  Angular56 mode).
- ``y``: the non-dispersive (slit) direction, ``y = z × x``.

An electron accepted at detector angle ``beta`` (along ``x``) and at angle
``gamma`` across the slit travels along
``k = (sin(beta) cos(gamma), sin(gamma), cos(beta) cos(gamma))``.
This parametrisation is a modelling choice: ``beta`` is the angle projected into
the dispersion plane (``k_x / k_z = tan(beta)``). The alternative
``(sin(beta), cos(beta) sin(gamma), cos(beta) cos(gamma))`` differs at second
order in the small angles: ``O(beta gamma^2)`` in ``k_x`` and ``O(beta^2 gamma)``
in ``k_y``.

``beta`` is the detector angle in the order the emission arrays use: the value
after the row-to-channel ``flip`` of ``HAXPESInstrumentConfig``. With
``tilt = 0`` and ``gamma = 0`` the emission angle is then ``polar + beta``,
which is exactly what ``emission_angles_prebinned()`` computes as
``flip(detector) + SAMPLE_POLAR_DEG``.

Manipulator
-----------
Stacking is **assumed** to be bottom to top: ``polar``, then ``tilt``, then
``azimuth``. The owner confirmed the standard HAXPES-Lab geometry; the stacking
order of a future multi-axis manipulator is not verified here.
- ``polar`` rotates about ``y``. Its axis is perpendicular to the dispersion
  plane, so with ``tilt = 0`` and ``gamma = 0`` changing it shifts every
  channel's signed emission angle ``polar + beta`` by the same amount. With a
  non-zero tilt or slit angle the shift differs from channel to channel, and
  the unsigned angle of ``emission_angle_deg`` also breaks it where
  ``polar + beta`` changes sign.
- ``tilt`` rotates about the sample's in-plane axis that lies along ``x`` when
  ``polar = 0``.
- ``azimuth`` rotates about the sample normal. It changes no output of this
  module (neither ``cos(theta)`` nor ``alpha``); it only rotates the sample's
  in-plane axes, which ``sample_frame`` exposes.

All three are zero when the sample normal points along the lens axis.
Composition is ``n = R_polar · R_tilt · R_azimuth · ẑ``: the lower stage
rotates everything above it.

The value ``SAMPLE_POLAR_DEG`` in ``deppro.haxpes.config`` (formerly named
``SAMPLE_TILT_DEG``) is the **polar** angle from the lens axis to the sample
normal. Tilt was 0° in the measurements analysed so far. Do not pass
``SAMPLE_POLAR_DEG`` as ``tilt_deg``: that silently gives angles wrong by tens
of degrees.

``emission_cos`` can be zero or negative (theta >= 90°) for extreme settings.
The attenuation ``exp(-z / (lambda cos theta))`` is not defined there; callers
must exclude such channels.

Closed forms (checked by the tests):
- polar bottom with tilt on top:
  ``cos(theta) = cos(T) cos(gamma) cos(P + beta) + sin(T) sin(gamma)``
  (the sign of ``T`` is a convention: positive ``T`` tilts the normal towards ``+y``).
- At ``gamma = 0`` this is ``cos(T) cos(P + beta)``. It is not
  ``cos(P + beta + T)``.

Photon–electron angle
---------------------
``cos(alpha) = s · k``, where ``s`` is the X-ray propagation direction in the
lab frame. ``alpha`` depends only on the lab geometry and on the detector
channel. It does **not** depend on polar, tilt or azimuth, so the non-dipole
factor of a channel is unchanged by sample rotation. This holds for an
unpolarized source (e.g. Ga Kα); for a polarized source the angle to the
polarization vector also enters, which is also lab-fixed and not computed
here.

The default ``s = +x`` (X-ray perpendicular to the lens axis, in the
dispersion plane) gives ``alpha = 90 - beta``. With polar = 30 that is
``alpha = 120 - theta``, the convention of the SIA non-dipole check. Here
``theta`` is the signed angle ``polar + beta`` at ``tilt = 0`` and
``gamma = 0``, not the unsigned value ``emission_angle_deg`` returns: where
``polar + beta < 0`` the two differ in sign and the identity fails if the
returned value is substituted. The sign
of ``s`` (incidence from +x or -x) is not fixed by the owner's facts; it is
chosen to reproduce that convention.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "electron_direction",
    "sample_frame",
    "sample_normal",
    "emission_cos",
    "emission_angle_deg",
    "photon_electron_angle_deg",
    "NO_TILT",
]

_XRAY_DEFAULT = (1.0, 0.0, 0.0)

#: Default tilt-axis angle. Tilt was 0° in the measurements analysed so far.
#: Named so that no tilt-like parameter takes a numeric literal (see
#: test_haxpes_polar_offset). Not the polar offset; that is
#: ``SAMPLE_POLAR_DEG`` in ``deppro.haxpes.config``.
NO_TILT = 0.0


def _rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def electron_direction(beta_deg, gamma_deg=0.0) -> np.ndarray:
    """Unit vectors ``k`` for detector angle(s) ``beta`` and slit angle(s) ``gamma`` (degrees).

    Returns an array of shape ``broadcast(beta, gamma).shape + (3,)``.
    """
    b = np.deg2rad(np.asarray(beta_deg, dtype=float))
    g = np.deg2rad(np.asarray(gamma_deg, dtype=float))
    b, g = np.broadcast_arrays(b, g)
    return np.stack([np.sin(b) * np.cos(g), np.sin(g), np.cos(b) * np.cos(g)], axis=-1)


def sample_frame(polar_deg: float, tilt_deg: float = NO_TILT,
                 azimuth_deg: float = 0.0) -> np.ndarray:
    """Rotation matrix whose columns are the sample's x, y and z (normal) axes in the lab frame.

    The stack is polar (bottom), tilt, azimuth (top). ``polar`` is signed so
    that, with ``tilt = 0``, the emission angle is ``polar + beta``.
    """
    return (_ry(-np.deg2rad(polar_deg)) @ _rx(-np.deg2rad(tilt_deg))
            @ _rz(np.deg2rad(azimuth_deg)))


def sample_normal(polar_deg: float, tilt_deg: float = NO_TILT, azimuth_deg: float = 0.0) -> np.ndarray:
    """Sample-normal unit vector in the lab frame (third column of ``sample_frame``)."""
    return sample_frame(polar_deg, tilt_deg, azimuth_deg) @ np.array([0.0, 0.0, 1.0])


def emission_cos(beta_deg, polar_deg: float, tilt_deg: float = NO_TILT,
                 azimuth_deg: float = 0.0, gamma_deg=0.0) -> np.ndarray:
    """cos(theta), theta measured from the sample normal, for each detector channel.

    This is the quantity the attenuation ``exp(-z / (lambda cos theta))``
    needs. A finite acceptance across the slit is handled by averaging the
    attenuation over ``gamma``, not by averaging ``theta``.
    """
    k = electron_direction(beta_deg, gamma_deg)
    return k @ sample_normal(polar_deg, tilt_deg, azimuth_deg)


def emission_angle_deg(beta_deg, polar_deg: float, tilt_deg: float = NO_TILT,
                       azimuth_deg: float = 0.0, gamma_deg=0.0) -> np.ndarray:
    """Emission angle theta in degrees, in [0, 180], from the sample normal.

    The angle is unsigned. Where ``polar + beta < 0``, the existing signed
    convention gives a negative value; its magnitude equals this angle.
    """
    c = np.clip(emission_cos(beta_deg, polar_deg, tilt_deg, azimuth_deg, gamma_deg), -1.0, 1.0)
    return np.rad2deg(np.arccos(c))


def photon_electron_angle_deg(beta_deg, gamma_deg=0.0, xray_direction=_XRAY_DEFAULT) -> np.ndarray:
    """Angle alpha (degrees) between the X-ray propagation direction and the electron.

    Independent of polar, tilt and azimuth, which is why this function does not
    take them.
    """
    s = np.asarray(xray_direction, dtype=float)
    if s.shape != (3,):
        raise ValueError(f"xray_direction must be a 3-vector, got shape {s.shape}")
    norm = np.linalg.norm(s)
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError(f"xray_direction must be a non-zero finite vector, got {xray_direction!r}")
    s = s / norm
    c = np.clip(electron_direction(beta_deg, gamma_deg) @ s, -1.0, 1.0)
    return np.rad2deg(np.arccos(c))
