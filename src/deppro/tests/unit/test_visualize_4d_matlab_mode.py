"""
Tests for the MATLAB-compatible (EzDeP4D sp=2) render mode
===========================================================

Verifies the geometric fidelity contract of
``deppro.visualize_4d.MatlabIsosurfaceRenderer``:

- per-frame, per-element threshold = max(V) * isovalue_degree
- isosurface + six-face interpolated isocaps exist
- MATLAB physical coordinates (meshgrid(y, x, z), depth in Angstroms,
  no display normalisation) and depth-axis inversion
- per-element colour, 0.75 * face_alpha opacity, no edges, Gouraud
- actors, camera and light are reused across frames

Uses only small synthetic in-memory data; nothing is written to disk.
"""

import numpy as np
import pytest

pv = pytest.importorskip("pyvista")

from deppro.visualize_4d import (  # noqa: E402  (after importorskip)
    AnimationConfig,
    DepthProfile4DData,
    IsosurfaceRenderer,
    MatlabIsosurfaceRenderer,
    _make_renderer,
)

NX, NY, NZ, NE, NT = 6, 5, 8, 2, 2
STEP = 30.0  # Angstrom


def _synthetic_data() -> DepthProfile4DData:
    """Two-element layered structure on an anisotropic grid.

    Element 0: top slab, concentration 1.0, full lateral extent.
      Frame 0 occupies depth indices 0-3, frame 1 indices 0-2
      (the interface moves between frames).
    Element 1: bottom slab, concentration 0.4 (its per-element
      threshold is 0.2 — it disappears if a global max is used).
    """
    depth = (np.arange(NZ) + 0.5) * STEP  # slice-centre convention, in A
    data = np.zeros((NT, NX, NY, NZ, NE))
    data[0, :, :, :4, 0] = 1.0
    data[1, :, :, :3, 0] = 1.0
    data[0, :, :, 4:, 1] = 0.4
    data[1, :, :, 3:, 1] = 0.4
    colors = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    return DepthProfile4DData(
        data=data,
        depth=depth,
        element_names=["A", "B"],
        colors=colors,
        step_size=STEP,
        nx=NX,
        ny=NY,
        nt=NT,
        n_depth=NZ,
        n_elem=NE,
    )


def _renderer(**config_kwargs) -> MatlabIsosurfaceRenderer:
    cfg = AnimationConfig(
        render_mode="matlab", depth_profile="none", **config_kwargs
    )
    return MatlabIsosurfaceRenderer(_synthetic_data(), cfg)


# ---------------------------------------------------------------------------
# Factory / default mode
# ---------------------------------------------------------------------------


def test_factory_default_is_not_matlab_mode():
    data = _synthetic_data()
    default = _make_renderer(data, AnimationConfig())
    assert isinstance(default, IsosurfaceRenderer)
    assert not isinstance(default, MatlabIsosurfaceRenderer)

    matlab = _make_renderer(data, AnimationConfig(render_mode="matlab"))
    assert isinstance(matlab, MatlabIsosurfaceRenderer)


# ---------------------------------------------------------------------------
# Mesh geometry (isosurface + isocaps)
# ---------------------------------------------------------------------------


def test_surface_and_caps_exist_per_element():
    meshes = _renderer().compute_frame_meshes(0)
    for elem_idx in (0, 1):
        surf, caps = meshes[elem_idx]
        assert surf is not None and surf.n_points > 0, (
            f"element {elem_idx}: missing isosurface"
        )
        assert caps is not None and caps.n_points > 0, (
            f"element {elem_idx}: missing isocaps"
        )


def test_per_element_threshold_uses_element_max():
    # Element 1 peaks at 0.4.  With isovalue_degree=0.5 its threshold is
    # 0.2 (max(V_elem) * degree); a global-max threshold (0.5) would
    # exceed the element's own maximum and produce nothing.
    surf, caps = _renderer(isovalue_degree=0.5).compute_frame_meshes(0)[1]
    assert surf is not None and surf.n_points > 0
    assert caps is not None and caps.n_points > 0


def test_matlab_axis_convention_and_physical_coords():
    # meshgrid(y, x, z): world X = y index (1..NY), world Y = x index
    # (1..NX).  NX != NY so a swapped mapping is detectable.
    meshes = _renderer().compute_frame_meshes(0)
    surf, caps = meshes[0]
    for mesh in (surf, caps):
        xmin, xmax, ymin, ymax = mesh.bounds[:4]
        assert xmin >= 1.0 - 1e-6 and xmax <= NY + 1e-6
        assert ymin >= 1.0 - 1e-6 and ymax <= NX + 1e-6

    # Element 0's iso-interface: concentration crosses 0.5 midway
    # between depth[3]=105 A and depth[4]=135 A -> exactly 120 A.
    # Physical depth is preserved (no normalisation to the spatial
    # extent), inverted in sign (z_reversed default).
    assert np.allclose(surf.points[:, 2], -120.0, atol=1e-6)


def test_caps_lie_on_boundary_faces_only():
    data = _synthetic_data()
    depth = data.depth
    meshes = _renderer().compute_frame_meshes(0)

    def on_boundary(pts):
        wx, wy, wz = pts[:, 0], pts[:, 1], pts[:, 2]
        return (
            np.isclose(wx, 1.0) | np.isclose(wx, NY)
            | np.isclose(wy, 1.0) | np.isclose(wy, NX)
            | np.isclose(wz, -depth[0]) | np.isclose(wz, -depth[-1])
        )

    # All cap points sit on one of the six boundary planes
    for elem_idx in (0, 1):
        caps = meshes[elem_idx][1]
        assert on_boundary(caps.points).all()

    # Element 0 (top slab): cap on the top face, none on the bottom
    caps0 = meshes[0][1].points[:, 2]
    assert np.any(np.isclose(caps0, -depth[0]))
    assert not np.any(np.isclose(caps0, -depth[-1]))

    # Element 1 (bottom slab): cap on the bottom face, none on the top
    caps1 = meshes[1][1].points[:, 2]
    assert np.any(np.isclose(caps1, -depth[-1]))
    assert not np.any(np.isclose(caps1, -depth[0]))


def test_depth_axis_inversion_flag():
    surf_rev, _ = _renderer(z_reversed=True).compute_frame_meshes(0)[0]
    assert np.allclose(surf_rev.points[:, 2], -120.0, atol=1e-6)

    surf_fwd, _ = _renderer(z_reversed=False).compute_frame_meshes(0)[0]
    assert np.allclose(surf_fwd.points[:, 2], 120.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Scene: colours, opacity, shading, actor/camera/light reuse
# ---------------------------------------------------------------------------


@pytest.fixture()
def scene():
    renderer = _renderer(face_alpha=0.8)
    plotter = pv.Plotter(off_screen=True, window_size=(300, 240))
    try:
        renderer.render_frame(0, plotter)
        yield renderer, plotter
    finally:
        plotter.close()


MESH_ACTOR_PREFIXES = ("matlab_iso_surf_", "matlab_iso_cap_")


def _mesh_actor_names(plotter):
    return [
        n for n in plotter.actors if n.startswith(MESH_ACTOR_PREFIXES)
    ]


def test_actor_color_opacity_shading(scene):
    renderer, plotter = scene
    expected_colors = {0: (1.0, 0.0, 0.0), 1: (0.0, 0.0, 1.0)}

    # surface + cap actor per element (title text actor is separate)
    assert sorted(_mesh_actor_names(plotter)) == sorted(
        f"matlab_iso_{kind}_{e}" for e in (0, 1) for kind in ("surf", "cap")
    )

    for elem_idx, color in expected_colors.items():
        for kind in ("surf", "cap"):
            actor = plotter.actors[f"matlab_iso_{kind}_{elem_idx}"]
            # The element colour is carried by ambient + diffuse.  `prop.color`
            # is a blend of all three material colours, and the specular one
            # is deliberately white (MATLAB SpecularColorReflectance = 1), so
            # it no longer reports the element colour on its own.
            assert np.allclose(
                actor.prop.ambient_color.float_rgb, color, atol=1 / 255
            )
            assert np.allclose(
                actor.prop.diffuse_color.float_rgb, color, atol=1 / 255
            )
            assert actor.prop.opacity == pytest.approx(0.75 * 0.8)
            assert not actor.prop.show_edges
            assert actor.prop.GetInterpolationAsString() == "Gouraud"


def test_actor_camera_light_reused_across_frames(scene):
    renderer, plotter = scene

    names = _mesh_actor_names(plotter)
    # plotter.actors re-wraps on each access, so compare the underlying
    # VTK object addresses, not Python id()s
    actor_addrs = {
        n: plotter.actors[n].GetAddressAsString("vtkActor") for n in names
    }
    camera = plotter.camera
    cam_pos = tuple(camera.position)
    cam_focal = tuple(camera.focal_point)
    lights = list(plotter.renderer.lights)
    surf_ph = renderer._placeholders[(0, "surf")]
    z_frame0 = float(surf_ph.points[:, 2].mean())

    renderer.render_frame(1, plotter)

    # Same actor objects, same camera state, same lights
    assert {
        n: plotter.actors[n].GetAddressAsString("vtkActor") for n in names
    } == actor_addrs
    assert plotter.camera is camera
    assert tuple(plotter.camera.position) == cam_pos
    assert tuple(plotter.camera.focal_point) == cam_focal
    assert list(plotter.renderer.lights) == lights

    # ... but the mesh data was updated in place: the element-0
    # interface moves from 120 A (frame 0) to 90 A (frame 1)
    assert renderer._placeholders[(0, "surf")] is surf_ph
    z_frame1 = float(surf_ph.points[:, 2].mean())
    assert z_frame0 == pytest.approx(-120.0, abs=1e-6)
    assert z_frame1 == pytest.approx(-90.0, abs=1e-6)

    # Title actor supports and received the per-frame text update
    # (a CornerAnnotation would silently ignore it)
    assert hasattr(renderer._title_actor, "SetInput")
    assert "t = 2/2" in renderer._title_actor.GetInput()


def test_stretch_to_fill_is_display_level_only(scene):
    # MATLAB axes default (DataAspectRatio auto) stretches each axis to
    # the plot box.  The renderer replicates this via actor scale;
    # the mesh data itself stays in physical coordinates.
    renderer, plotter = scene
    depth = renderer.data.depth
    z_range = depth[-1] - depth[0]
    expected_scale = (1.0 / (NY - 1), 1.0 / (NX - 1), 1.0 / z_range)

    for name in _mesh_actor_names(plotter):
        assert np.allclose(
            plotter.actors[name].GetScale(), expected_scale, rtol=1e-12
        )

    # Placeholder mesh (mapper input) keeps physical Angstrom depth
    surf_ph = renderer._placeholders[(0, "surf")]
    assert np.allclose(surf_ph.points[:, 2], -120.0, atol=1e-6)


def test_material_matches_matlab_patch_defaults(scene):
    # Measured on MATLAB R2023b: AmbientStrength 0.3, DiffuseStrength 0.6,
    # SpecularStrength 0.9, SpecularExponent 10, SpecularColorReflectance 1
    # (the last one means the highlight takes the light's colour, i.e. white).
    renderer, plotter = scene
    for name in _mesh_actor_names(plotter):
        prop = plotter.actors[name].prop
        assert prop.ambient == pytest.approx(0.3)
        assert prop.diffuse == pytest.approx(0.6)
        assert prop.specular == pytest.approx(0.9)
        assert prop.specular_power == pytest.approx(10.0)
        assert np.allclose(prop.specular_color.float_rgb, (1.0, 1.0, 1.0))


def test_depth_peeling_on_when_translucent(scene):
    # MATLAB resolves transparency by sorting (SortMethod='depth', measured
    # on R2023b).  VTK's translucent pass is unsorted, so the matlab mode
    # turns on depth peeling to get the same order-independent result.
    renderer, plotter = scene
    assert renderer.config.face_alpha == 0.8      # translucent
    assert plotter.renderer.GetUseDepthPeeling()


def test_depth_peeling_skipped_when_opaque():
    # 0.75 * (4/3) == 1.0 exactly; fully opaque needs no peeling.
    renderer = _renderer(face_alpha=4 / 3)
    plotter = pv.Plotter(off_screen=True, window_size=(300, 240))
    try:
        renderer.render_frame(0, plotter)
        assert 0.75 * renderer.config.face_alpha == 1.0
        assert not plotter.renderer.GetUseDepthPeeling()
    finally:
        plotter.close()


def test_depth_peeling_can_be_disabled():
    renderer = _renderer(matlab_depth_peeling=False)
    plotter = pv.Plotter(off_screen=True, window_size=(300, 240))
    try:
        renderer.render_frame(0, plotter)
        assert not plotter.renderer.GetUseDepthPeeling()
    finally:
        plotter.close()


def test_plot_box_aspect_overrides_the_cube():
    # Default (None) = cube, which is what MATLAB's auto stretch-to-fill
    # gives for a square axes (measured: PlotBoxAspectRatio [1 1 1]).
    # An explicit ratio reproduces a non-square axes, e.g. the 327x220 px
    # tile of EzDeP4D's 2x2 layout at view([-105 5]) -> [0.713 1 0.685].
    depth = _synthetic_data().depth
    z_range = depth[-1] - depth[0]
    box = (0.713, 1.0, 0.685)

    renderer = _renderer(plot_box_aspect=box)
    plotter = pv.Plotter(off_screen=True, window_size=(300, 240))
    try:
        renderer.render_frame(0, plotter)
        expected = (box[0] / (NY - 1), box[1] / (NX - 1), box[2] / z_range)
        for name in _mesh_actor_names(plotter):
            assert np.allclose(
                plotter.actors[name].GetScale(), expected, rtol=1e-12
            )
        # mesh data still physical
        assert np.allclose(
            renderer._placeholders[(0, "surf")].points[:, 2], -120.0, atol=1e-6
        )
    finally:
        plotter.close()


def test_camera_and_light_match_matlab_angles(scene):
    renderer, plotter = scene
    cfg = renderer.config

    def direction(az_deg, el_deg):
        az, el = np.radians(az_deg), np.radians(el_deg)
        return np.array([
            np.sin(az) * np.cos(el),
            -np.cos(az) * np.cos(el),
            np.sin(el),
        ])

    # Camera: view([-105 5]) equivalent, orthographic
    cam = plotter.camera
    view_vec = np.asarray(cam.position) - np.asarray(cam.focal_point)
    view_vec /= np.linalg.norm(view_vec)
    assert np.allclose(view_vec, direction(-105.0, 5.0), atol=1e-6)
    assert cam.parallel_projection

    # Light: single fixed directional light, lightangle(-115, 25).
    # lightangle works in MATLAB data space (+z = depth); through
    # ZDir='reverse' the display-space direction has z negated —
    # the light sits slightly below the display horizon (measured
    # against R2023b calibration cubes).
    lights = plotter.renderer.lights
    assert len(lights) == 1
    light = lights[0]
    light_vec = np.asarray(light.position) - np.asarray(light.focal_point)
    light_vec /= np.linalg.norm(light_vec)
    expected = direction(cfg.light_azimuth, cfg.light_elevation)
    expected[2] *= -1.0
    assert np.allclose(light_vec, expected, atol=1e-6)
    assert not light.positional
