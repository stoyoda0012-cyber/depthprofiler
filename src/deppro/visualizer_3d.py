"""
3D/4D Depth Profile Visualizer - MATLAB Style
==============================================

Layer-stack visualization like MATLAB's Present3DProfile.m:
- Horizontal layer surfaces with spatial variation
- Side walls showing layer structure
- Time evolution animation (oxide growth)

Two backends available:
- Plotly: Interactive HTML output (recommended)
- PyVista: VTK-based rendering

Date: 2026-02-01
"""

# Annotations such as ``-> pv.PolyData`` must not be evaluated at import time:
# pyvista, plotly and napari are optional and are ``None`` when absent.
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
from pathlib import Path

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    go = None

try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False
    pv = None

try:
    import napari
    HAS_NAPARI = True
except ImportError:
    HAS_NAPARI = False
    napari = None


# =============================================================================
# Layer Colors (MATLAB-compatible)
# =============================================================================

LAYER_COLORS = {
    'AlOx': [0.75, 0.45, 0.20],   # Brown
    'TiOx': [0.25, 0.70, 0.35],   # Green
    'SiO2': [0.20, 0.75, 0.85],   # Cyan
    'Si': [0.25, 0.35, 0.85],     # Blue
    'default': [0.5, 0.5, 0.5],
}

# Plotly RGB strings
LAYER_COLORS_RGB = {
    'AlOx': 'rgb(191, 115, 51)',   # Brown
    'TiOx': 'rgb(64, 179, 89)',    # Green
    'SiO2': 'rgb(51, 191, 217)',   # Cyan
    'Si': 'rgb(64, 89, 217)',      # Blue
    'default': 'rgb(128, 128, 128)',
}

# Napari colormaps (RGBA, 0-1 range)
LAYER_COLORMAPS = {
    'AlOx': 'PiYG',
    'TiOx': 'green',
    'SiO2': 'cyan',
    'Si': 'blue',
}


# =============================================================================
# Layer Stack Simulator
# =============================================================================

@dataclass
class LayerParams:
    """Parameters for multi-layer stack simulation."""
    # Layer thicknesses (Å)
    cap_thickness: float = 15.0
    initial_sio2_thickness: float = 5.0
    final_sio2_thickness: float = 25.0
    substrate_depth: float = 50.0

    # Spatial variation
    amplitude: float = 2.0        # Interface roughness amplitude
    wavelength: float = 4.0       # Spatial wavelength


class LayerStackSimulator:
    """
    Multi-layer stack simulation with interface roughness.

    Generates layer interfaces as 2D surfaces with sin/cos variation.
    """

    def __init__(
        self,
        nx: int = 10,
        ny: int = 10,
        n_times: int = 20,
        params: Optional[LayerParams] = None,
    ):
        self.nx = nx
        self.ny = ny
        self.n_times = n_times
        self.params = params or LayerParams()

        # Grid coordinates (one extra for surface vertices)
        self.x = np.arange(nx + 1)
        self.y = np.arange(ny + 1)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='ij')

        # Layer names and colors
        self.layers = ['AlOx', 'SiO2', 'Si']
        self.colors = [LAYER_COLORS[name] for name in self.layers]

        # Interface positions for each time step
        # interfaces[t] = list of Z arrays (nx+1, ny+1)
        self.interfaces = []

    def _interface_surface(self, base_z: float, amplitude: float = None) -> np.ndarray:
        """Create interface surface with spatial variation."""
        p = self.params
        amp = amplitude if amplitude is not None else p.amplitude
        wl = p.wavelength

        z = base_z + amp * np.sin(2 * np.pi * self.X / wl) * np.cos(2 * np.pi * self.Y / wl)
        return z

    def generate(self):
        """Generate interface positions for all time steps."""
        p = self.params

        for t in range(self.n_times):
            t_frac = t / max(1, self.n_times - 1)

            # SiO2 thickness grows with time
            sio2_thick = p.initial_sio2_thickness + t_frac * (
                p.final_sio2_thickness - p.initial_sio2_thickness
            )

            # Interface positions (depth from surface)
            z0 = self._interface_surface(0, amplitude=0)  # Top surface (flat)
            z1 = self._interface_surface(p.cap_thickness, amplitude=p.amplitude * 0.8)  # AlOx/SiO2
            z2 = self._interface_surface(p.cap_thickness + sio2_thick, amplitude=p.amplitude)  # SiO2/Si
            z3 = self._interface_surface(p.substrate_depth, amplitude=0)  # Bottom (flat)

            self.interfaces.append([z0, z1, z2, z3])

        print(f"Generated {self.n_times} frames: SiO2 {p.initial_sio2_thickness:.0f}→{p.final_sio2_thickness:.0f} Å")

    def get_sio2_thickness(self, t: int) -> float:
        """Get average SiO2 thickness at time t."""
        z1 = self.interfaces[t][1]  # AlOx/SiO2
        z2 = self.interfaces[t][2]  # SiO2/Si
        return np.mean(z2 - z1)


# =============================================================================
# MATLAB-Style Visualizer
# =============================================================================

class MatlabStyleVisualizer:
    """
    MATLAB Present3DProfile.m style visualization.

    Features:
    - Each layer as a colored horizontal surface
    - Side walls (4 edges) showing layer cross-section
    - Smooth shading and lighting
    """

    def __init__(self, sim: LayerStackSimulator, window_size=(1200, 900)):
        if not HAS_PYVISTA:
            raise ImportError("PyVista required: pip install pyvista")
        self.sim = sim
        self.window_size = window_size

    def _create_layer_surface(self, z: np.ndarray) -> pv.PolyData:
        """Create a surface mesh from Z height array."""
        nx, ny = self.sim.nx, self.sim.ny
        X, Y = self.sim.X, self.sim.Y

        points = np.stack([X, Y, z], axis=-1).reshape(-1, 3)

        grid = pv.StructuredGrid()
        grid.points = points
        grid.dimensions = (nx + 1, ny + 1, 1)

        return grid.extract_surface()

    def _create_side_wall(
        self,
        z_top: np.ndarray,
        z_bot: np.ndarray,
        edge: str,
    ) -> pv.PolyData:
        """Create side wall surface between two interfaces."""
        nx, ny = self.sim.nx, self.sim.ny

        if edge == 'x0':
            # X = 0 edge
            n = ny + 1
            points = np.zeros((n * 2, 3))
            for j in range(n):
                points[j * 2] = [0, j, z_top[0, j]]
                points[j * 2 + 1] = [0, j, z_bot[0, j]]
            dims = (2, n, 1)

        elif edge == 'xn':
            # X = nx edge
            n = ny + 1
            points = np.zeros((n * 2, 3))
            for j in range(n):
                points[j * 2] = [nx, j, z_top[nx, j]]
                points[j * 2 + 1] = [nx, j, z_bot[nx, j]]
            dims = (2, n, 1)

        elif edge == 'y0':
            # Y = 0 edge
            n = nx + 1
            points = np.zeros((n * 2, 3))
            for i in range(n):
                points[i * 2] = [i, 0, z_top[i, 0]]
                points[i * 2 + 1] = [i, 0, z_bot[i, 0]]
            dims = (2, n, 1)

        elif edge == 'yn':
            # Y = ny edge
            n = nx + 1
            points = np.zeros((n * 2, 3))
            for i in range(n):
                points[i * 2] = [i, ny, z_top[i, ny]]
                points[i * 2 + 1] = [i, ny, z_bot[i, ny]]
            dims = (2, n, 1)

        else:
            raise ValueError(f"Unknown edge: {edge}")

        grid = pv.StructuredGrid()
        grid.points = points
        grid.dimensions = dims

        return grid.extract_surface()

    def _add_layer(
        self,
        plotter: pv.Plotter,
        z_top: np.ndarray,
        z_bot: np.ndarray,
        color: List[float],
        opacity: float = 0.95,
    ):
        """Add a single layer (top surface + 4 side walls)."""
        # Top surface
        surf = self._create_layer_surface(z_top)
        plotter.add_mesh(
            surf, color=color, opacity=opacity,
            smooth_shading=True, specular=0.3,
        )

        # Side walls (slightly darker)
        side_color = [c * 0.85 for c in color]
        for edge in ['x0', 'xn', 'y0', 'yn']:
            wall = self._create_side_wall(z_top, z_bot, edge)
            plotter.add_mesh(
                wall, color=side_color, opacity=opacity * 0.95,
                smooth_shading=True,
            )

    def plot_frame(
        self,
        t: int,
        azimuth: float = -50,
        elevation: float = 35,
    ) -> pv.Plotter:
        """Plot single frame."""
        plotter = pv.Plotter(window_size=self.window_size)
        plotter.set_background([0.98, 0.98, 1.0])

        interfaces = self.sim.interfaces[t]

        # Add each layer
        for i in range(len(interfaces) - 1):
            z_top = interfaces[i]
            z_bot = interfaces[i + 1]
            color = self.sim.colors[i]
            self._add_layer(plotter, z_top, z_bot, color)

        # Bottom surface (substrate bottom)
        z_bot = interfaces[-1]
        surf_bot = self._create_layer_surface(z_bot)
        plotter.add_mesh(
            surf_bot, color=self.sim.colors[-1],
            opacity=0.9, smooth_shading=True,
        )

        # Axes
        plotter.add_axes(xlabel='X', ylabel='Y', zlabel='Depth (Å)', line_width=2)

        # Title
        sio2 = self.sim.get_sio2_thickness(t)
        plotter.add_title(
            f'AlOx / SiO2 / Si  |  t={t}  |  SiO2 ≈ {sio2:.1f} Å',
            font_size=11,
        )

        # Camera
        plotter.camera_position = 'iso'
        plotter.camera.elevation = elevation
        plotter.camera.azimuth = azimuth
        plotter.camera.zoom(1.15)

        # Lighting
        plotter.add_light(pv.Light(position=(30, 30, -40), intensity=0.8))

        return plotter

    def animate(
        self,
        output_gif: Optional[str] = None,
        fps: int = 10,
        rotation_speed: float = 3.0,
    ):
        """Create animated GIF."""
        off_screen = output_gif is not None
        plotter = pv.Plotter(window_size=self.window_size, off_screen=off_screen)
        plotter.set_background([0.98, 0.98, 1.0])

        if output_gif:
            plotter.open_gif(output_gif, fps=fps)

        for frame in range(self.sim.n_times):
            plotter.clear()

            interfaces = self.sim.interfaces[frame]

            # Add layers
            for i in range(len(interfaces) - 1):
                z_top = interfaces[i]
                z_bot = interfaces[i + 1]
                color = self.sim.colors[i]
                self._add_layer(plotter, z_top, z_bot, color)

            # Bottom
            surf_bot = self._create_layer_surface(interfaces[-1])
            plotter.add_mesh(surf_bot, color=self.sim.colors[-1], opacity=0.9, smooth_shading=True)

            # Title
            sio2 = self.sim.get_sio2_thickness(frame)
            plotter.add_title(
                f'AlOx / SiO2 / Si  |  t={frame}/{self.sim.n_times-1}  |  SiO2 ≈ {sio2:.1f} Å',
                font_size=10,
            )

            # Camera with rotation
            plotter.camera_position = 'iso'
            plotter.camera.elevation = 35
            plotter.camera.azimuth = -50 + frame * rotation_speed
            plotter.camera.zoom(1.15)

            # Lighting
            plotter.add_light(pv.Light(position=(30, 30, -40), intensity=0.8))

            if output_gif:
                plotter.write_frame()

        if output_gif:
            plotter.close()
            print(f"Saved: {output_gif}")
        else:
            plotter.show()


# =============================================================================
# Plotly-Based Visualizer (Recommended)
# =============================================================================

class PlotlyLayerVisualizer:
    """
    Plotly-based solid layer visualization.

    Features:
    - Interactive HTML output
    - Solid layers with side walls
    - No transparency issues
    - Works in any browser
    """

    def __init__(self, sim: LayerStackSimulator):
        if not HAS_PLOTLY:
            raise ImportError("Plotly required: pip install plotly")
        self.sim = sim

    def _rgb_to_colorscale(self, rgb: str) -> list:
        """Convert RGB string to Plotly colorscale."""
        return [[0, rgb], [1, rgb]]

    def _add_side_wall(
        self,
        fig: go.Figure,
        z_top: np.ndarray,
        z_bot: np.ndarray,
        edge: str,
        color: str,
        opacity: float = 1.0,
    ):
        """Add a side wall surface."""
        nx, ny = self.sim.nx, self.sim.ny

        if edge == 'x0':
            # X = 0 edge
            y_line = np.arange(ny + 1)
            z_line_top = z_top[0, :]
            z_line_bot = z_bot[0, :]
            X_wall = np.zeros((2, ny + 1))
            Y_wall = np.vstack([y_line, y_line])
            Z_wall = np.vstack([z_line_top, z_line_bot])

        elif edge == 'xn':
            # X = nx edge
            y_line = np.arange(ny + 1)
            z_line_top = z_top[nx, :]
            z_line_bot = z_bot[nx, :]
            X_wall = np.full((2, ny + 1), nx)
            Y_wall = np.vstack([y_line, y_line])
            Z_wall = np.vstack([z_line_top, z_line_bot])

        elif edge == 'y0':
            # Y = 0 edge
            x_line = np.arange(nx + 1)
            z_line_top = z_top[:, 0]
            z_line_bot = z_bot[:, 0]
            X_wall = np.vstack([x_line, x_line])
            Y_wall = np.zeros((2, nx + 1))
            Z_wall = np.vstack([z_line_top, z_line_bot])

        elif edge == 'yn':
            # Y = ny edge
            x_line = np.arange(nx + 1)
            z_line_top = z_top[:, ny]
            z_line_bot = z_bot[:, ny]
            X_wall = np.vstack([x_line, x_line])
            Y_wall = np.full((2, nx + 1), ny)
            Z_wall = np.vstack([z_line_top, z_line_bot])

        else:
            raise ValueError(f"Unknown edge: {edge}")

        fig.add_trace(go.Surface(
            x=X_wall, y=Y_wall, z=-Z_wall,  # Negative Z for depth convention
            colorscale=self._rgb_to_colorscale(color),
            showscale=False,
            opacity=opacity,
            lighting=dict(ambient=0.6, diffuse=0.8, specular=0.2),
        ))

    def plot_frame(
        self,
        t: int = 0,
        output_html: Optional[str] = None,
        show: bool = True,
        title: Optional[str] = None,
    ) -> go.Figure:
        """
        Create solid layer visualization for a single frame.

        Parameters
        ----------
        t : int
            Time step index
        output_html : str, optional
            Path to save HTML file
        show : bool
            Whether to display the figure
        title : str, optional
            Custom title

        Returns
        -------
        go.Figure
            Plotly figure object
        """
        interfaces = self.sim.interfaces[t]
        X, Y = self.sim.X, self.sim.Y
        nx, ny = self.sim.nx, self.sim.ny

        # Layer definitions: (name, z_top_idx, z_bot_idx)
        layer_defs = [
            ('AlOx', 0, 1),
            ('SiO2', 1, 2),
            ('Si', 2, 3),
        ]

        fig = go.Figure()

        # Top surface (Z=0, flat)
        z0 = interfaces[0]
        top_color = LAYER_COLORS_RGB['AlOx']
        fig.add_trace(go.Surface(
            x=X, y=Y, z=-z0,
            colorscale=self._rgb_to_colorscale(top_color),
            showscale=False,
            opacity=1.0,
            lighting=dict(ambient=0.7, diffuse=0.8, specular=0.3),
        ))

        # Each layer: interface surface + 4 side walls
        for name, top_idx, bot_idx in layer_defs:
            z_top = interfaces[top_idx]
            z_bot = interfaces[bot_idx]
            color = LAYER_COLORS_RGB.get(name, LAYER_COLORS_RGB['default'])

            # Interface surface
            fig.add_trace(go.Surface(
                x=X, y=Y, z=-z_bot,
                colorscale=self._rgb_to_colorscale(color),
                showscale=False,
                opacity=1.0,
                lighting=dict(ambient=0.6, diffuse=0.8, specular=0.2),
            ))

            # Side walls
            for edge in ['x0', 'xn', 'y0', 'yn']:
                self._add_side_wall(fig, z_top, z_bot, edge, color, opacity=1.0)

        # Layout
        sio2_thick = self.sim.get_sio2_thickness(t)
        if title is None:
            title = f'AlOx / SiO2 / Si Layer Stack | t={t} | SiO2 ≈ {sio2_thick:.1f} Å'

        fig.update_layout(
            title=dict(text=title, x=0.5),
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Depth (Å)',
                aspectmode='manual',
                aspectratio=dict(x=1, y=1, z=0.6),
                camera=dict(
                    eye=dict(x=1.2, y=1.2, z=1.3),
                    up=dict(x=0, y=0, z=1),
                ),
                bgcolor='rgb(250, 250, 255)',
            ),
            margin=dict(l=0, r=0, t=50, b=0),
            width=1000,
            height=800,
        )

        if output_html:
            fig.write_html(output_html)
            print(f"Saved: {output_html}")

        if show:
            fig.show()

        return fig

    def create_animation_frames(self) -> List[go.Frame]:
        """Create animation frames for all time steps."""
        frames = []

        for t in range(self.sim.n_times):
            # Create figure for this frame
            fig_t = self.plot_frame(t, show=False)
            frames.append(go.Frame(
                data=fig_t.data,
                name=str(t),
            ))

        return frames

    def animate(
        self,
        output_html: Optional[str] = None,
        show: bool = True,
        frame_duration: int = 200,
    ) -> go.Figure:
        """
        Create animated visualization with play/pause controls.

        Parameters
        ----------
        output_html : str, optional
            Path to save HTML file
        show : bool
            Whether to display the figure
        frame_duration : int
            Milliseconds per frame

        Returns
        -------
        go.Figure
            Plotly figure with animation
        """
        # Start with first frame
        fig = self.plot_frame(0, show=False)

        # Add animation frames
        frames = self.create_animation_frames()
        fig.frames = frames

        # Add animation controls
        fig.update_layout(
            updatemenus=[
                dict(
                    type='buttons',
                    showactive=False,
                    y=0,
                    x=0.1,
                    xanchor='right',
                    yanchor='top',
                    buttons=[
                        dict(
                            label='▶ Play',
                            method='animate',
                            args=[None, dict(
                                frame=dict(duration=frame_duration, redraw=True),
                                fromcurrent=True,
                                mode='immediate',
                            )],
                        ),
                        dict(
                            label='⏸ Pause',
                            method='animate',
                            args=[[None], dict(
                                frame=dict(duration=0, redraw=False),
                                mode='immediate',
                            )],
                        ),
                    ],
                ),
            ],
            sliders=[
                dict(
                    active=0,
                    yanchor='top',
                    xanchor='left',
                    currentvalue=dict(
                        font=dict(size=12),
                        prefix='Time step: ',
                        visible=True,
                        xanchor='right',
                    ),
                    pad=dict(b=10, t=50),
                    len=0.9,
                    x=0.1,
                    y=0,
                    steps=[
                        dict(
                            args=[[str(t)], dict(
                                frame=dict(duration=frame_duration, redraw=True),
                                mode='immediate',
                            )],
                            label=str(t),
                            method='animate',
                        )
                        for t in range(self.sim.n_times)
                    ],
                ),
            ],
        )

        if output_html:
            fig.write_html(output_html)
            print(f"Saved animated visualization: {output_html}")

        if show:
            fig.show()

        return fig


# =============================================================================
# Napari-Based Visualizer (4D Volume Viewer)
# =============================================================================

class NapariLayerVisualizer:
    """
    Napari-based volume visualization.

    Features:
    - 4D viewer (time + 3D space)
    - Interactive slicing
    - Volume rendering with multiple layers
    - Built-in animation controls
    """

    def __init__(self, sim: LayerStackSimulator):
        if not HAS_NAPARI:
            raise ImportError("Napari required: pip install napari[all]")
        self.sim = sim

    def _create_concentration_volume(self, t: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create 3D concentration volumes for each layer at time t.

        Returns (cap_vol, sio2_vol, si_vol) each with shape (nz, ny, nx)
        """
        nx, ny = self.sim.nx, self.sim.ny
        interfaces = self.sim.interfaces[t]

        # Z grid (depth from 0 to substrate_depth)
        max_depth = self.sim.params.substrate_depth
        nz = int(max_depth) + 1
        z_grid = np.arange(nz)

        # Create X, Y, Z grids
        X, Y = self.sim.X, self.sim.Y  # (nx+1, ny+1)

        # Initialize concentration volumes
        cap_vol = np.zeros((nz, ny + 1, nx + 1), dtype=np.float32)
        sio2_vol = np.zeros((nz, ny + 1, nx + 1), dtype=np.float32)
        si_vol = np.zeros((nz, ny + 1, nx + 1), dtype=np.float32)

        # Interface positions
        z_cap_sio2 = interfaces[1]  # AlOx/SiO2 interface
        z_sio2_si = interfaces[2]    # SiO2/Si interface

        # Fill volumes based on interface positions
        for iz, z in enumerate(z_grid):
            # AlOx: z < z_cap_sio2
            cap_vol[iz] = np.where(z < z_cap_sio2, 1.0, 0.0)

            # SiO2: z_cap_sio2 <= z < z_sio2_si
            sio2_vol[iz] = np.where((z >= z_cap_sio2) & (z < z_sio2_si), 1.0, 0.0)

            # Si: z >= z_sio2_si
            si_vol[iz] = np.where(z >= z_sio2_si, 1.0, 0.0)

        return cap_vol, sio2_vol, si_vol

    def _create_4d_volumes(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create 4D volumes (time, z, y, x) for all time steps.
        """
        volumes_cap = []
        volumes_sio2 = []
        volumes_si = []

        for t in range(self.sim.n_times):
            cap, sio2, si = self._create_concentration_volume(t)
            volumes_cap.append(cap)
            volumes_sio2.append(sio2)
            volumes_si.append(si)

        return (
            np.stack(volumes_cap, axis=0),
            np.stack(volumes_sio2, axis=0),
            np.stack(volumes_si, axis=0),
        )

    def view(
        self,
        t: Optional[int] = None,
        mode: str = 'labels',
    ) -> 'napari.Viewer':
        """
        Open napari viewer with layer visualization.

        Parameters
        ----------
        t : int, optional
            Time step. If None, shows all time steps (4D)
        mode : str
            'labels': Segmented labels (discrete)
            'image': Concentration images (continuous)
            'surface': 3D surface meshes

        Returns
        -------
        napari.Viewer
            Napari viewer instance
        """
        viewer = napari.Viewer(title='AlOx/SiO2/Si Layer Stack')

        if t is not None:
            # Single time step - 3D view
            cap, sio2, si = self._create_concentration_volume(t)

            if mode == 'labels':
                # Combined label volume
                labels = np.zeros_like(cap, dtype=np.uint8)
                labels[cap > 0.5] = 1  # AlOx
                labels[sio2 > 0.5] = 2  # SiO2
                labels[si > 0.5] = 3    # Si

                viewer.add_labels(
                    labels,
                    name='Layers',
                    opacity=0.8,
                )

            elif mode == 'image':
                # Separate image layers
                viewer.add_image(
                    cap, name='AlOx',
                    colormap='orange', opacity=0.7,
                    blending='additive',
                )
                viewer.add_image(
                    sio2, name='SiO2',
                    colormap='cyan', opacity=0.7,
                    blending='additive',
                )
                viewer.add_image(
                    si, name='Si',
                    colormap='blue', opacity=0.5,
                    blending='additive',
                )

            elif mode == 'surface':
                # Surface meshes from marching cubes
                self._add_surfaces(viewer, t)

        else:
            # All time steps - 4D view
            cap_4d, sio2_4d, si_4d = self._create_4d_volumes()

            if mode == 'labels':
                labels_4d = np.zeros_like(cap_4d, dtype=np.uint8)
                labels_4d[cap_4d > 0.5] = 1
                labels_4d[sio2_4d > 0.5] = 2
                labels_4d[si_4d > 0.5] = 3

                viewer.add_labels(
                    labels_4d,
                    name='Layers (4D)',
                    opacity=0.8,
                )
            else:
                viewer.add_image(
                    cap_4d, name='AlOx',
                    colormap='orange', opacity=0.7,
                    blending='additive',
                )
                viewer.add_image(
                    sio2_4d, name='SiO2',
                    colormap='cyan', opacity=0.7,
                    blending='additive',
                )
                viewer.add_image(
                    si_4d, name='Si',
                    colormap='blue', opacity=0.5,
                    blending='additive',
                )

            print(f"4D volume shape: {cap_4d.shape} (t, z, y, x)")
            print("Use the slider to navigate through time steps")

        # Set 3D rendering mode
        viewer.dims.ndisplay = 3

        # Adjust camera
        viewer.camera.angles = (30, -45, 0)
        viewer.camera.zoom = 0.8

        return viewer

    def _add_surfaces(self, viewer: 'napari.Viewer', t: int):
        """Add surface meshes using marching cubes."""
        try:
            from skimage import measure
        except ImportError:
            print("skimage required for surface mode: pip install scikit-image")
            return

        cap, sio2, si = self._create_concentration_volume(t)

        # AlOx surface
        if cap.max() > 0.5:
            verts, faces, _, _ = measure.marching_cubes(cap, level=0.5)
            viewer.add_surface(
                (verts, faces),
                name='AlOx surface',
                colormap='orange',
                opacity=0.8,
            )

        # SiO2 surface
        if sio2.max() > 0.5:
            verts, faces, _, _ = measure.marching_cubes(sio2, level=0.5)
            viewer.add_surface(
                (verts, faces),
                name='SiO2 surface',
                colormap='cyan',
                opacity=0.8,
            )

        # Si surface (top only)
        if si.max() > 0.5:
            verts, faces, _, _ = measure.marching_cubes(si, level=0.5)
            viewer.add_surface(
                (verts, faces),
                name='Si surface',
                colormap='blue',
                opacity=0.6,
            )


def view_in_napari(
    sim: Optional[LayerStackSimulator] = None,
    t: Optional[int] = None,
    mode: str = 'labels',
    **kwargs,
) -> 'napari.Viewer':
    """
    Quick function to view layer stack in napari.

    Parameters
    ----------
    sim : LayerStackSimulator, optional
        Simulator instance. If None, creates default simulation.
    t : int, optional
        Time step. If None, shows all time steps (4D).
    mode : str
        'labels', 'image', or 'surface'
    **kwargs
        Additional arguments for LayerParams

    Returns
    -------
    napari.Viewer

    Examples
    --------
    >>> viewer = view_in_napari()  # Default 4D view
    >>> viewer = view_in_napari(t=10, mode='surface')  # Single time, surfaces
    """
    if sim is None:
        params = LayerParams(
            cap_thickness=kwargs.get('cap_thickness', 15.0),
            initial_sio2_thickness=kwargs.get('initial_sio2', 5.0),
            final_sio2_thickness=kwargs.get('final_sio2', 25.0),
            substrate_depth=50.0,
            amplitude=kwargs.get('amplitude', 2.5),
            wavelength=kwargs.get('wavelength', 4.0),
        )
        sim = LayerStackSimulator(
            nx=kwargs.get('nx', 10),
            ny=kwargs.get('ny', 10),
            n_times=kwargs.get('n_times', 20),
            params=params,
        )
        sim.generate()

    vis = NapariLayerVisualizer(sim)
    return vis.view(t=t, mode=mode)


# =============================================================================
# Convenience Functions
# =============================================================================

def create_layer_visualization(
    layers: List[Tuple[str, float]],
    nx: int = 10,
    ny: int = 10,
    amplitude: float = 2.0,
    wavelength: float = 4.0,
    output_html: Optional[str] = None,
    backend: str = 'plotly',
) -> go.Figure:
    """
    Create a static layer stack visualization.

    Parameters
    ----------
    layers : list of (name, thickness) tuples
        Layer stack from top to bottom, e.g.:
        [('AlOx', 15.0), ('SiO2', 10.0), ('Si', 50.0)]
    nx, ny : int
        Grid resolution
    amplitude : float
        Interface roughness amplitude (Å)
    wavelength : float
        Spatial wavelength of roughness
    output_html : str, optional
        Path to save HTML file
    backend : str
        'plotly' (recommended) or 'pyvista'

    Returns
    -------
    go.Figure or pv.Plotter
        Visualization object

    Examples
    --------
    >>> fig = create_layer_visualization(
    ...     [('AlOx', 15), ('SiO2', 20), ('Si', 50)],
    ...     output_html='my_stack.html',
    ... )
    """
    # Convert layers to params
    if len(layers) >= 3:
        params = LayerParams(
            cap_thickness=layers[0][1],
            initial_sio2_thickness=layers[1][1],
            final_sio2_thickness=layers[1][1],
            substrate_depth=sum(l[1] for l in layers),
            amplitude=amplitude,
            wavelength=wavelength,
        )
    else:
        params = LayerParams(amplitude=amplitude, wavelength=wavelength)

    sim = LayerStackSimulator(nx=nx, ny=ny, n_times=1, params=params)
    sim.generate()

    if backend == 'plotly':
        vis = PlotlyLayerVisualizer(sim)
        return vis.plot_frame(0, output_html=output_html)
    else:
        vis = MatlabStyleVisualizer(sim)
        return vis.plot_frame(0)


def create_oxidation_animation(
    initial_sio2: float = 5.0,
    final_sio2: float = 25.0,
    cap_thickness: float = 15.0,
    n_times: int = 24,
    nx: int = 10,
    ny: int = 10,
    output_html: Optional[str] = None,
    backend: str = 'plotly',
) -> go.Figure:
    """
    Create an animation of oxide growth.

    Parameters
    ----------
    initial_sio2 : float
        Initial SiO2 thickness (Å)
    final_sio2 : float
        Final SiO2 thickness (Å)
    cap_thickness : float
        Cap layer (AlOx) thickness (Å)
    n_times : int
        Number of animation frames
    nx, ny : int
        Grid resolution
    output_html : str, optional
        Path to save HTML file
    backend : str
        'plotly' (recommended) or 'pyvista'

    Returns
    -------
    go.Figure or pv.Plotter
        Animation object

    Examples
    --------
    >>> fig = create_oxidation_animation(
    ...     initial_sio2=5, final_sio2=30,
    ...     output_html='oxidation.html',
    ... )
    """
    params = LayerParams(
        cap_thickness=cap_thickness,
        initial_sio2_thickness=initial_sio2,
        final_sio2_thickness=final_sio2,
        substrate_depth=50.0,
        amplitude=2.5,
        wavelength=4.0,
    )

    sim = LayerStackSimulator(nx=nx, ny=ny, n_times=n_times, params=params)
    sim.generate()

    if backend == 'plotly':
        vis = PlotlyLayerVisualizer(sim)
        return vis.animate(output_html=output_html)
    else:
        vis = MatlabStyleVisualizer(sim)
        vis.animate()
        return None


# =============================================================================
# Demo
# =============================================================================

def run_demo(
    nx: int = 10,
    ny: int = 10,
    n_times: int = 24,
    output_gif: str = None,
    output_html: str = None,
    backend: str = 'plotly',
):
    """
    Run MATLAB-style layer visualization demo.

    Parameters
    ----------
    nx, ny : int
        Grid resolution
    n_times : int
        Number of animation frames
    output_gif : str, optional
        Path for GIF output (PyVista only)
    output_html : str, optional
        Path for HTML output (Plotly only)
    backend : str
        'plotly' (recommended) or 'pyvista'
    """
    print("=" * 60)
    print("AlOx/SiO2/Si Layer Stack - 3D Visualization")
    print(f"Backend: {backend}")
    print("=" * 60)

    params = LayerParams(
        cap_thickness=15.0,
        initial_sio2_thickness=5.0,
        final_sio2_thickness=25.0,
        substrate_depth=50.0,
        amplitude=2.5,
        wavelength=4.0,
    )

    sim = LayerStackSimulator(nx=nx, ny=ny, n_times=n_times, params=params)
    sim.generate()

    if backend == 'plotly':
        vis = PlotlyLayerVisualizer(sim)
        if output_html:
            vis.animate(output_html=output_html, show=False)
        else:
            vis.animate(show=True)
    else:
        vis = MatlabStyleVisualizer(sim)
        vis.animate(output_gif=output_gif, fps=10, rotation_speed=3.0)

    return sim


if __name__ == '__main__':
    # Default to Plotly
    from deppro.paths import ensure_outputs_dir
    run_demo(output_html=str(ensure_outputs_dir() / 'layer_animation.html'), backend='plotly')
