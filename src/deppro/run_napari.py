#!/usr/bin/env python
"""
Launch napari viewer for layer stack visualization.

Usage:
    python -m deppro.run_napari           # 4D labels view (default)
    python -m deppro.run_napari --t 5     # 3D view at t=5
    python -m deppro.run_napari --mode image
    python -m deppro.run_napari --mode surface
"""

import argparse
from .visualizer_3d import view_in_napari


def main():
    parser = argparse.ArgumentParser(description='Napari layer stack viewer')
    parser.add_argument('--t', type=int, default=None, help='Time step (None for 4D)')
    parser.add_argument('--mode', choices=['labels', 'image', 'surface'],
                        default='labels', help='Visualization mode')
    parser.add_argument('--nx', type=int, default=10, help='Grid size X')
    parser.add_argument('--ny', type=int, default=10, help='Grid size Y')
    parser.add_argument('--n_times', type=int, default=20, help='Number of time steps')

    args = parser.parse_args()

    print(f"Opening napari viewer (mode={args.mode}, t={args.t})...")
    viewer = view_in_napari(
        t=args.t,
        mode=args.mode,
        nx=args.nx,
        ny=args.ny,
        n_times=args.n_times,
    )

    # Keep viewer open
    import napari
    napari.run()


if __name__ == '__main__':
    main()
