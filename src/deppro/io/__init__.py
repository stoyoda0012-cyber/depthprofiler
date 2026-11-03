"""
deppro.io - MATLAB-compatible I/O

Reading and writing H5 files compatible with the DepthProfiler GUI.
"""

from .h5_writer import H5Writer, create_gui_compatible_h5

__all__ = ['H5Writer', 'create_gui_compatible_h5']
