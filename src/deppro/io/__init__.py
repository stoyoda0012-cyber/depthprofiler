"""
deppro.io - MATLAB互換 I/O モジュール

DepthProfiler GUI と互換性のある H5 ファイルの読み書き。
"""

from .h5_writer import H5Writer, create_gui_compatible_h5

__all__ = ['H5Writer', 'create_gui_compatible_h5']
