"""Python-only second producer for DepthProfiler datasets.

Generates ground truth, observations, HDF5 and a manifest from a cfg without
MATLAB, the MATLAB Engine, the MATLAB Runtime, or the MATLAB CLI. Nothing is
copied out of an existing dataset: every array here is computed.

Placed OUTSIDE ``deppro.app`` on purpose. ``deppro.app.__init__`` imports
``fitpara_reader`` -> ``toyomacro.io.schema`` at module level, so anything under
it cannot be imported in an environment without the Python-side Toyomacro. This
package depends only on numpy, h5py and ``deppro.imfp`` / ``deppro.transfer_matrix``,
which import cleanly on their own.

The MATLAB producer stays the reference. This one is compared against it stage by
stage (materials -> lambda_basic -> GT -> unnormalized I -> normalized X), and
each stage reports its own agreement rather than one end-to-end number.
"""

from deppro.producer.materials import (
    CompoundTableError,
    MaterialConstants,
    lambda_basic,
    read_compound_table,
)
from deppro.producer.scene import SceneFlags, feature_vector_for_stacks
from deppro.producer.profile import ProfileSpec, depth_profiles

__all__ = [
    "CompoundTableError", "MaterialConstants", "lambda_basic", "read_compound_table",
    "SceneFlags", "feature_vector_for_stacks",
    "ProfileSpec", "depth_profiles",
]
