"""sensmaps — Interactive GUI for diffuse-optical sensitivity maps.

Public API (stable from v1):
    make_s             — compute S for a measurement type
    slice_s            — slice a 3D S array for 2D plotting
    make_color_limits  — compute saturated colormap and limits
    OpticalProperties  — dataclass of optical properties
    GridParams         — dataclass of voxel-grid parameters

More symbols are importable from sensmaps.physics / compute / views / gui
but are not part of the v1 public-API guarantee.
"""

from sensmaps.compute import GridParams, make_s
from sensmaps.physics import OpticalProperties
from sensmaps.views import make_color_limits, slice_s

__version__ = "1.0.0"

__all__ = [
    "GridParams",
    "OpticalProperties",
    "__version__",
    "make_color_limits",
    "make_s",
    "slice_s",
]
