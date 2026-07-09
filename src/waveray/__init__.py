"""Fast last-stage nearshore spectral wave transformation.

Backward ray-traced linear transfer operators (refraction + shoaling over
local bathymetry, multi-point boundary spectra) with parametric depth-limited
breaking at the target.
"""

from ._version import __version__
from .bathymetry import LocalGrid, fetch_datamesh_bathymetry
from .export import ray_paths_geojson
from .model import SiteModel
from .operator import TransferOperator, build_operator

__all__ = [
    "LocalGrid",
    "SiteModel",
    "TransferOperator",
    "__version__",
    "build_operator",
    "fetch_datamesh_bathymetry",
    "ray_paths_geojson",
]
