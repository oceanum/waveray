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
from .rays import BoundaryLine
from .spectra import set_wavespectra_attrs, to_specdataset

__all__ = [
    "BoundaryLine",
    "LocalGrid",
    "SiteModel",
    "TransferOperator",
    "__version__",
    "build_operator",
    "fetch_datamesh_bathymetry",
    "ray_paths_geojson",
    "set_wavespectra_attrs",
    "to_specdataset",
]
