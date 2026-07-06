"""Local bathymetry grids for ray tracing.

A :class:`LocalGrid` is a regular grid of water depth [m, positive down] on a
local tangent plane (x east, y north, metres). It is typically built from a
geographic bathymetry ``xarray.DataArray`` (e.g. a Datamesh datasource) via
:meth:`LocalGrid.from_dataarray`, or directly from arrays for testing.

Land is any node with depth <= 0 or NaN; for interpolation those nodes are
treated as depth 0 so that rays approaching land see the depth drop below
the stopping threshold.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import xarray as xr

# Metres per degree of latitude (spherical earth, adequate for domains <~100 km)
_M_PER_DEG_LAT = 111_320.0


@dataclass
class LocalGrid:
    """Regular local-metres depth grid.

    Parameters
    ----------
    x, y : 1D ascending coordinates [m] on the local tangent plane.
    depth : 2D array (ny, nx), water depth [m] positive down. Land <= 0 / NaN.
    lon0, lat0 : geographic origin of the tangent plane (x=0, y=0), optional.
    """

    x: np.ndarray
    y: np.ndarray
    depth: np.ndarray
    lon0: float | None = None
    lat0: float | None = None
    land: np.ndarray = field(init=False)
    _depth_filled: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.x = np.asarray(self.x, dtype=float)
        self.y = np.asarray(self.y, dtype=float)
        self.depth = np.asarray(self.depth, dtype=float)
        if self.depth.shape != (self.y.size, self.x.size):
            raise ValueError(
                f"depth shape {self.depth.shape} != (ny, nx) = {(self.y.size, self.x.size)}"
            )
        if self.x.size < 2 or self.y.size < 2:
            raise ValueError("grid needs at least 2 nodes per axis")
        if not (np.all(np.diff(self.x) > 0) and np.all(np.diff(self.y) > 0)):
            raise ValueError("x and y must be strictly ascending")
        self.land = ~(self.depth > 0)
        self._depth_filled = np.where(self.land, 0.0, self.depth)

    # ------------------------------------------------------------------ #
    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """(xmin, xmax, ymin, ymax) [m]."""
        return float(self.x[0]), float(self.x[-1]), float(self.y[0]), float(self.y[-1])

    @property
    def spacing(self) -> tuple[float, float]:
        """Median (dx, dy) [m]."""
        return float(np.median(np.diff(self.x))), float(np.median(np.diff(self.y)))

    # ------------------------------------------------------------------ #
    @classmethod
    def from_dataarray(cls, da: xr.DataArray, positive: str = "down") -> LocalGrid:
        """Build from a geographic DataArray with 1D lon/lat coordinates.

        Parameters
        ----------
        da : DataArray with dims/coords named (lat, lon), (latitude, longitude)
            or (y, x) holding depth or elevation.
        positive : "down" if values are depths (positive under water),
            "up" if values are elevations (negative under water, e.g. GEBCO).
        """
        names = {n.lower(): n for n in da.dims}
        lon_name = next((names[k] for k in ("lon", "longitude", "x") if k in names), None)
        lat_name = next((names[k] for k in ("lat", "latitude", "y") if k in names), None)
        if lon_name is None or lat_name is None:
            raise ValueError(f"cannot identify lon/lat dims in {da.dims}")
        da = da.transpose(lat_name, lon_name)
        lon = np.asarray(da[lon_name].values, dtype=float)
        lat = np.asarray(da[lat_name].values, dtype=float)
        vals = np.asarray(da.values, dtype=float)
        if lat[0] > lat[-1]:  # descending latitude -> flip
            lat = lat[::-1]
            vals = vals[::-1, :]
        if lon[0] > lon[-1]:
            lon = lon[::-1]
            vals = vals[:, ::-1]
        depth = -vals if positive == "up" else vals

        lon0 = float(lon.mean())
        lat0 = float(lat.mean())
        x = (lon - lon0) * _M_PER_DEG_LAT * np.cos(np.deg2rad(lat0))
        y = (lat - lat0) * _M_PER_DEG_LAT
        return cls(x=x, y=y, depth=depth, lon0=lon0, lat0=lat0)

    # ------------------------------------------------------------------ #
    def to_local(self, lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Geographic degrees -> local metres."""
        if self.lon0 is None or self.lat0 is None:
            raise ValueError("grid has no geographic origin (built from local coords)")
        lon = np.asarray(lon, dtype=float)
        lat = np.asarray(lat, dtype=float)
        x = (lon - self.lon0) * _M_PER_DEG_LAT * np.cos(np.deg2rad(self.lat0))
        y = (lat - self.lat0) * _M_PER_DEG_LAT
        return x, y

    def sample_depth(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Bilinear depth [m, positive down] at local points; land contributes 0."""
        return bilinear(self._depth_filled, self.x, self.y, x, y)


def bilinear(
    grid_vals: np.ndarray,
    gx: np.ndarray,
    gy: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """Bilinear interpolation of ``grid_vals`` (ny, nx) at points (x, y).

    Assumes gx, gy strictly ascending (may be non-uniform). Points outside the
    grid are clamped to the edge (callers stop rays at the boundary anyway).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ix = np.clip(np.searchsorted(gx, x) - 1, 0, gx.size - 2)
    iy = np.clip(np.searchsorted(gy, y) - 1, 0, gy.size - 2)
    tx = np.clip((x - gx[ix]) / (gx[ix + 1] - gx[ix]), 0.0, 1.0)
    ty = np.clip((y - gy[iy]) / (gy[iy + 1] - gy[iy]), 0.0, 1.0)
    return (
        grid_vals[iy, ix] * (1 - tx) * (1 - ty)
        + grid_vals[iy, ix + 1] * tx * (1 - ty)
        + grid_vals[iy + 1, ix] * (1 - tx) * ty
        + grid_vals[iy + 1, ix + 1] * tx * ty
    )


def fetch_datamesh_bathymetry(
    datasource: str,
    bbox: tuple[float, float, float, float],
    variable: str | None = None,
    positive: str = "up",
    token: str | None = None,
) -> LocalGrid:
    """Fetch a bathymetry subset from an Oceanum Datamesh datasource.

    Parameters
    ----------
    datasource : Datamesh datasource id (e.g. a GEBCO or regional grid).
    bbox : (west, south, east, north) in degrees.
    variable : data variable to use; defaults to the first data variable.
    positive : "up" for elevation sources (GEBCO convention), "down" for depth.
    token : Datamesh token; defaults to the DATAMESH_TOKEN environment variable.

    Requires the ``oceanum`` package (``pip install nearshore-transform[datamesh]``).
    """
    from oceanum.datamesh import Connector  # noqa: PLC0415 (optional dependency)

    conn = Connector(token=token) if token else Connector()
    query = {
        "datasource": datasource,
        "geofilter": {
            "type": "bbox",
            "geom": [bbox[0], bbox[1], bbox[2], bbox[3]],
        },
    }
    ds = conn.query(query)
    if isinstance(ds, xr.DataArray):
        da = ds
    else:
        if variable is None:
            variable = next(iter(ds.data_vars))
        da = ds[variable]
    return LocalGrid.from_dataarray(da, positive=positive)
