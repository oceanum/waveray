"""SWAN-formulation wind forcing for the transfer operator.

Wind input follows SWAN's third-generation source term ``S_in = A + B E``
(SWAN technical documentation, GEN3 KOMEN defaults):

- exponential growth ``B`` of Komen et al. (1984),

      B = max[0, 0.25 (rho_a/rho_w) (28 u*/c cos(theta - theta_w) - 1)] sigma

  which is linear in E and therefore folds into the ray-path exponent of the
  transfer operator exactly like bottom friction, with opposite sign;

- optional linear growth ``A`` of Cavaleri and Malanotte-Rizzoli (1981) with
  the low-frequency filter of Tolman (1992) — SWAN's AGROW keyword,

      A = 1.5e-3 / (2 pi g^2) (u* max[0, cos(theta - theta_w)])^4 H
      H = exp(-(sigma / sigma_PM)^-4),  sigma_PM = 2 pi 0.13 g / (28 u*)

  which is independent of E and integrates along each ray into an additive
  spectrum stored on the operator (it seeds locally generated wind sea in
  bins that carry no boundary energy).

The friction velocity u* comes from U10 through the drag law of Zijlema,
van Vledder and Holthuijsen (2012), SWAN's default since version 41.01.

Only the wind *input* term is represented: the nonlinear sinks that balance
it in SWAN (whitecapping, quadruplets) cannot live in a linear operator, so
the formulation suits the short downscale fetches this package targets, not
full-fetch wave growth. Wind directions at the API are coming-from nautical
degrees (meteorological convention), matching the package's wave directions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr

from .bathymetry import LocalGrid, bilinear

#: rho_air / rho_water used in the Komen growth term. SWAN's documented
#: default; a SWAN run started from its own RHOA = 1.28, RHOW = 1025 reports
#: RHOAW = 0.0012488, 0.1 % lower — far below the drag-law uncertainty.
RHO_AIR_WATER = 0.00125

# Reference wind speed of the Zijlema et al. (2012) drag fit [m/s].
_U_REF = 31.5

_COMPONENT_PAIRS = (("u10", "v10"), ("ugrd10m", "vgrd10m"), ("uwnd", "vwnd"), ("u", "v"))
_SPEED_DIR_PAIR = ("wspd", "wdir")


def drag_coefficient(u10: np.ndarray | float) -> np.ndarray:
    """Wind drag coefficient of Zijlema et al. (2012), SWAN default.

    ``Cd = (0.55 + 2.97 U~ - 1.49 U~^2) * 1e-3`` with ``U~ = U10 / 31.5``.
    Clipped at zero (the fit goes negative above ~68 m/s).
    """
    u = np.asarray(u10, dtype=float)
    ut = u / _U_REF
    return np.maximum((0.55 + 2.97 * ut - 1.49 * ut**2) * 1e-3, 0.0)


def friction_velocity(u10: np.ndarray | float) -> np.ndarray:
    """Friction velocity u* [m/s] from U10 via ``u*^2 = Cd U10^2``."""
    u = np.asarray(u10, dtype=float)
    return np.sqrt(drag_coefficient(u)) * u


def _dir_to_theta(dir_nautical_deg: np.ndarray | float) -> np.ndarray:
    """Coming-from nautical degrees -> going-to math radians.

    Same convention as :func:`waveray.operator.dir_to_theta` (duplicated here
    to keep this module free of an import cycle with ``operator``).
    """
    return np.deg2rad((270.0 - np.asarray(dir_nautical_deg, dtype=float)) % 360.0)


@dataclass
class WindField:
    """Friction-velocity vector on the LocalGrid nodes.

    ``usx``, ``usy`` are the (ny, nx) components of u* [m/s] along the wind's
    going-to direction in math convention (x east, y north). Storing the
    vector (rather than speed and direction separately) keeps interpolation
    well-behaved across direction wrap.
    """

    usx: np.ndarray
    usy: np.ndarray

    def __post_init__(self) -> None:
        self.usx = np.asarray(self.usx, dtype=float)
        self.usy = np.asarray(self.usy, dtype=float)
        if self.usx.shape != self.usy.shape or self.usx.ndim != 2:
            raise ValueError("usx and usy must be matching 2D arrays")

    # ------------------------------------------------------------------ #
    @classmethod
    def uniform(cls, grid: LocalGrid, speed: float, direction: float) -> WindField:
        """Spatially uniform wind: U10 ``speed`` [m/s] coming from
        ``direction`` [nautical deg]."""
        us = float(friction_velocity(speed))
        th = float(_dir_to_theta(direction))
        shape = (grid.y.size, grid.x.size)
        return cls(usx=np.full(shape, us * np.cos(th)), usy=np.full(shape, us * np.sin(th)))

    @classmethod
    def from_dataset(cls, grid: LocalGrid, ds: xr.Dataset) -> WindField:
        """Gridded wind interpolated onto the LocalGrid nodes.

        ``ds`` must hold a single wind snapshot (the operator is stationary)
        as either eastward/northward 10-m components (variable pairs
        ``u10/v10``, ``ugrd10m/vgrd10m``, ``uwnd/vwnd`` or ``u/v``) or as
        speed/direction ``wspd``/``wdir`` (coming-from nautical degrees), on
        1D ``lon``/``lat`` (geographic grids) or ``x``/``y`` (local metres)
        coordinates. Points outside the wind grid take the nearest edge value.
        """
        names = {n.lower(): n for n in ds.data_vars}
        pair = next(
            (p for p in _COMPONENT_PAIRS if p[0] in names and p[1] in names),
            None,
        )
        if pair is not None:
            u_da, v_da = ds[names[pair[0]]], ds[names[pair[1]]]
        elif all(k in names for k in _SPEED_DIR_PAIR):
            wspd, wdir = ds[names["wspd"]], ds[names["wdir"]]
            th = _dir_to_theta(wdir.values)
            u_da = wspd.copy(data=wspd.values * np.cos(th))
            v_da = wspd.copy(data=wspd.values * np.sin(th))
        else:
            raise ValueError(
                "wind Dataset needs component variables "
                f"{' / '.join('+'.join(p) for p in _COMPONENT_PAIRS)} or wspd+wdir; "
                f"got {sorted(ds.data_vars)}"
            )

        u, cx, cy, geographic = _snapshot_grid(u_da)
        v, *_ = _snapshot_grid(v_da)
        if u.shape != v.shape:
            raise ValueError("wind component variables have mismatching shapes")

        # target points: LocalGrid nodes in the wind grid's coordinate system
        gx, gy = np.meshgrid(grid.x, grid.y)
        if geographic:
            if grid.lon0 is None:
                raise ValueError(
                    "wind Dataset has lon/lat coordinates but the grid has no "
                    "geographic origin; supply wind on x/y local-metre coordinates"
                )
            px, py = grid.to_lonlat(gx.ravel(), gy.ravel())
        else:
            px, py = gx.ravel(), gy.ravel()

        u10 = bilinear(u, cx, cy, px, py).reshape(gx.shape)
        v10 = bilinear(v, cx, cy, px, py).reshape(gx.shape)
        if not (np.isfinite(u10).all() and np.isfinite(v10).all()):
            # a single NaN would otherwise poison every operator coefficient
            # without a word (Datamesh wind grids are masked over land)
            raise ValueError(
                "wind field has non-finite values over the bathymetry grid; fill or "
                "mask them before building (e.g. ds.interpolate_na / ds.fillna)"
            )

        speed = np.hypot(u10, v10)
        with np.errstate(invalid="ignore", divide="ignore"):
            scale = np.where(speed > 0, friction_velocity(speed) / np.maximum(speed, 1e-12), 0.0)
        return cls(usx=u10 * scale, usy=v10 * scale)


def _snapshot_grid(da: xr.DataArray) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """One wind component -> (values(ny, nx), x-coord, y-coord, geographic?).

    Squeezes singleton dims (e.g. a length-1 time); rejects a real time axis
    because the operator is stationary. Coordinates are returned ascending
    with the values flipped to match.
    """
    da = da.squeeze(drop=True)
    if "time" in da.dims:
        raise ValueError(
            f"wind must be a single snapshot (stationary operator); got "
            f"{da.sizes['time']} times — select one first"
        )
    names = {n.lower(): n for n in da.dims}
    lon_name = next((names[k] for k in ("lon", "longitude") if k in names), None)
    lat_name = next((names[k] for k in ("lat", "latitude") if k in names), None)
    if lon_name is not None and lat_name is not None:
        geographic = True
    else:
        lon_name = names.get("x")
        lat_name = names.get("y")
        geographic = False
        if lon_name is None or lat_name is None:
            raise ValueError(f"cannot identify lon/lat or x/y dims in wind dims {da.dims}")
    da = da.transpose(lat_name, lon_name)
    cx = np.asarray(da[lon_name].values, dtype=float)
    cy = np.asarray(da[lat_name].values, dtype=float)
    vals = np.asarray(da.values, dtype=float)
    if cy[0] > cy[-1]:
        cy = cy[::-1]
        vals = vals[::-1, :]
    if cx[0] > cx[-1]:
        cx = cx[::-1]
        vals = vals[:, ::-1]
    return vals, cx, cy, geographic


def as_wind_field(
    wind: WindField | tuple[float, float] | xr.Dataset | None, grid: LocalGrid
) -> WindField | None:
    """Normalise the ``wind`` argument of ``build_operator`` to a WindField."""
    if wind is None:
        return None
    if isinstance(wind, WindField):
        expected = (grid.y.size, grid.x.size)
        if wind.usx.shape != expected:
            raise ValueError(f"WindField shape {wind.usx.shape} != grid shape {expected}")
        return wind
    if isinstance(wind, xr.Dataset):
        return WindField.from_dataset(grid, wind)
    if isinstance(wind, (tuple, list)) and len(wind) == 2:
        return WindField.uniform(grid, float(wind[0]), float(wind[1]))
    raise TypeError(
        "wind must be None, a (speed, direction) pair, an xarray Dataset or a WindField; "
        f"got {type(wind).__name__}"
    )
