"""High-level site model: build once, transform whole hindcasts.

Typical use::

    from nearshore_transform import SiteModel

    model = SiteModel.build(
        bathy=bathy_dataarray,          # or a LocalGrid
        target=(114.5961, -28.7767),    # lon, lat (or x, y for local grids)
        boundary_points=[(114.40, -28.90), (114.40, -28.65)],
        freqs=efth.freq.values,
        dirs=efth.dir.values,
    )
    efth_near = model.transform(efth, tide=tide_series)

``efth`` follows the wavespectra convention: dims (..., freq, dir) plus a
``site`` dimension holding the K boundary points (omitted when K == 1),
units m^2 / Hz / deg, dir = coming-from nautical degrees.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr

from .bathymetry import LocalGrid
from .breaking import apply_breaking
from .operator import TransferOperator, build_operator


@dataclass
class SiteModel:
    """A built nearshore transformation for one target point."""

    operator: TransferOperator
    gamma: float = 0.73
    breaking_method: str = "miche"

    # ------------------------------------------------------------------ #
    @classmethod
    def build(
        cls,
        bathy: xr.DataArray | LocalGrid,
        target: tuple[float, float],
        boundary_points: list[tuple[float, float]] | np.ndarray,
        freqs: np.ndarray,
        dirs: np.ndarray,
        positive: str = "down",
        gamma: float = 0.73,
        breaking_method: str = "miche",
        **ray_kwargs,
    ) -> SiteModel:
        """Build the transfer operator for one target site.

        ``target`` and ``boundary_points`` are (lon, lat) when ``bathy`` is
        geographic (DataArray, or LocalGrid with an origin), else (x, y) in
        grid metres. ``ray_kwargs`` pass through to
        :func:`nearshore_transform.operator.build_operator` (nsub, ds,
        max_steps, d_min).
        """
        grid = (
            bathy
            if isinstance(bathy, LocalGrid)
            else LocalGrid.from_dataarray(bathy, positive=positive)
        )
        bpts = np.atleast_2d(np.asarray(boundary_points, dtype=float))
        if grid.lon0 is not None:
            tx, ty = grid.to_local(np.array([target[0]]), np.array([target[1]]))
            bx, by = grid.to_local(bpts[:, 0], bpts[:, 1])
            target_xy = (float(tx[0]), float(ty[0]))
            boundary_xy = np.column_stack([bx, by])
        else:
            target_xy = (float(target[0]), float(target[1]))
            boundary_xy = bpts

        op = build_operator(
            grid,
            target_xy=target_xy,
            boundary_xy=boundary_xy,
            freqs=np.asarray(freqs, dtype=float),
            dirs=np.asarray(dirs, dtype=float),
            **ray_kwargs,
        )
        if grid.lon0 is not None:
            # retained so transform() can verify boundary spectra site order
            op.attrs["bp_lon"] = [float(v) for v in bpts[:, 0]]
            op.attrs["bp_lat"] = [float(v) for v in bpts[:, 1]]
        return cls(operator=op, gamma=gamma, breaking_method=breaking_method)

    # ------------------------------------------------------------------ #
    def transform(
        self,
        efth: xr.DataArray,
        site_dim: str = "site",
        tide: xr.DataArray | np.ndarray | float | None = None,
        breaking: bool = True,
    ) -> xr.DataArray:
        """Transform boundary spectra to the target point.

        Parameters
        ----------
        efth : boundary spectra, dims (..., [site_dim,] freq, dir). The
            ``site_dim`` size must equal the operator's K and follow the
            same order as the boundary points given at build time; it may be
            omitted when K == 1.
        tide : water level offset [m] applied to the target depth for the
            breaking cap: scalar, or array/DataArray matching the leading
            (non-spectral) dims of efth.
        breaking : apply the depth-limited cap (default True).
        """
        op = self.operator
        if "freq" not in efth.dims or "dir" not in efth.dims:
            raise ValueError("efth must have 'freq' and 'dir' dims (wavespectra convention)")
        if not np.allclose(efth["freq"].values, op.freq, rtol=1e-4):
            raise ValueError("efth freq coordinates do not match the operator")
        if not np.allclose(efth["dir"].values % 360.0, op.dir_b % 360.0, atol=1e-3):
            raise ValueError("efth dir coordinates do not match the operator")

        had_site = site_dim in efth.dims
        if not had_site:
            if op.n_boundary != 1:
                raise ValueError(
                    f"operator has K={op.n_boundary} boundary points; efth needs a "
                    f"'{site_dim}' dimension"
                )
            efth = efth.expand_dims({site_dim: 1})
        if efth.sizes[site_dim] != op.n_boundary:
            raise ValueError(
                f"efth {site_dim} size {efth.sizes[site_dim]} != operator K {op.n_boundary}"
            )
        if (
            had_site
            and "bp_lon" in op.attrs
            and "lon" in efth.coords
            and "lat" in efth.coords
            and efth["lon"].size == op.n_boundary
        ):
            # guard against silently reordered boundary spectra
            if not (
                np.allclose(np.asarray(efth["lon"].values), op.attrs["bp_lon"], atol=5e-3)
                and np.allclose(np.asarray(efth["lat"].values), op.attrs["bp_lat"], atol=5e-3)
            ):
                raise ValueError(
                    f"efth {site_dim} lon/lat coordinates do not match the operator's "
                    "boundary points (wrong sites or wrong order)"
                )

        ordered = efth.transpose(..., site_dim, "freq", "dir")
        lead_dims = ordered.dims[:-3]
        out = op.apply(ordered.values)

        scale = None
        if breaking:
            tide_arr: np.ndarray | float
            if tide is None:
                tide_arr = 0.0
            elif isinstance(tide, xr.DataArray):
                tide_arr = tide.transpose(*[d for d in lead_dims if d in tide.dims]).values
            else:
                tide_arr = np.asarray(tide, dtype=float)
            out, scale = apply_breaking(
                out,
                op.freq,
                op.dir_t,
                depth=op.depth_target,
                tide=tide_arr,
                gamma=self.gamma,
                method=self.breaking_method,
            )

        coords = {d: ordered[d] for d in lead_dims if d in ordered.coords}
        coords["freq"] = op.freq
        coords["dir"] = op.dir_t
        result = xr.DataArray(
            out,
            dims=(*lead_dims, "freq", "dir"),
            coords=coords,
            name="efth",
            attrs={
                **efth.attrs,
                "target_x": op.target_x,
                "target_y": op.target_y,
                "depth_target": op.depth_target,
                "breaking": f"{self.breaking_method} gamma={self.gamma}" if breaking else "off",
            },
        )
        if scale is not None:
            result.attrs["breaking_scale_min"] = float(np.min(scale))
        return result

    # ------------------------------------------------------------------ #
    def to_netcdf(self, path: str) -> None:
        ds = self.operator.to_dataset()
        ds.attrs["gamma"] = self.gamma
        ds.attrs["breaking_method"] = self.breaking_method
        ds.to_netcdf(path)

    @classmethod
    def from_netcdf(cls, path: str) -> SiteModel:
        with xr.open_dataset(path) as raw:
            ds = raw.load()
        gamma = float(ds.attrs.pop("gamma", 0.73))
        method = str(ds.attrs.pop("breaking_method", "miche"))
        return cls(operator=TransferOperator.from_dataset(ds), gamma=gamma, breaking_method=method)
