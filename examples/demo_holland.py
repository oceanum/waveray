"""Real-hindcast example: Dutch coast (Noordwijk aan Zee).

Downscales Oceanum's SWAN 1km Dutch waters hindcast spectra
(oceanum_wave_dutch_era5_v1_spec, hourly, ERA5-forced) through the last-stage
nearshore transformation to a point off Noordwijk aan Zee, over GEBCO 2023
bathymetry (the same bathymetry source the parent SWAN model used). Covers
storms Pia and Henk (Dec 2023 - Jan 2024).

Also self-scores: the nearest nearshore hindcast site is transformed too and
compared against SWAN's own spectra there.

Requires DATAMESH_TOKEN and the datamesh extra:

    uv sync --extra datamesh
    uv run python examples/demo_holland.py
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import xarray as xr

from nearshore_transform import SiteModel, fetch_datamesh_bathymetry
from nearshore_transform.breaking import hm0

BBOX = (3.85, 52.00, 4.55, 52.50)  # southern Holland coast around Noordwijk
TARGET = (4.42, 52.24)  # ~1.5 km off Noordwijk aan Zee
BATHY_DATASOURCE = "gebco_2023"  # matches the parent SWAN model's bathymetry
SPEC_DATASOURCE = "oceanum_wave_dutch_era5_v1_spec"
TIMES = ["2023-12-15T00:00:00Z", "2024-01-15T00:00:00Z"]  # storms Pia + Henk
N_BOUNDARY = 6
BOUNDARY_MIN_DEPTH = 16.0  # [m] offshore arc from which boundary sites are drawn


def fetch_sites(conn, site_lonlats: list[tuple[float, float]]) -> xr.DataArray:
    """Fetch full-period spectra for specific sites, one small query each."""
    parts = []
    for lon, lat in site_lonlats:
        ds = conn.query(
            {
                "datasource": SPEC_DATASOURCE,
                "geofilter": {
                    "type": "bbox",
                    "geom": [lon - 0.004, lat - 0.004, lon + 0.004, lat + 0.004],
                },
                "timefilter": {"times": TIMES},
            }
        )
        site_dim = next(d for d in ds["efth"].dims if d not in ("time", "freq", "dir"))
        parts.append(ds["efth"].isel({site_dim: 0}).rename({site_dim: "site"}))
    return xr.concat(parts, dim="site").transpose("time", "site", "freq", "dir")


def main() -> int:
    if not os.environ.get("DATAMESH_TOKEN"):
        print("DATAMESH_TOKEN not set - cannot run the live example")
        return 1

    from oceanum.datamesh import Connector

    conn = Connector()

    print(f"Fetching bathymetry {BATHY_DATASOURCE} {BBOX} ...")
    grid = fetch_datamesh_bathymetry(BATHY_DATASOURCE, bbox=BBOX, positive="up")
    print(f"  grid: {grid.x.size} x {grid.y.size} nodes, spacing ~{grid.spacing[0]:.0f} m")

    # one-timestep probe to locate the hindcast sites inside the domain
    probe = conn.query(
        {
            "datasource": SPEC_DATASOURCE,
            "geofilter": {"type": "bbox", "geom": list(BBOX)},
            "timefilter": {"times": [TIMES[0], TIMES[0]]},
        }
    )
    lons = np.atleast_1d(probe["lon"].values)
    lats = np.atleast_1d(probe["lat"].values)
    xs, ys = grid.to_local(lons, lats)
    depths = grid.sample_depth(xs, ys)
    print(f"  {lons.size} hindcast sites in domain")

    # nudge target to a wet cell (GEBCO ~450 m does not resolve the beach step)
    tx_lon, tx_lat = TARGET
    for _ in range(40):
        x, y = grid.to_local(np.array([tx_lon]), np.array([tx_lat]))
        if grid.sample_depth(x, y)[0] >= 3.0:
            break
        tx_lon -= 0.002  # walk offshore (west)
    tx, ty = float(x[0]), float(y[0])

    # boundary sites: offshore arc (deep enough), spread by azimuth around target
    deep = np.flatnonzero(depths >= BOUNDARY_MIN_DEPTH)
    if deep.size < N_BOUNDARY:
        raise RuntimeError(f"only {deep.size} sites deeper than {BOUNDARY_MIN_DEPTH} m")
    az = np.arctan2(ys[deep] - ty, xs[deep] - tx)
    order = deep[np.argsort(az)]
    picks = order[np.linspace(0, order.size - 1, N_BOUNDARY).astype(int)]
    bpts = [(float(lons[i]), float(lats[i])) for i in picks]

    # truth site: shallowest site not on the boundary and safely inside the
    # grid (probe-bbox edge sites fall outside the GEBCO cell-centre extent)
    xmin, xmax, ymin, ymax = grid.bounds
    margin = 2000.0
    inside = (
        (xs > xmin + margin) & (xs < xmax - margin) & (ys > ymin + margin) & (ys < ymax - margin)
    )
    interior = [
        i for i in range(lons.size) if i not in set(picks) and inside[i] and 4.0 < depths[i] < 12.0
    ]
    truth = min(interior, key=lambda i: depths[i])
    print(f"  boundary sites: {[(round(a, 2), round(b, 2)) for a, b in bpts]}")
    print(f"  truth site: ({lons[truth]:.3f}, {lats[truth]:.3f}) depth {depths[truth]:.1f} m")

    print(f"Fetching hourly spectra {TIMES[0][:10]} -> {TIMES[1][:10]} ...")
    efth_b = fetch_sites(conn, bpts)
    efth_truth = fetch_sites(conn, [(float(lons[truth]), float(lats[truth]))]).isel(site=0)
    nt = efth_b.sizes["time"]
    print(f"  {nt} timesteps, {efth_b.sizes['freq']} freqs x {efth_b.sizes['dir']} dirs")

    freqs = efth_b["freq"].values
    dirs = efth_b["dir"].values

    # --- transform to the Noordwijk target -------------------------------- #
    t0 = time.perf_counter()
    model = SiteModel.build(
        bathy=grid, target=(tx_lon, tx_lat), boundary_points=bpts, freqs=freqs, dirs=dirs
    )
    print(
        f"Operator built in {time.perf_counter() - t0:.1f} s "
        f"(target depth {model.operator.depth_target:.1f} m, "
        f"lost rays {100 * model.operator.attrs['lost_fraction']:.1f}%)"
    )
    t0 = time.perf_counter()
    out = model.transform(efth_b)
    print(f"Transformed {nt} timesteps in {1e3 * (time.perf_counter() - t0):.0f} ms")

    hs_off = hm0(efth_b.isel(site=0).values, freqs, dirs)
    hs_near = hm0(out.values, freqs, dirs)
    ipk = int(np.nanargmax(hs_off))
    print(f"\n  Noordwijk target ({tx_lon:.3f}E, {tx_lat:.3f}N):")
    print("  time                Hs_offshore  Hs_target")
    for i in sorted({0, ipk - 24, ipk, ipk + 24, nt - 1} & set(range(nt))):
        t = str(efth_b["time"].values[i])[:16]
        print(f"  {t}    {hs_off[i]:5.2f} m     {hs_near[i]:5.2f} m")

    # --- self-score against SWAN's own nearshore site --------------------- #
    model_v = SiteModel.build(
        bathy=grid,
        target=(float(lons[truth]), float(lats[truth])),
        boundary_points=bpts,
        freqs=freqs,
        dirs=dirs,
    )
    hs_ray = hm0(model_v.transform(efth_b).values, freqs, dirs)
    hs_swan = hm0(efth_truth.values, freqs, dirs)
    ok = np.isfinite(hs_ray) & np.isfinite(hs_swan)
    r = np.corrcoef(hs_ray[ok], hs_swan[ok])[0, 1]
    bias = float(np.mean(hs_ray[ok] - hs_swan[ok]))
    rmse = float(np.sqrt(np.mean((hs_ray[ok] - hs_swan[ok]) ** 2)))
    print(
        f"\n  vs SWAN at the truth site ({depths[truth]:.1f} m, n={int(ok.sum())}): "
        f"r={r:.3f}  bias={bias:+.2f} m  rmse={rmse:.2f} m  "
        f"(SWAN mean Hs {np.mean(hs_swan[ok]):.2f} m)"
    )

    model.to_netcdf("noordwijk_operator.nc")
    print("\nOperator saved to noordwijk_operator.nc")
    return 0


if __name__ == "__main__":
    sys.exit(main())
