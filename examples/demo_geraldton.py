"""End-to-end demo: SWAN nowcast spectra -> Geraldton nearshore.

Pulls GEBCO 2025 bathymetry and Oceanum Abrolhos 500m SWAN spectra from
Datamesh, builds a transfer operator for a point off Geraldton, transforms
the most recent 2 days of spectra and prints offshore vs nearshore Hs.

Requires DATAMESH_TOKEN in the environment and the datamesh extra:

    uv sync --extra datamesh
    uv run python examples/demo_geraldton.py
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

from nearshore_transform import SiteModel, fetch_datamesh_bathymetry
from nearshore_transform.breaking import hm0

BBOX = (114.30, -28.95, 114.65, -28.60)  # local domain around Geraldton
TARGET = (114.58, -28.775)  # just seaward of the port
BATHY_DATASOURCE = "gebco_2025"
SPEC_DATASOURCE = "oceanum_wave_ec_abrol500m_spec_nowcast"
N_BOUNDARY = 3  # SWAN spectra sites to use as boundary points


def main() -> int:
    if not os.environ.get("DATAMESH_TOKEN"):
        print("DATAMESH_TOKEN not set - cannot run the live demo")
        return 1

    from oceanum.datamesh import Connector

    conn = Connector()

    print(f"Fetching bathymetry {BATHY_DATASOURCE} {BBOX} ...")
    grid = fetch_datamesh_bathymetry(BATHY_DATASOURCE, bbox=BBOX, positive="up")
    print(f"  grid: {grid.x.size} x {grid.y.size} nodes, spacing ~{grid.spacing[0]:.0f} m")

    print(f"Fetching SWAN spectra {SPEC_DATASOURCE} ...")
    ds = conn.query(
        {
            "datasource": SPEC_DATASOURCE,
            "geofilter": {"type": "bbox", "geom": list(BBOX)},
            "timefilter": {"times": ["2026-07-04T00:00:00Z", "2026-07-06T00:00:00Z"]},
        }
    )
    site_dim = next(d for d in ds["efth"].dims if d not in ("time", "freq", "dir"))
    lons = np.atleast_1d(ds["lon"].values)
    lats = np.atleast_1d(ds["lat"].values)
    print(f"  {lons.size} spectra sites in domain, {ds.sizes['time']} timesteps")

    # choose the westernmost (most offshore) N sites as boundary points
    order = np.argsort(lons)[:N_BOUNDARY]
    bpts = [(float(lons[i]), float(lats[i])) for i in order]
    efth = ds["efth"].isel({site_dim: order}).rename({site_dim: "site"})
    efth = efth.transpose("time", "site", "freq", "dir")
    print(f"  boundary points: {[(round(x, 3), round(y, 3)) for x, y in bpts]}")

    # nudge the target west until it is wet on this grid (GEBCO is ~450 m;
    # the harbour itself is not resolved)
    tx, ty = TARGET
    for _ in range(40):
        x, y = grid.to_local(np.array([tx]), np.array([ty]))
        if grid.sample_depth(x, y)[0] >= 2.0:
            break
        tx -= 0.002
    print(f"  target: ({tx:.4f}, {ty:.4f})")

    t0 = time.perf_counter()
    model = SiteModel.build(
        bathy=grid,
        target=(tx, ty),
        boundary_points=bpts,
        freqs=efth["freq"].values,
        dirs=efth["dir"].values,
    )
    t_build = time.perf_counter() - t0
    print(f"Operator built in {t_build:.1f} s (target depth {model.operator.depth_target:.1f} m)")

    t0 = time.perf_counter()
    out = model.transform(efth)
    t_apply = time.perf_counter() - t0
    nt = out.sizes["time"]
    print(
        f"Transformed {nt} timesteps in {t_apply * 1e3:.0f} ms "
        f"({nt / max(t_apply, 1e-9):,.0f} spectra/s)"
    )

    freqs = efth["freq"].values
    dirs = efth["dir"].values
    hs_off = hm0(efth.isel(site=0).values, freqs, dirs)
    hs_near = hm0(out.values, freqs, dirs)
    print("\n  time                 Hs_offshore  Hs_nearshore")
    for i in range(0, nt, max(1, nt // 8)):
        t = str(ds["time"].values[i])[:16]
        print(f"  {t}     {hs_off[i]:5.2f} m      {hs_near[i]:5.2f} m")

    model.to_netcdf("geraldton_demo_operator.nc")
    print("\nOperator saved to geraldton_demo_operator.nc")
    return 0


if __name__ == "__main__":
    sys.exit(main())
