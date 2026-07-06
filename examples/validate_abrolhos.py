"""Validation against the parent SWAN model.

Uses Oceanum's Abrolhos 500m SWAN nowcast spectra: the westernmost sites in a
local domain become boundary points, an interior nearshore SWAN site is the
truth, and the ray-traced transfer of the boundary spectra is scored against
SWAN's own spectrum at that site. This measures exactly what the operator is
meant to do: reproduce the parent model's last-stage nearshore transformation
at a tiny fraction of its cost.

    uv run python examples/validate_abrolhos.py
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

from nearshore_transform import SiteModel, fetch_datamesh_bathymetry
from nearshore_transform.breaking import hm0

BBOX = (114.30, -28.95, 114.65, -28.60)
BATHY_DATASOURCE = "gebco_2025"
SPEC_DATASOURCE = "oceanum_wave_ec_abrol500m_spec_nowcast"
N_BOUNDARY = 5
TIMES = ["2026-06-20T00:00:00Z", "2026-07-06T00:00:00Z"]


def main() -> int:
    if not os.environ.get("DATAMESH_TOKEN"):
        print("DATAMESH_TOKEN not set - cannot run validation")
        return 1

    from oceanum.datamesh import Connector

    grid = fetch_datamesh_bathymetry(BATHY_DATASOURCE, bbox=BBOX, positive="up")
    ds = Connector().query(
        {
            "datasource": SPEC_DATASOURCE,
            "geofilter": {"type": "bbox", "geom": list(BBOX)},
            "timefilter": {"times": TIMES},
        }
    )
    site_dim = next(d for d in ds["efth"].dims if d not in ("time", "freq", "dir"))
    lons = np.atleast_1d(ds["lon"].values)
    lats = np.atleast_1d(ds["lat"].values)
    freqs = ds["freq"].values
    dirs = ds["dir"].values
    nt = ds.sizes["time"]
    print(f"{lons.size} SWAN sites, {nt} timesteps {str(ds['time'].values[0])[:10]} ...")

    # boundary = westernmost sites; candidate truths = everything else that is
    # wet on our grid and at least ~3 km inside the boundary line
    order = np.argsort(lons)
    b_idx = order[:N_BOUNDARY]
    bpts = [(float(lons[i]), float(lats[i])) for i in b_idx]
    efth_b = (ds["efth"].isel({site_dim: b_idx}).rename({site_dim: "site"})).transpose(
        "time", "site", "freq", "dir"
    )

    x_all, y_all = grid.to_local(lons, lats)
    depth_all = grid.sample_depth(x_all, y_all)
    x_bnd = x_all[b_idx].max()
    candidates = [
        i for i in order if i not in set(b_idx) and depth_all[i] > 3.0 and x_all[i] > x_bnd + 3000.0
    ]
    # score the 3 shallowest interior sites (the most nearshore transformation)
    candidates = sorted(candidates, key=lambda i: depth_all[i])[:3]

    print(f"boundary: {[(round(x, 2), round(y, 2)) for x, y in bpts]}")
    print("\nsite (lon, lat)      depth   r      bias    rmse    si     hs_mean(swan)")
    for i in candidates:
        t0 = time.perf_counter()
        model = SiteModel.build(
            bathy=grid,
            target=(float(lons[i]), float(lats[i])),
            boundary_points=bpts,
            freqs=freqs,
            dirs=dirs,
        )
        t_build = time.perf_counter() - t0
        out = model.transform(efth_b)

        hs_ray = hm0(out.values, freqs, dirs)
        hs_swan = hm0(
            ds["efth"].isel({site_dim: i}).transpose("time", "freq", "dir").values,
            freqs,
            dirs,
        )
        ok = np.isfinite(hs_ray) & np.isfinite(hs_swan)
        r = np.corrcoef(hs_ray[ok], hs_swan[ok])[0, 1]
        bias = np.mean(hs_ray[ok] - hs_swan[ok])
        rmse = np.sqrt(np.mean((hs_ray[ok] - hs_swan[ok]) ** 2))
        si = rmse / np.mean(hs_swan[ok])
        print(
            f"({lons[i]:.3f}, {lats[i]:.3f})  {depth_all[i]:5.1f}m  "
            f"{r:5.3f}  {bias:+5.2f}m  {rmse:5.2f}m  {si:5.2f}   "
            f"{np.mean(hs_swan[ok]):.2f} m   (build {t_build:.0f}s)"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
