"""Export backward-traced wave ray paths as GeoJSON.

Diagnostic/visualisation companion to the transfer operator: the same rays
that build an operator can be exported as a GeoJSON FeatureCollection (one
MultiLineString feature per (frequency, direction) bin, sub-rays as members)
for inspection in QGIS, EIDOS, or any web map.

Coordinates are WGS84 lon/lat when the grid has a geographic origin (built
via ``LocalGrid.from_dataarray`` / ``fetch_datamesh_bathymetry``), otherwise
local metres (structurally valid GeoJSON, no CRS).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .bathymetry import LocalGrid
from .dispersion import wavenumber
from .operator import dir_to_theta
from .rays import STATUS_EXITED, STATUS_LANDED, SpeedField, trace_backward

_STATUS_NAMES = {STATUS_EXITED: "exited", STATUS_LANDED: "landed"}


def ray_paths_geojson(
    grid: LocalGrid,
    target_xy: tuple[float, float],
    freqs: np.ndarray,
    dirs: np.ndarray,
    lonlat: bool | None = None,
    nsub: int = 1,
    ds: float | None = None,
    max_steps: int | None = None,
    d_min: float = 0.3,
    cf_jonswap: float | None = 0.038,
    stride: int = 5,
    path: str | Path | None = None,
) -> dict:
    """Trace backward rays from a target and return them as GeoJSON.

    Parameters mirror :func:`nearshore_transform.operator.build_operator`
    (same rays, same physics); ``target_xy`` is in grid metres — use
    ``grid.to_local(lon, lat)`` first for geographic targets.

    Parameters
    ----------
    lonlat : emit WGS84 coordinates (default: True when the grid has a
        geographic origin). False emits local metres.
    nsub : sub-rays per direction bin (1 gives one clean line per bin).
    stride : keep every ``stride``-th vertex (plus first and last).
    path : optionally write the FeatureCollection to this file.

    Returns
    -------
    dict : GeoJSON FeatureCollection. One Feature per (freq, dir) bin with a
        MultiLineString of the bin's sub-ray paths (from the target outward)
        and properties freq [Hz], period [s], dir [coming-from deg, nautical],
        statuses per sub-ray, and the mean path-integrated friction decay.
    """
    freqs = np.atleast_1d(np.asarray(freqs, dtype=float))
    dirs = np.atleast_1d(np.asarray(dirs, dtype=float))
    if lonlat is None:
        lonlat = grid.lon0 is not None

    dx, dy = grid.spacing
    if ds is None:
        ds = min(dx, dy) / 3.0
    xmin, xmax, ymin, ymax = grid.bounds
    if max_steps is None:
        max_steps = int(np.ceil(1.5 * 2.0 * ((xmax - xmin) + (ymax - ymin)) / ds))

    tx, ty = float(target_xy[0]), float(target_xy[1])
    ndirs = dirs.size
    offsets = (np.arange(nsub) + 0.5) / nsub - 0.5
    bin_width = 360.0 / ndirs
    theta0 = (dir_to_theta(dirs)[:, None] + np.deg2rad(offsets * bin_width)[None, :]).ravel()

    features = []
    for f in freqs:
        omega = 2.0 * np.pi * f
        fld = SpeedField.build(grid, omega, d_min=d_min, cf_jonswap=cf_jonswap)
        fan = trace_backward(
            fld, tx, ty, theta0, ds=ds, max_steps=max_steps, d_min=d_min, record_paths=True
        )
        depth_t = float(grid.sample_depth(np.array([tx]), np.array([ty]))[0])
        wavelength = float(2.0 * np.pi / wavenumber(np.array(omega), np.array(depth_t)))

        for j, d in enumerate(dirs):
            lines = []
            statuses = []
            attens = []
            for m in range(nsub):
                i = j * nsub + m
                pts = fan.paths[i]
                keep = np.r_[np.arange(0, pts.shape[0] - 1, stride), pts.shape[0] - 1]
                pts = pts[keep]
                if pts.shape[0] < 2:
                    continue
                if lonlat:
                    lon, lat = grid.to_lonlat(pts[:, 0], pts[:, 1])
                    coords = np.column_stack([lon, lat]).round(6)
                else:
                    coords = pts.round(2)
                lines.append(coords.tolist())
                statuses.append(_STATUS_NAMES.get(int(fan.status[i]), "lost"))
                attens.append(float(fan.atten[i]))
            if not lines:
                continue
            features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "MultiLineString", "coordinates": lines},
                    "properties": {
                        "freq": round(float(f), 5),
                        "period": round(1.0 / float(f), 2),
                        "dir": round(float(d), 1),
                        "wavelength_at_target": round(wavelength, 1),
                        "status": statuses,
                        "mean_friction_decay": round(float(np.exp(-np.mean(attens))), 4),
                    },
                }
            )

    collection = {"type": "FeatureCollection", "features": features}
    if path is not None:
        Path(path).write_text(json.dumps(collection))
    return collection
