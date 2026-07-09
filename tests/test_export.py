"""GeoJSON ray-path export."""

import json

import numpy as np
import pytest
import xarray as xr

from waveray import LocalGrid, ray_paths_geojson
from waveray.rays import SpeedField, trace_backward

FREQS = np.array([0.06, 0.1])
DIRS = np.arange(0.0, 360.0, 30.0)


@pytest.fixture(scope="module")
def geo_grid():
    """Plane beach as a geographic elevation DataArray, shore to the east."""
    lon = np.linspace(4.0, 4.5, 101)
    lat = np.linspace(52.0, 52.4, 81)
    depth = np.tile(30.0 * (1.0 - (lon - lon[0]) / (lon[-1] - lon[0])), (lat.size, 1))
    da = xr.DataArray(-depth, dims=("lat", "lon"), coords={"lat": lat, "lon": lon})
    return LocalGrid.from_dataarray(da, positive="up")


def test_geojson_structure_and_geography(geo_grid):
    tx, ty = geo_grid.to_local(np.array([4.35]), np.array([52.2]))
    gj = ray_paths_geojson(geo_grid, (float(tx[0]), float(ty[0])), FREQS, DIRS)

    assert gj["type"] == "FeatureCollection"
    assert len(gj["features"]) == FREQS.size * DIRS.size
    json.dumps(gj)  # must be serialisable

    for feat in gj["features"]:
        geom = feat["geometry"]
        assert geom["type"] == "MultiLineString"
        assert len(geom["coordinates"]) >= 1
        for line in geom["coordinates"]:
            assert len(line) >= 2
            arr = np.asarray(line)
            # WGS84, within the grid bbox
            assert arr[:, 0].min() >= 4.0 - 1e-6 and arr[:, 0].max() <= 4.5 + 1e-6
            assert arr[:, 1].min() >= 52.0 - 1e-6 and arr[:, 1].max() <= 52.4 + 1e-6
            # every line starts at the target
            assert np.hypot(arr[0, 0] - 4.35, arr[0, 1] - 52.2) < 1e-3
        props = feat["properties"]
        assert set(props) >= {"freq", "period", "dir", "status", "mean_friction_decay"}
        assert all(s in ("exited", "landed", "lost") for s in props["status"])


def test_shoreward_rays_land_and_seaward_rays_exit(geo_grid):
    tx, ty = geo_grid.to_local(np.array([4.35]), np.array([52.2]))
    gj = ray_paths_geojson(geo_grid, (float(tx[0]), float(ty[0])), np.array([0.08]), DIRS)
    by_dir = {f["properties"]["dir"]: f["properties"]["status"][0] for f in gj["features"]}
    # waves coming FROM the west (270) → backward ray heads west (offshore) → exits
    assert by_dir[270.0] == "exited"
    # waves coming FROM the east (90) → backward ray heads east into the beach → lands
    assert by_dir[90.0] == "landed"


def test_multiline_nsub_and_file_output(geo_grid, tmp_path):
    tx, ty = geo_grid.to_local(np.array([4.35]), np.array([52.2]))
    out = tmp_path / "rays.geojson"
    gj = ray_paths_geojson(
        geo_grid, (float(tx[0]), float(ty[0])), np.array([0.08]), DIRS, nsub=3, path=out
    )
    assert out.exists()
    assert json.loads(out.read_text()) == gj
    lens = [len(f["geometry"]["coordinates"]) for f in gj["features"]]
    assert max(lens) == 3  # sub-rays grouped into the bin's MultiLineString


def test_record_paths_off_by_default(geo_grid):
    fld = SpeedField.build(geo_grid, 2 * np.pi * 0.08, d_min=0.3)
    tx, ty = geo_grid.to_local(np.array([4.35]), np.array([52.2]))
    fan = trace_backward(fld, float(tx[0]), float(ty[0]), np.array([0.0]), ds=50.0, max_steps=5000)
    assert fan.paths is None
    fan2 = trace_backward(
        fld,
        float(tx[0]),
        float(ty[0]),
        np.array([0.0]),
        ds=50.0,
        max_steps=5000,
        record_paths=True,
    )
    assert fan2.paths is not None and fan2.paths[0].shape[1] == 2
    # recorded endpoint must equal the reported exit state
    assert np.allclose(fan2.paths[0][-1], [fan2.x[0], fan2.y[0]])
