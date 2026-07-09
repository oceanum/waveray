# Installation

waveray requires **Python 3.12 or newer**.

```bash
pip install waveray
```

## Extras

| Extra | Installs | Needed for |
|---|---|---|
| *(none)* | numpy, xarray, wavespectra, netcdf4 | Core: grids, operators, transformation, breaking, GeoJSON export |
| `datamesh` | `oceanum` | `fetch_datamesh_bathymetry`, pulling spectra from Oceanum Datamesh |
| `notebooks` | jupyter, matplotlib, nbconvert | Running the bundled notebooks |

```bash
pip install "waveray[datamesh]"              # bathymetry + spectra from Datamesh
pip install "waveray[datamesh,notebooks]"    # everything, including the notebooks
```

`wavespectra` is a **core** dependency, not an extra: waveray's spectral
conventions and its integrated wave parameters are defined by wavespectra, so
it is always present. See [wavespectra interoperability](wavespectra.md).

## Development install

```bash
git clone https://github.com/oceanum/waveray
cd waveray
uv sync --extra datamesh --extra notebooks
uv run pytest -q
uv run ruff check . && uv run ruff format --check .
```

The test suite is a physics-validation suite: analytic Snell refraction and
energy-flux shoaling on a plane beach, island sheltering, friction attenuation,
breaking caps, and exact agreement with wavespectra's integrated parameters. It
needs no network access and runs in well under a minute.

## Datamesh credentials

The `datamesh` extra reads a token from the environment:

```bash
export DATAMESH_TOKEN="..."
```

Everything except `fetch_datamesh_bathymetry` works without it — you can supply
bathymetry as a plain `xarray.DataArray` from any source (GEBCO, EMODnet, a
national survey grid, a `.nc` file on disk).
