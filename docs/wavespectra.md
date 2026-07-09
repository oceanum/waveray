# wavespectra interoperability

[wavespectra](https://github.com/oceanum/wavespectra) is a **core dependency**,
not an optional extra. Its conventions are waveray's contract: anything
wavespectra can read is valid input, and everything waveray returns is a valid
wavespectra spectrum.

## The convention

A directional spectrum is an `xarray.DataArray`:

| Aspect | Value |
|---|---|
| Name | `efth` |
| Dims | `(..., freq, dir)`, optionally with `site` and `time` |
| `freq` | frequency in Hz |
| `dir` | direction the waves come **from**, nautical degrees (0 = from north, 90 = from east) |
| Values | variance density, `m2 s degree-1` |

waveray stamps the matching CF attributes (`standard_name`, `units`) on the
spectrum and its coordinates, so the `.spec` accessor works on the output with
no conversion:

```python
import wavespectra              # registers the .spec accessor
efth_near = model.transform(efth_boundary)

efth_near.spec.hs()             # significant wave height
efth_near.spec.tp()             # peak period
efth_near.spec.dpm()            # mean direction at the peak
efth_near.spec.dspr()           # directional spread
efth_near.spec.partition(...)   # wave partitioning
```

## Integrated parameters agree exactly

waveray's own `hm0` uses **the same quadrature as wavespectra** — directions
summed with a constant bin width `dd = |dir[1] − dir[0]|`, frequencies with
`df = np.gradient(freq)` (not trapezoidal, which would halve the two edge
bins). Therefore:

```python
from waveray.breaking import hm0
assert np.allclose(hm0(efth.values, freqs, dirs),
                   efth.spec.hs(tail=False).values)   # exact to 1e-12
```

This is enforced by `tests/test_wavespectra_compat.py`. It matters because the
breaking cap is applied to `Hm0`: if waveray's `Hm0` disagreed with the one
you compute downstream with `.spec.hs()`, the cap would appear to be applied at
the wrong height.

> [!NOTE]
> `spec.hs()` defaults to `tail=True`, which adds an analytic high-frequency
> tail when `freq[-1] > 0.333 Hz`. waveray does not add a tail — it transforms
> only the resolved bins. Compare against `spec.hs(tail=False)`.

## Input

Any of these work:

```python
model.transform(efth_dataarray)          # DataArray named efth
model.transform(spec_dataset)            # Dataset holding an 'efth' variable
model.transform(wavespectra.read_ww3("spec.nc"))   # a wavespectra reader's output
```

Direction coordinates do **not** need to be sorted. Operational hindcasts
frequently ship a wrapped, descending `dir` axis (e.g. 265°→5°, then 355°→275°);
waveray sorts internally with circular interpolation and returns the spectrum on
your original direction axis.

> [!WARNING]
> If you plot such a spectrum yourself with `pcolormesh` on polar axes, sort
> the direction axis first — `pcolormesh` requires a monotonic theta and will
> otherwise smear energy across the circle. The bundled notebook shows this.

## Output helpers

```python
from waveray import to_specdataset, set_wavespectra_attrs

ds = to_specdataset(efth_near)      # Dataset usable by wavespectra writers
ds.to_netcdf("nearshore_spectra.nc")

set_wavespectra_attrs(some_efth)    # stamp CF attrs onto a spectrum in place
```

Round-tripping through netCDF preserves the accessor: reopening the file with
`xr.open_dataset` gives a Dataset whose `.spec.hs()` matches the original.
