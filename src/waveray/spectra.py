"""Interoperability with ``wavespectra``.

waveray speaks the wavespectra convention throughout: a directional spectrum is
an ``xarray.DataArray`` named ``efth`` with dims ``(..., freq, dir)``, where
``freq`` is in Hz, ``dir`` is the *coming-from* direction in nautical degrees,
and the values are variance density in ``m2 s degree-1``.

Anything ``wavespectra`` can read is therefore valid input, and every spectrum
waveray returns carries the CF attributes wavespectra expects, so the ``.spec``
accessor works directly on it::

    efth_near = model.transform(efth_boundary)
    efth_near.spec.hs()        # significant wave height
    efth_near.spec.tp()        # peak period
    efth_near.spec.dpm()       # mean direction at the peak

Integrated parameters computed here (:func:`waveray.breaking.hm0`) use the same
quadrature as wavespectra, so they agree exactly with ``.spec.hs(tail=False)``.
"""

from __future__ import annotations

import xarray as xr

# Canonical CF attributes, mirroring wavespectra.core.attributes.attrs["ATTRS"].
# Duplicated as literals (rather than imported) so the exact strings waveray
# writes are visible here and stable if wavespectra reorganises its internals.
EFTH_ATTRS = {
    "standard_name": "sea_surface_wave_directional_variance_spectral_density",
    "units": "m2 s degree-1",
}
FREQ_ATTRS = {"standard_name": "sea_surface_wave_frequency", "units": "Hz"}
DIR_ATTRS = {"standard_name": "sea_surface_wave_from_direction", "units": "degree"}


def set_wavespectra_attrs(efth: xr.DataArray) -> xr.DataArray:
    """Stamp the CF attributes wavespectra expects onto a spectrum in place."""
    efth.attrs = {**EFTH_ATTRS, **efth.attrs}
    if "freq" in efth.coords:
        efth["freq"].attrs = {**FREQ_ATTRS, **efth["freq"].attrs}
    if "dir" in efth.coords:
        efth["dir"].attrs = {**DIR_ATTRS, **efth["dir"].attrs}
    return efth


def to_specdataset(efth: xr.DataArray) -> xr.Dataset:
    """Wrap a transformed spectrum as a Dataset usable by wavespectra writers.

    The result supports the ``.spec`` accessor and wavespectra's output
    backends, e.g. ``to_specdataset(efth).spec.to_netcdf("out.nc")``.
    """
    return set_wavespectra_attrs(efth.rename("efth")).to_dataset()
