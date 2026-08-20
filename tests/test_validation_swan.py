"""waveray against stationary SWAN, run in the official SWAN docker image.

These tests are marked ``swan`` and excluded from the default suite (they need
docker and take minutes). Run them with::

    uv run pytest -m swan -s

Design notes
------------
Both models are given the *same* boundary spectrum and spectral grid, so a
difference at the target is a difference in the transformation.

For the no-wind cases SWAN is run with quadruplets and whitecapping off, which
matches what the linear operator contains, and the agreement is tight. The
wind cases cannot be run that way: SWAN raises an error for a third-generation
wind without quadruplets and, when forced past it, never converges and returns
zero. So the wind cases compare against SWAN's *full* physics and quantify the
cost of carrying the wind input without the nonlinear sinks that balance it —
the limitation documented in the wind forcing guide. The tolerances below are
the measured behaviour, not aspirations.
"""

from __future__ import annotations

import importlib.util
import os
import warnings

import numpy as np
import pytest
from validation.cases import (
    island_case,
    plane_beach_case,
    wind_growth_case,
    wind_on_swell_case,
)
from validation.swan import swan_image_available

pytestmark = [
    pytest.mark.swan,
    pytest.mark.skipif(
        not swan_image_available(pull=False),
        reason="SWAN docker image not available (docker pull delftwaves/swan:latest)",
    ),
]


def _run_both(case, tmp_path, **build_kwargs):
    swan_out = case.run_swan(tmp_path / case.name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the growth-clip warning is asserted elsewhere
        wr_out = case.run_waveray(**build_kwargs)
    return swan_out, wr_out


def _report(case, sw, wr, label: str) -> None:
    print(f"\n{label}")
    print(
        f"{'depth':>8} {'SWAN Hs':>9} {'waveray':>9} {'diff %':>8} {'SWAN dir':>9} {'wray dir':>9}"
    )
    for i in range(len(case.targets)):
        d = 100.0 * (wr["HSIGN"][i] - sw["HSIGN"][i]) / max(sw["HSIGN"][i], 1e-9)
        print(
            f"{sw['DEP'][i]:8.2f} {sw['HSIGN'][i]:9.3f} {wr['HSIGN'][i]:9.3f} "
            f"{d:8.1f} {sw['DIR'][i]:9.1f} {wr['DIR'][i]:9.1f}"
        )


# ------------------------------------------------------------------ #
# Propagation: refraction and shoaling, no wind
# ------------------------------------------------------------------ #
def test_plane_beach_shore_normal(tmp_path):
    """Shore-normal swell up a plane beach: shoaling must match SWAN closely."""
    case = plane_beach_case(name="beach_normal", dpm=270.0)
    sw, wr = _run_both(case, tmp_path)
    _report(case, sw, wr, "plane beach, shore-normal swell")

    rel = np.abs(wr["HSIGN"] - sw["HSIGN"]) / sw["HSIGN"]
    assert np.all(rel < 0.03), f"Hs differs by {rel.max():.1%} (measured ~0.3%)"
    assert np.all(np.abs(wr["DIR"] - sw["DIR"]) < 2.0)


def test_plane_beach_oblique_refraction(tmp_path):
    """Oblique swell: the refracted direction must track SWAN's."""
    case = plane_beach_case(name="beach_oblique", dpm=240.0)
    sw, wr = _run_both(case, tmp_path)
    _report(case, sw, wr, "plane beach, 30 deg oblique swell")

    rel = np.abs(wr["HSIGN"] - sw["HSIGN"]) / sw["HSIGN"]
    assert np.all(rel < 0.03), f"Hs differs by {rel.max():.1%} (measured ~0.3%)"
    # both models must turn the waves shoreward, and agree on how far
    assert np.all(np.diff(sw["DIR"]) > 0) and np.all(np.diff(wr["DIR"]) > 0)
    assert np.all(np.abs(wr["DIR"] - sw["DIR"]) < 3.0), "refracted direction drifted from SWAN"


def test_island_sheltering(tmp_path):
    """Island lee: both models shelter strongly; the deep shadow is where a
    ray model and a spectral model legitimately disagree (no diffraction)."""
    case = island_case()
    sw, wr = _run_both(case, tmp_path)
    _report(case, sw, wr, "circular island: lee / beside / up-wave")

    # up-wave of the island the two must agree tightly
    assert abs(wr["HSIGN"][2] - sw["HSIGN"][2]) / sw["HSIGN"][2] < 0.03
    # both shelter the lee to well under half the up-wave height
    assert wr["HSIGN"][0] < 0.5 * wr["HSIGN"][2]
    assert sw["HSIGN"][0] < 0.5 * sw["HSIGN"][2]
    # ... and stay within a factor of two of each other there (measured ~1.5)
    assert 0.5 < wr["HSIGN"][0] / sw["HSIGN"][0] < 2.0


# ------------------------------------------------------------------ #
# Wind input
# ------------------------------------------------------------------ #
def test_wind_on_swell_bounded_departure(tmp_path):
    """The realistic configuration: swell plus wind over a nearshore fetch.

    waveray carries only the input term, so it must sit above the no-wind
    solution and above SWAN (which also dissipates), but by a bounded amount.
    """
    case = wind_on_swell_case(name="windswell")
    sw, wr = _run_both(case, tmp_path)
    _report(case, sw, wr, "plane beach, swell + 12 m/s onshore wind (SWAN full physics)")

    nowind = plane_beach_case(name="windswell_ref", dpm=270.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wr_nowind = nowind.run_waveray()

    assert np.all(wr["HSIGN"] > wr_nowind["HSIGN"]), "wind input must add energy"
    ratio = wr["HSIGN"] / sw["HSIGN"]
    # SWAN stops on a 99.5%-of-points / 1% criterion, so the nonlinear wind
    # case lands slightly differently run to run: observed 1.03-1.30.
    print(f"  waveray/SWAN ratio: {np.round(ratio, 3)} (measured 1.03-1.30 across runs)")
    assert np.all(ratio > 0.9), "waveray fell below SWAN despite the missing sinks"
    assert np.all(ratio < 1.6), f"departure from SWAN grew to {ratio.max():.2f}"


def test_wind_growth_from_zero(tmp_path):
    """Fetch-limited growth from a calm sea, seeded by the AGROW term.

    Without quadruplets there is no downshifting, so the input-only operator
    under-predicts a balanced model over a resolved (<= 0.4 Hz) spectrum. The
    test pins the direction of the effect and the measured bound.
    """
    case = wind_growth_case(name="windsea")
    sw, wr = _run_both(case, tmp_path)
    _report(case, sw, wr, "flat bottom, 12 m/s wind, zero boundary energy")

    assert np.all(np.diff(wr["HSIGN"]) > 0), "waveray must grow with fetch"
    assert np.all(np.diff(sw["HSIGN"]) > 0), "SWAN must grow with fetch"
    ratio = wr["HSIGN"] / sw["HSIGN"]
    print(f"  waveray/SWAN ratio: {np.round(ratio, 3)} (measured 0.54-0.70)")
    assert np.all(ratio > 0.3), "wind-sea generation collapsed"
    assert np.all(ratio < 1.5), "wind-sea generation ran away"


def test_growth_clip_reports_when_it_binds(tmp_path):
    """The runaway guard must announce itself on a long high-frequency fetch."""
    case = wind_growth_case(name="windsea_clip", length=6_000.0)
    case.freqs = np.array([0.5, 0.7, 0.9])  # beyond the resolved swell range
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = case.run_waveray()
    msgs = [str(w.message) for w in caught if "max_growth" in str(w.message)]
    assert msgs, "no growth-clip warning on a case that must run away"
    assert np.all(np.isfinite(out["HSIGN"]))


# ------------------------------------------------------------------ #
# Real bathymetry
# ------------------------------------------------------------------ #
@pytest.mark.skipif(not os.environ.get("DATAMESH_TOKEN"), reason="needs DATAMESH_TOKEN")
@pytest.mark.skipif(
    importlib.util.find_spec("oceanum") is None,
    reason="needs the datamesh extra (uv sync --extra datamesh)",
)
def test_real_bathymetry_against_swan(tmp_path):
    """Real GEBCO bathymetry off the Dutch coast: the operator must track
    SWAN through genuine depth contours, not just an analytic slope."""
    from validation.cases import real_bathymetry_case

    case = real_bathymetry_case()
    sw, wr = _run_both(case, tmp_path)
    _report(case, sw, wr, "GEBCO bathymetry (Noordwijk, NL), swell only")

    rel = np.abs(wr["HSIGN"] - sw["HSIGN"]) / sw["HSIGN"]
    print(f"  max Hs difference: {rel.max():.1%}")
    assert np.all(rel < 0.20), f"Hs differs by {rel.max():.1%} on real bathymetry"
    assert np.all(np.abs((wr["DIR"] - sw["DIR"] + 180) % 360 - 180) < 15.0)
