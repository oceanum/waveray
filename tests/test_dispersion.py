"""Dispersion solver against known limits."""

import numpy as np

from waveray.dispersion import GRAV, ccg, group_speed, phase_speed, wavenumber


def test_dispersion_residual_across_regimes():
    f = np.array([0.03, 0.05, 0.1, 0.2, 0.5, 1.0])[:, None]
    d = np.array([1.0, 5.0, 20.0, 100.0, 4000.0])[None, :]
    omega = 2 * np.pi * f
    k = wavenumber(omega, d)
    residual = GRAV * k * np.tanh(k * d) - omega**2
    assert np.max(np.abs(residual) / omega**2) < 1e-10


def test_deep_water_limit():
    omega = 2 * np.pi * 0.1
    k = wavenumber(np.array(omega), np.array(4000.0))
    assert np.isclose(k, omega**2 / GRAV, rtol=1e-8)
    assert np.isclose(
        group_speed(np.array(omega), np.array(4000.0)),
        0.5 * phase_speed(np.array(omega), np.array(4000.0)),
        rtol=1e-6,
    )


def test_shallow_water_limit():
    omega = 2 * np.pi * 0.02
    d = np.array(2.0)
    c = phase_speed(np.array(omega), d)
    cg = group_speed(np.array(omega), d)
    assert np.isclose(c, np.sqrt(GRAV * 2.0), rtol=1e-2)
    assert np.isclose(cg, c, rtol=1e-2)


def test_ccg_monotonic_shoaling():
    # c*cg decreases into shallow water for swell -> transfer coefficient
    # (ccg_deep / ccg_shallow) > 1, i.e. spectral density grows shoreward.
    omega = 2 * np.pi * 0.07
    depths = np.array([50.0, 30.0, 20.0, 10.0, 5.0])
    vals = ccg(omega, depths)
    assert np.all(np.diff(vals) < 0)
