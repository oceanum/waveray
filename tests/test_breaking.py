"""Depth-limited breaking cap."""

import numpy as np

from nearshore_transform.breaking import apply_breaking, hm0, hm0_max

FREQS = np.linspace(0.04, 0.4, 19)
DIRS = np.arange(0.0, 360.0, 15.0)


def gaussian_spectrum(hs, fp=0.1, dp=270.0):
    """Simple separable spectrum with an exact target Hm0."""
    ef = np.exp(-0.5 * ((FREQS - fp) / 0.02) ** 2)
    dd = np.exp(-0.5 * (((DIRS - dp + 180.0) % 360.0 - 180.0) / 30.0) ** 2)
    e = ef[:, None] * dd[None, :]
    m0 = hm0(e, FREQS, DIRS) ** 2 / 16.0
    return e * (hs**2 / 16.0) / m0


def test_hm0_of_constructed_spectrum():
    e = gaussian_spectrum(hs=2.5)
    assert np.isclose(hm0(e, FREQS, DIRS), 2.5, rtol=1e-6)


def test_no_breaking_in_deep_water():
    e = gaussian_spectrum(hs=2.0)
    out, scale = apply_breaking(e, FREQS, DIRS, depth=30.0, method="gamma")
    assert np.allclose(out, e)
    assert np.allclose(scale, 1.0)


def test_gamma_cap_enforced():
    e = gaussian_spectrum(hs=4.0)
    out, scale = apply_breaking(e, FREQS, DIRS, depth=3.0, gamma=0.73, method="gamma")
    assert np.isclose(hm0(out, FREQS, DIRS), 0.73 * 3.0, rtol=1e-6)
    assert scale < 1.0


def test_tide_raises_the_cap():
    e = gaussian_spectrum(hs=4.0)
    low, _ = apply_breaking(e, FREQS, DIRS, depth=3.0, tide=0.0, method="gamma")
    high, _ = apply_breaking(e, FREQS, DIRS, depth=3.0, tide=1.5, method="gamma")
    assert hm0(high, FREQS, DIRS) > hm0(low, FREQS, DIRS)


def test_miche_cap_shallow_matches_gamma():
    # long waves in very shallow water: tanh(x) ~ x -> Hmax -> gamma * d
    d = 2.0
    cap = hm0_max(np.array(d), fm=np.array(0.05), gamma=0.73, method="miche")
    assert np.isclose(cap, 0.73 * d, rtol=5e-2)


def test_miche_cap_applied_when_exceeded():
    e = gaussian_spectrum(hs=5.0)
    out, scale = apply_breaking(e, FREQS, DIRS, depth=4.0, method="miche")
    h_out = hm0(out, FREQS, DIRS)
    assert h_out < 5.0
    assert scale < 1.0


def test_vectorised_over_time():
    e = np.stack([gaussian_spectrum(hs=1.0), gaussian_spectrum(hs=4.0)])  # (2, nf, nd)
    out, scale = apply_breaking(e, FREQS, DIRS, depth=3.0, method="gamma")
    assert scale.shape == (2,)
    assert scale[0] == 1.0 and scale[1] < 1.0
    # tide per time step
    out2, scale2 = apply_breaking(
        e, FREQS, DIRS, depth=3.0, tide=np.array([0.0, 2.0]), method="gamma"
    )
    assert scale2[1] > scale[1]
