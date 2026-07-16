"""Transfer operator against analytic linear wave transformation.

Truth values are classic 1D results (Snell refraction + energy-flux shoaling
on a plane beach) computed from the independently-tested dispersion module.
"""

import numpy as np
import pytest

from waveray.bathymetry import LocalGrid
from waveray.dispersion import group_speed, phase_speed
from waveray.operator import (
    TransferOperator,
    build_operator,
    dir_to_theta,
    theta_to_dir,
)

DIRS = np.arange(0.0, 360.0, 10.0)
DDIR = 10.0


def flat_grid(depth=20.0, size=10_000.0, n=101):
    x = np.linspace(-size / 2, size / 2, n)
    y = np.linspace(-size / 2, size / 2, n)
    return LocalGrid(x=x, y=y, depth=np.full((n, n), depth))


def plane_beach_grid(d_off=30.0, length=20_000.0, width=24_000.0, nx=201, ny=241):
    x = np.linspace(0.0, length, nx)
    y = np.linspace(-width / 2, width / 2, ny)
    depth = np.tile(d_off * (1.0 - x / length), (ny, 1))
    return LocalGrid(x=x, y=y, depth=depth)


def test_direction_convention_roundtrip():
    d = np.array([0.0, 90.0, 180.0, 270.0, 355.0])
    assert np.allclose(theta_to_dir(dir_to_theta(d)), d)
    # coming-from North (0 deg) propagates southward: theta = -90 deg = 270 deg math
    assert np.isclose(np.rad2deg(dir_to_theta(np.array([0.0])))[0] % 360.0, 270.0)


def test_flat_bathymetry_is_identity():
    """Constant depth: no refraction, no shoaling -> spectrum passes through."""
    grid = flat_grid()
    freqs = np.array([0.06, 0.1])
    op = build_operator(
        grid,
        target_xy=(0.0, 0.0),
        boundary_xy=np.array([[-5000.0, 0.0]]),
        freqs=freqs,
        dirs=DIRS,
        nsub=9,
        cf_jonswap=None,
    )
    # smooth directional spectrum: cos^2 spreading around coming-from 270
    spread = np.cos(np.deg2rad((DIRS - 270.0 + 180.0) % 360.0 - 180.0) / 2.0) ** 4
    e_b = np.tile(spread, (1, freqs.size, 1))  # (K=1, nf, ndir)
    e_t = op.apply(e_b)

    tot_in = e_b[0].sum(axis=-1)
    tot_out = e_t.sum(axis=-1)
    assert np.allclose(tot_out, tot_in, rtol=2e-2)
    # peak direction preserved
    assert DIRS[np.argmax(e_t[0])] == 270.0


def test_plane_beach_normal_incidence_shoaling():
    """Unit density in the shore-normal bin: energy ratio must equal cg_b/cg_t."""
    grid = plane_beach_grid()
    f = 0.08
    x_t = 20_000.0 * (1.0 - 10.0 / 30.0)  # 10 m depth
    op = build_operator(
        grid,
        target_xy=(x_t, 0.0),
        boundary_xy=np.array([[0.0, 0.0]]),
        freqs=np.array([f]),
        dirs=DIRS,
        nsub=15,
        cf_jonswap=None,
    )
    e_b = np.zeros((1, 1, DIRS.size))
    e_b[0, 0, DIRS == 270.0] = 1.0  # coming-from W = going +x = shoreward
    e_t = op.apply(e_b)

    omega = 2 * np.pi * f
    d_b = grid.sample_depth(np.array([0.0]), np.array([0.0]))[0]
    ratio_expected = group_speed(omega, np.array(d_b)) / group_speed(omega, np.array(10.0))
    ratio = e_t.sum() / e_b.sum()
    assert np.isclose(ratio, ratio_expected, rtol=5e-2), (ratio, ratio_expected)


def test_plane_beach_oblique_refraction():
    """Oblique swell: energy ratio must equal Ks^2 * Kr^2 from Snell."""
    grid = plane_beach_grid()
    f = 0.08
    x_t = 20_000.0 * (1.0 - 10.0 / 30.0)
    op = build_operator(
        grid,
        target_xy=(x_t, 0.0),
        boundary_xy=np.array([[0.0, 0.0]]),
        freqs=np.array([f]),
        dirs=DIRS,
        nsub=15,
        cf_jonswap=None,
    )
    # boundary bin: going-to 30 deg off shore-normal -> coming-from 240
    e_b = np.zeros((1, 1, DIRS.size))
    e_b[0, 0, DIRS == 240.0] = 1.0
    e_t = op.apply(e_b)

    omega = 2 * np.pi * f
    d_b = grid.sample_depth(np.array([0.0]), np.array([0.0]))[0]
    c_b = phase_speed(omega, np.array(d_b))
    c_t = phase_speed(omega, np.array(10.0))
    cg_b = group_speed(omega, np.array(d_b))
    cg_t = group_speed(omega, np.array(10.0))
    th_b = np.deg2rad(30.0)
    th_t = np.arcsin(np.sin(th_b) * c_t / c_b)  # Snell
    ratio_expected = (cg_b * np.cos(th_b)) / (cg_t * np.cos(th_t))
    ratio = e_t.sum() / e_b.sum()
    assert np.isclose(ratio, ratio_expected, rtol=7e-2), (ratio, ratio_expected)
    # refraction must pull the peak toward shore-normal (270), i.e. between
    # the input bin (240) and 270
    peak = DIRS[np.argmax(e_t[0])]
    assert 240.0 < peak <= 270.0


def test_multi_boundary_point_interpolation():
    """Alongshore-varying boundary spectra are picked up by exit location."""
    grid = flat_grid()
    freqs = np.array([0.08])
    bps = np.array([[-5000.0, 0.0], [5000.0, 0.0]])  # west edge, east edge
    op = build_operator(
        grid, target_xy=(0.0, 0.0), boundary_xy=bps, freqs=freqs, dirs=DIRS, nsub=9, cf_jonswap=None
    )
    # west bp: uniform density 1; east bp: uniform density 2
    e_b = np.stack(
        [np.ones((1, DIRS.size)), 2.0 * np.ones((1, DIRS.size))], axis=0
    )  # (K, nf, ndir)
    e_t = op.apply(e_b)

    # waves coming from W (270) exit the west edge backward -> west bp value
    assert np.isclose(e_t[0, DIRS == 270.0][0], 1.0, rtol=0.1)
    # waves coming from E (90) exit the east edge backward -> east bp value
    assert np.isclose(e_t[0, DIRS == 90.0][0], 2.0, rtol=0.1)


def test_single_bp_equals_duplicated_bps():
    """K=1 must be the degenerate case of K=2 with identical spectra."""
    grid = flat_grid()
    freqs = np.array([0.08])
    spread = 1.0 + np.cos(np.deg2rad(DIRS))  # arbitrary smooth spectrum
    op1 = build_operator(
        grid,
        target_xy=(0.0, 0.0),
        boundary_xy=np.array([[-5000.0, 0.0]]),
        freqs=freqs,
        dirs=DIRS,
        nsub=9,
        cf_jonswap=None,
    )
    op2 = build_operator(
        grid,
        target_xy=(0.0, 0.0),
        boundary_xy=np.array([[-5000.0, 0.0], [5000.0, 0.0]]),
        freqs=freqs,
        dirs=DIRS,
        nsub=9,
        cf_jonswap=None,
    )
    e1 = op1.apply(np.tile(spread, (1, 1, 1)))
    e2 = op2.apply(np.tile(spread, (2, 1, 1)))
    assert np.allclose(e1, e2, rtol=1e-10, atol=1e-12)


def test_friction_attenuates_along_shallow_paths():
    """JONSWAP friction must reduce energy on a long shallow approach and be
    negligible on a short deep one."""
    beach = plane_beach_grid()
    x_t = 20_000.0 * (1.0 - 10.0 / 30.0)
    kwargs = dict(
        target_xy=(x_t, 0.0),
        boundary_xy=np.array([[0.0, 0.0]]),
        freqs=np.array([0.08]),
        dirs=DIRS,
        nsub=9,
    )
    op_free = build_operator(beach, cf_jonswap=None, **kwargs)
    op_fric = build_operator(beach, cf_jonswap=0.038, **kwargs)
    e_b = np.zeros((1, 1, DIRS.size))
    e_b[0, 0, DIRS == 270.0] = 1.0
    loss = op_fric.apply(e_b).sum() / op_free.apply(e_b).sum()
    assert 0.3 < loss < 0.98  # attenuated, but not annihilated

    deep = flat_grid(depth=200.0)
    op_deep = build_operator(
        deep,
        target_xy=(0.0, 0.0),
        boundary_xy=np.array([[-5000.0, 0.0]]),
        freqs=np.array([0.08]),
        dirs=DIRS,
        nsub=9,
        cf_jonswap=0.038,
    )
    op_deep_free = build_operator(
        deep,
        target_xy=(0.0, 0.0),
        boundary_xy=np.array([[-5000.0, 0.0]]),
        freqs=np.array([0.08]),
        dirs=DIRS,
        nsub=9,
        cf_jonswap=None,
    )
    ratio = op_deep.apply(e_b).sum() / op_deep_free.apply(e_b).sum()
    assert np.isclose(ratio, 1.0, rtol=1e-4)


def test_line_mode_samples_line_depth_not_bbox_edge():
    """With boundary_mode='line' the shoaling ratio is sampled where the ray
    crosses the site line (20 m), not at the deeper grid edge (30 m)."""
    grid = plane_beach_grid()  # x: 0..20000, depth 30 -> 0
    f = 0.08
    x_t = 20_000.0 * (1.0 - 10.0 / 30.0)  # target at 10 m
    x_b = 20_000.0 * (1.0 - 20.0 / 30.0)  # boundary line at 20 m
    bxy = np.array([[x_b, -11_000.0], [x_b, 11_000.0]])  # vertical, spans the fan
    # Only the shore-normal (offshore-going) rays cross this one-sided line;
    # alongshore/shoreward rays escape or land, which is expected here — they
    # carry no energy in the shore-normal boundary bin we probe below.
    with pytest.warns(UserWarning, match="without crossing the boundary line"):
        op = build_operator(
            grid,
            target_xy=(x_t, 0.0),
            boundary_xy=bxy,
            freqs=np.array([f]),
            dirs=DIRS,
            nsub=15,
            cf_jonswap=None,
            boundary_mode="line",
        )
    e_b = np.zeros((2, 1, DIRS.size))
    e_b[:, 0, DIRS == 270.0] = 1.0  # uniform shoreward swell at both sites
    ratio = op.apply(e_b).sum()  # transferred density in the shore-normal bin

    omega = 2 * np.pi * f
    at_line = group_speed(omega, np.array(20.0)) / group_speed(omega, np.array(10.0))
    at_edge = group_speed(omega, np.array(30.0)) / group_speed(omega, np.array(10.0))
    assert np.isclose(ratio, at_line, rtol=5e-2), (ratio, at_line)
    # the whole point: the line result is NOT the deeper bbox-edge shoaling
    assert abs(ratio - at_line) < abs(ratio - at_edge)
    assert op.attrs["boundary_mode"] == "line"


def test_ring_mode_encloses_target_no_escapes():
    """A target inside a ring of sites: every ray crosses the ring, nothing
    escapes, and a uniform boundary field is conserved (flat bathymetry)."""
    grid = flat_grid(size=16_000.0)
    r = 5_000.0
    ang = np.deg2rad(np.arange(0.0, 360.0, 45.0))
    bxy = np.column_stack([r * np.cos(ang), r * np.sin(ang)])  # 8-point ring
    op = build_operator(
        grid,
        target_xy=(0.0, 0.0),
        boundary_xy=bxy,
        freqs=np.array([0.08]),
        dirs=DIRS,
        nsub=9,
        cf_jonswap=None,
        boundary_mode="ring",
    )
    assert op.attrs["escaped_fraction"] == 0.0
    e_b = np.ones((bxy.shape[0], 1, DIRS.size))  # uniform density 1 everywhere
    e_t = op.apply(e_b)
    # flat depth -> no shoaling: uniform-in, uniform-out per direction bin
    assert np.allclose(e_t, 1.0, atol=2e-2)


def test_line_mode_escape_falls_back_to_nearest_site_and_warns():
    """Rays that miss a short open line fall back to the nearest site and the
    build warns about the escaped fraction."""
    grid = flat_grid(size=16_000.0)
    # short line near the west edge spanning only |y| < 1500: rays toward N/E/S
    # never cross it and must fall back.
    bxy = np.array([[-5_000.0, -1_500.0], [-5_000.0, 1_500.0]])
    with pytest.warns(UserWarning, match="without crossing the boundary line"):
        op = build_operator(
            grid,
            target_xy=(0.0, 0.0),
            boundary_xy=bxy,
            freqs=np.array([0.08]),
            dirs=DIRS,
            nsub=5,
            cf_jonswap=None,
            boundary_mode="line",
        )
    assert op.attrs["escaped_fraction"] > 0.0
    # still a usable operator: uniform field transfers to finite, ~uniform out
    e_b = np.ones((2, 1, DIRS.size))
    e_t = op.apply(e_b)
    assert np.all(np.isfinite(e_t))
    assert np.allclose(e_t, 1.0, atol=2e-2)


def test_invalid_boundary_mode_raises():
    grid = flat_grid()
    with pytest.raises(ValueError, match="boundary_mode"):
        build_operator(
            grid,
            target_xy=(0.0, 0.0),
            boundary_xy=np.array([[-5000.0, 0.0]]),
            freqs=np.array([0.08]),
            dirs=DIRS,
            boundary_mode="perimeter",
        )


def test_operator_netcdf_roundtrip(tmp_path):
    grid = flat_grid(n=41)
    op = build_operator(
        grid,
        target_xy=(0.0, 0.0),
        boundary_xy=np.array([[-5000.0, 0.0]]),
        freqs=np.array([0.1]),
        dirs=DIRS,
        nsub=3,
    )
    path = str(tmp_path / "op.nc")
    op.to_netcdf(path)
    op2 = TransferOperator.from_netcdf(path)
    assert np.allclose(op.T, op2.T)
    assert np.allclose(op.freq, op2.freq)
    assert op2.depth_target == op.depth_target
    assert op2.n_boundary == 1
