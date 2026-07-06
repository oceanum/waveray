"""Ray integrator on a plane beach: Snell's law is the analytic truth."""

import numpy as np

from nearshore_transform.bathymetry import LocalGrid
from nearshore_transform.dispersion import phase_speed
from nearshore_transform.rays import STATUS_EXITED, SpeedField, trace_backward


def plane_beach_grid(d_off=30.0, length=20_000.0, width=16_000.0, nx=201, ny=161):
    """Depth decreasing linearly from d_off at x=0 to 0 at x=length."""
    x = np.linspace(0.0, length, nx)
    y = np.linspace(-width / 2, width / 2, ny)
    depth = np.tile(d_off * (1.0 - x / length), (ny, 1))
    return LocalGrid(x=x, y=y, depth=depth)


def test_snell_invariant_on_plane_beach():
    """sin(alpha)/c must be conserved along rays when contours are parallel."""
    grid = plane_beach_grid()
    f = 0.08
    omega = 2 * np.pi * f
    fld = SpeedField.build(grid, omega, d_min=0.3)

    # Target at 10 m depth; alpha measured from shore-normal (-x is shoreward,
    # so a wave travelling +x-ward has propagation angle theta=0 == alpha=0
    # toward shore... here waves travel toward +x (shore at x=length).
    x_t = 20_000.0 * (1.0 - 10.0 / 30.0)  # depth 10 m
    d_t = grid.sample_depth(np.array([x_t]), np.array([0.0]))[0]
    assert abs(d_t - 10.0) < 0.1

    for alpha_deg in (0.0, 15.0, 30.0, 45.0):
        theta_t = np.deg2rad(alpha_deg)  # propagation direction at target
        fan = trace_backward(fld, x_t, 0.0, np.array([theta_t]), ds=20.0, max_steps=20_000)
        assert fan.status[0] == STATUS_EXITED

        c_t = phase_speed(omega, np.array(d_t))
        d_b = grid.sample_depth(fan.x[:1], fan.y[:1])[0]
        c_b = phase_speed(omega, np.array(d_b))
        # Snell: sin(theta)/c conserved (theta measured from the +x axis,
        # which is the shore-normal of this beach)
        lhs = np.sin(theta_t) / c_t
        rhs = np.sin(fan.theta[0]) / c_b
        assert np.isclose(lhs, rhs, rtol=2e-2, atol=1e-6), f"alpha={alpha_deg}"


def test_rays_block_on_land():
    """Backward rays pointing shoreward (from the target toward the beach)
    must run aground, not exit."""
    grid = plane_beach_grid()
    omega = 2 * np.pi * 0.08
    fld = SpeedField.build(grid, omega, d_min=0.3)
    x_t = 20_000.0 * (1.0 - 10.0 / 30.0)
    # propagation direction -x (offshore-ward): its backward ray marches
    # toward +x, i.e. toward the beach, and must land.
    fan = trace_backward(fld, x_t, 0.0, np.array([np.pi]), ds=20.0, max_steps=20_000)
    assert fan.status[0] != STATUS_EXITED
