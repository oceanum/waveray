"""Ray integrator on a plane beach: Snell's law is the analytic truth."""

import numpy as np
import pytest

from waveray.bathymetry import LocalGrid
from waveray.dispersion import phase_speed
from waveray.rays import (
    STATUS_EXITED,
    BoundaryLine,
    SpeedField,
    _first_line_crossing,
    trace_backward,
)


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


def flat_grid(depth=20.0, half=8_000.0, n=161):
    x = np.linspace(-half, half, n)
    y = np.linspace(-half, half, n)
    return LocalGrid(x=x, y=y, depth=np.full((n, n), depth))


def test_boundary_line_segment_counts():
    line = BoundaryLine(x=np.array([0.0, 1.0, 2.0]), y=np.array([0.0, 1.0, 0.0]))
    assert line.n_seg == 2  # open: K-1
    assert list(line.seg_a) == [0, 1] and list(line.seg_b) == [1, 2]
    ring = BoundaryLine(x=np.array([0.0, 1.0, 2.0]), y=np.array([0.0, 1.0, 0.0]), closed=True)
    assert ring.n_seg == 3  # closed: wraps K-1 -> 0
    assert list(ring.seg_b) == [1, 2, 0]
    with pytest.raises(ValueError):
        BoundaryLine(x=np.array([0.0, 1.0]), y=np.array([0.0, 1.0]), closed=True)


def test_first_line_crossing_geometry():
    """A step crossing a known vertical segment returns the right t, seg, u."""
    line = BoundaryLine(x=np.array([-5.0, -5.0]), y=np.array([-10.0, 10.0]))
    px, py = np.array([0.0]), np.array([0.0])
    qx, qy = np.array([-10.0]), np.array([0.0])  # step from 0 toward -x
    t, seg, u = _first_line_crossing(px, py, qx, qy, line)
    assert np.isclose(t[0], 0.5)  # crosses x=-5 at half the step
    assert seg[0] == 0
    assert np.isclose(u[0], 0.5)  # y=0 is the midpoint of the -10..10 segment
    # a step that does not reach the segment: no crossing
    t2, seg2, _ = _first_line_crossing(px, py, np.array([-2.0]), qy, line)
    assert not np.isfinite(t2[0]) and seg2[0] == -1


def test_ray_stops_on_interior_line_not_bbox():
    """With a boundary line inside the grid, the ray terminates on the line
    (well inside the bbox), recording the crossed segment."""
    grid = flat_grid(half=8_000.0)
    fld = SpeedField.build(grid, 2 * np.pi * 0.08, d_min=0.3)
    # vertical line at x = -5000, spanning y; target at origin.
    line = BoundaryLine(x=np.array([-5_000.0, -5_000.0]), y=np.array([-7_000.0, 7_000.0]))
    # theta0 = 0 -> going +x; backward ray marches toward -x into the line.
    fan = trace_backward(
        fld, 0.0, 0.0, np.array([0.0]), ds=30.0, max_steps=2_000, boundary_line=line
    )
    assert fan.status[0] == STATUS_EXITED
    assert np.isclose(fan.x[0], -5_000.0, atol=30.0)  # on the line, not the -8000 edge
    assert fan.seg[0] == 0
    assert np.isclose(fan.u[0], 0.5, atol=0.02)


def test_ray_escapes_open_line_reaches_bbox():
    """A ray missing a short open line runs on to the bbox: seg stays -1."""
    grid = flat_grid(half=8_000.0)
    fld = SpeedField.build(grid, 2 * np.pi * 0.08, d_min=0.3)
    # short segment near x=-5000 spanning only |y| < 500; a ray along +y misses it.
    line = BoundaryLine(x=np.array([-5_000.0, -5_000.0]), y=np.array([-500.0, 500.0]))
    # theta0 = pi/2 -> going +y; backward ray marches toward -y, never meets the line.
    fan = trace_backward(
        fld, 0.0, 0.0, np.array([np.pi / 2]), ds=30.0, max_steps=2_000, boundary_line=line
    )
    assert fan.status[0] == STATUS_EXITED  # exited via the bbox
    assert fan.seg[0] == -1  # but did not cross the line
