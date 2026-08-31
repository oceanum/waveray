"""Performance guardrails: the cost of wind input in the operator build.

Wind adds three bilinear samples per active ray step (five with agrow) on
top of the friction sample, so the build should slow down by a bounded
constant factor, not change complexity class. These tests measure a
representative build and print the ratios (run pytest with ``-s`` to see
them); the assertions are deliberately generous so scheduler noise cannot
flake CI, while still catching an accidental per-step blow-up.
"""

import time

import numpy as np
import pytest

from waveray.bathymetry import LocalGrid
from waveray.operator import build_operator

DIRS = np.arange(0.0, 360.0, 15.0)
FREQS = np.array([0.06, 0.1, 0.15])


@pytest.fixture(scope="module")
def beach_grid():
    """Plane beach, large enough that rays take a realistic number of steps."""
    nx, ny = 121, 121
    x = np.linspace(0.0, 12_000.0, nx)
    y = np.linspace(-6_000.0, 6_000.0, ny)
    depth = np.tile(25.0 * (1.0 - x / 15_000.0), (ny, 1))
    return LocalGrid(x=x, y=y, depth=depth)


def _build(grid, reps=3, **kwargs):
    """Best-of-reps wall time [s] — the minimum is the least noisy estimator."""
    best = np.inf
    for _ in range(reps):
        t0 = time.perf_counter()
        build_operator(
            grid,
            target_xy=(9_000.0, 0.0),
            boundary_xy=np.array([[0.0, -4_000.0], [0.0, 4_000.0]]),
            freqs=FREQS,
            dirs=DIRS,
            nsub=5,
            **kwargs,
        )
        best = min(best, time.perf_counter() - t0)
    return best


def test_wind_build_overhead(beach_grid):
    t_base = _build(beach_grid, cf_jonswap=0.038)
    t_wind = _build(beach_grid, cf_jonswap=0.038, wind=(15.0, 270.0))
    t_agrow = _build(beach_grid, cf_jonswap=0.038, wind=(15.0, 270.0), agrow=True)

    r_wind = t_wind / t_base
    r_agrow = t_agrow / t_base
    print(
        f"\noperator build: base {t_base:.3f} s, wind {t_wind:.3f} s "
        f"(x{r_wind:.2f}), wind+agrow {t_agrow:.3f} s (x{r_agrow:.2f})"
    )

    # Bounded constant-factor overhead: wind must not change complexity class.
    # Measured on a dev machine: wind x1.20, wind+agrow x1.47.
    assert r_wind < 3.0, f"wind overhead x{r_wind:.2f} exceeds the x3 guardrail"
    assert r_agrow < 4.0, f"wind+agrow overhead x{r_agrow:.2f} exceeds the x4 guardrail"


def test_apply_speed_unaffected_by_wind(beach_grid):
    """Runtime transform cost must not depend on wind: same einsum, plus at
    most one broadcast add for E0."""
    op_base = build_operator(
        beach_grid,
        target_xy=(9_000.0, 0.0),
        boundary_xy=np.array([[0.0, -4_000.0], [0.0, 4_000.0]]),
        freqs=FREQS,
        dirs=DIRS,
        nsub=3,
        cf_jonswap=0.038,
    )
    op_wind = build_operator(
        beach_grid,
        target_xy=(9_000.0, 0.0),
        boundary_xy=np.array([[0.0, -4_000.0], [0.0, 4_000.0]]),
        freqs=FREQS,
        dirs=DIRS,
        nsub=3,
        cf_jonswap=0.038,
        wind=(15.0, 270.0),
        agrow=True,
    )
    e_b = np.random.default_rng(0).random((2_000, 2, FREQS.size, DIRS.size))

    def timeit(op):
        best = np.inf
        for _ in range(3):
            t0 = time.perf_counter()
            op.apply(e_b)
            best = min(best, time.perf_counter() - t0)
        return best

    t_base = timeit(op_base)
    t_wind = timeit(op_wind)
    print(f"\napply 2000 steps: base {t_base * 1e3:.1f} ms, wind+agrow {t_wind * 1e3:.1f} ms")
    assert t_wind < 3.0 * t_base + 0.05  # +50 ms absolute slack for tiny timings
