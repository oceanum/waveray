"""Island sheltering: rays that ground on an island carry no energy.

Blocking is a depth test (depth < d_min -> STATUS_LANDED), so an island in the
middle of the domain shelters its lee exactly as a shoreline does. Ray theory
has no diffraction, so the shadow is sharper than reality; these tests pin the
geometric behaviour, not surf-zone realism.
"""

import numpy as np

from nearshore_transform.bathymetry import LocalGrid
from nearshore_transform.operator import build_operator

DIRS = np.arange(0.0, 360.0, 10.0)
FREQS = np.array([0.07, 0.09])
BPTS = np.array([[-10_000.0, 0.0], [0.0, -10_000.0], [10_000.0, 0.0], [0.0, 10_000.0]])

ISLAND_CENTRE = (-3000.0, 0.0)
ISLAND_RADIUS = 1500.0
TARGET_IN_LEE = (3000.0, 0.0)  # due east of the island


def island_grid(with_island: bool = True, ramp: float = 800.0, n: int = 201) -> LocalGrid:
    """Flat 30 m domain, optionally with a circular island west of centre."""
    x = y = np.linspace(-10_000.0, 10_000.0, n)
    gx, gy = np.meshgrid(x, y)
    depth = np.full_like(gx, 30.0)
    if with_island:
        r = np.hypot(gx - ISLAND_CENTRE[0], gy - ISLAND_CENTRE[1])
        depth = np.minimum(30.0, 30.0 * np.clip((r - ISLAND_RADIUS) / ramp, 0.0, 1.0))
    return LocalGrid(x=x, y=y, depth=depth)


def operator(grid: LocalGrid, target=TARGET_IN_LEE):
    return build_operator(
        grid,
        target_xy=target,
        boundary_xy=BPTS,
        freqs=FREQS,
        dirs=DIRS,
        nsub=9,
        cf_jonswap=None,
    )


def transfer_by_direction(op) -> np.ndarray:
    """Total transfer coefficient per target direction bin, first frequency."""
    return op.T[0].sum(axis=(1, 2))


def test_island_blocks_its_lee_and_leaves_other_directions_intact():
    t_island = transfer_by_direction(operator(island_grid(with_island=True)))
    t_free = transfer_by_direction(operator(island_grid(with_island=False)))

    j = {d: int(np.flatnonzero(DIRS == d)[0]) for d in (270.0, 280.0, 90.0, 0.0)}

    # Flat bathymetry: pure identity, every direction passes through untouched.
    assert np.allclose(t_free, 1.0, atol=1e-6)

    # Waves coming from the west travel east through the island -> fully blocked.
    assert t_island[j[270.0]] == 0.0
    assert t_island[j[280.0]] == 0.0

    # Directions that never cross the island are untouched.
    assert np.isclose(t_island[j[90.0]], 1.0, atol=1e-6)
    assert np.isclose(t_island[j[0.0]], 1.0, atol=1e-6)

    # Some rays landed on the island; none were lost to the step budget.
    op = operator(island_grid(with_island=True))
    assert op.attrs["landed_fraction"] > 0.05
    assert op.attrs["lost_fraction"] == 0.0


def test_lee_shadow_is_deeper_than_beside_the_island():
    grid = island_grid(with_island=True)
    lee = transfer_by_direction(operator(grid, target=TARGET_IN_LEE))
    beside = transfer_by_direction(operator(grid, target=(3000.0, 8000.0)))
    # summed over all directions, the sheltered point receives strictly less
    assert lee.sum() < beside.sum()


def test_sharp_cliff_island_stays_finite():
    """A vertical-sided island (no shoaling rim) must not break the integrator."""
    x = y = np.linspace(-10_000.0, 10_000.0, 201)
    gx, gy = np.meshgrid(x, y)
    r = np.hypot(gx - ISLAND_CENTRE[0], gy - ISLAND_CENTRE[1])
    grid = LocalGrid(x=x, y=y, depth=np.where(r <= ISLAND_RADIUS, 0.0, 30.0))

    op = operator(grid)
    assert np.all(np.isfinite(op.T))
    assert op.attrs["lost_fraction"] == 0.0
    assert transfer_by_direction(op)[int(np.flatnonzero(DIRS == 270.0)[0])] == 0.0


def test_unresolved_islet_does_not_block():
    """A sub-grid islet is smoothed away by bilinear depth sampling.

    Guards the practical trap: GEBCO (~450 m) cannot shelter behind a small
    reef or islet, so sheltering by such features needs a finer bathymetry.
    """
    j270 = int(np.flatnonzero(DIRS == 270.0)[0])

    def t270(islet_radius: float, spacing: float) -> float:
        n = int(16_000 / spacing) + 1
        x = y = np.linspace(-8000.0, 8000.0, n)
        gx, gy = np.meshgrid(x, y)
        r = np.hypot(gx + 3000.0, gy)
        grid = LocalGrid(x=x, y=y, depth=np.where(r <= islet_radius, 0.0, 30.0))
        bpts = np.array([[-8000.0, 0.0], [0.0, -8000.0], [8000.0, 0.0], [0.0, 8000.0]])
        op = build_operator(
            grid,
            target_xy=(3000.0, 0.0),
            boundary_xy=bpts,
            freqs=FREQS,
            dirs=DIRS,
            nsub=9,
            cf_jonswap=None,
        )
        return float(transfer_by_direction(op)[j270])

    assert t270(1500.0, 100.0) == 0.0  # well resolved -> blocks completely
    assert t270(300.0, 1000.0) > 0.75  # sub-grid islet -> nearly transparent
