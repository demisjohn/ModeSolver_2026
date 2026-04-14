"""Tests for refractive_index_grid and plot_refractive_index_profile."""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("MPLBACKEND", "Agg")

from matplotlib.colors import LogNorm, Normalize

from ModeSolver_2026 import Material, Slice, Waveguide, plot_refractive_index_profile


@pytest.fixture
def simple_wg():
    """Three-column WG: left clad, higher-index core strip, right clad."""
    low = Material(1.45)
    high = Material(2.0)
    clad = Slice(low(2.0))
    core = Slice(low(1.0) + high(0.5) + low(1.0))
    return Waveguide(clad(1.0) + core(0.5) + clad(1.0))


def test_refractive_index_grid_shape(simple_wg):
    nx, ny = 41, 31
    x, y, n = simple_wg.refractive_index_grid(nx=nx, ny=ny)
    assert x.shape == (nx,)
    assert y.shape == (ny,)
    assert n.shape == (nx, ny)
    assert np.allclose(x[0], 0.0) and np.allclose(x[-1], simple_wg._width_um())
    assert np.allclose(y[0], 0.0) and np.allclose(y[-1], simple_wg._height_um())


def test_refractive_index_grid_values_clad_and_core(simple_wg):
    x, y, n = simple_wg.refractive_index_grid(nx=200, ny=200)
    # Center of middle column (core width 0.5 between x=1 and 1.5)
    ix = int(np.argmin(np.abs(x - 1.25)))
    # Upper high-index slab: y in [1.0, 1.5] from bottom in core slice
    iy_hi = int(np.argmin(np.abs(y - 1.25)))
    assert n[ix, iy_hi] == pytest.approx(2.0, rel=0, abs=1e-9)
    # Lower low-index in core column
    iy_lo = int(np.argmin(np.abs(y - 0.5)))
    assert n[ix, iy_lo] == pytest.approx(1.45, rel=0, abs=1e-9)
    # Left clad column, mid height — uniform low
    ix_left = int(np.argmin(np.abs(x - 0.5)))
    assert n[ix_left, iy_lo] == pytest.approx(1.45, rel=0, abs=1e-9)


def test_plot_refractive_index_profile_smoke(simple_wg):
    fig, ax = plot_refractive_index_profile(simple_wg, nx=48, ny=40, log_scale=True)
    assert fig is not None
    assert ax.get_title() != ""
    # contourf collection
    assert len(ax.collections) >= 1
    assert ax.collections[0].norm.vmin < ax.collections[0].norm.vmax


def test_plot_refractive_index_profile_lognorm_when_log_scale(simple_wg):
    _, ax = plot_refractive_index_profile(simple_wg, nx=32, ny=24, log_scale=True)
    assert isinstance(ax.collections[0].norm, LogNorm)


def test_plot_refractive_index_profile_linear_norm(simple_wg):
    _, ax = plot_refractive_index_profile(simple_wg, nx=32, ny=24, log_scale=False)
    assert isinstance(ax.collections[0].norm, Normalize)


def test_plot_refractive_index_profile_respects_vmin_vmax(simple_wg):
    _, ax = plot_refractive_index_profile(
        simple_wg,
        nx=20,
        ny=20,
        log_scale=True,
        vmin=1.4,
        vmax=2.1,
    )
    norm = ax.collections[0].norm
    assert norm.vmin == pytest.approx(1.4)
    assert norm.vmax == pytest.approx(2.1)
