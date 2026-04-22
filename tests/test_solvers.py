"""Smoke tests for the SVFD (default) and VFD EMpy solvers."""

from __future__ import annotations

import pytest

from ModeSolver_2026 import Material, Slice, Waveguide


@pytest.fixture
def simple_wg():
    low = Material(1.45)
    high = Material(2.0)
    clad = Slice(low(2.0))
    core = Slice(low(1.0) + high(0.5) + low(1.0))
    return Waveguide(clad(1.0) + core(0.5) + clad(1.0))


def test_calc_svfd_default_and_fd_synonym(simple_wg):
    simple_wg.calc(wavelength_um=1.55, neigs=1, nx=14, ny=14)
    assert simple_wg._solver_backend == "svfd"
    simple_wg.calc(wavelength_um=1.55, neigs=1, nx=14, ny=14, solver="fd")
    assert simple_wg._solver_backend == "svfd"


def test_calc_vfd_synonym(simple_wg):
    simple_wg.calc(
        wavelength_um=1.55,
        neigs=1,
        nx=14,
        ny=14,
        solver="vectorial",
        tol=0.05,
    )
    assert simple_wg._solver_backend == "vfd"
