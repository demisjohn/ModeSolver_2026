"""Smoke test for ``solver='eme'`` (EMEpy ``MSEMpy``), when ``emepy`` imports."""

from __future__ import annotations

import numpy as np
import pytest

from ModeSolver_2026 import Material, Slice, Waveguide


@pytest.fixture
def simple_wg():
    low = Material(1.45)
    high = Material(2.0)
    clad = Slice(low(2.0))
    core = Slice(low(1.0) + high(0.5) + low(1.0))
    return Waveguide(clad(1.0) + core(0.5) + clad(1.0))


def test_calc_solver_eme_neffs_and_modes(simple_wg):
    try:
        from emepy.fd import MSEMpy  # noqa: F401
    except ImportError:
        pytest.skip(
            "emepy.fd (MSEMpy) is not importable; install emepy and its dependencies."
        )
    # Small grid and loose eigensolver tolerance keep CI fast when emepy is present.
    simple_wg.calc(
        wavelength_um=1.55,
        neigs=1,
        nx=16,
        ny=16,
        boundary="0000",
        solver="eme",
        tol=0.05,
    )
    assert simple_wg._solver_backend == "eme"
    n = simple_wg.neffs
    assert n.shape == (1,)
    assert np.all(np.isfinite(n))
    m0 = simple_wg.mode(0)
    assert np.isfinite(m0.neff.real) or np.isfinite(m0.neff.imag)


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
