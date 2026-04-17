"""PML boundary option and preprocessing."""

from __future__ import annotations

import numpy as np
import pytest

from ModeSolver_2026 import Material, Slice, Waveguide
from ModeSolver_2026.pml import (
    boundary_for_empy,
    power_attenuation_from_neff_imag,
    validate_calc_boundary,
)


@pytest.fixture
def simple_wg():
    low = Material(1.45)
    high = Material(2.0)
    clad = Slice(low(2.0))
    core = Slice(low(1.0) + high(0.5) + low(1.0))
    return Waveguide(clad(1.0) + core(0.5) + clad(1.0))


def test_validate_boundary_rejects_bad_char():
    with pytest.raises(ValueError, match="invalid character"):
        validate_calc_boundary("X000")


def test_validate_boundary_rejects_length():
    with pytest.raises(ValueError, match="exactly 4"):
        validate_calc_boundary("000")


def test_boundary_for_empy_maps_p():
    assert boundary_for_empy("PP00") == "0000"
    assert boundary_for_empy("P0A0") == "00A0"


def test_calc_pp00_svfd_smoke(simple_wg):
    simple_wg.calc(
        wavelength_um=1.55,
        neigs=1,
        nx=24,
        ny=24,
        boundary="PP00",
        pml_cells=4,
        tol=0.05,
    )
    assert simple_wg._solver_backend == "svfd"
    n = simple_wg.neffs
    assert n.shape == (1,)
    assert np.isfinite(n[0].real) and np.isfinite(n[0].imag)


def test_calc_0000_regression(simple_wg):
    simple_wg.calc(
        wavelength_um=1.55,
        neigs=1,
        nx=24,
        ny=24,
        boundary="0000",
        tol=0.05,
    )
    assert simple_wg._solver_backend == "svfd"
    n = simple_wg.neffs
    assert np.isfinite(n[0].real)


def test_power_attenuation_from_neff_imag():
    alpha = power_attenuation_from_neff_imag(1e-5, 1.55)
    assert alpha > 0 and np.isfinite(alpha)
