"""
Eigenmode backend for ``Waveguide.calc(..., solver=\"eme\")``.

The `emepy` package on PyPI documents a class ``MSEMpy`` that solves the 2-D
cross-section using EMpy's vector finite-difference (VFD) engine—the same
fields used in eigenmode-expansion (EME) workflows. Importing ``emepy`` pulls
in optional dependencies that often fail on current stacks; this module
therefore calls **ElectromagneticPython** :class:`EMpy.modesolvers.FD.VFDModeSolver`
directly with the waveguide's permittivity, which matches that numerical path
without vendoring or patching upstream ``emepy``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from EMpy.modesolvers.FD import VFDModeSolver

if TYPE_CHECKING:
    from .ModeSolver import Waveguide


@dataclass(frozen=True)
class EMEVFDResult:
    """Result of the EME-style cross-section solve (vector FD, EMpy)."""

    solver: VFDModeSolver


def solve_cross_section_eme(
    waveguide: "Waveguide",
    wavelength_um: float,
    neigs: int,
    nx: int,
    ny: int,
    boundary: str,
    eme_accuracy: float,
    index_guess: float | None,
) -> EMEVFDResult:
    """
    Run vector finite-difference modes on the same grid as ``Waveguide.calc`` FD mode.

    Parameters
    ----------
    wavelength_um, neigs, nx, ny, boundary
        Same meaning as :meth:`Waveguide.calc`.
    eme_accuracy
        Passed to :meth:`VFDModeSolver.solve` as ``tol`` (eigenvalue tolerance).
    index_guess
        Optional sigma shift for the sparse eigensolver (max index if omitted).
    """
    w = waveguide._width_um()
    h = waveguide._height_um()
    x = np.linspace(0.0, w, nx)
    y = np.linspace(0.0, h, ny)
    nmax = max(slab.n for st in waveguide._strips for slab in st.stack.slabs)
    guess = index_guess if index_guess is not None else nmax - 1e-3

    solver = VFDModeSolver(wavelength_um, x, y, waveguide._make_epsfunc(), boundary).solve(
        neigs=neigs, tol=eme_accuracy, guess=guess
    )
    return EMEVFDResult(solver=solver)
