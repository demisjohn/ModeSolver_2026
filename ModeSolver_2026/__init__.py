from __future__ import annotations

import numpy as np

# NumPy 1.x compatibility: ElectromagneticPython 2.x uses ``numpy.trapezoid`` (NumPy 2+).
# The ``[eme]`` extra pins NumPy 1.24.x for unmodified ``EMpy_gpu`` (``numpy.testing.Tester``).
if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz  # type: ignore[attr-defined, assignment]

"""
ModeSolver_2026 — pyFIMM-like geometry API with EMpy mode solving.

Backend: `ElectromagneticPython` (EMpy) finite-difference mode solvers
(:class:`SVFDModeSolver` by default, optional :class:`VFDModeSolver`), see
https://github.com/lbolla/EMpy .

Optional EMEpy cross-section modes: pass ``solver="eme"`` (or ``"EMEpy"``) to
:class:`~ModeSolver_2026.ModeSolver.Waveguide.calc` to use EMEpy's ``MSEMpy``
transverse solver. That path requires the ``emepy`` package and its dependencies
(see ``pip install mode-solver-2026[eme]`` or the `emepy` project README).
"""

from .ModeSolver import Material, Mode, Slice, Waveguide


def plot_refractive_index_profile(waveguide: Waveguide, **kwargs):
    """Plot :math:`n(x,y)`; same as :meth:`Waveguide.plot_refractive_index_profile`."""
    return waveguide.plot_refractive_index_profile(**kwargs)



from .__version import version as __version__
from .__version import versiondate as __versiondate__
from .__version import author as __author__
print( "\nModeSolver_2026   (v."  +  __version__  +  "   "  +  __versiondate__ + ") \nby " + __author__ + "\n")


__all__ = [
    "Material",
    "Mode",
    "Slice",
    "Waveguide",
    "plot_refractive_index_profile",
    "__version__",
]

from .__globals import * # global variables/methods to the module.

