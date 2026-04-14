from __future__ import annotations

"""
ModeSolver_2026 — pyFIMM-like geometry API with EMpy mode solving.

Backend: `ElectromagneticPython` (EMpy) finite-difference mode solvers
(:class:`SVFDModeSolver` by default, optional :class:`VFDModeSolver`), see
https://github.com/lbolla/EMpy .

Optional EME-style cross-section modes: pass ``solver="eme"`` to
:class:`~ModeSolver_2026.ModeSolver.Waveguide.calc` to use the vector FD path
aligned with EMEPy's ``MSEMpy`` engine (same EMpy VFD kernel; no ``emepy`` import
required). To install the upstream ``emepy`` package for its full EME stack, use
``pip install mode-solver-2026[eme]`` (optional; may need extra dependencies).
"""

from .ModeSolver import Material, Slice, Waveguide


def plot_refractive_index_profile(waveguide: Waveguide, **kwargs):
    """Plot :math:`n(x,y)`; same as :meth:`Waveguide.plot_refractive_index_profile`."""
    return waveguide.plot_refractive_index_profile(**kwargs)



from .__version import version as __version__
from .__version import versiondate as __versiondate__
from .__version import author as __author__
print( "\nModeSolver_2026   (v."  +  __version__  +  "   "  +  __versiondate__ + ") \nby " + __author__ + "\n")


__all__ = [
    "Material",
    "Slice",
    "Waveguide",
    "plot_refractive_index_profile",
    "__version__",
]

from .__globals import * # global variables/methods to the module.

