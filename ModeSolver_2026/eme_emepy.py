"""
EMEpy cross-section backend for ``Waveguide.calc(..., solver='eme')``.

Uses :class:`emepy.fd.MSEMpy`, the mode solver packaged with EMEpy for
eigenmode-expansion workflows (vector finite-difference modes on the
transverse grid). This is the solver path documented for EMEpy; it is **not**
the same as running a multi-layer :class:`emepy.eme.EME` propagation simulation
on a single arbitrary cross-section (that engine expects ``Layer`` geometry).

``MSEMpy`` internally uses ``EMpy_gpu``'s :class:`VFDModeSolver` when available.

Import strategy
---------------
We load ``emepy.fd`` (and its dependencies ``mode``, ``materials``, ``tools``)
**without** executing ``emepy/__init__.py``. Upstream ``emepy``'s package
``__init__`` pulls in the full library (``eme``, ``ann``/PyTorch, etc.); only
the FD path is needed here. This keeps optional dependencies smaller and avoids
import-order issues with unrelated subsystems.

NumPy version
-------------
``EMpy_gpu`` (a dependency of EMEpy's FD stack) still imports
``numpy.testing.Tester``, which was removed in NumPy 2.0. The ``[eme]`` extra
pins NumPy to 1.24.x so third-party packages remain **unmodified**.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

import numpy as np

if TYPE_CHECKING:
    from .ModeSolver import Waveguide


@dataclass(frozen=True)
class EMEMSEMPyResult:
    """Holds the vector FD solver instance produced by ``MSEMpy.solve``."""

    solver: Any


def _get_msempy_class() -> Callable[..., Any]:
    """Return ``MSEMpy`` without importing ``emepy`` package ``__init__``."""
    if "emepy.fd" in sys.modules:
        fd_mod = sys.modules["emepy.fd"]
        return fd_mod.MSEMpy

    spec_pkg = importlib.util.find_spec("emepy")
    if spec_pkg is None or not spec_pkg.submodule_search_locations:
        raise ImportError(
            "The 'emepy' distribution is not installed (cannot find package emepy)."
        )
    emepy_root = Path(spec_pkg.submodule_search_locations[0])

    def _exec_submodule(name: str, filename: str) -> Any:
        full_name = f"emepy.{name}"
        path = emepy_root / filename
        if not path.is_file():
            raise ImportError(f"Expected {path} from emepy distribution.")
        sp = importlib.util.spec_from_file_location(
            full_name,
            path,
            submodule_search_locations=[str(emepy_root)],
        )
        if sp is None or sp.loader is None:
            raise ImportError(f"Cannot load {full_name} from {path}")
        mod = importlib.util.module_from_spec(sp)
        mod.__name__ = full_name
        mod.__package__ = "emepy"
        sys.modules[full_name] = mod
        sp.loader.exec_module(mod)
        return mod

    if "emepy" not in sys.modules:
        emepy_pkg = types.ModuleType("emepy")
        emepy_pkg.__path__ = [str(emepy_root)]  # type: ignore[attr-defined]
        sys.modules["emepy"] = emepy_pkg

    _exec_submodule("mode", "mode.py")
    _exec_submodule("materials", "materials.py")
    _exec_submodule("tools", "tools.py")
    fd_mod = _exec_submodule("fd", "fd.py")
    return fd_mod.MSEMpy


def solve_cross_section_emepy_msempy(
    waveguide: "Waveguide",
    wavelength_um: float,
    neigs: int,
    x: np.ndarray,
    y: np.ndarray,
    boundary: str,
    tol: float,
    epsfunc: Optional[Callable[..., np.ndarray]] = None,
) -> EMEMSEMPyResult:
    """
    Run EMEpy's ``MSEMpy`` on the same ``(x, y)`` grid as :meth:`Waveguide.calc`.

    Parameters
    ----------
    wavelength_um, neigs, x, y, boundary
        Same meaning as :meth:`Waveguide.calc` (``boundary`` must already map
        ``P`` → ``0`` for EMpy if PML is used).
    tol
        Eigenvalue tolerance (``MSEMpy`` ``accuracy`` argument).
    epsfunc
        Optional relative permittivity callback; defaults to ``waveguide._make_epsfunc()``.
    """
    try:
        MSEMpy = _get_msempy_class()
    except ImportError as exc:
        raise ImportError(
            "solver='eme' requires the ``emepy`` package (``MSEMpy`` cross-section "
            "solver). Install with:\n"
            "  pip install 'mode-solver-2026[eme]'\n"
            "or ``pip install emepy``.\n"
            "Use NumPy 1.24.x (see project ``[eme]`` pins): ``EMpy_gpu`` requires "
            "``numpy.testing.Tester`` (removed in NumPy 2.0)."
        ) from exc

    # Non-None width/thickness ensure ``MSEMpy.__init__`` keeps the supplied
    # ``epsfunc`` instead of rebuilding a rectangular profile (see ``emepy.fd``).
    epsf = epsfunc if epsfunc is not None else waveguide._make_epsfunc()
    ms = MSEMpy(
        wl=wavelength_um,
        width=1.0,
        thickness=1.0,
        num_modes=neigs,
        x=x,
        y=y,
        epsfunc=epsf,
        boundary=boundary,
        accuracy=tol,
        subpixel=False,
    )
    ms.solve()
    if ms.solver is None:
        raise RuntimeError("EMEpy MSEMpy.solve() did not set a solver instance.")
    return EMEMSEMPyResult(solver=ms.solver)
