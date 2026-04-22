"""
Optical mode classes.

Classes
-------
Mode
    Public abstract base class for all mode views. Provides attenuation helpers
    (``get_alpha``, ``get_alpha_dB``); subclasses supply ``neff`` and ``_wg``.
_ScalarModeView
    Mode view for the semi-vectorial FD (SVFD) solver — wraps an
    ``EMpy.SVFDModeSolver`` result and exposes scalar-field plotting.
_VectorModeView
    Mode view for the full-vectorial FD (VFD) solver — wraps an
    ``EMpy.FDMode`` object and exposes vector-field plotting.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import EMpy.utils
from EMpy.modesolvers.FD import SVFDModeSolver

if TYPE_CHECKING:
    from .Waveguide import Waveguide


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class Mode:
    """
    Public base class for all backend optical Mode views.

    Provides attenuation methods shared by :class:`_ScalarModeView` and
    :class:`_VectorModeView`. Subclasses must expose a ``neff`` property and
    store the parent :class:`Waveguide` as ``self._wg``.
    """

    _wg: "Waveguide"

    @property
    def neff(self) -> complex:
        raise NotImplementedError  # implemented by subclasses

    def get_alpha(self) -> float:
        """
        Power attenuation coefficient α in 1/m derived from Im(neff).

        Uses the power-decay convention: P(z) = P(0)·exp(−α·z), so
        α = 2·k₀·|Im(neff)| where k₀ = 2π/λ (in metres).

        Returns
        -------
        float
            Non-negative attenuation coefficient in 1/m.
        """
        k0 = 2.0 * np.pi / (self._wg._wl_um * 1e-6)
        return 2.0 * k0 * abs(self.neff.imag)

    def get_alpha_dB(self) -> float:
        """
        Power attenuation in dB/m derived from Im(neff).

        Converts :meth:`get_alpha` from Np/m to dB/m via the identity
        1 Np/m (power) = 10/ln(10) dB/m ≈ 4.343 dB/m.

        Returns
        -------
        float
            Non-negative attenuation in dB/m.
        """
        return (10 * np.log10(np.e)) * self.get_alpha()
        # return self.get_alpha() * 10.0 / np.log(10.0)


# ---------------------------------------------------------------------------
# Semi-vectorial (SVFD) mode view
# ---------------------------------------------------------------------------

class _ScalarModeView(Mode):
    """SVFD eigenmode: principal field component / scalar wavefunction."""

    def __init__(self, solver: SVFDModeSolver, index: int, wg: "Waveguide") -> None:
        """
        Initialize a scalar mode view from SVFD solver results.

        Parameters
        ----------
        solver : SVFDModeSolver
            EMpy semi-vectorial FD solver instance after solve().
        index : int
            Mode index (0-based) after EMpy sorting.
        wg : Waveguide
            Parent waveguide instance.
        """
        self._solver = solver
        self._index = index
        self._wg = wg

    @property
    def neff(self) -> complex:
        """
        Effective index of this mode.

        Returns
        -------
        complex
            Complex effective index (may have imaginary part with PML).
        """
        return self._solver.neff[self._index]

    def get_field(self, name: str):
        """
        Retrieve a field component array for this mode.

        Parameters
        ----------
        name : str
            Field name (case-insensitive). Options: 'phi', 'scalar', 'psi' for
            the scalar wavefunction; 'ex' or 'ey' if solver method matches.

        Returns
        -------
        np.ndarray
            2-D field array on the FD grid.

        Raises
        ------
        KeyError
            If the requested field is not available for this solver method.
        """
        n = name.lower()
        if n in ("phi", "scalar", "psi"):
            return self._solver.phi[self._index]
        if self._solver.method == "Ex" and n == "ex":
            return self._solver.Ex[self._index]
        if self._solver.method == "Ey" and n == "ey":
            return self._solver.Ey[self._index]
        raise KeyError(
            f"Field {name!r} not available for method={self._solver.method!r}"
        )

    def plot(
        self,
        field: str = "intensity",
        ax=None,
        title: str | None = None,
        cmap: str = "hot",
    ):
        """
        Plot a field magnitude or scalar intensity on the FD grid.

        Parameters
        ----------
        field : str, default='intensity'
            Field to plot. 'intensity' or 'i' plots |φ|² from scalar wavefunction.
            Other names are passed to get_field() and magnitude is shown.
        ax : matplotlib.axes.Axes or None, default=None
            Target axes. If None, creates a new figure.
        title : str or None, default=None
            Plot title. If None, shows neff to 4 decimal places.
        cmap : str, default='hot'
            Matplotlib colormap name for contourf.

        Returns
        -------
        matplotlib.axes.Axes
            The axes containing the plot with colorbar attached.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()
        xc = EMpy.utils.centered1d(self._solver.x)
        yc = EMpy.utils.centered1d(self._solver.y)
        if field.lower() in ("intensity", "i"):
            z = np.abs(self._solver.phi[self._index]) ** 2
            label = "|φ|² (scalar FD)"
        else:
            z = np.abs(self.get_field(field))
            label = f"|{field}|"

        xm, ym = np.meshgrid(xc, yc, indexing="ij")
        cf = ax.contourf(xm, ym, z, levels=32, cmap=cmap)
        ax.set_aspect("equal")
        ax.set_xlabel("x (µm)")
        ax.set_ylabel("y (µm)")
        ax.set_title(
            title or (
                f"neff = {self.neff.real:.4f}"
                f"\nα = {self.get_alpha_dB():.2f} dB/m"
            )
        )
        plt.colorbar(cf, ax=ax, label=label)
        return ax

    def plot_intensity(self, ax=None, title: str | None = None):
        """
        Plot the scalar intensity |φ|² for this mode.

        Parameters
        ----------
        ax : matplotlib.axes.Axes or None, default=None
            Optional axes to draw into.
        title : str or None, default=None
            Optional plot title.

        Returns
        -------
        matplotlib.axes.Axes
            Axes with the intensity plot.
        """
        return self.plot(field="intensity", ax=ax, title=title)


# ---------------------------------------------------------------------------
# Full-vectorial (VFD) mode view
# ---------------------------------------------------------------------------

class _VectorModeView(Mode):
    """Full-vectorial EMpy :class:`FDMode` with pyFIMM-like plotting."""

    def __init__(self, fdmode, wg: "Waveguide") -> None:
        """
        Initialize a vector mode view from VFD solver results.

        Parameters
        ----------
        fdmode
            EMpy mode object with neff, get_field, intensity methods.
        wg : Waveguide
            Parent waveguide instance.
        """
        self._m = fdmode
        self._wg = wg

    @property
    def neff(self) -> complex:
        """
        Effective index of this vector mode.

        Returns
        -------
        complex
            Complex effective index from the underlying EMpy mode object.
        """
        return self._m.neff

    def get_field(self, name: str):
        """
        Retrieve a vector field component from the underlying EMpy mode.

        Parameters
        ----------
        name : str
            Field component name (e.g., 'Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz').

        Returns
        -------
        np.ndarray
            2-D field array from EMpy's get_field method.
        """
        return self._m.get_field(name)

    def plot(
        self,
        field: str = "intensity",
        ax=None,
        title: str | None = None,
        cmap: str = "hot",
    ):
        """
        Plot intensity or a vector field magnitude.

        Parameters
        ----------
        field : str, default='intensity'
            Field to plot. 'intensity' or 'i' uses real(intensity()).
            'ex', 'ey', 'ez' plot magnitudes of those components (case-insensitive).
        ax : matplotlib.axes.Axes or None, default=None
            Optional target axes. If None, creates a new figure.
        title : str or None, default=None
            Plot title. If None, shows neff to 4 decimal places.
        cmap : str, default='hot'
            Matplotlib colormap name for contourf.

        Returns
        -------
        matplotlib.axes.Axes
            Axes with the plotted field and colorbar.

        Raises
        ------
        ValueError
            If field is not a recognized name.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()
        xc = EMpy.utils.centered1d(self._m.x)
        yc = EMpy.utils.centered1d(self._m.y)
        if field.lower() in ("intensity", "i"):
            z = np.real(self._m.intensity())
            label = "|E|² (normalized)"
        elif field.lower() == "ex":
            z = np.abs(self._m.get_field("Ex"))
            label = "|Ex|"
        elif field.lower() == "ey":
            z = np.abs(self._m.get_field("Ey"))
            label = "|Ey|"
        elif field.lower() == "ez":
            z = np.abs(self._m.get_field("Ez"))
            label = "|Ez|"
        else:
            raise ValueError(f"Unknown field {field!r}")

        xm, ym = np.meshgrid(xc, yc, indexing="ij")
        cf = ax.contourf(xm, ym, z, levels=32, cmap=cmap)
        ax.set_aspect("equal")
        ax.set_xlabel("x (µm)")
        ax.set_ylabel("y (µm)")
        ax.set_title(
            title or (
                f"neff = {self.neff.real:.4f}"
                f"\nα = {self.get_alpha_dB():.2f} dB/m"
            )
        )
        plt.colorbar(cf, ax=ax, label=label)
        return ax

    def plot_intensity(self, ax=None, title: str | None = None):
        """
        Plot the |E|² intensity for this vector mode.

        Parameters
        ----------
        ax : matplotlib.axes.Axes or None, default=None
            Optional axes to draw into.
        title : str or None, default=None
            Optional plot title.

        Returns
        -------
        matplotlib.axes.Axes
            Axes with the intensity plot.
        """
        return self.plot(field="intensity", ax=ax, title=title)
