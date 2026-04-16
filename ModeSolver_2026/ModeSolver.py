"""Rectangular cross-section construction (pyFIMM-style) and EMpy coupling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Literal, Optional, Sequence, Tuple, Union

import numpy as np

import EMpy.utils
from EMpy.modesolvers.FD import SVFDModeSolver, VFDModeSolver

from .eme_emepy import solve_cross_section_eme

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


@dataclass(frozen=True)
class _Slab:
    """Single horizontal layer in a vertical slice (bottom toward top)."""

    n: float
    thickness_um: float

    def __add__(self, other: Union["_Slab", "_SlabStack"]) -> "_SlabStack":
        if isinstance(other, _SlabStack):
            return _SlabStack((self,) + other.slabs)
        return _SlabStack((self, other))


@dataclass(frozen=True)
class _SlabStack:
    slabs: tuple

    def __add__(self, other: Union[_Slab, "_SlabStack"]) -> "_SlabStack":
        if isinstance(other, _Slab):
            return _SlabStack(self.slabs + (other,))
        return _SlabStack(self.slabs + other.slabs)


class Material:
    """Isotropic material with refractive index ``n`` (dimensionless)."""

    def __init__(self, n: float) -> None:
        self.n = float(n)

    def __call__(self, thickness_um: float) -> _Slab:
        return _Slab(self.n, float(thickness_um))


def _as_stack(vertical: Union[_Slab, _SlabStack]) -> _SlabStack:
    if isinstance(vertical, _SlabStack):
        return vertical
    return _SlabStack((vertical,))


@dataclass
class _Strip:
    """One vertical slice placed with a horizontal width (microns)."""

    stack: _SlabStack
    width_um: float

    def __add__(self, other: Union["_Strip", "_StripChain"]) -> "_StripChain":
        if isinstance(other, _StripChain):
            return _StripChain([self] + other._strips)
        return _StripChain([self, other])


class Slice:
    """
    Vertical 1-D stack (bottom → top), same idea as pyFIMM ``Slice``.

    Example::

        clad = Slice(SiO(15.75))
        core = Slice(SiO(10.0) + SiN(2.5) + SiO(5.0))
    """

    def __init__(self, vertical: Union[_Slab, _SlabStack]) -> None:
        self._stack = _as_stack(vertical)

    def __call__(self, width_um: float) -> _Strip:
        return _Strip(self._stack, float(width_um))


class _StripChain:
    def __init__(self, strips: Sequence[_Strip]) -> None:
        self._strips = list(strips)

    def __add__(self, other: Union[_Strip, "_StripChain"]) -> "_StripChain":
        if isinstance(other, _StripChain):
            return _StripChain(self._strips + other._strips)
        return _StripChain(self._strips + [other])


class Waveguide:
    """
    2-D cross-section from left-to-right strips, pyFIMM ``Waveguide`` style.

    Example::

        WG = Waveguide(clad(3.0) + core(1.0) + clad(4.0))
        WG.calc(wavelength_um=1.55, neigs=5)
        WG.plot("all")
        WG.mode(0).plot_intensity()
    """

    def __init__(self, layout: Union[_Strip, _StripChain]) -> None:
        if isinstance(layout, _StripChain):
            self._strips = list(layout._strips)
        else:
            self._strips = [layout]
        self._solver: SVFDModeSolver | VFDModeSolver | None = None
        self._vectorial: bool = False
        self._solver_backend: Literal["fd", "eme"] = "fd"
        self._wl_um: float | None = None
        self._x: np.ndarray | None = None
        self._y: np.ndarray | None = None

    def _height_um(self) -> float:
        return max(
            sum(slab.thickness_um for slab in st.stack.slabs) for st in self._strips
        )

    def _width_um(self) -> float:
        return sum(st.width_um for st in self._strips)

    def _n_at(self, x_um: float, y_um: float) -> float:
        """Piecewise-constant index; origin bottom-left of bounding box."""
        if x_um < 0 or y_um < 0:
            raise ValueError("coordinates outside structure")
        x0 = 0.0
        for st in self._strips:
            x1 = x0 + st.width_um
            if x_um < x1 or st is self._strips[-1]:
                yb = 0.0
                for slab in st.stack.slabs:
                    yt = yb + slab.thickness_um
                    if y_um < yt or slab is st.stack.slabs[-1]:
                        return slab.n
                    yb = yt
                return st.stack.slabs[-1].n
            x0 = x1
        return self._strips[-1].stack.slabs[-1].n

    def refractive_index_grid(
        self, 
        nx: int = 200, 
        ny: int = 200
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample the piecewise-constant refractive index on a uniform grid.
        
        Parameters
        ----------
        nx, ny: int, number grid points in each dimension.
        
        Returns
        -------
        x, y
            1D coordinates in µm (horizontal ``x``, vertical ``y``, bottom-left origin).
        n
            2D array ``n[ix, iy]`` at ``(x[ix], y[iy])`` (``ij`` indexing).
        """
        print("waveguide.plot_refractive_index_profile(): plotting RIX...")
        w = self._width_um()
        h = self._height_um()
        x = np.linspace(0.0, w, int(nx))
        y = np.linspace(0.0, h, int(ny))
        n = np.empty((x.size, y.size), dtype=np.float64)
        for ix, xv in enumerate(x):
            for iy, yv in enumerate(y):
                n[ix, iy] = self._n_at(float(xv), float(yv))
        return x, y, n

    def _make_epsfunc(self) -> Callable:
        def epsfunc(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            if x.ndim != 1 or y.ndim != 1:
                raise ValueError("EMpy passes 1D center coordinates for x and y.")
            eps = np.empty((x.size, y.size), dtype=np.float64)
            for ix, xv in enumerate(x):
                for iy, yv in enumerate(y):
                    eps[ix, iy] = self._n_at(float(xv), float(yv)) ** 2
            return eps

        return epsfunc

    def calc(
        self,
        wavelength_um: float = 1.55,
        neigs: int = 5,
        nx: int = 120,
        ny: int = 120,
        boundary: str = "0000",
        vectorial: bool = False,
        fd_method: str = "scalar",
        index_guess: float | None = None,
        tol: float = 0.0,
        solver: Literal["fd", "eme"] = "fd",
        eme_accuracy: float = 1e-8,
    ) -> "Waveguide":
        """
        Run a mode solver on the cross-section.

        ``solver="fd"`` (default) uses EMpy finite-difference solvers. By default
        uses :class:`SVFDModeSolver` (scalar or ``Ex`` / ``Ey`` semi-vectorial),
        which is much faster than the full-vectorial :class:`VFDModeSolver`.
        Set ``vectorial=True`` for vectorial modes and full E/H components (see EMpy).

        ``solver="eme"`` uses the same vector finite-difference engine as the
        EMEPy ``MSEMpy`` cross-section path (full E/H via :class:`VFDModeSolver`),
        suitable for eigenmode-expansion workflows, without importing the optional
        ``emepy`` package.

        Parameters
        ----------
        wavelength_um
            Vacuum wavelength in microns.
        neigs
            Number of eigenmodes to compute.
        nx, ny
            Vertex counts along the simulation window edges.
        boundary
            Per EMpy: four chars N,S,E,W each ``'0'``, ``'S'``, or ``'A'``.  See EMpy.modesolves.FD documentation.  Pasted here: The following options are available:
           'A' - Hx is antisymmetric, Hy is symmetric.
           'S' - Hx is symmetric and, Hy is antisymmetric.
           '0' - Hx and Hy are zero immediately outside of the boundary.
        vectorial
            If True, use full vectorial solver (slower, more accurate) via :class:`VFDModeSolver`. Defaults to False. Ignored when ``solver="eme"``.
        fd_method
            For ``vectorial=False`` and ``solver="fd"``: ``'scalar'``, ``'Ex'``, or ``'Ey'`` (SVFD).
        index_guess
            For vectorial or EME solver: sigma shift; default uses max index.
        tol
            Eigenvalue tolerance for FD solvers (``solver="fd"``).
        solver
            ``"fd"`` for semi-vectorial or vectorial FD as controlled by ``vectorial``;
            ``"eme"`` for vector FD with ``eme_accuracy``.
        eme_accuracy
            Eigenvalue tolerance when ``solver="eme"`` (passed to :class:`VFDModeSolver`).
        """
        w = self._width_um()
        h = self._height_um()
        x = np.linspace(0.0, w, nx)
        y = np.linspace(0.0, h, ny)
        nmax = max(slab.n for st in self._strips for slab in st.stack.slabs)
        guess = index_guess if index_guess is not None else nmax - 1e-3

        print("Waveguide.calc(): Calculating Modes...")
        if solver == "eme":
            result = solve_cross_section_eme(
                self,
                wavelength_um=wavelength_um,
                neigs=neigs,
                nx=nx,
                ny=ny,
                boundary=boundary,
                eme_accuracy=eme_accuracy,
                index_guess=index_guess,
            )
            self._solver = result.solver
            self._vectorial = True
            self._solver_backend = "eme"
        elif vectorial:
            solver = VFDModeSolver(
                wavelength_um, x, y, self._make_epsfunc(), boundary
            ).solve(neigs=neigs, tol=tol, guess=guess)
            self._solver = solver
            self._vectorial = True
            self._solver_backend = "fd"
        else:
            solver = SVFDModeSolver(
                wavelength_um, x, y, self._make_epsfunc(), boundary, method=fd_method
            ).solve(neigs=neigs, tol=tol)
            self._solver = solver
            self._vectorial = False
            self._solver_backend = "fd"

        self._wl_um = wavelength_um
        self._x = x
        self._y = y
        return self

    @property
    def modes(self):
        if self._solver is None:
            raise RuntimeError("Call calc() before accessing modes.")
        if self._vectorial:
            return self._solver.modes
        raise AttributeError(
            "Scalar/SVFD solution has no .modes list; use mode(i) or neffs."
        )

    @property
    def neffs(self) -> np.ndarray:
        if self._solver is None:
            raise RuntimeError("Call calc() before accessing neffs.")
        if self._vectorial:
            return np.asarray([complex(m.neff) for m in self._solver.modes])
        return np.asarray(self._solver.neff)

    def mode(self, index: int | Sequence[int]):
        if self._solver is None:
            raise RuntimeError("Call calc() before accessing modes.")
        if self._vectorial:
            if isinstance(index, int):
                return _VectorModeView(self._solver.modes[index], self)
            return tuple(_VectorModeView(self._solver.modes[i], self) for i in index)
        if isinstance(index, int):
            return _ScalarModeView(self._solver, index, self)
        return tuple(_ScalarModeView(self._solver, i, self) for i in index)

    def neff_dataframe(self):
        """Effective indices as a :class:`pandas.DataFrame` (requires pandas)."""
        import pandas as pd

        if self._solver is None:
            raise RuntimeError("Call calc() first.")
        if self._vectorial:
            vals = [complex(m.neff) for m in self._solver.modes]
        else:
            vals = [complex(z) for z in self._solver.neff]
        return pd.DataFrame({"mode": range(len(vals)), "neff": vals})

    def plot(
        self,
        which: Literal["all"],
        *,
        figsize: Tuple[float, float] = (11.0, 7.0),
        suptitle: str | None = None,
    ) -> Tuple["Figure", Any]:
        """
        Multi-panel mode field plots after :meth:`calc`.

        ``which="all"`` reproduces the Example1 layout: a ``2×3`` grid of
        intensity maps for the first five modes (fewer if fewer were computed),
        with unused panels hidden.

        Parameters
        ----------
        which
            Currently only ``"all"`` is supported.
        figsize
            Figure size in inches (width, height).
        suptitle
            Optional figure title; a default is chosen from the solver backend.

        Returns
        -------
        fig, axes
            Matplotlib figure and ``2×3`` array of axes.
        """
        if which != "all":
            raise ValueError(f"plot(which=...): unknown {which!r}; use 'all'.")

        if self._solver is None:
            raise RuntimeError("Call calc() before plot().")

        import matplotlib.pyplot as plt

        n_modes = int(self.neffs.size)
        n_plot = min(5, n_modes)
        if n_plot == 0:
            raise RuntimeError("No modes to plot; increase neigs in calc().")

        fig, axes = plt.subplots(2, 3, figsize=figsize, constrained_layout=True)
        axes_flat = axes.ravel()
        for i in range(n_plot):
            m = self.mode(i)
            m.plot_intensity(
                ax=axes_flat[i], title=f"Mode {i}, neff = {m.neff.real:.4f}"
            )
        for j in range(n_plot, 6):
            axes_flat[j].axis("off")

        if suptitle is None:
            backend = self._solver_backend
            if backend == "eme":
                sub = (
                    f"First {n_plot} mode(s)\n"
                    'ModeSolver_2026 · solver="eme" '
                    "(vector FD / EME-style cross-section)"
                )
            else:
                sub = (
                    f"First {n_plot} mode(s)\n"
                    'ModeSolver_2026 · solver="fd" '
                    "(finite-difference cross-section)"
                )
            fig.suptitle(sub, fontsize=11)
        else:
            fig.suptitle(suptitle, fontsize=11)

        return fig, axes

    def plot_refractive_index_profile(
        self,
        waveguide: Optional["Waveguide"] = None,  # allow user to pass a different WG obj.
        *,
        nx: int = 200,
        ny: int = 200,
        ax: Optional["Axes"] = None,
        cmap: str = "coolwarm",
        log_scale: bool = False,
        title: str | None = None,
        xlabel: str = "x (µm)",
        ylabel: str = "y (µm)",
        colorbar_label: str = "Refractive index n",
        vmin: float | None = None,
        vmax: float | None = None,
    ) -> Tuple["Figure", "Axes"]:
        """
        Plot the 2D refractive-index cross-section (x horizontal, y vertical).

        Uses a diverging *hot–cold* style colormap (default ``coolwarm``: blue → red)
        so cladding and core contrasts read clearly. When ``log_scale`` is True,
        color mapping uses :class:`matplotlib.colors.LogNorm` so small index
        differences remain visible across orders of magnitude.

        Parameters
        ----------
        waveguide
            Constructed :class:`~ModeSolver_2026.geometry.Waveguide` (``calc()`` not required). Defaults to `self`.
        nx, ny
            Sample counts along x and y.
        ax
            Optional existing matplotlib axes.
        cmap
            Matplotlib colormap name (Defaults to `coolwarm` for hot–cold).
        log_scale
            If True, map colors with logarithmic scaling of *n*. Defaults to False.
        title, xlabel, ylabel, colorbar_label
            Axis and colorbar labels.
        vmin, vmax
            Optional bounds for the norm (linear or log). Defaults to data range.
        
        Returns
        -------
        fig, ax matplotlib objects for the generated plot.
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm, Normalize
        
        waveguide = waveguide if (waveguide is not None) else self
        
        x, y, n = waveguide.refractive_index_grid(nx=nx, ny=ny)
        xm, ym = np.meshgrid(x, y, indexing="ij")
        
        import reprlib
        
        print('wg.plot_RIX(): x =')
        print(reprlib.repr( x ))
        
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        n_min = float(np.min(n))
        n_max = float(np.max(n))
        if vmin is None:
            vmin = n_min
        if vmax is None:
            vmax = n_max
        if log_scale:
            vmin = max(vmin, 1e-15)
            vmax = max(vmax, vmin * 1.000001)
            if vmin >= vmax:
                vmax = vmin * (1.0 + 1e-6)
            norm: LogNorm | Normalize = LogNorm(vmin=vmin, vmax=vmax)
        else:
            if vmin >= vmax:
                vmax = vmin + 1e-9
            norm = Normalize(vmin=vmin, vmax=vmax)

        cf = ax.contourf(
            xm,
            ym,
            n,
            levels=80,
            cmap=cmap,
            norm=norm,
        )
        ax.set_aspect("equal")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title or "Refractive index profile")
        fig.colorbar(cf, ax=ax, label=colorbar_label)
        return fig, ax
    
    # alias:
    plot_RIX = plot_refractive_index_profile

#end class waveguide



class _ScalarModeView:
    """SVFD eigenmode: principal field component / scalar wavefunction."""

    def __init__(self, solver: SVFDModeSolver, index: int, wg: Waveguide) -> None:
        self._solver = solver
        self._index = index
        self._wg = wg

    @property
    def neff(self) -> complex:
        return self._solver.neff[self._index]

    def get_field(self, name: str):
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
        ax.set_title(title or f"neff = {self.neff.real:.4f}")
        plt.colorbar(cf, ax=ax, label=label)
        return ax

    def plot_intensity(self, ax=None, title: str | None = None):
        return self.plot(field="intensity", ax=ax, title=title)
    


class _VectorModeView:
    """Full-vectorial EMpy :class:`FDMode` with pyFIMM-like plotting."""

    def __init__(self, fdmode, wg: Waveguide) -> None:
        self._m = fdmode
        self._wg = wg

    @property
    def neff(self) -> complex:
        return self._m.neff

    def get_field(self, name: str):
        return self._m.get_field(name)

    def plot(
        self,
        field: str = "intensity",
        ax=None,
        title: str | None = None,
        cmap: str = "hot",
    ):
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
        ax.set_title(title or f"neff = {self.neff.real:.4f}")
        plt.colorbar(cf, ax=ax, label=label)
        return ax

    def plot_intensity(self, ax=None, title: str | None = None):
        return self.plot(field="intensity", ax=ax, title=title)
