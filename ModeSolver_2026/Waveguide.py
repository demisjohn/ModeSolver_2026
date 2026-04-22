"""2-D waveguide cross-section construction and mode solving."""

from __future__ import annotations

from typing import Any, Callable, Literal, Optional, Sequence, Tuple, TYPE_CHECKING, Union

import numpy as np

import EMpy.utils
from EMpy.modesolvers.FD import SVFDModeSolver, VFDModeSolver

from .structure import _Strip, _StripChain
from .pml import (
    boundary_for_empy,
    extend_vertex_axes,
    has_pml,
    make_pml_epsfunc,
    normalize_pml_cells,
    validate_calc_boundary,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


def _normalize_calc_solver(solver: str) -> Literal["svfd", "vfd"]:
    """
    Map ``calc(solver=...)`` to an internal backend key.

    SVFD (default): ``"SVFD"``, ``"EMpy-SVFD"``, ``"semi-vectorial"``, ``"fd"``.
    VFD: ``"VFD"``, ``"EMpy-VFD"``, ``"vectorial"``.
    """
    s = (
        str(solver).strip().lower().replace(" ", "").replace("-", "").replace("_", "")
    )
    if s in ("svfd", "empysvfd", "semivectorial", "fd"):
        return "svfd"
    if s in ("vfd", "empyvfd", "vectorial", "fullvectorial"):
        return "vfd"
    raise ValueError(
        f"Unknown calc(solver={solver!r}). "
        "Use 'SVFD' (default), 'EMpy-SVFD', 'semi-vectorial', 'fd'; or "
        "'VFD', 'EMpy-VFD', 'vectorial'."
    )


def _waveguide_plot_kind(which: str | None) -> Literal["rix", "all"]:
    """Map ``plot(which=...)`` to ``'rix'`` (index profile) or ``'all'`` (modes)."""
    if which is None or (isinstance(which, str) and which.strip() == ""):
        return "rix"
    if not isinstance(which, str):
        raise TypeError(
            "Waveguide.plot(which=...): expected str or None, got "
            f"{type(which).__name__}"
        )
    normalized = (
        which.strip().lower().replace(" ", "").replace("-", "").replace("_", "")
    )
    if normalized in ("rix", "refractiveindex"):
        return "rix"
    if normalized == "all":
        return "all"
    raise ValueError(
        "Waveguide.plot(which=...): unknown "
        f"{which!r}; use None (default), 'RIX', 'refractiveindex', or 'all'."
    )


class Waveguide:
    """
    2-D cross-section from left-to-right strips, pyFIMM ``Waveguide`` style.

    Example::

        WG = Waveguide(clad(3.0) + core(1.0) + clad(4.0))
        WG.plot()
        WG.calc(wavelength_um=1.55, neigs=5)
        WG.plot("all")
        WG.mode(0).plot_intensity()
    """

    def __init__(self, layout: Union[_Strip, _StripChain]) -> None:
        """
        Initialize a 2-D waveguide cross-section from a horizontal layout.

        Parameters
        ----------
        layout : _Strip or _StripChain
            Single strip or chain of strips defining the left-to-right geometry.

        Notes
        -----
        Internal state includes solver results (_solver), backend type, wavelength,
        and grid coordinates, all initialized to None until calc() is called.
        """
        if isinstance(layout, _StripChain):
            self._strips = list(layout._strips)
        else:
            self._strips = [layout]
        self._solver: SVFDModeSolver | VFDModeSolver | None = None
        self._vectorial: bool = False
        self._solver_backend: Literal["svfd", "vfd"] = "svfd"
        self._wl_um: float | None = None
        self._x: np.ndarray | None = None
        self._y: np.ndarray | None = None

    def __str__(self) -> str:
        """Return a human-readable summary of the waveguide geometry and solver state.

        Output mirrors the pyFIMM ``Waveguide.__str__()`` style: strips are listed
        left-to-right with positional labels (Leftmost / Middle / Rightmost), and
        slabs within each strip are listed bottom-to-top (Bottom / Middle / Top).
        When ``calc()`` has been run, wavelength, solver backend, and effective
        indices are also shown.

        Example
        -------
        Waveguide Cross-section
          Total width:  8.0000 µm,   Height: 18.2500 µm
          Wavelength:   1.5500 µm   [solver: SVFD]
          Modes:  5
          neff:   1.7523   1.7200   1.6891   1.6504   1.6038

        ----- Leftmost Strip: -----
        width = 3.0000 µm
        *** Bottom Layer: ***
        n = 1.4440,  thickness = 10.0000 µm
        *** Top Layer: ***
        n = 1.4440,  thickness = 8.2500 µm

        ----- Middle Strip 1: -----
        width = 1.0000 µm
        ...
        """
        lines: list[str] = []

        # --- Header: geometry summary ---
        lines.append("Waveguide Cross-section")
        lines.append(
            f"  Total width: {self._width_um():8.4f} µm,   "
            f"Height: {self._height_um():.4f} µm"
        )

        # --- Solver state (only when calc() has been called) ---
        if self._solver is not None:
            backend_label = {
                "svfd": "SVFD (semi-vectorial FD)",
                "vfd":  "VFD  (full-vectorial FD)",
            }.get(self._solver_backend, self._solver_backend.upper())
            lines.append(
                f"  Wavelength:  {self._wl_um:.4f} µm   "
                f"[solver: {backend_label}]"
            )
            neff_arr = self.neffs
            lines.append(f"  Modes found: {neff_arr.size}")
            neff_strs = "   ".join(
                f"{v.real:.6f}" if v.imag == 0.0 else f"{v.real:.6f}{v.imag:+.2e}j"
                for v in neff_arr
            )
            lines.append(f"  neff:   {neff_strs}")

        lines.append("")  # blank line before strip list

        # --- Strip / slab detail (pyFIMM style) ---
        n_strips = len(self._strips)
        for n, strip in enumerate(self._strips):
            # Strip position label
            if n == 0:
                label = "Leftmost Strip:"
            elif n == n_strips - 1:
                label = "Rightmost Strip:"
            else:
                label = f"Middle Strip {n}:"
            lines.append(5 * "-" + f" {label} " + 5 * "-")
            lines.append(f"width = {strip.width_um:.4f} µm")

            # Slab (layer) detail within this strip
            slabs = strip.stack.slabs
            n_slabs = len(slabs)
            for i, slab in enumerate(slabs):
                if i == 0:
                    layer_label = "Bottom Layer:"
                elif i == n_slabs - 1:
                    layer_label = "Top Layer:"
                else:
                    layer_label = f"Middle Layer {i}:"
                lines.append(3 * "*" + f" {layer_label} " + 3 * "*")
                lines.append(str(slab))

        return "\n".join(lines)

    def _height_um(self) -> float:
        """
        Compute the maximum vertical extent of the waveguide cross-section.

        Returns
        -------
        float
            Height in microns (tallest strip's total thickness).
        """
        return max(
            sum(slab.thickness_um for slab in st.stack.slabs) for st in self._strips
        )

    def _width_um(self) -> float:
        """
        Compute the total horizontal extent of the waveguide cross-section.

        Returns
        -------
        float
            Width in microns (sum of all strip widths).
        """
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

    def _n_at_bounded(self, x_um: float, y_um: float) -> float:
        """Refractive index with (x,y) clamped to the physical bounding box (for PML padding)."""
        w = self._width_um()
        h = self._height_um()
        return self._n_at(
            float(np.clip(x_um, 0.0, w)),
            float(np.clip(y_um, 0.0, h)),
        )

    def refractive_index_grid(
        self,
        nx: int = 200,
        ny: int = 200
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample the piecewise-constant refractive index on a uniform grid.

        Parameters
        ----------
        nx, ny : int, default=200
            Number of grid points in each dimension.

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
        """
        Build a real relative-permittivity callback for EMpy on the physical window.

        Returns
        -------
        callable
            Function epsfunc(x, y) returning real ε = n² for EMpy FD solvers.

        Notes
        -----
        Used for non-PML simulations. For PML, calc() replaces this with a
        complex callback from pml.make_pml_epsfunc().
        """
        def epsfunc(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            """
            Compute relative permittivity at EMpy cell-center coordinates.

            Parameters
            ----------
            x : np.ndarray
                1-D array of x-coordinates in microns (cell centers).
            y : np.ndarray
                1-D array of y-coordinates in microns (cell centers).

            Returns
            -------
            np.ndarray
                Real relative permittivity ε_r = n² with shape (x.size, y.size).

            Raises
            ------
            ValueError
                If x or y is not 1-dimensional.
            """
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
        fd_method: str = "scalar",
        index_guess: float | None = None,
        tol: float = 0.0,
        solver: str = "SVFD",
        pml_cells: int | tuple[int, int, int, int] = 10,
        pml_m: int = 3,
        pml_R: float = 1e-8,
        pml_sigma_max_geom: float | None = None,
    ) -> "Waveguide":
        """
        Run a mode solver on the cross-section.

        The ``solver`` string selects the backend (case- and separator-insensitive):

        * **SVFD** (default): ElectromagneticPython :class:`SVFDModeSolver`
          (scalar or ``Ex`` / ``Ey`` semi-vectorial). Synonyms: ``"EMpy-SVFD"``,
          ``"semi-vectorial"``, ``"fd"``.
        * **VFD**: full-vectorial :class:`VFDModeSolver` (all field components).
          Synonyms: ``"EMpy-VFD"``, ``"vectorial"``.

        Parameters
        ----------
        wavelength_um : float, default=1.55
            Vacuum wavelength in microns.
        neigs : int, default=5
            Number of eigenmodes to compute.
        nx : int, default=120
            Vertex count along x-axis (**physical** window ``[0, width]``).
            If any boundary is ``'P'``, extra vertices are appended outside that
            window for the PML; interior spacing matches ``linspace`` on the box.
        ny : int, default=120
            Vertex count along y-axis (**physical** window ``[0, height]``).
            If any boundary is ``'P'``, extra vertices are appended outside that
            window for the PML; interior spacing matches ``linspace`` on the box.
        boundary : str, default="0000"
            Four characters representing each side of the simulation window: `N,S,E,W` (north/top, south/bottom, east/right, west/left). Each may be:

            * ``'0'`` — Hx and Hy zero immediately outside the boundary (EMpy).
            * ``'S'`` / ``'A'`` — symmetry / antisymmetry (EMpy); see EMpy FD docs.
            * ``'P'`` — perfectly matched layer (PML): absorption via complex
              :math:`\\varepsilon` on padded cells; EMpy still sees outer ``'0'`` on
              that edge.

            Example: ``"PP00"`` is PML on top and bottom only, with zero-field boundaries on left and right.

            Case-insensitive.
            If only one character is provided, it is expanded to all four sides,
            e.g. ``"P"`` → ``"PPPP"``, ``"0"`` → ``"0000"``, ``"S"`` → ``"SSSS"``,
            ``"A"`` → ``"AAAA"``.

            ``'P'`` (PML) is only implemented via the complex-ε
            preprocessor (same path for SVFD and VFD).
        fd_method : str, default="scalar"
            For ``solver`` SVFD only: ``'scalar'``, ``'Ex'``, or ``'Ey'``.
        index_guess : float or None, default=None
            For VFD only: optional sigma shift for the sparse eigensolver.
            If ``None``, uses max index minus a small offset.
        tol : float, default=0.0
            Eigenvalue tolerance for SVFD and VFD.
        solver : str, default="SVFD"
            ``"SVFD"`` or ``"VFD"`` (plus synonyms above).
        pml_cells : int or tuple[int, int, int, int], default=10
            When any side uses ``'P'``: integer (same count on every ``P`` side) or
            ``(N, S, E, W)`` integer tuple for PML thickness in **FD cells** on each
            ``P`` side (minimum 1 cell per active ``P``).
        pml_m : int, default=3
            Polynomial grade :math:`m` in :math:`\\sigma(u) \\propto (u/d)^m`.
        pml_R : float, default=1e-8
            Target reflectivity :math:`R` for the default :math:`\\sigma_{\\max}` choice.
        pml_sigma_max_geom : float or None, default=None
            If set, overrides the geometric :math:`\\sigma_{\\max}` (in ``1/m`` at full
            depth) for every active PML slab; otherwise use the default from
            :math:`(m+1)/(2d)\\ln(1/R)` with physical thickness :math:`d`.
        """
        w = self._width_um()
        h = self._height_um()
        b_upper = validate_calc_boundary(boundary)
        b_empy = boundary_for_empy(b_upper)

        if has_pml(b_upper):
            pml_nswe = normalize_pml_cells(pml_cells, b_upper)
            x, y, meta = extend_vertex_axes(w, h, nx, ny, b_upper, pml_nswe)
            epsfunc = make_pml_epsfunc(
                self._n_at_bounded,
                w,
                h,
                wavelength_um,
                b_upper,
                meta["d_north_um"],
                meta["d_south_um"],
                meta["d_east_um"],
                meta["d_west_um"],
                m=pml_m,
                R=pml_R,
                sigma_max_geom_override=pml_sigma_max_geom,
            )
        else:
            x = np.linspace(0.0, w, nx)
            y = np.linspace(0.0, h, ny)
            epsfunc = self._make_epsfunc()

        nmax = max(slab.n for st in self._strips for slab in st.stack.slabs)
        guess = index_guess if index_guess is not None else nmax - 1e-3

        kind = _normalize_calc_solver(solver)
        print("Waveguide.calc(): Calculating Modes...")
        if kind == "vfd":
            fd_solver = VFDModeSolver(
                wavelength_um, x, y, epsfunc, b_empy
            ).solve(neigs=neigs, tol=tol, guess=guess)
            self._solver = fd_solver
            self._vectorial = True
            self._solver_backend = "vfd"
        else:
            fd_solver = SVFDModeSolver(
                wavelength_um, x, y, epsfunc, b_empy, method=fd_method
            ).solve(neigs=neigs, tol=tol)
            self._solver = fd_solver
            self._vectorial = False
            self._solver_backend = "svfd"

        self._wl_um = wavelength_um
        self._x = x
        self._y = y
        return self

    @property
    def modes(self) -> list:
        """
        All solved modes as a list of :class:`Mode` objects.

        Works for every backend (SVFD and VFD). Each element supports
        ``.neff``, ``.plot()``, ``.plot_intensity()``, ``.get_field()``,
        ``.get_alpha()``, and ``.get_alpha_dB()``.

        Returns
        -------
        list of Mode
            One entry per computed eigenmode, ordered by decreasing real(neff).

        Raises
        ------
        RuntimeError
            If :meth:`calc` has not been called yet.
        """
        if self._solver is None:
            raise RuntimeError("Call calc() before accessing modes.")
        return [self.mode(i) for i in range(len(self.neffs))]

    @property
    def neffs(self) -> np.ndarray:
        """
        Effective indices for all computed modes.

        Returns
        -------
        np.ndarray
            1-D array of complex effective indices with length equal to neigs
            from the last calc() call. May be complex with PML or lossy media.

        Raises
        ------
        RuntimeError
            If calc() has not been called yet.
        """
        if self._solver is None:
            raise RuntimeError("Call calc() before accessing neffs.")
        if self._vectorial:
            return np.asarray([complex(m.neff) for m in self._solver.modes])
        return np.asarray(self._solver.neff)

    def mode(self, index: int | Sequence[int]):
        """
        Access one or more modes with a plotting-friendly wrapper.

        Parameters
        ----------
        index : int or sequence of int
            Mode index or indices in range [0, neigs). Sorted by effective index
            (highest neff first).

        Returns
        -------
        Mode or tuple of Mode
            Single :class:`Mode` for int index, or tuple of :class:`Mode` for
            a sequence. Returns a :class:`_ScalarModeView` for SVFD and a
            :class:`_VectorModeView` for VFD (both subclass :class:`Mode`).

        Raises
        ------
        RuntimeError
            If calc() has not been called yet.
        IndexError
            If index is out of range for the computed modes.
        """
        if self._solver is None:
            raise RuntimeError("Call calc() before accessing modes.")
        if self._vectorial:
            from .Mode import _VectorModeView  # deferred to avoid circular import
            if isinstance(index, int):
                return _VectorModeView(self._solver.modes[index], self)
            return tuple(_VectorModeView(self._solver.modes[i], self) for i in index)
        from .Mode import _ScalarModeView  # deferred to avoid circular import
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

    def plot_refractive_index_profile(
        self,
        waveguide: Optional["Waveguide"] = None,  # allow user to pass a different WG obj.
        *,
        nx: int = 200,
        ny: int = 200,
        ax: Optional["Axes"] = None,
        figsize: Tuple[float, float] | None = None,
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
        waveguide : Waveguide or None, default=None
            Constructed :class:`~ModeSolver_2026.Waveguide` (``calc()`` not required). If ``None``, uses `self`.
        nx : int, default=200
            Sample count along x-axis.
        ny : int, default=200
            Sample count along y-axis.
        ax : matplotlib.axes.Axes or None, default=None
            Optional existing matplotlib axes.
        figsize
            Figure ``(width, height)`` in inches when ``ax`` is None. If omitted,
            matplotlib's default size is used.
        cmap
            Matplotlib colormap name (Defaults to `coolwarm` for hot–cold).
        log_scale
            If True, map colors with logarithmic scaling of *n*. Defaults to False.
        title : str or None, default=None
            Plot title.
        xlabel : str, default="x (µm)"
            X-axis label.
        ylabel : str, default="y (µm)"
            Y-axis label.
        colorbar_label : str, default="Refractive index n"
            Colorbar label.
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
        print(reprlib.repr(x))

        if ax is None:
            sub_kw: dict[str, Any] = {}
            if figsize is not None:
                sub_kw["figsize"] = figsize
            fig, ax = plt.subplots(**sub_kw)
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

    def plot(
        self,
        which: str | None = None,
        *,
        waveguide: Optional["Waveguide"] = None,
        nx: int = 200,
        ny: int = 200,
        ax: Optional["Axes"] = None,
        figsize: Tuple[float, float] | None = None,
        cmap: str = "coolwarm",
        log_scale: bool = False,
        title: str | None = None,
        xlabel: str = "x (µm)",
        ylabel: str = "y (µm)",
        colorbar_label: str = "Refractive index n",
        vmin: float | None = None,
        vmax: float | None = None,
        suptitle: str | None = None,
    ) -> Tuple["Figure", Any]:
        """
        Plotting entry point: refractive index and/or solved modes.

        * **Default** (``which`` omitted, ``None``, or ``""``): same as
          :meth:`plot_refractive_index_profile` — 2D :math:`n(x,y)` cross-section.
          Aliases: ``"RIX"``, ``"refractiveindex"`` (spacing and case ignored).
        * ``which="all"``: after :meth:`calc`, a ``2×3`` grid of mode intensities
          (first five modes, or fewer if fewer were computed).

        Parameters
        ----------
        which : str or None, default=None
            ``None`` / ``""`` / ``"RIX"`` / ``"refractiveindex"`` for the refractive index profile;
            ``"all"`` for the multi-mode intensity figure.
        waveguide : Waveguide or None, default=None
            Constructed :class:`~ModeSolver_2026.Waveguide` (``calc()`` not required). If ``None``, uses `self`.
        nx : int, default=200
            Sample count along x-axis (for RIX plots).
        ny : int, default=200
            Sample count along y-axis (for RIX plots).
        ax : matplotlib.axes.Axes or None, default=None
            Target axes for RIX plots; ignored for ``"all"``.
        figsize : tuple or None, default=None
            Figure size: for RIX plots, optional size when ``ax`` is None; for ``"all"``, defaults to ``(11, 7)`` when omitted.
        cmap : str, default="coolwarm"
            Matplotlib colormap name for RIX plots.
        log_scale : bool, default=False
            If True, use logarithmic color scaling for RIX plots.
        title : str or None, default=None
            Plot title for RIX plots.
        xlabel : str, default="x (µm)"
            X-axis label for RIX plots.
        ylabel : str, default="y (µm)"
            Y-axis label for RIX plots.
        colorbar_label : str, default="Refractive index n"
            Colorbar label for RIX plots.
        vmin : float or None, default=None
            Lower bound for RIX color norm; defaults to data minimum.
        vmax : float or None, default=None
            Upper bound for RIX color norm; defaults to data maximum.
        suptitle : str or None, default=None
            Figure super-title for ``which="all"``; if omitted, a default based on solver backend is chosen.

        Returns
        -------
        fig, ax_or_axes
            ``(fig, ax)`` for RIX; ``(fig, axes)`` with a ``2×3`` *axes* array for
            ``"all"``.
        """
        kind = _waveguide_plot_kind(which)
        if kind == "rix":
            return self.plot_refractive_index_profile(
                waveguide,
                nx=nx,
                ny=ny,
                ax=ax,
                figsize=figsize,
                cmap=cmap,
                log_scale=log_scale,
                title=title,
                xlabel=xlabel,
                ylabel=ylabel,
                colorbar_label=colorbar_label,
                vmin=vmin,
                vmax=vmax,
            )

        if self._solver is None:
            raise RuntimeError('Call calc() before plot("all").')

        import matplotlib.pyplot as plt

        n_modes = int(self.neffs.size)
        n_plot = min(5, n_modes)
        if n_plot == 0:
            raise RuntimeError("No modes to plot; increase neigs in calc().")

        all_figsize = figsize if figsize is not None else (11.0, 7.0)
        fig, axes = plt.subplots(
            2, 3, figsize=all_figsize, constrained_layout=True
        )
        axes_flat = axes.ravel()
        for i in range(n_plot):
            m = self.mode(i)
            m.plot_intensity(
                ax=axes_flat[i],
                title=(
                    f"Mode {i},  neff = {m.neff.real:.4f}"
                    f"\nα = {m.get_alpha_dB():.2f} dB/m"
                ),
            )
        for j in range(n_plot, 6):
            axes_flat[j].axis("off")

        if suptitle is None:
            backend = self._solver_backend
            if backend == "vfd":
                sub = (
                    f"First {n_plot} mode(s)\n"
                    'ModeSolver_2026 · solver="VFD" (EMPy VFDModeSolver)'
                )
            else:
                sub = (
                    f"First {n_plot} mode(s)\n"
                    'ModeSolver_2026 · solver="SVFD" (EMPy SVFDModeSolver)'
                )
            fig.suptitle(sub, fontsize=11)
        else:
            fig.suptitle(suptitle, fontsize=11)

        return fig, axes

    # alias:
    plot_RIX = plot_refractive_index_profile
