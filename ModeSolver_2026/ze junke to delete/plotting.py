"""Visualization helpers for cross-section refractive-index profiles."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from ModeSolver_2026.geometry import Waveguide


def plot_refractive_index_profile(
    waveguide: "Waveguide",
    *,
    nx: int = 200,
    ny: int = 200,
    ax: Optional["Axes"] = None,
    cmap: str = "coolwarm",
    log_scale: bool = True,
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
        Constructed :class:`~ModeSolver_2026.geometry.Waveguide` (``calc()`` not required).
    nx, ny
        Sample counts along x and y.
    ax
        Optional existing matplotlib axes.
    cmap
        Matplotlib colormap name (``coolwarm`` recommended for hot–cold).
    log_scale
        If True, map colors with logarithmic scaling of *n*.
    title, xlabel, ylabel, colorbar_label
        Axis and colorbar labels.
    vmin, vmax
        Optional bounds for the norm (linear or log). Defaults to data range.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize

    x, y, n = waveguide.refractive_index_grid(nx=nx, ny=ny)
    xm, ym = np.meshgrid(x, y, indexing="ij")

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
