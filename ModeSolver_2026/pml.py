"""PML preprocessing via stretched-coordinate complex grid for FD mode solvers."""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
from scipy import constants as const

_ALLOWED_BOUNDARY = frozenset("AS0P")


def validate_calc_boundary(boundary: str) -> str:
    """Return normalized NSEW boundary string (uppercase, 4 chars).

    A single allowed character is treated as shorthand for all four sides,
    e.g. ``"p"`` → ``"PPPP"``, ``"0"`` → ``"0000"``, ``"S"`` → ``"SSSS"``,
    ``"A"`` → ``"AAAA"``.
    """
    s = boundary.strip().upper()
    if len(s) == 1 and s in _ALLOWED_BOUNDARY:
        s = s * 4
    if len(s) != 4:
        raise ValueError(
            "boundary must be exactly 1 or 4 characters (N,S,E,W order), "
            f"each one of A, S, 0, P; got {boundary!r}."
        )
    bad = [c for c in s if c not in _ALLOWED_BOUNDARY]
    if bad:
        raise ValueError(
            "boundary: invalid character(s) "
            f"{set(bad)!r}; allowed: A, S, 0, P. Full string was {boundary!r}."
        )
    return s


def boundary_for_empy(boundary_upper: str) -> str:
    """Map PML letter P to EMpy-compatible zero-field outer boundary 0."""
    return "".join("0" if c == "P" else c for c in boundary_upper)


def has_pml(boundary_upper: str) -> bool:
    """
    Check if any edge uses PML (letter 'P' in boundary string).

    Parameters
    ----------
    boundary_upper : str
        Four-character boundary string in uppercase (from validate_calc_boundary).

    Returns
    -------
    bool
        True if any character is 'P', indicating PML on at least one edge.
    
    Notes
    -----
    Used by calc() to branch between real and complex permittivity paths.
    """
    return "P" in boundary_upper


def normalize_pml_cells(
    pml_cells: int | Sequence[int],
    boundary_upper: str,
) -> tuple[int, int, int, int]:
    """
    Return effective PML cell counts (N, S, E, W) for sides marked ``P``; 0 elsewhere.
    """
    if isinstance(pml_cells, int):
        base = (int(pml_cells),) * 4
    else:
        seq = tuple(pml_cells)
        if len(seq) != 4:
            raise ValueError(
                "pml_cells as a sequence must have 4 integers (N, S, E, W); "
                f"got length {len(seq)}."
            )
        base = tuple(int(x) for x in seq)
    out: list[int] = []
    for i, ch in enumerate(boundary_upper):
        if ch == "P":
            out.append(max(1, base[i]))
        else:
            out.append(0)
    return (out[0], out[1], out[2], out[3])


def extend_vertex_axes(
    w_um: float,
    h_um: float,
    nx: int,
    ny: int,
    boundary_upper: str,
    pml_cells_nswe: tuple[int, int, int, int],
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """
    Build extended vertex coordinates (µm) with PML padding outside [0,w]×[0,h].

    Interior ``linspace(0,w,nx)`` / ``linspace(0,h,ny)`` is unchanged when PML is present.
    """
    if nx < 2 or ny < 2:
        raise ValueError("PML requires nx >= 2 and ny >= 2.")
    x_core = np.linspace(0.0, w_um, nx)
    y_core = np.linspace(0.0, h_um, ny)
    dx = w_um / (nx - 1)
    dy = h_um / (ny - 1)
    b = boundary_upper
    n_n, n_s, n_e, n_w = pml_cells_nswe

    parts_x: list[np.ndarray] = []
    if b[3] == "P" and n_w > 0:
        parts_x.append(np.linspace(-n_w * dx, 0.0, n_w + 1)[:-1])
    parts_x.append(x_core)
    if b[2] == "P" and n_e > 0:
        parts_x.append(np.linspace(w_um, w_um + n_e * dx, n_e + 1)[1:])
    x = np.concatenate(parts_x)

    parts_y: list[np.ndarray] = []
    if b[1] == "P" and n_s > 0:
        parts_y.append(np.linspace(-n_s * dy, 0.0, n_s + 1)[:-1])
    parts_y.append(y_core)
    if b[0] == "P" and n_n > 0:
        parts_y.append(np.linspace(h_um, h_um + n_n * dy, n_n + 1)[1:])
    y = np.concatenate(parts_y)

    meta = {
        "dx_um": dx,
        "dy_um": dy,
        "d_north_um": float(n_n * dy),
        "d_south_um": float(n_s * dy),
        "d_east_um": float(n_e * dx),
        "d_west_um": float(n_w * dx),
    }
    return x, y, meta


def _sigma_max_geom(d_m: float, m: int, R: float) -> float:
    """
    Compute default geometric conductivity σ_max for PML grading.

    Internal helper implementing the formula σ_max = (m+1)/(2d) * ln(1/R)
    for target reflectivity R at PML thickness d.

    Parameters
    ----------
    d_m : float
        PML thickness in meters (not microns).
    m : int
        Polynomial grading order.
    R : float
        Target reflectivity (e.g., 1e-8 for low reflection).

    Returns
    -------
    float
        Geometric conductivity σ_max in 1/m, or 0.0 if d_m <= 0.
    
    Notes
    -----
    Not part of the public API. Override via pml_sigma_max_geom parameter
    in make_pml_epsfunc or Waveguide.calc() if manual tuning is needed.
    """
    if d_m <= 0.0:
        return 0.0
    return (m + 1.0) / (2.0 * d_m) * np.log(1.0 / R)


def _s_stretch(
    u_um: float,
    d_um: float,
    m: int,
    R: float,
    omega: float,
    sigma_max_geom_override: float | None,
) -> complex:
    """1D complex stretching factor s(u) at depth u_um into a PML of thickness d_um."""
    if d_um <= 0.0 or u_um <= 0.0:
        return 1.0 + 0.0j
    d_m = d_um * 1e-6
    u_m = min(u_um, d_um) * 1e-6
    sigma_max_geom = (
        sigma_max_geom_override
        if sigma_max_geom_override is not None
        else _sigma_max_geom(d_m, m, R)
    )
    sigma_geom = sigma_max_geom * (u_m / d_m) ** m
    sigma_rad = const.c * sigma_geom
    return 1.0 - 1.0j * sigma_rad / omega


def make_pml_complex_grid(
    x_real: np.ndarray,
    y_real: np.ndarray,
    w_um: float,
    h_um: float,
    wavelength_um: float,
    boundary_upper: str,
    d_north_um: float,
    d_south_um: float,
    d_east_um: float,
    d_west_um: float,
    m: int = 3,
    R: float = 1e-8,
    sigma_max_geom_override: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert real extended vertex axes into complex stretched-coordinate arrays.

    Inside the physical domain ``[0, w] x [0, h]`` the coordinates stay purely
    real.  In the PML padding the cell spacings acquire an imaginary part that
    encodes the PML absorption via the Chew & Weedon (1994) stretched-coordinate
    formulation.  When passed to EMpy's FD solver, ``dx = numpy.diff(x_complex)``
    is complex in PML cells, so the finite-difference stencil coefficients
    (which are proportional to ``1/dx`` and ``1/dx**2``) automatically include
    the correct ``1/s`` and ``1/s**2`` PML factors without modifying EMpy's code.

    Parameters
    ----------
    x_real, y_real : np.ndarray
        Real vertex coordinates (µm) from :func:`extend_vertex_axes`.
    w_um, h_um : float
        Physical domain width and height in µm.
    wavelength_um : float
        Vacuum wavelength in µm.
    boundary_upper : str
        Four-character NSEW boundary string (from :func:`validate_calc_boundary`).
    d_north_um, d_south_um, d_east_um, d_west_um : float
        PML slab thickness on each side in µm (0 when that side is not PML).
    m : int, default 3
        Polynomial grading order for :math:`\\sigma(u) \\propto (u/d)^m`.
    R : float, default 1e-8
        Target reflectivity for the default :math:`\\sigma_{\\max}`.
    sigma_max_geom_override : float or None, default None
        If set, overrides the automatic :math:`\\sigma_{\\max}`.

    Returns
    -------
    x_complex, y_complex : np.ndarray (dtype=complex128)
        Vertex arrays whose ``diff()`` values carry the PML stretching.
    """
    omega = 2.0 * np.pi * const.c / (wavelength_um * 1e-6)
    b = boundary_upper

    def _stretch_axis(
        coords: np.ndarray,
        lo: float,
        hi: float,
        d_lo_um: float,
        d_hi_um: float,
        b_lo: str,
        b_hi: str,
    ) -> np.ndarray:
        n = len(coords)
        if n < 2:
            return coords.astype(np.complex128)
        dx_real = np.diff(coords)
        dx_complex = np.empty(n - 1, dtype=np.complex128)
        for i in range(n - 1):
            mid = 0.5 * (coords[i] + coords[i + 1])
            s: complex = 1.0 + 0.0j
            if b_hi == "P" and d_hi_um > 0.0:
                u = max(0.0, mid - hi)
                if u > 0.0:
                    s *= _s_stretch(u, d_hi_um, m, R, omega, sigma_max_geom_override)
            if b_lo == "P" and d_lo_um > 0.0:
                u = max(0.0, lo - mid)
                if u > 0.0:
                    s *= _s_stretch(u, d_lo_um, m, R, omega, sigma_max_geom_override)
            # _s_stretch returns s = 1 - jσ/ω (ε-convention: multiplying ε by s
            # adds loss).  EMpy's FD stencil needs Im(dx) > 0 for absorption
            # (see EMpy.stretchmesh which forces Im >= 0).  Conjugating gives
            # s_grid = 1 + jσ/ω so that dx_complex has the correct sign.
            dx_complex[i] = dx_real[i] * s.conjugate()
        out = np.empty(n, dtype=np.complex128)
        out[0] = coords[0]
        np.cumsum(dx_complex, out=out[1:])
        out[1:] += coords[0]
        return out

    # x: west PML is the "lo" side (x < 0), east PML is the "hi" side (x > w)
    x_complex = _stretch_axis(x_real, 0.0, w_um, d_west_um, d_east_um, b[3], b[2])
    # y: south PML is the "lo" side (y < 0), north PML is the "hi" side (y > h)
    y_complex = _stretch_axis(y_real, 0.0, h_um, d_south_um, d_north_um, b[1], b[0])
    return x_complex, y_complex


# ---- legacy / deprecated -------------------------------------------------

def make_pml_epsfunc(
    n_at_bounded: Callable[[float, float], float],
    w_um: float,
    h_um: float,
    wavelength_um: float,
    boundary_upper: str,
    d_north_um: float,
    d_south_um: float,
    d_east_um: float,
    d_west_um: float,
    m: int = 3,
    R: float = 1e-8,
    sigma_max_geom_override: float | None = None,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Return ``epsfunc(x, y)`` with isotropic complex-ε PML.

    .. deprecated::
        This applies PML via ``ε_eff = n² · s_x · s_y``, an isotropic
        approximation that adds spurious loss to guided modes.  Use
        :func:`make_pml_complex_grid` (stretched-coordinate approach) instead.
    """
    import warnings
    warnings.warn(
        "make_pml_epsfunc() uses an isotropic ε approximation that adds "
        "spurious loss to guided modes.  Use make_pml_complex_grid() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    omega = 2.0 * np.pi * const.c / (wavelength_um * 1e-6)
    b = boundary_upper

    def epsfunc(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        if x.ndim != 1 or y.ndim != 1:
            raise ValueError("EMpy passes 1D center coordinates for x and y.")
        out = np.empty((x.size, y.size), dtype=np.complex128)
        for ix, xv in enumerate(x):
            for iy, yv in enumerate(y):
                n0 = n_at_bounded(float(xv), float(yv))
                u_n = max(0.0, float(yv) - h_um) if b[0] == "P" else 0.0
                u_s = max(0.0, -float(yv)) if b[1] == "P" else 0.0
                u_e = max(0.0, float(xv) - w_um) if b[2] == "P" else 0.0
                u_w = max(0.0, -float(xv)) if b[3] == "P" else 0.0

                sx = 1.0 + 0.0j
                if u_w > 0.0 and b[3] == "P":
                    sx *= _s_stretch(
                        u_w, d_west_um, m, R, omega, sigma_max_geom_override
                    )
                if u_e > 0.0 and b[2] == "P":
                    sx *= _s_stretch(
                        u_e, d_east_um, m, R, omega, sigma_max_geom_override
                    )

                sy = 1.0 + 0.0j
                if u_n > 0.0 and b[0] == "P":
                    sy *= _s_stretch(
                        u_n, d_north_um, m, R, omega, sigma_max_geom_override
                    )
                if u_s > 0.0 and b[1] == "P":
                    sy *= _s_stretch(
                        u_s, d_south_um, m, R, omega, sigma_max_geom_override
                    )

                n_pml = n0 * np.sqrt(sx * sy)
                out[ix, iy] = n_pml**2
        return out

    return epsfunc


def power_attenuation_from_neff_imag(neff_imag: float, wavelength_um: float) -> float:
    """
    Power-law attenuation constant α = 2 k₀ Im(n_eff) with k₀ = 2π/λ [1/µm].

    Multiply by 1e4 for cm⁻¹ or convert to dB/m with 10/ln(10) * α_m.
    """
    k0 = 2.0 * np.pi / wavelength_um
    return 2.0 * k0 * neff_imag
