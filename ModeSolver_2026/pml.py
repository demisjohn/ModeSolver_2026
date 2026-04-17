"""PML preprocessing: complex isotropic permittivity on a padded real grid."""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
from scipy import constants as const

_ALLOWED_BOUNDARY = frozenset("AS0P")


def validate_calc_boundary(boundary: str) -> str:
    """Return normalized NSEW boundary string (uppercase)."""
    s = boundary.strip().upper()
    if len(s) != 4:
        raise ValueError(
            "boundary must be exactly 4 characters (N,S,E,W order), "
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
    """
    Return ``epsfunc(x, y)`` suitable for EMpy: relative permittivity ε_r = n² (complex).
    """
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
