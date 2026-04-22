"""Isotropic optical material defined by its refractive index."""

from __future__ import annotations

from .structure import _Slab


class Material:
    """Isotropic material with refractive index ``n`` (dimensionless)."""

    def __init__(self, n: float) -> None:
        """
        Initialize an optical material with a specified refractive index.

        Parameters
        ----------
        n : float
            Refractive index (dimensionless) at the wavelength of interest.
        """
        self.n = float(n)

    def __call__(self, thickness_um: float) -> _Slab:
        """
        Create a horizontal slab layer with this material.

        Parameters
        ----------
        thickness_um : float
            Layer thickness in microns (positive value expected).

        Returns
        -------
        _Slab
            A slab with this material's index and specified thickness.
        """
        return _Slab(self.n, float(thickness_um))
