"""
Waveguide cross-section geometry primitives.

Classes
-------
_Slab
    Single horizontal material layer defined by refractive index and thickness.
_SlabStack
    Ordered vertical sequence of slabs (bottom → top).
_Strip
    A _SlabStack placed at a horizontal position with a given width.
_StripChain
    Ordered horizontal sequence of strips (left → right).
Slice
    pyFIMM-style vertical stack; callable to produce a _Strip with a given width.

Helper
------
_as_stack
    Normalise a bare _Slab or existing _SlabStack to a _SlabStack.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Union


# ---------------------------------------------------------------------------
# Vertical geometry: slabs and stacks
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _Slab:
    """Single horizontal layer in a vertical slice (bottom toward top)."""

    n: float
    thickness_um: float

    def __str__(self) -> str:
        """Return a concise single-line description of this slab."""
        return f"n = {self.n:.4f},  thickness = {self.thickness_um:.4f} µm"

    def __add__(self, other: Union["_Slab", "_SlabStack"]) -> "_SlabStack":
        """
        Concatenate this slab with another slab or stack to build vertical layers.

        Parameters
        ----------
        other : _Slab or _SlabStack
            Another slab or existing stack to append above this slab.

        Returns
        -------
        _SlabStack
            A new stack containing this slab followed by the other slab(s).
        """
        if isinstance(other, _SlabStack):
            return _SlabStack((self,) + other.slabs)
        return _SlabStack((self, other))


@dataclass(frozen=True)
class _SlabStack:
    slabs: tuple

    def __add__(self, other: Union[_Slab, "_SlabStack"]) -> "_SlabStack":
        """
        Append a slab or merge another stack into this vertical stack.

        Parameters
        ----------
        other : _Slab or _SlabStack
            Slab or stack to append at the top of this stack.

        Returns
        -------
        _SlabStack
            A new stack containing all layers in bottom-to-top order.
        """
        if isinstance(other, _Slab):
            return _SlabStack(self.slabs + (other,))
        return _SlabStack(self.slabs + other.slabs)


def _as_stack(vertical: Union[_Slab, _SlabStack]) -> _SlabStack:
    """
    Normalize a single slab or existing stack to a _SlabStack.

    Helper function to ensure vertical geometry is always represented as a stack.

    Parameters
    ----------
    vertical : _Slab or _SlabStack
        Single slab or existing stack of slabs.

    Returns
    -------
    _SlabStack
        The input stack if already a _SlabStack, or a new single-element stack.
    """
    if isinstance(vertical, _SlabStack):
        return vertical
    return _SlabStack((vertical,))


# ---------------------------------------------------------------------------
# Horizontal geometry: strips and chains
# ---------------------------------------------------------------------------

@dataclass
class _Strip:
    """One vertical slice placed with a horizontal width (microns)."""

    stack: _SlabStack
    width_um: float

    def __add__(self, other: Union["_Strip", "_StripChain"]) -> "_StripChain":
        """
        Place another strip to the right of this strip to build horizontal layout.

        Parameters
        ----------
        other : _Strip or _StripChain
            Strip or chain to append to the right (larger x) of this strip.

        Returns
        -------
        _StripChain
            A new chain containing strips in left-to-right order.
        """
        if isinstance(other, _StripChain):
            return _StripChain([self] + other._strips)
        return _StripChain([self, other])


class _StripChain:
    def __init__(self, strips: Sequence[_Strip]) -> None:
        """
        Initialize a horizontal chain of strips forming a waveguide layout.

        Parameters
        ----------
        strips : Sequence[_Strip]
            Ordered sequence of strips from left to right.
        """
        self._strips = list(strips)

    def __add__(self, other: Union[_Strip, "_StripChain"]) -> "_StripChain":
        """
        Append a strip or concatenate another chain to the right.

        Parameters
        ----------
        other : _Strip or _StripChain
            Strip or chain to place at the right end of this chain.

        Returns
        -------
        _StripChain
            A new chain with all strips in left-to-right order.
        """
        if isinstance(other, _StripChain):
            return _StripChain(self._strips + other._strips)
        return _StripChain(self._strips + [other])


# ---------------------------------------------------------------------------
# Public geometry entry point
# ---------------------------------------------------------------------------

class Slice:
    """
    Vertical 1-D stack (bottom → top), same idea as pyFIMM ``Slice``.

    Example::

        clad = Slice(SiO(15.75))
        core = Slice(SiO(10.0) + SiN(2.5) + SiO(5.0))
    """

    def __init__(self, vertical: Union[_Slab, _SlabStack]) -> None:
        """
        Initialize a vertical slice of materials (bottom to top).

        Parameters
        ----------
        vertical : _Slab or _SlabStack
            Vertical material composition for this slice column.
        """
        self._stack = _as_stack(vertical)

    def __call__(self, width_um: float) -> _Strip:
        """
        Assign a horizontal width to this slice to create a strip.

        Parameters
        ----------
        width_um : float
            Width in microns along the x-axis.

        Returns
        -------
        _Strip
            A strip with this slice's vertical composition and specified width.
        """
        return _Strip(self._stack, float(width_um))
