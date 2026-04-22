"""Backwards-compatibility re-exports. Classes now live in their own modules."""

from .Material import Material
from .Mode import Mode, _ScalarModeView, _VectorModeView
from .structure import Slice, _Slab, _SlabStack, _Strip, _StripChain
from .Waveguide import Waveguide

__all__ = [
    "Material",
    "Mode",
    "Slice",
    "Waveguide",
    "_Slab",
    "_SlabStack",
    "_Strip",
    "_StripChain",
    "_ScalarModeView",
    "_VectorModeView",
]
