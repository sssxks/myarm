from __future__ import annotations
from typing import Iterable
import sympy as sp
from .sym import Sym


class Mat:
    def __init__(self, data: Iterable[Iterable[Sym]]):
        self._m = sp.Matrix([[s.e for s in row] for row in data])  # type: ignore

    def T(self) -> Mat:
        return Mat([[Sym(x) for x in self._m.T]])  # type: ignore

    def det(self) -> Sym:
        return Sym(self._m.det())  # type: ignore

    def inv(self) -> Mat:
        return Mat([[Sym(x) for x in self._m.inv()]])  # type: ignore

    def __matmul__(self, other: Mat) -> Mat:
        return Mat([[Sym(x) for x in (self._m @ other._m)]]) # type: ignore
