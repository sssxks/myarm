from __future__ import annotations
from dataclasses import dataclass
from typing import Any, cast
import sympy as sp

from .types import Num

type ExprLike = Any

@dataclass(frozen=True)
class Sym:
    e: sp.Expr

    @staticmethod
    def of(x: Any) -> Sym:
        if isinstance(x, Sym):
            return x
        
        return Sym(cast(sp.Expr, sp.sympify(x))) # type: ignore

    def __add__(self, other: Sym | Num) -> Sym:
        o = other.e if isinstance(other, Sym) else other # type: ignore
        return Sym(self.e + sp.sympify(o)) # type: ignore
    
    __radd__ = __add__

    def __mul__(self, other: Sym | Num) -> Sym:
        o = other.e if isinstance(other, Sym) else other # type: ignore
        return Sym(self.e * sp.sympify(o)) # type: ignore
    def __rmul__(self, other: Sym | Num) -> Sym:
        return Sym.of(other).__mul__(self)

    def __sub__(self, other: Sym | Num) -> Sym:
        return self + (Sym.of(other) * -1)
    def __rsub__(self, other: Sym | Num) -> Sym:
        return Sym.of(other) - self

    def __truediv__(self, other: Sym | Num) -> Sym:
        o = other.e if isinstance(other, Sym) else other # type: ignore
        return Sym(self.e / sp.sympify(o)) # type: ignore
    def __rtruediv__(self, other: Sym | Num) -> Sym:
        return Sym.of(other).__truediv__(self)

    def __pow__(self, p: Sym | Num) -> Sym:
        o = p.e if isinstance(p, Sym) else p # type: ignore
        return Sym(self.e ** sp.sympify(o)) # type: ignore
