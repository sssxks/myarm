from __future__ import annotations
from typing import Any
import sympy as sp
from .sym import Sym

def symbols(names: str) -> tuple[Sym, ...]:
    syms = sp.symbols(names) # type: ignore
    if isinstance(syms, tuple):
        return tuple(Sym(s) for s in syms) # type: ignore
    return (Sym(syms),)

def simplify(x: Sym) -> Sym:
    return Sym(sp.simplify(x.e)) # type: ignore

def sin(x: Sym) -> Sym: return Sym(sp.sin(x.e)) # type: ignore
def cos(x: Sym) -> Sym: return Sym(sp.cos(x.e)) # type: ignore
def exp(x: Sym) -> Sym: return Sym(sp.exp(x.e)) # type: ignore
def diff(x: Sym, *syms: Sym, **kw: Any) -> Sym:
    return Sym(sp.diff(x.e, *[s.e for s in syms], **kw)) # type: ignore

def integrate(x: Sym, *args: Any, **kw: Any) -> Sym:
    # 区间积分等参数透传
    conv = []
    for a in args:
        conv.append(a.e if isinstance(a, Sym) else a) # type: ignore
    return Sym(sp.integrate(x.e, *conv, **kw)) # type: ignore

# @overload
# def lambdify(args: Sym, expr: Sym, *, modules: Any = ...) -> Callable[[float], float]: ...
# @overload
# def lambdify(args: Sequence[Sym], expr: Sym, *, modules: Any = ...) -> Callable[..., float]: ...
# def lambdify(args, expr, *, modules="math"):
#     def unwrap(a): return a.e if isinstance(a, Sym) else a
#     a_unwrapped = [unwrap(a) for a in (args if isinstance(args, (list, tuple)) else [args])]
#     f = sp.lambdify(a_unwrapped, expr.e, modules=modules)
#     return f
