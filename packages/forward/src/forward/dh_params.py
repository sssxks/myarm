from .type_utils import HALF_PI, Num
from .solver import fk_standard
import sympy as sp
from typing import cast

def demo_standard_6R() -> tuple[sp.Matrix, list[sp.Symbol], dict[str, list[Num]]]:
    """Convenience constructor for the ZJU‑I 6‑DoF arm used in this project.

    Returns
    -------
    T06 : sympy.Matrix (4x4)
        Homogeneous transform from base to tool.
    theta_syms : list[sympy.Symbol]
        [th1..th6] joint symbols.
    params : dict
        Dict with keys ``a``, ``alpha``, ``d``, ``theta`` (the DH lists).
    """
    th1, th2, th3, th4, th5, th6 = cast(
        tuple[sp.Symbol, sp.Symbol, sp.Symbol, sp.Symbol, sp.Symbol, sp.Symbol],
        sp.symbols("th1 th2 th3 th4 th5 th6", real=True),
    )
    a: list[Num] = [0, 185, 170, 0, 0, 0]
    alpha: list[Num] = [-HALF_PI, 0, 0, HALF_PI, HALF_PI, 0]
    d: list[Num] = [230, -54, 0, 77, 77, 85.5]
    theta: list[Num] = [
        cast(sp.Expr, th1),
        cast(sp.Expr, sp.Add(cast(sp.Expr, th2), sp.Mul(-1, HALF_PI))),
        cast(sp.Expr, th3),
        cast(sp.Expr, sp.Add(cast(sp.Expr, th4), HALF_PI)),
        cast(sp.Expr, sp.Add(cast(sp.Expr, th5), HALF_PI)),
        cast(sp.Expr, th6),
    ]
    T06 = fk_standard(a, alpha, d, theta)
    return (
        T06,
        [th1, th2, th3, th4, th5, th6],
        {"a": a, "alpha": alpha, "d": d, "theta": theta},
    )
