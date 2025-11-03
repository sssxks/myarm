import sympy as sp
from typing import cast

type Num = int | float | sp.Expr

# Avoid direct division on sp.pi to keep mypy happy with SymPy stubs.
# Use SymPy constructors and cast the result.
HALF_PI: sp.Expr = cast(sp.Expr, sp.Mul(sp.pi, sp.Rational(1, 2)))
