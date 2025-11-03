from __future__ import annotations

import math
from typing import cast

import numpy as np
import sympy as sp
from numpy.typing import NDArray

from .type_utils import Num

Matrix33 = NDArray[np.float64]


def rotation_xy_dash_z_numeric(alpha: float, beta: float, gamma: float) -> Matrix33:
    """Return intrinsic XY'Z' rotation matrix with NumPy backend."""
    a = float(alpha)
    b = float(beta)
    g = float(gamma)
    ca, sa = math.cos(a), math.sin(a)
    cb, sb = math.cos(b), math.sin(b)
    cg, sg = math.cos(g), math.sin(g)
    return np.array(
        [
            [cb * cg, -cb * sg, sb],
            [sa * sb * cg + ca * sg, ca * cg - sa * sb * sg, -sa * cb],
            [-ca * sb * cg + sa * sg, sa * cg + ca * sb * sg, ca * cb],
        ],
        dtype=float,
    )


def rotation_xy_dash_z_symbolic(alpha: Num, beta: Num, gamma: Num) -> sp.Matrix:
    """Return intrinsic XY'Z' rotation matrix with SymPy backend."""
    ca = cast(sp.Expr, sp.cos(alpha))
    sa = cast(sp.Expr, sp.sin(alpha))
    cb = cast(sp.Expr, sp.cos(beta))
    sb = cast(sp.Expr, sp.sin(beta))
    cg = cast(sp.Expr, sp.cos(gamma))
    sg = cast(sp.Expr, sp.sin(gamma))
    # Use SymPy's multiplication
    return sp.Matrix(
        [
            [cb.mul(cg), -cb.mul(sg), sb],
            [sa.mul(sb).mul(cg) + ca.mul(sg), ca.mul(cg) - sa.mul(sb).mul(sg), -sa.mul(cb)],
            [-ca.mul(sb).mul(cg) + sa.mul(sg), sa.mul(cg) + ca.mul(sb).mul(sg), ca.mul(cb)],
        ]
    )

