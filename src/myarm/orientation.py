from __future__ import annotations

import math
from typing import Literal, overload, cast

import numpy as np
import sympy as sp
from numpy.typing import NDArray

from .type_utils import Num

Matrix33 = NDArray[np.float64]


@overload
def rotation_xy_dash_z(alpha: float, beta: float, gamma: float, *, numeric: Literal[True]) -> Matrix33: ...


@overload
def rotation_xy_dash_z(alpha: Num, beta: Num, gamma: Num, *, numeric: Literal[False] = False) -> sp.Matrix: ...


def rotation_xy_dash_z(
    alpha: Num | float,
    beta: Num | float,
    gamma: Num | float,
    *,
    numeric: bool = False,
) -> sp.Matrix | Matrix33:
    """Return intrinsic XY'Z' rotation matrix with either SymPy or NumPy backend."""
    if numeric:
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

    ca = cast(sp.Expr, sp.cos(alpha))
    sa = cast(sp.Expr, sp.sin(alpha))
    cb = cast(sp.Expr, sp.cos(beta))
    sb = cast(sp.Expr, sp.sin(beta))
    cg = cast(sp.Expr, sp.cos(gamma))
    sg = cast(sp.Expr, sp.sin(gamma))
    return sp.Matrix(
        [
            [cb * cg, -cb * sg, sb],
            [sa * sb * cg + ca * sg, ca * cg - sa * sb * sg, -sa * cb],
            [-ca * sb * cg + sa * sg, sa * cg + ca * sb * sg, ca * cb],
        ]
    )


def rotation_xy_dash_z_numeric(alpha: float, beta: float, gamma: float) -> Matrix33:
    """Helper wrapper for numeric rotation matrix."""
    return rotation_xy_dash_z(alpha, beta, gamma, numeric=True)


def rotation_xy_dash_z_symbolic(alpha: Num, beta: Num, gamma: Num) -> sp.Matrix:
    """Helper wrapper for symbolic rotation matrix."""
    return cast(sp.Matrix, rotation_xy_dash_z(alpha, beta, gamma, numeric=False))

