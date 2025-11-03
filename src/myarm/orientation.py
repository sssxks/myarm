from __future__ import annotations

import math

import numpy as np
import sympy as sp
from numpy.typing import NDArray

from .type_utils import Num

type Matrix33 = NDArray[np.float64]


def rotation_xy_dash_z_numeric(alpha: float, beta: float, gamma: float) -> Matrix33:
    """Return intrinsic XY'Z' rotation matrix with NumPy backend."""
    cos, sin = math.cos, math.sin
    ca, sa = cos(alpha), sin(alpha)
    cb, sb = cos(beta), sin(beta)
    cg, sg = cos(gamma), sin(gamma)
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
    cos, sin = sp.cos, sp.sin
    ca, sa = cos(alpha), sin(alpha)
    cb, sb = cos(beta), sin(beta)
    cg, sg = cos(gamma), sin(gamma)

    return sp.Matrix(
        [
            [cb * cg, -cb * sg, sb],  # type:ignore
            [sa * sb * cg + ca * sg, ca * cg - sa * sb * sg, -sa * cb],  # type:ignore
            [-ca * sb * cg + sa * sg, sa * cg + ca * sb * sg, ca * cb],  # type:ignore
        ],
    )
