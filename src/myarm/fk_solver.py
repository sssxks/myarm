"""
Symbolic Forward Kinematics from DH parameters (Standard + Modified)

Angles in radians.
Functions:
- fk_standard(a, alpha, d, theta)
- fk_modified(a_prev, alpha_prev, d, theta)
- demo_standard_6R()  # returns a symbolic T06 and joint symbols for the ZJU‑I arm
- rot_to_euler_xy_dash_z(R)           # intrinsic XY'Z' (≡ extrinsic Z‑Y‑X)
- T_to_euler_xy_dash_z(T, safe=True)  # convenience wrapper for a 4x4 T
"""

import math
from collections.abc import Iterable, Sequence
from typing import Any, cast
import sympy as sp

from .type_utils import Num

def Rx(alpha: Num) -> sp.Matrix:
    ca = cast(sp.Expr, sp.cos(alpha))
    sa = cast(sp.Expr, sp.sin(alpha))
    return sp.Matrix(
        [[1, 0, 0, 0], [0, ca, -sa, 0], [0, sa, ca, 0], [0, 0, 0, 1]]
    )


def Ry(beta: Num) -> sp.Matrix:
    cb = cast(sp.Expr, sp.cos(beta))
    sb = cast(sp.Expr, sp.sin(beta))
    return sp.Matrix([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])


def Rz(theta: Num) -> sp.Matrix:
    ct = cast(sp.Expr, sp.cos(theta))
    st = cast(sp.Expr, sp.sin(theta))
    return sp.Matrix(
        [[ct, -st, 0, 0], [st, ct, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    )


def Tx(a: Num) -> sp.Matrix:
    return sp.Matrix([[1, 0, 0, a], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])


def Tz(d: Num) -> sp.Matrix:
    return sp.Matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, d], [0, 0, 0, 1]])


def simplify_T(T: sp.Matrix) -> sp.Matrix:
    # SymPy is untyped; cast result to Matrix for mypy.
    return cast(sp.Matrix, sp.simplify(T))  # type: ignore[no-untyped-call]


def fk_standard(
    a: Sequence[Num],
    alpha: Sequence[Num],
    d: Sequence[Num],
    theta: Sequence[Num],
    simplify: bool = True,
) -> sp.Matrix:
    assert len(a) == len(alpha) == len(d) == len(theta)
    T = cast(sp.Matrix, sp.eye(4))  # type: ignore[no-untyped-call]
    for ai, al, di, th in zip(a, alpha, d, theta):
        Ti = Rz(th) * Tz(di) * Tx(ai) * Rx(al)
        T = T * Ti
    return simplify_T(T) if simplify else T


def rot_to_euler_xy_dash_z(R: sp.Matrix) -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    """Return intrinsic XY'Z' Euler angles (alpha, beta, gamma) from a 3x3 rotation.

    Convention:
    - Intrinsic sequence X → Y' → Z' (rotate about body axes)
    - Equivalent to extrinsic Z → Y → X applied to fixed axes.
    - We reconstruct as R = Rx(alpha) * Ry(beta) * Rz(gamma).

    Correct element-to-angle mapping for the above product (non-singular |cos(beta)|>0):
        beta  = asin( R[0,2])         # r13 = sin(beta)
        alpha = atan2(-R[1,2], R[2,2])# (-r23, r33) → alpha
        gamma = atan2(-R[0,1], R[0,0])# (-r12, r11) → gamma

    Notes:
    - Returns sympy Expr objects; for numeric inputs, prefer the safe wrapper
      `T_to_euler_xy_dash_z(..., safe=True)` which adds gimbal-lock handling.
    """
    if R.shape != (3, 3):
        raise ValueError("R must be 3x3 rotation submatrix")

    r11 = cast(sp.Expr, R[0, 0])
    r12 = cast(sp.Expr, R[0, 1])
    r13 = cast(sp.Expr, R[0, 2])
    r23 = cast(sp.Expr, R[1, 2])
    r33 = cast(sp.Expr, R[2, 2])

    beta = cast(sp.Expr, sp.asin(r13))
    alpha = cast(sp.Expr, sp.atan2(-r23, r33))
    gamma = cast(sp.Expr, sp.atan2(-r12, r11))
    return alpha, beta, gamma


def _rot_to_euler_xy_dash_z_safe(
    R: sp.Matrix, eps: float = 1e-9
) -> tuple[float, float, float]:
    """Numeric-safe XY'Z' extraction with gimbal-lock handling for XY'Z'.

    Uses the same mapping as `rot_to_euler_xy_dash_z`, and handles the
    singular case |cos(beta)| ≈ 0 (i.e., |r13| ≈ 1) by setting gamma = 0 and
    folding the yaw into alpha via atan2 on (r21, r22).
    """
    r11, r12, r13 = [
        float(sp.N(R[0, 0])),  # type: ignore[no-untyped-call]
        float(sp.N(R[0, 1])),  # type: ignore[no-untyped-call]
        float(sp.N(R[0, 2])),  # type: ignore[no-untyped-call]
    ]
    r21, r22, r23 = [
        float(sp.N(R[1, 0])),  # type: ignore[no-untyped-call]
        float(sp.N(R[1, 1])),  # type: ignore[no-untyped-call]
        float(sp.N(R[1, 2])),  # type: ignore[no-untyped-call]
    ]
    r33 = float(sp.N(R[2, 2]))  # type: ignore[no-untyped-call]

    # Detect gimbal lock at beta = ±pi/2, where cos(beta) ≈ 0 and |r13| = |sin(beta)| ≈ 1
    if abs(abs(r13) - 1.0) < eps:
        beta = math.copysign(math.pi / 2, r13)
        gamma = 0.0
        # For beta = +pi/2: alpha = atan2(r21, r22)
        # For beta = -pi/2: alpha = atan2(-r21, r22)
        if r13 > 0:
            alpha = math.atan2(r21, r22)
        else:
            alpha = math.atan2(-r21, r22)
        return float(alpha), float(beta), float(gamma)

    beta = float(math.asin(r13))
    alpha = float(math.atan2(-r23, r33))
    gamma = float(math.atan2(-r12, r11))
    return alpha, beta, gamma


def T_to_euler_xy_dash_z(
    T: sp.Matrix, safe: bool = True
) -> tuple[sp.Expr, sp.Expr, sp.Expr] | tuple[float, float, float]:
    """Extract intrinsic XY'Z' Euler angles from a 4x4 homogeneous transform.

    Parameters
    ----------
    T : 4x4 sympy Matrix
        Homogeneous transformation matrix.
    safe : bool
        If True and the matrix is numeric, use a gimbal-lock-aware path.

    Returns
    -------
    (alpha, beta, gamma) : tuple of sympy Expr or floats
        Intrinsic XY'Z' angles in radians.
    """
    if T.shape != (4, 4):
        raise ValueError("T must be 4x4 homogeneous transform")
    R = sp.Matrix(
        [
            [T[0, 0], T[0, 1], T[0, 2]],
            [T[1, 0], T[1, 1], T[1, 2]],
            [T[2, 0], T[2, 1], T[2, 2]],
        ]
    )

    # Decide numeric vs symbolic path
    is_numeric = all(getattr(e, "is_number", False) for e in cast(Iterable[Any], R))
    if safe and is_numeric:
        return _rot_to_euler_xy_dash_z_safe(R)
    return rot_to_euler_xy_dash_z(R)


def fk_modified(
    a_prev: Sequence[Num],
    alpha_prev: Sequence[Num],
    d: Sequence[Num],
    theta: Sequence[Num],
    simplify: bool = True,
) -> sp.Matrix:
    assert len(a_prev) == len(alpha_prev) == len(d) == len(theta)
    T = cast(sp.Matrix, sp.eye(4))  # type: ignore[no-untyped-call]
    for ap, alp, di, th in zip(a_prev, alpha_prev, d, theta):
        Ti = Rx(alp) * Tx(ap) * Rz(th) * Tz(di)
        T = T * Ti
    return simplify_T(T) if simplify else T


