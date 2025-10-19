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
from __future__ import annotations
import math
import sympy as sp

def Rx(alpha):
    ca, sa = sp.cos(alpha), sp.sin(alpha)
    return sp.Matrix([[1, 0, 0, 0],
                      [0, ca, -sa, 0],
                      [0, sa,  ca, 0],
                      [0, 0, 0, 1]])

def Ry(beta):
    cb, sb = sp.cos(beta), sp.sin(beta)
    return sp.Matrix([[ cb, 0,  sb],
                      [  0, 1,   0],
                      [-sb, 0,  cb]])

def Rz(theta):
    ct, st = sp.cos(theta), sp.sin(theta)
    return sp.Matrix([[ ct, -st, 0, 0],
                      [ st,  ct, 0, 0],
                      [  0,   0, 1, 0],
                      [  0,   0, 0, 1]])

def Tx(a):
    return sp.Matrix([[1, 0, 0, a],
                      [0, 1, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])

def Tz(d):
    return sp.Matrix([[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, d],
                      [0, 0, 0, 1]])

def simplify_T(T):
    return sp.simplify(T)

def fk_standard(a, alpha, d, theta, simplify=True):
    assert len(a) == len(alpha) == len(d) == len(theta)
    T = sp.eye(4)
    for ai, al, di, th in zip(a, alpha, d, theta):
        Ti = Rz(th) * Tz(di) * Tx(ai) * Rx(al)
        T = T * Ti
    return simplify_T(T) if simplify else T

def rot_to_euler_xy_dash_z(R: sp.Matrix):
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

    r11, r12, r13 = R[0,0], R[0,1], R[0,2]
    r21, r22, r23 = R[1,0], R[1,1], R[1,2]
    r31, r32, r33 = R[2,0], R[2,1], R[2,2]

    beta = sp.asin(r13)
    alpha = sp.atan2(-r23, r33)
    gamma = sp.atan2(-r12, r11)
    return alpha, beta, gamma

def _rot_to_euler_xy_dash_z_safe(R: sp.Matrix, eps: float = 1e-9):
    """Numeric-safe XY'Z' extraction with gimbal-lock handling for XY'Z'.

    Uses the same mapping as `rot_to_euler_xy_dash_z`, and handles the
    singular case |cos(beta)| ≈ 0 (i.e., |r13| ≈ 1) by setting gamma = 0 and
    folding the yaw into alpha via atan2 on (r21, r22).
    """
    r11, r12, r13 = [float(R[0,0]), float(R[0,1]), float(R[0,2])]
    r21, r22, r23 = [float(R[1,0]), float(R[1,1]), float(R[1,2])]
    r31, r32, r33 = [float(R[2,0]), float(R[2,1]), float(R[2,2])]

    # Detect gimbal lock at beta = ±pi/2, where cos(beta) ≈ 0 and |r13| = |sin(beta)| ≈ 1
    if abs(abs(r13) - 1.0) < eps:
        beta = math.copysign(math.pi/2, r13)
        gamma = 0.0
        # For beta = +pi/2: alpha = atan2(r21, r22)
        # For beta = -pi/2: alpha = atan2(-r21, r22)
        if r13 > 0:
            alpha = float(sp.atan2(r21, r22))
        else:
            alpha = float(sp.atan2(-r21, r22))
        return float(alpha), float(beta), float(gamma)

    beta = float(sp.asin(r13))
    alpha = float(sp.atan2(-r23, r33))
    gamma = float(sp.atan2(-r12, r11))
    return alpha, beta, gamma

def T_to_euler_xy_dash_z(T: sp.Matrix, safe: bool = True):
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
    R = sp.Matrix([[T[0,0], T[0,1], T[0,2]],
                   [T[1,0], T[1,1], T[1,2]],
                   [T[2,0], T[2,1], T[2,2]]])

    # Decide numeric vs symbolic path
    is_numeric = all(getattr(e, 'is_number', False) for e in R)
    if safe and is_numeric:
        return _rot_to_euler_xy_dash_z_safe(R)
    return rot_to_euler_xy_dash_z(R)

def fk_modified(a_prev, alpha_prev, d, theta, simplify=True):
    assert len(a_prev) == len(alpha_prev) == len(d) == len(theta)
    T = sp.eye(4)
    for ap, alp, di, th in zip(a_prev, alpha_prev, d, theta):
        Ti = Rx(alp) * Tx(ap) * Rz(th) * Tz(di)
        T = T * Ti
    return simplify_T(T) if simplify else T

def demo_standard_6R():
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
    th1, th2, th3, th4, th5, th6 = sp.symbols('th1 th2 th3 th4 th5 th6', real=True)
    a     = [0,   185,   170,   0, 0, 0]
    alpha = [-sp.pi/2, 0, 0, sp.pi/2, sp.pi/2, 0]
    d     = [230, -54, 0, 77, 77, 85.5]
    theta = [th1, th2-sp.pi/2, th3, th4+sp.pi/2, th5+sp.pi/2, th6]
    T06 = fk_standard(a, alpha, d, theta)
    return T06, [th1, th2, th3, th4, th5, th6], {"a": a, "alpha": alpha, "d": d, "theta": theta}
