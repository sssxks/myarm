"""
Inverse kinematics (IK) utilities for the ZJU‑I 6R arm.

Design goals
------------
- Reuse the FK/DH setup from solver.py (same DH + theta offsets)
- Pure, typed functions; no I/O here (keep I/O in CLI/verify modules)
- Provide a robust numeric IK (damped least squares) with small, modular pieces

Notes
-----
The arm uses standard DH with a4=a5=a6=0 (spherical wrist). However d4 and d5
are non‑zero, which makes a closed‑form decoupling less straightforward for a
short implementation. This module implements a reliable geometric‑Jacobian‑based
IK that converges quickly and returns one or more solutions from multiple seeds.
It’s structured so that we can later swap in a closed‑form branch when desired.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from functools import lru_cache
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from .dh_params import demo_standard_6R_num

if TYPE_CHECKING:
    import sympy as sp

# currently, keep it as static var to simplify code
dh_num = demo_standard_6R_num()


@lru_cache(maxsize=1)
def _symbolic_fk() -> tuple["sp.Matrix", list["sp.Symbol"]]:
    from .dh_params import demo_standard_6R

    T_sym, th_syms, _ = demo_standard_6R()
    return T_sym, th_syms

# ---- Small numeric helpers ----
def _rotz(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    M = np.eye(4)
    M[0, 0], M[0, 1] = c, -s
    M[1, 0], M[1, 1] = s, c
    return M


def _rotx(alpha: float) -> np.ndarray:
    c, s = math.cos(alpha), math.sin(alpha)
    M = np.eye(4)
    M[1, 1], M[1, 2] = c, -s
    M[2, 1], M[2, 2] = s, c
    return M


def _tz(d: float) -> np.ndarray:
    M = np.eye(4)
    M[2, 3] = d
    return M


def _tx(a: float) -> np.ndarray:
    M = np.eye(4)
    M[0, 3] = a
    return M


def rotation_xy_dash_z(alpha: float, beta: float, gamma: float) -> np.ndarray:
    """Return rotation matrix for intrinsic XY'Z' (≡ extrinsic Z-Y-X) angles."""
    ca, sa = math.cos(alpha), math.sin(alpha)
    cb, sb = math.cos(beta), math.sin(beta)
    cg, sg = math.cos(gamma), math.sin(gamma)

    Rx = np.array([[1.0, 0.0, 0.0], [0.0, ca, -sa], [0.0, sa, ca]], dtype=float)
    Ry = np.array([[cb, 0.0, sb], [0.0, 1.0, 0.0], [-sb, 0.0, cb]], dtype=float)
    Rz = np.array([[cg, -sg, 0.0], [sg, cg, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    return Rx @ Ry @ Rz


def pose_from_xyz_euler(
    x_mm: float, y_mm: float, z_mm: float, alpha: float, beta: float, gamma: float
) -> np.ndarray:
    """Build a homogeneous transform from XYZ translation (mm) and XY'Z' angles (rad)."""
    T = np.eye(4, dtype=float)
    T[:3, 3] = np.array([x_mm, y_mm, z_mm], dtype=float)
    T[:3, :3] = rotation_xy_dash_z(alpha, beta, gamma)
    return T

def _wrap_to_pi(v: float) -> float:
    """Wrap an angle to [-pi, pi]."""
    x = (v + math.pi) % (2 * math.pi)
    if x < 0:
        x += 2 * math.pi
    return x - math.pi


# ---- Forward kinematics (fast numeric, consistent with solver.py) ----
def dh_T(a: float, alpha: float, d: float, theta: float) -> np.ndarray:
    """Standard DH: Rz(theta) Tz(d) Tx(a) Rx(alpha)."""
    return _rotz(theta) @ _tz(d) @ _tx(a) @ _rotx(alpha)


def fk_numeric(q: Sequence[float] | np.ndarray) -> np.ndarray:
    """Compute T06 (4x4) numerically from joint angles q (rad).

    This uses the exact same DH lists and theta offsets as demo_standard_6R().
    """
    if len(q) != 6:
        raise ValueError("q must have length 6")
    th = np.asarray(q, dtype=float) + dh_num.theta_offset
    T = np.eye(4)
    for i in range(6):
        T = T @ dh_T(float(dh_num.a[i]), float(dh_num.alpha[i]), float(dh_num.d[i]), float(th[i]))
    return T


def transforms_0_to_i(q: Sequence[float] | np.ndarray) -> list[np.ndarray]:
    """Return [T00, T01, …, T06] for current q."""
    th = np.asarray(q, dtype=float) + dh_num.theta_offset
    Ts: list[np.ndarray] = [np.eye(4)]
    T = np.eye(4)
    for i in range(6):
        T = T @ dh_T(float(dh_num.a[i]), float(dh_num.alpha[i]), float(dh_num.d[i]), float(th[i]))
        Ts.append(T)
    return Ts


def geometric_jacobian(q: Sequence[float] | np.ndarray) -> np.ndarray:
    """6x6 geometric Jacobian at q using base frame coordinates.

    J = [ Jv; Jw ], where for revolute joints i (i=1..6):
        z = z_{i-1}
        o = o_{i-1}
        Jv_i = z × (o_6 - o)
        Jw_i = z
    """
    Ts = transforms_0_to_i(q)
    o_n = Ts[6][:3, 3]
    J = np.zeros((6, 6), dtype=float)
    for i in range(6):
        Ti = Ts[i]
        z = Ti[:3, 2]
        o = Ti[:3, 3]
        J[:3, i] = np.cross(z, o_n - o)
        J[3:, i] = z
    return J


def rotation_error_vee(R_cur: np.ndarray, R_des: np.ndarray) -> np.ndarray:
    """Return so(3) error vector e such that small e ≈ minimal rotation from R_cur→R_des.

    Uses the rotation logarithm to stay well-behaved near 180° differences.
    """
    R_err = R_des.T @ R_cur
    angle = rotation_angle(R_cur, R_des)
    if angle < 1e-9:
        return np.zeros(3, dtype=float)

    skew = R_err - R_err.T
    if abs(math.pi - angle) < 1e-3:
        eigvals, eigvecs = np.linalg.eig(R_err)
        idx = int(np.argmin(np.abs(eigvals - 1.0)))
        axis = np.real(eigvecs[:, idx])
        norm = float(np.linalg.norm(axis))
        if norm < 1e-9:
            axis = np.array([1.0, 0.0, 0.0], dtype=float)
        else:
            axis = axis / norm
        return axis * angle

    factor = angle / (2.0 * math.sin(angle))
    return factor * np.array([skew[2, 1], skew[0, 2], skew[1, 0]], dtype=float)


def rotation_angle(RA: np.ndarray, RB: np.ndarray) -> float:
    """Angle between two rotations in radians."""
    R = RA.T @ RB
    tr = float(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0))
    return float(math.acos(tr))


class IKOptions(NamedTuple):
    max_iter: int = 200
    lambda_dls: float = 1e-3  # damping factor for DLS
    w_pos: float = 1.0        # weight (mm)
    w_rot: float = 200.0      # weight (rad) → scale radians to ~mm
    tol_pos: float = 1e-3     # mm
    tol_rot: float = 1e-3     # rad
    step_clip: float = 0.5    # max |Δq| per iter (rad)


def _ik_once_dls(
    T_des: np.ndarray, q0: Sequence[float] | np.ndarray, opts: IKOptions
) -> tuple[np.ndarray, float, float, int]:
    """Run one DLS solve from seed q0. Returns (q, pos_err, rot_err, iters)."""
    q = np.asarray(q0, dtype=float).copy()
    q = np.array([_wrap_to_pi(float(v)) for v in q], dtype=float)

    for it in range(1, opts.max_iter + 1):
        T = fk_numeric(q)
        p_cur = T[:3, 3]
        R_cur = T[:3, :3]
        p_des = T_des[:3, 3]
        R_des = T_des[:3, :3]

        e_pos = p_des - p_cur
        e_rot = rotation_error_vee(R_cur, R_des)
        pos_err = float(np.linalg.norm(e_pos))
        rot_err = rotation_angle(R_cur, R_des)
        if pos_err <= opts.tol_pos and rot_err <= opts.tol_rot:
            return q, pos_err, rot_err, it

        J = geometric_jacobian(q)
        # Weighted task: W * e where W=diag(w_pos,w_pos,w_pos,w_rot,w_rot,w_rot)
        W = np.diag([opts.w_pos] * 3 + [opts.w_rot] * 3)
        e = np.hstack((e_pos, e_rot))
        JW = W @ J
        eW = W @ e

        # Damped least squares: Δq = (JW^T JW + λ^2 I)^-1 JW^T eW
        JT = JW.T
        H = JT @ JW
        H += (opts.lambda_dls ** 2) * np.eye(6)
        g = JT @ eW
        try:
            dq = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            dq = np.linalg.lstsq(H, g, rcond=None)[0]

        # Clip step to avoid large jumps
        maxabs = float(np.max(np.abs(dq)))
        if maxabs > opts.step_clip:
            dq = dq * (opts.step_clip / maxabs)

        q = q + dq
        q = np.array([_wrap_to_pi(float(v)) for v in q], dtype=float)

    # Return last iterate if not converged
    T = fk_numeric(q)
    pos_err = float(np.linalg.norm(T[:3, 3] - T_des[:3, 3]))
    rot_err = rotation_angle(T[:3, :3], T_des[:3, :3])
    return q, pos_err, rot_err, opts.max_iter


def _seed_from_pose(T_des: np.ndarray) -> np.ndarray:
    """Crude geometric seed: face base toward target XY, shoulder/elbow neutral."""
    px, py, pz = float(T_des[0, 3]), float(T_des[1, 3]), float(T_des[2, 3])
    q = np.zeros(6, dtype=float)
    q[0] = math.atan2(py, px)
    # Aim shoulder toward target height
    q[1] = -math.pi / 4 if pz < 0.0 else math.pi / 4
    q[2] = 0.0
    # Point wrist roughly to target orientation using Rz component
    q[5] = math.atan2(float(T_des[1, 0]), float(T_des[0, 0]))
    return q


def unique_solutions(solutions: Iterable[np.ndarray], tol: float = 1e-3) -> list[np.ndarray]:
    """Merge nearly identical angle sets (L_inf metric)."""
    uniq: list[np.ndarray] = []
    for q in solutions:
        keep = True
        for u in uniq:
            if float(np.max(np.abs(q - u))) <= tol:
                keep = False
                break
        if keep:
            uniq.append(q)
    return uniq


def solve_ik(
    T_des: np.ndarray,
    seeds: Sequence[Sequence[float]] | None = None,
    opts: IKOptions | None = None,
) -> list[tuple[np.ndarray, float, float, int]]:
    """Try multiple seeds; return converged solutions with metrics.

    Returns list of tuples (q, pos_err_mm, rot_err_rad, iters). The list is
    deduplicated by angle proximity and sorted by increasing errors.
    """
    if opts is None:
        opts = IKOptions()
    if T_des.shape != (4, 4):
        raise ValueError("T_des must be 4x4")

    # Default seed set: crude pose‑based + a few canonical postures
    seed_list: list[np.ndarray] = []
    if seeds:
        for seed in seeds:
            if len(seed) != 6:
                raise ValueError("seed must have length 6")
            seed_arr = np.asarray(seed, dtype=float)
            seed_list.append(seed_arr)
    else:
        seed_list.append(_seed_from_pose(T_des))
        seed_list.append(np.zeros(6, dtype=float))
        seed_list.append(np.array([math.pi / 2, 0, 0, 0, 0, 0], dtype=float))
        seed_list.append(np.array([-math.pi / 2, 0, 0, 0, 0, 0], dtype=float))
        seed_list.append(np.array([0, math.pi / 3, 0, 0, 0, 0], dtype=float))
        seed_list.append(np.array([0, -math.pi / 3, 0, 0, 0, 0], dtype=float))

    results: list[tuple[np.ndarray, float, float, int]] = []
    tried: list[np.ndarray] = []
    for s_arr in seed_list:
        # Avoid re‑trying very similar seeds
        if tried and any(float(np.max(np.abs(s_arr - t))) < 1e-3 for t in tried):
            continue
        tried.append(s_arr)
        q, pe, re, it = _ik_once_dls(T_des, s_arr, opts)
        results.append((q, pe, re, it))

    # Keep converged (within tolerance) first; still return best few otherwise
    converged = [r for r in results if r[1] <= opts.tol_pos and r[2] <= opts.tol_rot]
    if not converged:
        # Sort all by a lexicographic key (pos first then rot)
        results.sort(key=lambda r: (r[1], r[2]))
        # Deduplicate and return top 1–3 even if not within tol
        uni_q = unique_solutions([r[0] for r in results])
        best = []
        for u in uni_q[:3]:
            for r in results:
                if np.allclose(r[0], u, atol=1e-3, rtol=0.0):
                    best.append(r)
                    break
        return best

    # Deduplicate converged solutions and sort by errors
    uni_q = unique_solutions([r[0] for r in converged])
    out: list[tuple[np.ndarray, float, float, int]] = []
    for u in uni_q:
        for r in converged:
            if np.allclose(r[0], u, atol=1e-3, rtol=0.0):
                out.append(r)
                break
    out.sort(key=lambda r: (r[1], r[2], r[3]))
    return out


def solve_ik_nsolve(T_des: np.ndarray, guess: Sequence[float]) -> np.ndarray:
    """Refine an IK guess using SymPy's nsolve."""
    import sympy as sp

    from .solver import T_to_euler_xy_dash_z

    T_sym, th_syms = _symbolic_fk()
    T_target = sp.Matrix(T_des.tolist())
    alpha_sym, beta_sym, gamma_sym = T_to_euler_xy_dash_z(T_sym, safe=False)
    alpha_tar, beta_tar, gamma_tar = T_to_euler_xy_dash_z(T_target, safe=False)
    eqs = [
        T_sym[0, 3] - T_target[0, 3],
        T_sym[1, 3] - T_target[1, 3],
        T_sym[2, 3] - T_target[2, 3],
        alpha_sym - alpha_tar,
        beta_sym - beta_tar,
        gamma_sym - gamma_tar,
    ]

    guess_vec = [float(v) for v in guess]
    sol = sp.nsolve(eqs, th_syms, guess_vec, tol=1e-12, maxsteps=200)
    values = [float(_wrap_to_pi(float(s))) for s in sol]
    return np.array(values, dtype=float)


__all__ = [
    "IKOptions",
    "fk_numeric",
    "geometric_jacobian",
    "pose_from_xyz_euler",
    "rotation_xy_dash_z",
    "rotation_angle",
    "solve_ik",
    "solve_ik_nsolve",
]
