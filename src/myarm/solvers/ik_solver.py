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
from functools import partial
from typing import Callable, NamedTuple, cast

import numpy as np
from numpy.typing import NDArray

from myarm.core.orientation import rotation_xy_dash_z_numeric
from myarm.model.dh_params import DHParamsNum, demo_standard_6R_num
from myarm.model.jacobian import (
    forward_chain_numeric,
    geometric_jacobian_numeric,
)


Matrix44 = NDArray[np.float64]
Matrix33 = NDArray[np.float64]
Vector3 = NDArray[np.float64]


def pose_from_xyz_euler(
    x_mm: float, y_mm: float, z_mm: float, alpha: float, beta: float, gamma: float
) -> Matrix44:
    """Build a homogeneous transform from XYZ translation (mm) and XY'Z' angles (rad)."""
    T = np.eye(4, dtype=float)
    T[:3, 3] = np.array([x_mm, y_mm, z_mm], dtype=float)
    T[:3, :3] = rotation_xy_dash_z_numeric(alpha, beta, gamma)
    return T

def _wrap_to_pi(v: float) -> float:
    """Wrap an angle to [-pi, pi]."""
    x = (v + math.pi) % (2 * math.pi)
    if x < 0:
        x += 2 * math.pi
    return x - math.pi


def _angle_delta(
    a: Sequence[float] | np.ndarray, b: Sequence[float] | np.ndarray
) -> np.ndarray:
    """Return elementwise wrapped angle difference a-b (rad)."""
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    wrapped = np.array([_wrap_to_pi(v) for v in diff], dtype=float)
    return wrapped


# ---- Forward kinematics ----
def fk_numeric(q: Sequence[float] | np.ndarray, dh_params: DHParamsNum) -> Matrix44:
    """Compute T06 (4x4) numerically from joint angles q (rad).

    This uses the exact same DH lists and theta offsets as demo_standard_6R().
    """
    if len(q) != 6:
        raise ValueError("q must have length 6")
    chain = forward_chain_numeric(dh_params, q)
    return chain[-1]


def transforms_0_to_i(q: Sequence[float] | np.ndarray, dh_params: DHParamsNum) -> list[Matrix44]:
    """Return [T00, T01, ., T06] for current q."""
    chain = forward_chain_numeric(dh_params, q)
    return chain


def geometric_jacobian(q: Sequence[float] | np.ndarray, dh_params: DHParamsNum) -> NDArray[np.float64]:
    """6x6 geometric Jacobian at q using base frame coordinates.

    J = [ Jv; Jw ], where for revolute joints i (i=1..6):
        z = z_{i-1}
        o = o_{i-1}
        Jv_i = z × (o_6 - o)
        Jw_i = z
    """
    return geometric_jacobian_numeric(q, dh=dh_params)


def rotation_error_vee(R_cur: Matrix33, R_des: Matrix33) -> Vector3:
    """Return so(3) error vector e such that small e ≈ minimal rotation from R_cur→R_des.

    Uses the rotation logarithm to stay well-behaved near 180° differences.
    """
    R_err = R_des @ R_cur.T
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


def rotation_angle(RA: Matrix33, RB: Matrix33) -> float:
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


def _ik_dls_step(
    q: np.ndarray,
    it: int,
    T_des: np.ndarray,
    opts: IKOptions,
    _fk: Callable[[np.ndarray], Matrix44],
    _jacobian: Callable[[np.ndarray], NDArray[np.float64]],
) -> tuple[np.ndarray, float, float, int]:
    """Recursive step for the DLS IK solver."""
    T = _fk(q)
    p_cur, R_cur = T[:3, 3], T[:3, :3]
    p_des, R_des = T_des[:3, 3], T_des[:3, :3]

    e_pos = p_des - p_cur
    e_rot = rotation_error_vee(R_cur, R_des)
    pos_err, rot_err = float(np.linalg.norm(e_pos)), rotation_angle(R_cur, R_des)

    if pos_err <= opts.tol_pos and rot_err <= opts.tol_rot or it >= opts.max_iter:
        return q, pos_err, rot_err, it

    J = _jacobian(q)
    W = np.diag([opts.w_pos] * 3 + [opts.w_rot] * 3)
    e = np.hstack((e_pos, e_rot))
    JW = W @ J
    eW = W @ e

    JT = JW.T
    H = JT @ JW + (opts.lambda_dls ** 2) * np.eye(6)
    g = JT @ eW
    try:
        dq = np.linalg.solve(H, g)
    except np.linalg.LinAlgError:
        dq = np.linalg.lstsq(H, g, rcond=None)[0]

    maxabs = float(np.max(np.abs(dq)))
    if maxabs > opts.step_clip:
        dq = dq * (opts.step_clip / maxabs)

    q_next = np.array([_wrap_to_pi(v) for v in (q + dq)], dtype=float)
    return _ik_dls_step(q_next, it + 1, T_des, opts, _fk, _jacobian)


def _ik_once_dls(
    T_des: np.ndarray, q0: Sequence[float] | np.ndarray, opts: IKOptions, dh_params: DHParamsNum
) -> tuple[np.ndarray, float, float, int]:
    """Run one DLS solve from seed q0. Returns (q, pos_err, rot_err, iters)."""
    q_initial = np.array([_wrap_to_pi(float(v)) for v in q0], dtype=float)

    _fk_numeric = partial(fk_numeric, dh_params=dh_params)
    _geometric_jacobian = partial(geometric_jacobian, dh_params=dh_params)

    return _ik_dls_step(q_initial, 1, T_des, opts, _fk_numeric, _geometric_jacobian)


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
            delta = _angle_delta(q, u)
            if float(np.max(np.abs(delta))) <= tol:
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

    dh_params = demo_standard_6R_num()

    # Default seed set: crude pose‑based + a few canonical postures
    seed_list: list[np.ndarray] = []
    if seeds:
        for seed in seeds:
            if len(seed) != 6:
                raise ValueError("seed must have length 6")
            seed_arr = np.asarray(seed, dtype=float)
            seed_list.append(seed_arr)
    else:
        seed_list = [
            _seed_from_pose(T_des),
            np.zeros(6, dtype=float),
            np.array([math.pi / 2, 0, 0, 0, 0, 0], dtype=float),
            np.array([-math.pi / 2, 0, 0, 0, 0, 0], dtype=float),
            np.array([0, math.pi / 3, 0, 0, 0, 0], dtype=float),
            np.array([0, -math.pi / 3, 0, 0, 0, 0], dtype=float),
        ]

    tried: list[np.ndarray] = []
    unique_seeds = []
    for s_arr in seed_list:
        if not tried or not any(float(np.max(np.abs(_angle_delta(s_arr, t)))) < 1e-3 for t in tried):
            unique_seeds.append(s_arr)
            tried.append(s_arr)

    results = [_ik_once_dls(T_des, s_arr, opts, dh_params) for s_arr in unique_seeds]

    converged = [r for r in results if r[1] <= opts.tol_pos and r[2] <= opts.tol_rot]

    if not converged:
        results.sort(key=lambda r: (r[1], r[2]))
        uni_q = unique_solutions([r[0] for r in results])

        best = []
        for u in uni_q[:3]:
            for r in results:
                if np.allclose(r[0], u, atol=1e-3, rtol=0.0):
                    best.append(r)
                    break
        return best

    uni_q = unique_solutions([r[0] for r in converged])

    out = []
    for u in uni_q:
        for r in converged:
            if np.allclose(r[0], u, atol=1e-3, rtol=0.0):
                out.append(r)
                break

    out.sort(key=lambda r: (r[1], r[2], r[3]))
    return out


__all__ = [
    "IKOptions",
    "fk_numeric",
    "geometric_jacobian",
    "pose_from_xyz_euler",
    "rotation_xy_dash_z_numeric",
    "rotation_angle",
    "solve_ik",
]
