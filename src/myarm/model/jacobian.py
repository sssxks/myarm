"""Jacobian helpers for the ZJU-I 6-DoF robotic arm."""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple, cast

import math
import numpy as np
import sympy as sp
from numpy.typing import NDArray

from .dh_params import DHParams, DHParamsNum, demo_standard_6R, demo_standard_6R_num
from myarm.solvers.fk_solver import Rx, Tx, Tz, Rz


type Matrix6x6 = NDArray[np.float64]


class JacobianSymbolic(NamedTuple):
    """Container for the symbolic ZJU-I geometric Jacobian."""

    J_linear: sp.Matrix
    J_angular: sp.Matrix
    thetas: tuple[sp.Symbol, ...]

    @property
    def geometric(self) -> sp.Matrix:
        """Return the stacked 6x6 Jacobian."""

        return cast(sp.Matrix, sp.Matrix.vstack(self.J_linear, self.J_angular))


def _forward_chain_symbolic(dh: DHParams) -> list[sp.Matrix]:
    """Return base-to-link transforms for each joint (including base frame)."""

    transforms: list[sp.Matrix] = [cast(sp.Matrix, sp.eye(4))]  # type: ignore[no-untyped-call]
    T = transforms[0]
    for ai, alpha_i, di, theta_i in zip(dh.params["a"], dh.params["alpha"], dh.params["d"], dh.params["theta"]):
        Ti = Rz(theta_i) * Tz(di) * Tx(ai) * Rx(alpha_i)
        T = cast(sp.Matrix, T * Ti)
        transforms.append(T)
    return transforms


def _extract_origins_and_axes(chain: Sequence[sp.Matrix]) -> tuple[list[sp.Matrix], list[sp.Matrix]]:
    origins: list[sp.Matrix] = []
    z_axes: list[sp.Matrix] = []
    for T in chain:
        origins.append(sp.Matrix(T[:3, 3]))
        z_axes.append(sp.Matrix(T[:3, 2]))
    return origins, z_axes


def symbolic_geometric_jacobian() -> JacobianSymbolic:
    """Return the symbolic geometric Jacobian for the ZJU-I arm."""

    dh = demo_standard_6R()
    chain = _forward_chain_symbolic(dh)
    origins, z_axes = _extract_origins_and_axes(chain)

    on = origins[-1]
    n_joints = len(dh.thetas)
    Jv_cols: list[sp.Matrix] = []
    Jw_cols: list[sp.Matrix] = []
    for i in range(n_joints):
        zi = z_axes[i]
        oi = origins[i]
        Jv_cols.append(zi.cross(on - oi))
        Jw_cols.append(zi)

    Jv = cast(sp.Matrix, sp.Matrix.hstack(*Jv_cols))
    Jw = cast(sp.Matrix, sp.Matrix.hstack(*Jw_cols))
    return JacobianSymbolic(J_linear=Jv, J_angular=Jw, thetas=tuple(dh.thetas))


def _numeric_transform(a: float, alpha: float, d: float, theta: float) -> NDArray[np.float64]:
    ca, sa = math.cos(alpha), math.sin(alpha)
    ct, st = math.cos(theta), math.sin(theta)
    return np.array(
        [
            [ct, -st * ca, st * sa, ct * a],
            [st, ct * ca, -ct * sa, st * a],
            [0.0, sa, ca, d],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def _forward_chain_numeric(dh: DHParamsNum, joint_angles: Sequence[float]) -> list[NDArray[np.float64]]:
    theta = np.asarray(joint_angles, dtype=float)
    if theta.shape != (dh.theta_offset.shape[0],):
        msg = f"Expected {dh.theta_offset.shape[0]} joint angles, received {theta.shape[0]}"
        raise ValueError(msg)
    theta_eff = theta + dh.theta_offset

    chain: list[NDArray[np.float64]] = [np.eye(4, dtype=float)]
    T = chain[0]
    for ai, alpha_i, di, th_i in zip(dh.a, dh.alpha, dh.d, theta_eff):
        Ti = _numeric_transform(ai, alpha_i, di, th_i)
        T = T @ Ti
        chain.append(T)
    return chain


def evaluate_numeric_geometric_jacobian(
    joint_angles: Sequence[float],
    dh: DHParamsNum | None = None,
) -> Matrix6x6:
    """Evaluate the 6x6 geometric Jacobian at a specific joint configuration."""

    params = dh if dh is not None else demo_standard_6R_num()
    chain = _forward_chain_numeric(params, joint_angles)

    origins = [T[:3, 3] for T in chain]
    z_axes = [T[:3, 2] for T in chain]
    on = origins[-1]
    n_joints = params.a.shape[0]
    Jv = np.zeros((3, n_joints), dtype=float)
    Jw = np.zeros((3, n_joints), dtype=float)
    for i in range(n_joints):
        zi = z_axes[i]
        oi = origins[i]
        Jv[:, i] = np.cross(zi, on - oi)
        Jw[:, i] = zi
    return np.vstack((Jv, Jw))


__all__ = [
    "JacobianSymbolic",
    "symbolic_geometric_jacobian",
    "evaluate_numeric_geometric_jacobian",
]
