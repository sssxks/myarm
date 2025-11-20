"""Numerical helpers shared by velocity-control routines.

Separated from `velocity_control.py` so the core math can be imported in tests
without pulling in simulator dependencies.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Vector = NDArray[np.float64]
Matrix = NDArray[np.float64]

__all__ = ["damped_pinv_step", "clip_joint_velocities"]


def damped_pinv_step(J: Matrix, twist: Vector, lam: float) -> Vector:
    """Map task-space twist to joint velocities using damped least squares.

    The damping term is scaled when the Jacobian is ill-conditioned to avoid
    spikes near singularities. Falls back to an SVD-based pseudo-inverse if
    the direct solve fails.
    """
    JT = J.T
    cond = float(np.linalg.cond(J))
    lam_eff = lam
    if cond > 1e3:
        lam_eff *= cond / 1e3
        lam_eff = min(lam_eff, 1e-1)

    JJt = J @ JT
    JJt += (lam_eff**2) * np.eye(J.shape[0])
    try:
        dq = JT @ np.linalg.solve(JJt, twist)
    except np.linalg.LinAlgError:
        U, S, Vt = np.linalg.svd(J)
        damped = S / (S**2 + lam_eff**2)
        dq = Vt.T @ (damped * (U.T @ twist))
    return dq


def clip_joint_velocities(dq: Vector, limit: float) -> Vector:
    """Uniformly scale joint velocities if any element exceeds ``limit``."""
    if limit <= 0:
        return dq
    max_abs = float(np.max(np.abs(dq)))
    if max_abs <= limit:
        return dq
    return dq * (limit / max_abs)
