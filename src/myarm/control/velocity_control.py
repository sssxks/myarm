"""Core Jacobian velocity-control helpers for executing simulator demos.

This module exposes the low-level control loop, drawing utilities, and gains
config so higher-level scripts/CLIs can reuse them without duplicating logic.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from myarm.adapters.coppelia_utils import get_joint_positions
from .control_utils import clip_joint_velocities, damped_pinv_step
from myarm.solvers.ik_solver import IKOptions, fk_numeric, geometric_jacobian, rotation_error_vee
from myarm.solvers.trajectories import TrajState

Vector3 = NDArray[np.float64]
Matrix33 = NDArray[np.float64]

HOST = "127.0.0.1"
PORT = 23000


@dataclass(frozen=True)
class ControlGains:
    kp_pos: float = 4.0  # mm/s per mm
    kp_rot: float = 4.0  # rad/s per rad
    lambda_dls: float = 1e-3
    qdot_clip: float = 2.0  # rad/s


def control_loop(
    sim,
    joints: list[int],
    tip: int | None,
    traj_fn: Callable[[float], TrajState],
    gains: ControlGains,
    duration: float,
    dt: float,
    draw_handle: int | None = None,
) -> None:
    """Core velocity-control loop used by both the CLI and scripts."""
    t0 = time.monotonic()
    next_tick = t0
    last_draw_pt: list[float] | None = None
    while True:
        now = time.monotonic()
        t = now - t0
        if t >= duration:
            break
        if now < next_tick:
            time.sleep(max(0.0, next_tick - now))
            continue
        next_tick += dt

        state = traj_fn(t)
        q = np.array(get_joint_positions(sim, joints), dtype=float)
        T_cur = fk_numeric(q)
        p_cur = T_cur[:3, 3]
        R_cur = T_cur[:3, :3]

        e_pos = state.pos_mm - p_cur
        e_rot = rotation_error_vee(R_cur, state.rot_des)

        v_cmd = state.vel_mm_s + gains.kp_pos * e_pos
        w_cmd = state.omega_rad_s + gains.kp_rot * e_rot
        twist = np.hstack((v_cmd, w_cmd))

        J = geometric_jacobian(q)
        dq = damped_pinv_step(J, twist, gains.lambda_dls)
        dq = clip_joint_velocities(dq, gains.qdot_clip)

        for h, v in zip(joints, dq):
            sim.setJointTargetVelocity(h, float(v))

        if draw_handle is not None and tip is not None:
            pt = sim.getObjectPosition(tip, -1)  # meters
            if last_draw_pt is not None:
                sim.addDrawingObjectItem(draw_handle, last_draw_pt + pt)
            last_draw_pt = pt

    for h in joints:
        sim.setJointTargetVelocity(h, 0.0)


def stop_robot(sim, joints: list[int]) -> None:
    """Safely zero velocities."""
    for h in joints:
        sim.setJointTargetVelocity(h, 0.0)


def configure_draw(sim, enable: bool, tip: int | None, draw_max: int) -> int | None:
    if not enable or tip is None:
        return None
    opts = sim.drawing_linestrip + sim.drawing_cyclic
    return sim.addDrawingObject(opts, 3, 0.0, -1, draw_max, [1.0, 0.0, 0.0])
