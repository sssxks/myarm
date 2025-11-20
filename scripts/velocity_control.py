"""Linear/rotational velocity control demos for Lab5 trajectories.

This script drives the ZJU-I arm in CoppeliaSim using Jacobian-based
velocity commands. It reuses the existing FK/IK/Jacobian utilities and
keeps configuration lightweight so you can tweak speeds/poses quickly.

Examples (from repo root):
  uv.exe run python scripts/velocity_control.py --traj square --speed 0.05
  uv.exe run python scripts/velocity_control.py --traj circle --speed 0.05
  uv.exe run python scripts/velocity_control.py --traj cone --ang-speed 0.8
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from myarm.coppelia_utils import (
    DEFAULT_JOINT_NAMES,
    DEFAULT_TIP_NAME,
    connect_coppelia,
    get_joint_positions,
    get_object_handle,
)
from myarm.ik_solver import IKOptions, fk_numeric, geometric_jacobian, rotation_error_vee, solve_ik
from myarm.orientation import rotation_xy_dash_z_numeric

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


def _damped_pinv_step(J: NDArray[np.float64], twist: NDArray[np.float64], lam: float) -> NDArray[np.float64]:
    """Return joint velocities from task twist using damped least squares."""
    JT = J.T
    JJt = J @ JT
    JJt += (lam ** 2) * np.eye(J.shape[0])
    dq = JT @ np.linalg.solve(JJt, twist)
    return dq


def _clip(dq: NDArray[np.float64], limit: float) -> NDArray[np.float64]:
    if limit <= 0:
        return dq
    max_abs = float(np.max(np.abs(dq)))
    if max_abs <= limit:
        return dq
    return dq * (limit / max_abs)


@dataclass
class TrajState:
    pos_mm: Vector3
    rot_des: Matrix33
    vel_mm_s: Vector3
    omega_rad_s: Vector3


# ---- Trajectory generators ----

def square_traj(center_mm: Vector3, side_mm: float, speed_m_s: float, t: float) -> TrajState:
    """Axis-aligned square in X-Y, fixed Z."""
    side = side_mm
    v = speed_m_s * 1000.0  # convert to mm/s
    if v <= 1e-6:
        raise ValueError("speed too small")

    # Four edges, each length=side, time per edge
    t_edge = side / v
    total_t = 4 * t_edge
    tau = t % total_t
    edge = int(tau // t_edge)
    local_t = tau - edge * t_edge
    start_xy = np.array([
        [ -0.5, -0.5 ],
        [  0.5, -0.5 ],
        [  0.5,  0.5 ],
        [ -0.5,  0.5 ],
    ]) * side
    dir_xy = np.array([
        [ 1.0, 0.0 ],
        [ 0.0, 1.0 ],
        [ -1.0, 0.0 ],
        [ 0.0, -1.0 ],
    ])
    p_xy = start_xy[edge] + dir_xy[edge] * v * local_t
    pos = center_mm.copy()
    pos[0] += p_xy[0]
    pos[1] += p_xy[1]
    vel = np.zeros(3, dtype=float)
    vel[0:2] = dir_xy[edge] * v
    rot = rotation_xy_dash_z_numeric(math.pi, 0.0, 0.0)  # tool facing down
    return TrajState(pos_mm=pos, rot_des=rot, vel_mm_s=vel, omega_rad_s=np.zeros(3, dtype=float))


def circle_traj(center_mm: Vector3, radius_mm: float, speed_m_s: float, t: float) -> TrajState:
    r = radius_mm
    v = speed_m_s * 1000.0
    if r <= 1e-6 or v <= 1e-6:
        raise ValueError("radius/speed too small")
    omega = v / r  # rad/s
    theta = omega * t
    pos = center_mm.copy()
    pos[0] += r * math.cos(theta)
    pos[1] += r * math.sin(theta)
    vel = np.array([-v * math.sin(theta), v * math.cos(theta), 0.0], dtype=float)
    rot = rotation_xy_dash_z_numeric(math.pi, 0.0, 0.0)
    return TrajState(pos_mm=pos, rot_des=rot, vel_mm_s=vel, omega_rad_s=np.zeros(3, dtype=float))


def cone_traj(apex_mm: Vector3, half_angle_deg: float, ang_speed: float, t: float) -> TrajState:
    """End-effector rotates about apex; position fixed, orientation spins around world Z."""
    theta = math.radians(half_angle_deg)
    phi = ang_speed * t
    # R = Rz(phi) * Ry(theta): tool z tilted theta away from world Z and spinning
    rot = rotation_xy_dash_z_numeric(0.0, theta, phi)
    omega = np.array([0.0, 0.0, ang_speed], dtype=float)  # world-frame spin about Z
    pos = apex_mm
    vel = np.zeros(3, dtype=float)
    return TrajState(pos_mm=pos, rot_des=rot, vel_mm_s=vel, omega_rad_s=omega)


def init_from_pose(sim, joints: list[int], pos_mm: Vector3, rot: Matrix33) -> np.ndarray:
    """Solve IK for starting pose and push to sim."""
    T_des = np.eye(4, dtype=float)
    T_des[:3, :3] = rot
    T_des[:3, 3] = pos_mm
    sols = solve_ik(T_des, opts=IKOptions(max_iter=150))
    if not sols:
        raise RuntimeError("No IK solution for start pose")
    q = sols[0][0]
    for h, val in zip(joints, q):
        sim.setJointPosition(h, float(val))
    return q


def control_loop(
    sim,
    joints: list[int],
    traj_fn: Callable[[float], TrajState],
    gains: ControlGains,
    duration: float,
    dt: float,
) -> None:
    t0 = time.time()
    last = t0
    while True:
        now = time.time()
        t = now - t0
        if t >= duration:
            break
        if now - last < dt:
            time.sleep(max(0.0, dt - (now - last)))
            continue
        last = now

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
        dq = _damped_pinv_step(J, twist, gains.lambda_dls)
        dq = _clip(dq, gains.qdot_clip)

        for h, v in zip(joints, dq):
            sim.setJointTargetVelocity(h, float(v))

    for h in joints:
        sim.setJointTargetVelocity(h, 0.0)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Jacobian velocity control demo")
    p.add_argument("--traj", choices=["square", "circle", "cone"], default="square")
    p.add_argument("--speed", type=float, default=0.05, help="linear speed (m/s) for square/circle")
    p.add_argument("--ang-speed", type=float, default=0.8, help="angular speed (rad/s) for cone spin")
    p.add_argument("--side", type=float, default=0.10, help="square side length (m)")
    p.add_argument("--radius", type=float, default=0.05, help="circle radius (m)")
    p.add_argument("--duration", type=float, default=20.0)
    p.add_argument("--dt", type=float, default=0.02)
    p.add_argument("--host", type=str, default=HOST)
    p.add_argument("--port", type=int, default=PORT)
    p.add_argument("--cone-angle", type=float, default=60.0, help="apex angle (deg)")
    p.add_argument("--center", type=float, nargs=3, default=[0.300, 0.000, 0.300], help="center/apex in meters")
    return p


def main() -> None:
    args = build_parser().parse_args()
    center_mm = np.array(args.center, dtype=float) * 1000.0

    _, sim = connect_coppelia(args.host, args.port)
    joints = [get_object_handle(sim, name) for name in DEFAULT_JOINT_NAMES]
    _ = get_object_handle(sim, DEFAULT_TIP_NAME)

    if args.traj == "square":
        side_mm = args.side * 1000.0
        traj_fn = lambda t: square_traj(center_mm.copy(), side_mm, args.speed, t)
    elif args.traj == "circle":
        radius_mm = args.radius * 1000.0
        traj_fn = lambda t: circle_traj(center_mm.copy(), radius_mm, args.speed, t)
    else:
        half_angle = args.cone_angle / 2.0
        traj_fn = lambda t: cone_traj(center_mm.copy(), half_angle, args.ang_speed, t)

    init_state = traj_fn(0.0)
    init_from_pose(sim, joints, init_state.pos_mm, init_state.rot_des)

    gains = ControlGains()
    print(f"Running {args.traj} trajectory for {args.duration}s ...")
    control_loop(sim, joints, traj_fn, gains, args.duration, args.dt)
    print("Done.")


if __name__ == "__main__":
    main()
