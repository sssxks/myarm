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
    # soften acceleration near corners to avoid overshoot when the direction flips
    ramp_frac = 0.12  # fraction of each edge spent ramping speed
    ramp_t = ramp_frac * t_edge
    if local_t < ramp_t:
        speed_scale = local_t / ramp_t
    elif local_t > t_edge - ramp_t:
        speed_scale = (t_edge - local_t) / ramp_t
    else:
        speed_scale = 1.0
    speed_scale = max(0.1, speed_scale)  # keep small feedforward so it doesn't stall

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
    vel[0:2] = dir_xy[edge] * v * speed_scale
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


def _orthonormal_basis_from_axis(axis: Vector3) -> tuple[Vector3, Vector3, Vector3]:
    """Return (n_hat, u, v) where u,v are orthonormal and ⟂ n_hat."""
    n_norm = float(np.linalg.norm(axis))
    if n_norm <= 1e-9:
        raise ValueError("spin axis must be non-zero")
    n_hat = axis / n_norm
    # pick a helper not parallel to n_hat
    helper = np.array([0.0, 0.0, 1.0], dtype=float)
    if abs(np.dot(helper, n_hat)) > 0.99:
        helper = np.array([0.0, 1.0, 0.0], dtype=float)
    u = np.cross(helper, n_hat)
    u /= np.linalg.norm(u)
    v = np.cross(n_hat, u)
    return n_hat, u, v


def cone_traj(
    apex_mm: Vector3,
    half_angle_deg: float,
    ang_speed: float,
    t: float,
    spin_axis: Vector3 | None = None,
) -> TrajState:
    """End-effector orientation traces a cone about the given world-frame spin axis."""
    theta = math.radians(half_angle_deg)
    phi = ang_speed * t

    n_hat, u, v = _orthonormal_basis_from_axis(spin_axis if spin_axis is not None else np.array([0.0, 0.0, 1.0]))

    # Cone generatrix direction (tool z): tilt by theta from n_hat, sweep with phi around n_hat.
    z_tool = math.cos(theta) * n_hat + math.sin(theta) * (math.cos(phi) * u + math.sin(phi) * v)

    # x_tool follows the precession around n_hat to keep R spinning with phi.
    x_tool = np.cross(z_tool, n_hat)
    norm_x = float(np.linalg.norm(x_tool))
    if norm_x <= 1e-9:  # theta=0 edge case: choose stable x aligned with u
        x_tool = u
    else:
        x_tool /= norm_x
    y_tool = np.cross(z_tool, x_tool)

    rot = np.column_stack((x_tool, y_tool, z_tool))
    # Angular velocity in world frame: spin about the chosen axis.
    omega = ang_speed * n_hat
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
    tip: int | None,
    traj_fn: Callable[[float], TrajState],
    gains: ControlGains,
    duration: float,
    dt: float,
    draw_handle: int | None = None,
) -> None:
    t0 = time.time()
    last = t0
    last_draw_pt: list[float] | None = None
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

        if draw_handle is not None and tip is not None:
            pt = sim.getObjectPosition(tip, -1)  # meters
            if last_draw_pt is not None:
                sim.addDrawingObjectItem(draw_handle, last_draw_pt + pt)
            last_draw_pt = pt

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
    p.add_argument("--center", type=float, nargs=3, default=[0.100, 0.100, 0.500], help="center/apex in meters")
    p.add_argument("--draw", dest="draw", action="store_true", default=True, help="enable path rendering in CoppeliaSim (default)")
    p.add_argument("--no-draw", dest="draw", action="store_false", help="disable path rendering in CoppeliaSim")
    p.add_argument("--draw-max", type=int, default=2000, help="max drawing segments to keep (cyclic)")
    p.add_argument("--axis", type=float, nargs=3, default=[0.0, 0.0, 1.0], help="spin axis (world frame, xyz)")
    return p


def _stop_robot(sim, joints: list[int]) -> None:
    """Safely zero velocities."""
    for h in joints:
        sim.setJointTargetVelocity(h, 0.0)


def main() -> None:
    args = build_parser().parse_args()
    center_mm = np.array(args.center, dtype=float) * 1000.0

    _, sim = connect_coppelia(args.host, args.port)
    joints = [get_object_handle(sim, name) for name in DEFAULT_JOINT_NAMES]
    tip = get_object_handle(sim, DEFAULT_TIP_NAME)

    if args.traj == "square":
        side_mm = args.side * 1000.0
        traj_fn = lambda t: square_traj(center_mm.copy(), side_mm, args.speed, t)
    elif args.traj == "circle":
        radius_mm = args.radius * 1000.0
        traj_fn = lambda t: circle_traj(center_mm.copy(), radius_mm, args.speed, t)
    else:
        half_angle = args.cone_angle / 2.0
        spin_axis = np.array(args.axis, dtype=float)
        traj_fn = lambda t: cone_traj(center_mm.copy(), half_angle, args.ang_speed, t, spin_axis=spin_axis)

    init_state = traj_fn(0.0)
    init_from_pose(sim, joints, init_state.pos_mm, init_state.rot_des)

    draw_handle = None
    if args.draw:
        opts = sim.drawing_linestrip + sim.drawing_cyclic
        draw_handle = sim.addDrawingObject(opts, 3, 0.0, -1, args.draw_max, [1.0, 0.0, 0.0])

    gains = ControlGains()
    print(f"Running {args.traj} trajectory for {args.duration}s ...")
    try:
        control_loop(sim, joints, tip, traj_fn, gains, args.duration, args.dt, draw_handle)
    except KeyboardInterrupt:
        print("Interrupted, stopping ...")
    finally:
        _stop_robot(sim, joints)
        if draw_handle is not None:
            sim.removeDrawingObject(draw_handle)
    print("Done.")


if __name__ == "__main__":
    main()
