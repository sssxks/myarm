"""Jacobian-based velocity control demos wired into the myarm CLI.

Sections:
- control data/config (gains, state dataclass)
- control math utilities (imported from ``control_utils``)
- simulator helpers + control loop
- CLI plumbing
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import NDArray

from myarm.coppelia_utils import (
    DEFAULT_JOINT_NAMES,
    DEFAULT_TIP_NAME,
    connect_coppelia,
    get_joint_positions,
    get_object_handle,
)
from myarm.control_utils import clip_joint_velocities, damped_pinv_step
from myarm.ik_solver import IKOptions, fk_numeric, geometric_jacobian, rotation_error_vee, solve_ik
from myarm.trajectories import TrajState, build_traj_fn

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


# ---- Simulator helpers ----

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


def _stop_robot(sim, joints: list[int]) -> None:
    """Safely zero velocities."""
    for h in joints:
        sim.setJointTargetVelocity(h, 0.0)


def _configure_draw(sim, enable: bool, tip: int | None, draw_max: int) -> int | None:
    if not enable or tip is None:
        return None
    opts = sim.drawing_linestrip + sim.drawing_cyclic
    return sim.addDrawingObject(opts, 3, 0.0, -1, draw_max, [1.0, 0.0, 0.0])


def configure_velocity_parser(parser: argparse.ArgumentParser) -> None:
    """Attach velocity-control options to a subparser."""
    parser.description = "Jacobian velocity-control demos in CoppeliaSim"
    defaults = ControlGains()
    parser.add_argument("--traj", choices=["square", "circle", "cone"], default="square")
    parser.add_argument("--speed", type=float, default=0.05, help="linear speed (m/s) for square/circle")
    parser.add_argument("--ang-speed", type=float, default=0.8, help="angular speed (rad/s) for cone spin")
    parser.add_argument("--side", type=float, default=0.10, help="square side length (m)")
    parser.add_argument("--radius", type=float, default=0.05, help="circle radius (m)")
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--host", type=str, default=HOST)
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--cone-angle", type=float, default=60.0, help="apex angle (deg)")
    parser.add_argument("--center", type=float, nargs=3, default=[0.100, 0.100, 0.500], help="center/apex in meters")
    parser.add_argument("--draw", dest="draw", action="store_true", default=True, help="enable path rendering in CoppeliaSim (default)")
    parser.add_argument("--no-draw", dest="draw", action="store_false", help="disable path rendering in CoppeliaSim")
    parser.add_argument("--draw-max", type=int, default=2000, help="max drawing segments to keep (cyclic)")
    parser.add_argument("--axis", type=float, nargs=3, default=[0.0, 0.0, 1.0], help="spin axis (world frame, xyz)")
    parser.add_argument("--kp-pos", type=float, default=defaults.kp_pos, help="pos gain (mm/s per mm)")
    parser.add_argument("--kp-rot", type=float, default=defaults.kp_rot, help="rot gain (rad/s per rad)")
    parser.add_argument("--lambda-dls", type=float, default=defaults.lambda_dls, help="damping for damped least squares")
    parser.add_argument("--qdot-clip", type=float, default=defaults.qdot_clip, help="joint velocity limit (rad/s)")


def run_velocity_control(args: argparse.Namespace) -> int:
    """Entry point for the CLI subcommand."""
    center_mm = np.array(args.center, dtype=float) * 1000.0

    _, sim = connect_coppelia(args.host, args.port)
    joints = [get_object_handle(sim, name) for name in DEFAULT_JOINT_NAMES]
    tip = get_object_handle(sim, DEFAULT_TIP_NAME)

    traj_fn = build_traj_fn(args, center_mm)

    init_state = traj_fn(0.0)
    init_from_pose(sim, joints, init_state.pos_mm, init_state.rot_des)

    draw_handle = _configure_draw(sim, args.draw, tip, args.draw_max)

    gains = ControlGains(
        kp_pos=args.kp_pos,
        kp_rot=args.kp_rot,
        lambda_dls=args.lambda_dls,
        qdot_clip=args.qdot_clip,
    )
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
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="myarm velocity", description="Velocity-control demo driver")
    configure_velocity_parser(parser)
    args = parser.parse_args(argv)
    return run_velocity_control(args)


if __name__ == "__main__":
    raise SystemExit(main())
