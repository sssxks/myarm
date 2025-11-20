"""CLI wiring for the velocity-control demo."""

from __future__ import annotations

import argparse

import numpy as np

from myarm.adapters.coppelia_utils import (
    DEFAULT_JOINT_NAMES,
    DEFAULT_TIP_NAME,
    connect_coppelia,
    get_object_handle,
)
from myarm.control.velocity_control import (
    ControlGains,
    configure_draw,
    control_loop,
    stop_robot,
    HOST,
    PORT,
)
from myarm.solvers.trajectories import build_traj_fn


def configure_velocity_parser(parser: argparse.ArgumentParser) -> None:
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


def cmd_velocity(args: argparse.Namespace) -> int:
    center_mm = np.array(args.center, dtype=float) * 1000.0

    _, sim = connect_coppelia(args.host, args.port)
    joints = [get_object_handle(sim, name) for name in DEFAULT_JOINT_NAMES]
    tip = get_object_handle(sim, DEFAULT_TIP_NAME)

    traj_fn = build_traj_fn(args, center_mm)

    draw_handle = configure_draw(sim, args.draw, tip, args.draw_max)

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
        stop_robot(sim, joints)
        if draw_handle is not None:
            sim.removeDrawingObject(draw_handle)
    print("Done.")
    return 0


def register_subparsers(subparsers: argparse._SubParsersAction) -> None:
    velocity = subparsers.add_parser("velocity", help="Jacobian velocity-control demos (requires CoppeliaSim)")
    configure_velocity_parser(velocity)
    velocity.set_defaults(func=cmd_velocity)
