from __future__ import annotations

import argparse
import math
import time
from collections.abc import Sequence
from typing import Any, cast

import numpy as np

from myarm.coppelia_utils import (
    DEFAULT_JOINT_NAMES,
    DEFAULT_TIP_NAME,
    connect_coppelia,
    get_joint_positions,
    get_matrix4,
    get_object_handle,
    rotation_angle_deg,
    set_joint_positions,
)
from myarm.ik_solver import IKOptions, fk_numeric, solve_ik


def configure_verify_ik_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Attach the shared IK verification CLI arguments."""
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=23000)
    parser.add_argument(
        "--joint",
        dest="joints",
        nargs=6,
        metavar=("J1", "J2", "J3", "J4", "J5", "J6"),
        help="override joint paths (provide 6 names)",
        default=None,
    )
    parser.add_argument("--tip", default=DEFAULT_TIP_NAME)
    parser.add_argument("--base", default=None)
    parser.add_argument("--unit-scale", type=float, default=0.001, help="FK(mm)↔Sim(m)")
    parser.add_argument("--tol-pos-mm", type=float, default=1e-1)
    parser.add_argument("--tol-rot-deg", type=float, default=0.2)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--lmbda", type=float, default=1e-3)
    parser.add_argument("--sleep", type=float, default=0.05)
    parser.add_argument("--apply", action="store_true", help="apply best IK q to sim")
    return parser


def _prepare_target(
    sim: Any,
    tip_handle: int,
    base_handle: int | None,
    unit_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (matrix_meters, matrix_mm) for the current simulation pose."""
    matrix_meters = get_matrix4(sim, tip_handle, base_handle)
    matrix_mm = matrix_meters.copy()
    matrix_mm[:3, 3] = matrix_mm[:3, 3] / float(unit_scale)
    return matrix_meters, matrix_mm


def _build_options(args: argparse.Namespace) -> IKOptions:
    return IKOptions(
        max_iter=int(args.max_iter),
        lambda_dls=float(args.lmbda),
        tol_pos=float(args.tol_pos_mm),
        tol_rot=math.radians(float(args.tol_rot_deg)),
    )


def run_verify_ik(args: argparse.Namespace) -> int:
    # for some reason, we can not call .close() on client. so just ignore that
    _, sim = connect_coppelia(args.host, args.port)

    joint_names = list(args.joints) if args.joints is not None else list(DEFAULT_JOINT_NAMES)
    if len(joint_names) != 6:
        raise SystemExit(f"Expected 6 joints, got {len(joint_names)}")
    joint_handles = [get_object_handle(sim, name) for name in joint_names]
    tip_handle = get_object_handle(sim, args.tip)
    base_handle = get_object_handle(sim, args.base) if args.base else None

    target_meters, target_mm = _prepare_target(
        sim, tip_handle, base_handle, float(args.unit_scale)
    )
    q_sim = get_joint_positions(sim, joint_handles)
    options = _build_options(args)

    results = solve_ik(target_mm, seeds=[q_sim], opts=options)
    if not results:
        print("No solution found.")
        return 1

    np.set_printoptions(precision=4, suppress=True)
    print("Target transform (meters from sim; mm internally for IK):")
    print(target_meters)

    best: tuple[np.ndarray, float, float] | None = None
    for index, (q, pos_err_mm, rot_err_rad, iterations) in enumerate(results, start=1):
        T_fk = fk_numeric(q)
        pos_err = float(np.linalg.norm(T_fk[:3, 3] - target_mm[:3, 3]))
        rot_err = rotation_angle_deg(T_fk[:3, :3], target_mm[:3, :3])
        print(
            f"\nSol {index}: iters={iterations}, pos_err={pos_err:.3e} mm, "
            f"rot_err={rot_err:.3f} deg"
        )
        print("  q(rad):", [round(float(value), 6) for value in q])
        print("  q(deg):", [round(math.degrees(float(value)), 3) for value in q])
        if best is None or (pos_err, rot_err) < (best[1], best[2]):
            best = (q, pos_err, rot_err)

    assert best is not None  # results is non-empty by this point
    passed = best[1] <= args.tol_pos_mm and best[2] <= args.tol_rot_deg
    print(
        f"\nBest: pos_err={best[1]:.3e} mm, rot_err={best[2]:.3f} deg → "
        f"{'PASS' if passed else 'FAIL'}"
    )

    if args.apply:
        set_joint_positions(
            sim, joint_handles, cast(Sequence[float], best[0]), mode="position"
        )
        time.sleep(max(0.0, float(args.sleep)))
        updated = get_matrix4(sim, tip_handle, base_handle)
        print("\nApplied best solution. Sim tip now:")
        print(updated)

    return 0
