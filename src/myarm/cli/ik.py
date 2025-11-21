"""CLI wiring for inverse-kinematics helpers."""

from __future__ import annotations

import argparse
import math
from math import degrees, radians
from typing import Sequence

import numpy as np
import sympy as sp

from myarm.cli.utils import pprint_matrix
from myarm.core.models import JointAngles, PoseTarget
from myarm.solvers.ik_solver import IKOptions, fk_numeric, pose_from_xyz_euler, solve_ik


def _print_ik_solutions(target: PoseTarget, results: list[tuple[np.ndarray, float, float, int]], limit: int) -> None:
    print("Target T06:")
    pprint_matrix(sp.Matrix(target.matrix.tolist()))
    print("\nSolutions (up to 8 unique):")
    for i, (q, pe, re, it) in enumerate(results[:limit], 1):
        qlist = [float(x) for x in q]
        qdeg = [round(degrees(v), 3) for v in qlist]
        print(f"\nSol {i}: iters={it}, pos_err={pe:.3e} mm, rot_err={degrees(re):.4f} deg")
        print("  q (rad):", [round(v, 6) for v in qlist])
        print("  q (deg):", qdeg)


def _build_ik_options(args: argparse.Namespace) -> IKOptions:
    return IKOptions(
        max_iter=int(args.max_iter),
        lambda_dls=float(args.lmbda),
        w_pos=float(args.w_pos),
        w_rot=float(args.w_rot),
        tol_pos=float(args.tol_pos),
        tol_rot=math.radians(float(args.tol_rot_deg)),
        step_clip=float(args.step_clip),
    )


def _collect_seeds(args: argparse.Namespace, deg: bool) -> list[JointAngles]:
    rows = getattr(args, "seed", None)
    if not rows:
        return []

    if any(len(row) != 6 for row in rows):
        raise SystemExit("--seed expects 6 values per entry")

    if deg:
        return [JointAngles(tuple(radians(v) for v in row)) for row in rows]
    else:
        return [JointAngles(tuple(float(v) for v in row)) for row in rows]


def _matrix16_to_np(vals: Sequence[float]) -> np.ndarray:
    if len(vals) != 16:
        raise SystemExit("--T requires 16 values (row-major 4x4)")
    matrix = np.array(list(vals), dtype=float).reshape(4, 4)
    return matrix


def _build_target_from_q(q: Sequence[float], deg: bool) -> np.ndarray:
    if len(q) != 6:
        raise SystemExit("--from-q requires 6 values")
    qrad = [math.radians(v) for v in q] if deg else list(q)
    return fk_numeric(qrad)


def _solve_ik(T_des, options, seeds):
    seed_values = [seed.as_list() for seed in seeds]
    return solve_ik(T_des, seeds=seed_values or None, opts=options)


def cmd_ik_solve(args: argparse.Namespace) -> int:
    if args.T is not None:
        T_des = _matrix16_to_np(args.T)
    elif args.from_q:
        T_des = _build_target_from_q(args.from_q, args.deg)
    else:
        raise SystemExit("Provide either --T 16vals or --from-q q1..q6")

    options = _build_ik_options(args)
    seeds = _collect_seeds(args, args.deg)
    results = _solve_ik(T_des, options, seeds)

    if not results:
        print("No solution found. Try adjusting seeds or tolerances.")
        return 1
    target = PoseTarget(T_des)
    _print_ik_solutions(target, results, int(args.limit))
    return 0


def _build_target_from_euler(args):
    if len(args.target) != 6:
        raise SystemExit("--target expects 6 values: x y z alpha beta gamma")

    x, y, z, alpha, beta, gamma = (float(v) for v in args.target)
    pos_unit = args.pos_unit.lower()
    if pos_unit == "m":
        scale = 1000.0
    elif pos_unit == "mm":
        scale = 1.0
    else:
        raise SystemExit("--pos-unit must be 'm' or 'mm'")

    x_mm, y_mm, z_mm = (scale * v for v in (x, y, z))
    if args.deg:
        alpha_r, beta_r, gamma_r = (radians(alpha), radians(beta), radians(gamma))
    else:
        alpha_r, beta_r, gamma_r = (alpha, beta, gamma)

    return pose_from_xyz_euler(x_mm, y_mm, z_mm, alpha_r, beta_r, gamma_r)


def _solve_ik_from_euler(args, T_des):
    options = _build_ik_options(args)
    seeds = _collect_seeds(args, args.deg)
    return _solve_ik(T_des, options, seeds)


def cmd_ik_euler(args: argparse.Namespace) -> int:
    T_des = _build_target_from_euler(args)
    results = _solve_ik_from_euler(args, T_des)

    if not results:
        print("No solution found. Try adjusting seeds or tolerances.")
        return 1

    x, y, z, alpha, beta, gamma = (float(v) for v in args.target)
    if args.deg:
        alpha_r, beta_r, gamma_r = (radians(alpha), radians(beta), radians(gamma))
    else:
        alpha_r, beta_r, gamma_r = (alpha, beta, gamma)
        
    limit = int(args.limit)
    if args.pos_unit == "m":
        xyz_vals = (x * 1000.0, y * 1000.0, z * 1000.0)
    else:
        xyz_vals = (x, y, z)
    xyz_mm = [round(v, 3) for v in xyz_vals]
    euler_rad = [round(v, 6) for v in (alpha_r, beta_r, gamma_r)]
    euler_deg = [round(degrees(v), 3) for v in (alpha_r, beta_r, gamma_r)]

    print(f"Target XYZ (mm): {xyz_mm}")
    print("Euler (rad):", euler_rad)
    print("Euler (deg):", euler_deg)
    target = PoseTarget(T_des)
    _print_ik_solutions(target, results, limit)
    return 0


def _add_ik_common_arguments(parser: argparse.ArgumentParser, deg_help: str) -> None:
    parser.add_argument("--deg", action="store_true", help=deg_help)
    parser.add_argument(
        "--seed",
        nargs=6,
        type=float,
        action="append",
        help="optional initial seed(s) q1..q6 (repeatable)",
    )
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--lmbda", type=float, default=1e-3, help="damping λ")
    parser.add_argument("--w-pos", type=float, default=1.0, help="weight for position (mm)")
    parser.add_argument("--w-rot", type=float, default=200.0, help="weight for rotation (rad)")
    parser.add_argument("--tol-pos", type=float, default=1e-2, help="pos tol (mm)")
    parser.add_argument("--tol-rot-deg", type=float, default=0.1, help="rot tol (deg)")
    parser.add_argument("--step-clip", type=float, default=0.5, help="max |Δq| per iter (rad)")
    parser.add_argument("--limit", type=int, default=8, help="print up to N solutions")


def register_subparsers(subparsers: argparse._SubParsersAction) -> None:
    ik = subparsers.add_parser("ik", help="inverse kinematics helpers")
    ik_sub = ik.add_subparsers(dest="ik_command", required=True)

    ik_solve = ik_sub.add_parser("solve", help="inverse kinematics for a target pose")
    ik_solve.add_argument("--T", nargs=16, type=float, help="target 4x4 (row-major) — 16 values")
    ik_solve.add_argument(
        "--from-q",
        nargs=6,
        type=float,
        help="build target from these joints (q1..q6)",
    )
    _add_ik_common_arguments(ik_solve, "interpret --from-q/--seed in degrees")
    ik_solve.set_defaults(func=cmd_ik_solve)

    ik_euler = ik_sub.add_parser("euler", help="inverse kinematics for XY'Z' target pose")
    ik_euler.add_argument(
        "--target",
        nargs=6,
        type=float,
        required=True,
        metavar=("x", "y", "z", "alpha", "beta", "gamma"),
        help="x y z (pos-unit) and XY'Z' Euler angles (rad by default)",
    )
    ik_euler.add_argument(
        "--pos-unit",
        choices=("m", "mm"),
        default="mm",
        help="units for x y z (default: millimeters)",
    )
    _add_ik_common_arguments(ik_euler, "interpret Euler angles and --seed in degrees")
    ik_euler.set_defaults(func=cmd_ik_euler)
