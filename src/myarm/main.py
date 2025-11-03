"""Unified command-line interface for the myarm manipulator toolkit.

Examples
--------
- Show symbolic T06:
    uv run myarm -- fk symbolic

- Evaluate presets (degrees):
    uv run myarm -- fk eval --preset 1 2 3 --deg

- Run 5 random FK checks avoiding gimbal lock:
    uv run myarm -- fk random --count 5 --seed 0

- Solve IK for a pose described by Euler angles and translation:
    uv run myarm -- ik solve --from-q 0 0 1.57 0 0 0

- Verify FK against CoppeliaSim:
    uv run myarm -- verify fk --tol-pos 1e-3 --tol-rot 0.5
"""

from __future__ import annotations

import argparse
import math
import random
from collections.abc import Sequence
from typing import TYPE_CHECKING

import sympy as sp

from myarm.dh_params import demo_standard_6R
from myarm.ik_solver import (
    IKOptions,
    fk_numeric,
    pose_from_xyz_euler,
    solve_ik,
)
from myarm.numerical_checker import check_numeric_once
from myarm.solver import T_to_euler_xy_dash_z
from myarm.verify_fk_coppelia import configure_parser as configure_verify_fk_parser
from myarm.verify_fk_coppelia import run_verify_fk
from myarm.verify_ik_coppelia import configure_parser as configure_verify_ik_parser
from myarm.verify_ik_coppelia import run_verify_ik

if TYPE_CHECKING:
    import numpy as np


def _pprint_matrix(matrix: sp.Matrix) -> None:
    sp.pprint(matrix, use_unicode=True)  # type: ignore[operator]


def _deg(value: float) -> float:
    return math.degrees(value)


def _rad(value: float) -> float:
    return math.radians(value)


def _parse_qs(q_list: list[list[float]], deg: bool) -> list[list[float]]:
    if deg:
        return [[_rad(entry) for entry in row] for row in q_list]
    return [list(row) for row in q_list]


PRESETS_DEG = [
    [30, 0, 90, 60, 0, 0],
    [-30, 0, 60, 60, -30, 0],
    [30, 0, 60, 60, 60, 0],
    [-30, 0, 60, 60, 15, 0],
    [15, 15, 15, 15, 15, 15],
]


def cmd_fk_symbolic(args: argparse.Namespace) -> int:
    T06, th_syms, params = demo_standard_6R()
    a = params["a"]
    alpha = params["alpha"]
    d = params["d"]
    theta = params["theta"]
    if args.steps:
        print("Stepwise Ti (evaluated at rest unless --no-eval):")
        rest = {s: 0.0 for s in th_syms}
        for i in range(6):
            Ti = fk_standard(a[: i + 1], alpha[: i + 1], d[: i + 1], theta[: i + 1])
            Mi = sp.N(Ti.subs(rest), 6) if args.eval else Ti  # type: ignore[no-untyped-call]
            print(f"\nT0{i + 1}:")
            _pprint_matrix(Mi)
        return 0

    print("Symbolic T06:")
    _pprint_matrix(T06)
    if args.eval:
        rest = {s: 0.0 for s in th_syms}
        print("\nT06 at rest (rad=0):")
        _pprint_matrix(sp.N(T06.subs(rest), 6))  # type: ignore[no-untyped-call]

    if args.euler:
        ea, eb, eg = T_to_euler_xy_dash_z(T06, safe=False)
        print("\nSymbolic XY'Z' (alpha, beta, gamma):")
        _pprint_matrix(sp.simplify(ea))  # type: ignore[no-untyped-call]
        _pprint_matrix(sp.simplify(eb))  # type: ignore[no-untyped-call]
        _pprint_matrix(sp.simplify(eg))  # type: ignore[no-untyped-call]
    return 0


def cmd_fk_eval(args: argparse.Namespace) -> int:
    T06, th_syms, _ = demo_standard_6R()

    qs: list[list[float]] = []
    if args.preset:
        for idx in args.preset:
            if not 1 <= idx <= len(PRESETS_DEG):
                raise SystemExit(f"--preset index out of range: {idx}")
            qs.append(PRESETS_DEG[idx - 1])
        qs = _parse_qs(qs, deg=True)
    if args.q:
        qs.extend(_parse_qs(args.q, deg=args.deg))
    if not qs:
        raise SystemExit("Provide --preset or --q …")

    for i, row in enumerate(qs, 1):
        subs = {s: float(v) for s, v in zip(th_syms, row)}
        T_num = sp.N(T06.subs(subs), 15)  # type: ignore[no-untyped-call]
        print(f"\nCase {i}:")
        _pprint_matrix(T_num)
        a, b, g = T_to_euler_xy_dash_z(T_num, safe=True)
        print("XY'Z' (rad):", float(a), float(b), float(g))
        print("XY'Z' (deg):", _deg(float(a)), _deg(float(b)), _deg(float(g)))
    return 0


def cmd_fk_random(args: argparse.Namespace) -> int:
    T06, th_syms, _ = demo_standard_6R()
    rng = random.Random(args.seed)
    n = int(args.count)
    printed = 0
    while printed < n:
        vals = [rng.uniform(-math.pi, math.pi) for _ in th_syms]
        subs = {s: v for s, v in zip(th_syms, vals)}
        T_num = sp.N(T06.subs(subs), 15)  # type: ignore[no-untyped-call]
        a, b, g = T_to_euler_xy_dash_z(T_num, safe=True)
        if abs(math.cos(float(b))) < 1e-6:
            continue  # skip near gimbal lock
        print(f"\nSample {printed + 1}:")
        check_numeric_once(T06, subs)
        printed += 1
    return 0


def cmd_fk_dh(_args: argparse.Namespace) -> int:
    _T, _th, params = demo_standard_6R()
    print("a:", params["a"])
    print("alpha:", params["alpha"])
    print("d:", params["d"])
    print("theta (symbolic):", params["theta"])
    return 0


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


def _collect_seeds(args: argparse.Namespace, deg: bool) -> list[list[float]]:
    seeds: list[list[float]] = []
    rows = getattr(args, "seed", None)
    if rows:
        for row in rows:
            if len(row) != 6:
                raise SystemExit("--seed expects 6 values per entry")
            converted = [_rad(v) for v in row] if deg else [float(v) for v in row]
            seeds.append(converted)
    return seeds


def _print_ik_solutions(T_des: "np.ndarray", results: list[tuple["np.ndarray", float, float, int]], limit: int) -> None:
    print("Target T06:")
    _pprint_matrix(sp.Matrix(T_des.tolist()))
    print("\nSolutions (up to 8 unique):")
    for i, (q, pe, re, it) in enumerate(results[:limit], 1):
        qlist = [float(x) for x in q]
        qdeg = [round(_deg(v), 3) for v in qlist]
        iter_label = f"iters={it}" if it >= 0 else "iters=nsolve"
        print(f"\nSol {i}: {iter_label}, pos_err={pe:.3e} mm, rot_err={_deg(re):.4f} deg")
        print("  q (rad):", [round(v, 6) for v in qlist])
        print("  q (deg):", qdeg)


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
    parser.add_argument("--tol-pos", type=float, default=1e-3, help="pos tol (mm)")
    parser.add_argument("--tol-rot-deg", type=float, default=0.1, help="rot tol (deg)")
    parser.add_argument("--step-clip", type=float, default=0.5, help="max |Δq| per iter (rad)")
    parser.add_argument("--limit", type=int, default=8, help="print up to N solutions")


def _matrix16_to_np(vals: Sequence[float]) -> "np.ndarray":
    import numpy as np  # local import to keep module import light

    if len(vals) != 16:
        raise SystemExit("--T requires 16 values (row-major 4x4)")
    matrix = np.array(list(vals), dtype=float).reshape(4, 4)
    return matrix


def _build_target_from_q(q: Sequence[float], deg: bool) -> "np.ndarray":
    import numpy as np

    if len(q) != 6:
        raise SystemExit("--from-q requires 6 values")
    qrad = [math.radians(v) for v in q] if deg else list(q)
    return fk_numeric(qrad)


def cmd_ik_solve(args: argparse.Namespace) -> int:
    if args.T is not None:
        T_des = _matrix16_to_np(args.T)
    elif args.from_q:
        T_des = _build_target_from_q(args.from_q, args.deg)
    else:
        raise SystemExit("Provide either --T 16vals or --from-q q1..q6")

    options = _build_ik_options(args)
    seeds = _collect_seeds(args, args.deg)
    results = solve_ik(T_des, seeds=seeds or None, opts=options)
    if not results:
        print("No solution found. Try adjusting seeds or tolerances.")
        return 1
    _print_ik_solutions(T_des, results, int(args.limit))
    return 0


def cmd_verify_fk(args: argparse.Namespace) -> int:
    return run_verify_fk(args)


def cmd_verify_ik(args: argparse.Namespace) -> int:
    return run_verify_ik(args)


def cmd_ik_euler(args: argparse.Namespace) -> int:
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
        alpha_r, beta_r, gamma_r = (_rad(alpha), _rad(beta), _rad(gamma))
    else:
        alpha_r, beta_r, gamma_r = (alpha, beta, gamma)

    T_des = pose_from_xyz_euler(x_mm, y_mm, z_mm, alpha_r, beta_r, gamma_r)
    options = _build_ik_options(args)
    seeds = _collect_seeds(args, args.deg)
    results = solve_ik(T_des, seeds=seeds or None, opts=options)
    if not results:
        print("No solution found. Try adjusting seeds or tolerances.")
        return 1
    limit = int(args.limit)

    xyz_mm = [round(v, 3) for v in (x_mm, y_mm, z_mm)]
    euler_rad = [round(v, 6) for v in (alpha_r, beta_r, gamma_r)]
    euler_deg = [round(_deg(v), 3) for v in (alpha_r, beta_r, gamma_r)]
    print(f"Target XYZ (mm): {xyz_mm}")
    print("Euler (rad):", euler_rad)
    print("Euler (deg):", euler_deg)
    _print_ik_solutions(T_des, results, limit)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="myarm", description="myarm robotics CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    fk = sub.add_parser("fk", help="forward kinematics utilities")
    fk_sub = fk.add_subparsers(dest="fk_command", required=True)

    fk_symbolic = fk_sub.add_parser("symbolic", help="show symbolic T06 or stepwise Ti")
    fk_symbolic.add_argument("--steps", action="store_true", help="show T01..T06")
    fk_symbolic.add_argument(
        "--no-eval", dest="eval", action="store_false", help="do not eval at rest"
    )
    fk_symbolic.add_argument("--euler", action="store_true", help="print symbolic XY'Z' angles")
    fk_symbolic.set_defaults(func=cmd_fk_symbolic, eval=True)

    fk_eval = fk_sub.add_parser("eval", help="evaluate T06 and XY'Z' for angles")
    fk_eval.add_argument("--preset", type=int, nargs="*", help="use preset 1..5 (deg)")
    fk_eval.add_argument("--q", nargs=6, type=float, action="append", help="custom angles q1..q6")
    fk_eval.add_argument("--deg", action="store_true", help="interpret --q in degrees")
    fk_eval.set_defaults(func=cmd_fk_eval)

    fk_random = fk_sub.add_parser("random", help="random numeric checks avoiding gimbal lock")
    fk_random.add_argument("--count", type=int, default=5)
    fk_random.add_argument("--seed", type=int, default=0)
    fk_random.set_defaults(func=cmd_fk_random)

    fk_dh = fk_sub.add_parser("dh", help="print DH parameters")
    fk_dh.set_defaults(func=cmd_fk_dh)

    ik = sub.add_parser("ik", help="inverse kinematics helpers")
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
        default="m",
        help="units for x y z (default: meters)",
    )
    ik_euler.add_argument(
        "--nsolve",
        action="store_true",
        help="refine solutions via sympy.nsolve starting from numeric IK",
    )
    _add_ik_common_arguments(ik_euler, "interpret Euler angles and --seed in degrees")
    ik_euler.set_defaults(func=cmd_ik_euler)

    verify = sub.add_parser("verify", help="compare against CoppeliaSim")
    verify_sub = verify.add_subparsers(dest="verify_command", required=True)

    verify_fk = verify_sub.add_parser("fk", help="verify FK using CoppeliaSim")
    configure_verify_fk_parser(verify_fk)
    verify_fk.set_defaults(func=cmd_verify_fk)

    verify_ik = verify_sub.add_parser("ik", help="verify IK using CoppeliaSim")
    configure_verify_ik_parser(verify_ik)
    verify_ik.set_defaults(func=cmd_verify_ik)

    return parser


def fk_standard(
    a: Sequence[int | float | sp.Expr],
    alpha: Sequence[int | float | sp.Expr],
    d: Sequence[int | float | sp.Expr],
    theta: Sequence[int | float | sp.Expr],
) -> sp.Matrix:  # keep backward import compatibility if used elsewhere
    from myarm.solver import fk_standard as _fk

    return _fk(a, alpha, d, theta)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
