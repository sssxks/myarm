"""CLI wiring for forward-kinematics utilities."""

from __future__ import annotations

import argparse
import math
import random
from collections.abc import Iterable, Sequence
from math import degrees, radians
from typing import Iterator
import sympy as sp

from myarm.cli.utils import pprint_matrix
from myarm.model.dh_params import demo_standard_6R
from myarm.model.presets import PRESETS_DEG
from myarm.solvers.fk_solver import T_to_euler_xy_dash_z, fk_standard
from myarm.solvers.numerical_checker import NumericCheckResult, check_numeric_once


def _parse_qs(q_list: Sequence[Sequence[float]], deg: bool) -> list[list[float]]:
    if deg:
        return [[radians(entry) for entry in row] for row in q_list]
    return [list(row) for row in q_list]


def _generate_symbolic_matrices(
    a: Sequence[sp.Expr | float],
    alpha: Sequence[sp.Expr | float],
    d: Sequence[sp.Expr | float],
    theta: Sequence[sp.Expr | float],
    rest: dict[sp.Symbol, float],
    evaluate: bool,
) -> list[sp.Matrix]:
    matrices = [
        fk_standard(a[: i + 1], alpha[: i + 1], d[: i + 1], theta[: i + 1])
        for i in range(6)
    ]
    if evaluate:
        return [sp.N(T.subs(rest), 6) for T in matrices]
    return matrices


def cmd_fk_symbolic(args: argparse.Namespace) -> int:
    T06, th_syms, params = demo_standard_6R()
    a = params["a"]
    alpha = params["alpha"]
    d = params["d"]
    theta = params["theta"]

    if args.steps:
        print("Stepwise Ti (evaluated at rest unless --no-eval):")
        rest = {s: 0.0 for s in th_syms}
        matrices = _generate_symbolic_matrices(a, alpha, d, theta, rest, args.eval)
        for i, T in enumerate(matrices):
            print(f"\nT0{i + 1}:")
            pprint_matrix(T)
        return 0

    print("Symbolic T06:")
    pprint_matrix(T06)
    if args.eval:
        rest = {s: 0.0 for s in th_syms}
        print("\nT06 at rest (rad=0):")
        pprint_matrix(sp.N(T06.subs(rest), 6))  # type: ignore[no-untyped-call]

    if args.euler:
        ea, eb, eg = T_to_euler_xy_dash_z(T06, safe=False)
        print("\nSymbolic XY'Z' (alpha, beta, gamma):")
        pprint_matrix(sp.simplify(ea))  # type: ignore[no-untyped-call]
        pprint_matrix(sp.simplify(eb))  # type: ignore[no-untyped-call]
        pprint_matrix(sp.simplify(eg))  # type: ignore[no-untyped-call]
    return 0


def _evaluate_fk(
    T06: sp.Matrix, th_syms: Sequence[sp.Symbol], qs: Sequence[Sequence[float]]
) -> list[tuple[sp.Matrix, tuple[float, float, float]]]:
    def _calculate(row: Sequence[float]) -> tuple[sp.Matrix, tuple[float, float, float]]:
        T_num = sp.N(T06.subs({s: float(v) for s, v in zip(th_syms, row)}), 15)
        a, b, g = T_to_euler_xy_dash_z(T_num, safe=True)
        return T_num, (float(a), float(b), float(g))

    return [_calculate(row) for row in qs]


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

    results = _evaluate_fk(T06, th_syms, qs)

    for i, (T_num, (a, b, g)) in enumerate(results, 1):
        print(f"\nCase {i}:")
        pprint_matrix(T_num)
        print("XY'Z' (rad):", float(a), float(b), float(g))
        print("XY'Z' (deg):", degrees(float(a)), degrees(float(b)), degrees(float(g)))
    return 0


def _generate_random_fk(T06: sp.Matrix, th_syms: Sequence[sp.Symbol], rng: random.Random) -> Iterator[NumericCheckResult]:
    while True:
        vals = [rng.uniform(-math.pi, math.pi) for _ in th_syms]
        subs = {s: v for s, v in zip(th_syms, vals)}
        T_num = sp.N(T06.subs(subs), 15)
        a, b, g = T_to_euler_xy_dash_z(T_num, safe=True)
        if abs(math.cos(float(b))) < 1e-6:
            continue
        yield check_numeric_once(T06, subs)


def cmd_fk_random(args: argparse.Namespace) -> int:
    T06, th_syms, _ = demo_standard_6R()
    rng = random.Random(args.seed)
    
    from itertools import islice

    for i, result in enumerate(islice(_generate_random_fk(T06, th_syms, rng), args.count)):
        print(f"\nSample {i + 1}:")
        print("XY'Z' (rad):", result.alpha, result.beta, result.gamma)
        print(
            f"||R-Rrec||_F = {result.err_F:.3e},  ||R-Rrec||_inf = {result.err_inf:.3e}"
        )
        sp.pprint(result.delta)  # type: ignore[operator]
    return 0


def cmd_fk_dh(_args: argparse.Namespace) -> int:
    _T, _th, params = demo_standard_6R()
    print("a:", params["a"])
    print("alpha:", params["alpha"])
    print("d:", params["d"])
    print("theta (symbolic):", params["theta"])
    return 0


def register_subparsers(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    fk = subparsers.add_parser("fk", help="forward kinematics utilities")
    fk_sub = fk.add_subparsers(dest="fk_command", required=True)

    fk_symbolic = fk_sub.add_parser("symbolic", help="show symbolic T06 or stepwise Ti")
    fk_symbolic.add_argument("--steps", action="store_true", help="show T01..T06")
    fk_symbolic.add_argument(
        "--no-eval", dest="eval", action="store_false", help="do not eval at rest"
    )
    fk_symbolic.add_argument(
        "--euler",
        action="store_true",
        help="print symbolic XY'Z' angles",
    )
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
