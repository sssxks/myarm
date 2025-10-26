"""Command‑line utilities for symbolic and numeric FK checks.

Subcommands
-----------
- symbolic: show T06 (or stepwise Ti) and optionally evaluate at a joint set
- eval:     evaluate T06 and XY'Z' for given joint angles
- random:   run numeric reconstruction checks for N random joint sets
- dh:       print the DH parameter lists used

Examples
--------
- Show symbolic T06:
    uv run forward -- symbolic

- Evaluate the 5 preset sets (degrees):
    uv run forward -- eval --preset 1 2 3 4 5 --deg

- Run 5 random checks avoiding gimbal lock:
    uv run forward -- random --count 5 --seed 0
"""

from __future__ import annotations

import argparse
import math
import random
from typing import List, Sequence

import sympy as sp

from forward.solver import (
    T_to_euler_xy_dash_z,
    demo_standard_6R,
)
from forward.numerical_checker import check_numeric_once


def _pprint_matrix(M: sp.Matrix) -> None:
    sp.pprint(M, use_unicode=True)  # type: ignore[operator]


def _deg(v: float) -> float:
    return math.degrees(v)


def _rad(v: float) -> float:
    return math.radians(v)


def _parse_qs(q_list: List[List[float]], deg: bool) -> List[List[float]]:
    if deg:
        return [[_rad(x) for x in row] for row in q_list]
    return [list(row) for row in q_list]


PRESETS_DEG = [
    [30, 0, 90, 60, 0, 0],
    [-30, 0, 60, 60, -30, 0],
    [30, 0, 60, 60, 60, 0],
    [-30, 0, 60, 60, 15, 0],
    [15, 15, 15, 15, 15, 15],
]


def cmd_symbolic(args: argparse.Namespace) -> int:
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
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    T06, th_syms, _ = demo_standard_6R()

    qs: List[List[float]] = []
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


def cmd_random(args: argparse.Namespace) -> int:
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


def cmd_dh(_args: argparse.Namespace) -> int:
    _T, _th, params = demo_standard_6R()
    print("a:", params["a"])
    print("alpha:", params["alpha"])
    print("d:", params["d"])
    print("theta (symbolic):", params["theta"])
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="forward", description="Forward kinematics CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("symbolic", help="show symbolic T06 or stepwise Ti")
    s.add_argument("--steps", action="store_true", help="show T01..T06")
    s.add_argument(
        "--no-eval", dest="eval", action="store_false", help="do not eval at rest"
    )
    s.add_argument("--euler", action="store_true", help="print symbolic XY'Z' angles")
    s.set_defaults(func=cmd_symbolic, eval=True)

    e = sub.add_parser("eval", help="evaluate T06 and XY'Z' for angles")
    e.add_argument("--preset", type=int, nargs="*", help="use preset 1..5 (deg)")
    e.add_argument(
        "--q", nargs=6, type=float, action="append", help="custom angles q1..q6"
    )
    e.add_argument("--deg", action="store_true", help="interpret --q in degrees")
    e.set_defaults(func=cmd_eval)

    r = sub.add_parser("random", help="random numeric checks avoiding gimbal lock")
    r.add_argument("--count", type=int, default=5)
    r.add_argument("--seed", type=int, default=0)
    r.set_defaults(func=cmd_random)

    d = sub.add_parser("dh", help="print DH parameters")
    d.set_defaults(func=cmd_dh)
    return p


def fk_standard(
    a: Sequence[int | float | sp.Expr],
    alpha: Sequence[int | float | sp.Expr],
    d: Sequence[int | float | sp.Expr],
    theta: Sequence[int | float | sp.Expr],
) -> sp.Matrix:  # keep backward import compatibility if used elsewhere
    from forward.solver import fk_standard as _fk

    return _fk(a, alpha, d, theta)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
