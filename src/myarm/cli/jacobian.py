"""CLI wiring for Jacobian helpers."""

from __future__ import annotations

import argparse
from math import radians

import numpy as np
import sympy as sp

from myarm.model.jacobian import (
    JacobianSymbolic,
    evaluate_numeric_geometric_jacobian,
    symbolic_geometric_jacobian,
)


def cmd_jacobian_symbolic(args: argparse.Namespace) -> int:
    jac: JacobianSymbolic = symbolic_geometric_jacobian()
    if args.block == "linear":
        matrix = jac.J_linear
        label = "Linear block (Jv)"
    elif args.block == "angular":
        matrix = jac.J_angular
        label = "Angular block (Jω)"
    else:
        matrix = jac.geometric
        label = "Full geometric Jacobian"
    if args.q is not None:
        q_vals = [float(val) for val in args.q]
        if args.deg:
            q_vals = [radians(val) for val in q_vals]
        subs = {sym: val for sym, val in zip(jac.thetas, q_vals)}
        matrix = sp.N(matrix.subs(subs), int(args.digits))  # type: ignore[no-untyped-call]
    print(label + (" (substituted)" if args.q is not None else ""))
    sp.pprint(matrix, use_unicode=True)  # type: ignore[operator]
    return 0


def cmd_jacobian_numeric(args: argparse.Namespace) -> int:
    q_vals = [float(val) for val in args.q]
    if args.deg:
        q_vals = [radians(val) for val in q_vals]
    full = evaluate_numeric_geometric_jacobian(q_vals)
    if args.block == "linear":
        matrix = full[:3, :]
        label = "Linear block (Jv)"
    elif args.block == "angular":
        matrix = full[3:, :]
        label = "Angular block (Jω)"
    else:
        matrix = full
        label = "Full geometric Jacobian"
    np.set_printoptions(precision=int(args.digits), suppress=not args.scientific)
    print(label)
    print(matrix)
    return 0


def register_subparsers(subparsers: argparse._SubParsersAction) -> None:
    jacobian = subparsers.add_parser("jacobian", help="geometric Jacobian helpers")
    jac_sub = jacobian.add_subparsers(dest="jacobian_command", required=True)

    jac_symbolic = jac_sub.add_parser("symbolic", help="print symbolic Jacobian blocks")
    jac_symbolic.add_argument(
        "--block",
        choices=("full", "linear", "angular"),
        default="full",
        help="choose which block to print",
    )
    jac_symbolic.add_argument(
        "--q",
        nargs=6,
        type=float,
        help="optionally substitute q1..q6 (rad unless --deg)",
    )
    jac_symbolic.add_argument("--deg", action="store_true", help="interpret --q in degrees")
    jac_symbolic.add_argument("--digits", type=int, default=6, help="digits for numeric eval")
    jac_symbolic.set_defaults(func=cmd_jacobian_symbolic)

    jac_numeric = jac_sub.add_parser("numeric", help="evaluate numeric Jacobian at q")
    jac_numeric.add_argument(
        "--q",
        nargs=6,
        required=True,
        type=float,
        help="joint angles q1..q6 (rad unless --deg)",
    )
    jac_numeric.add_argument("--deg", action="store_true", help="interpret --q in degrees")
    jac_numeric.add_argument(
        "--block",
        choices=("full", "linear", "angular"),
        default="full",
        help="choose which block to print",
    )
    jac_numeric.add_argument("--digits", type=int, default=5, help="print precision")
    jac_numeric.add_argument(
        "--scientific",
        action="store_true",
        help="use scientific notation for numeric output",
    )
    jac_numeric.set_defaults(func=cmd_jacobian_numeric)
