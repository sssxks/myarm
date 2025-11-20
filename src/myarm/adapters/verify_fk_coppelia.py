from __future__ import annotations

import argparse
import math
import time
from collections.abc import Sequence
from typing import cast

import numpy as np
import sympy as sp

from myarm.adapters.coppelia_utils import (
    DEFAULT_JOINT_NAMES,
    DEFAULT_TIP_NAME,
    connect_coppelia,
    get_matrix4,
    get_object_handle,
    rotation_angle_deg,
    set_joint_positions,
    translation_error,
)
from myarm.model.dh_params import demo_standard_6R
from myarm.model.presets import PRESETS_DEG


def build_T06_symbolic() -> tuple[sp.Matrix, list[sp.Symbol]]:
    """Return symbolic FK and corresponding joint symbols."""
    T06, th_syms, _ = demo_standard_6R()
    return T06, th_syms


def eval_T_numeric(
    T_sym: sp.Matrix,
    theta_syms: Sequence[sp.Symbol],
    q: Sequence[float],
) -> sp.Matrix:
    """Evaluate a symbolic transform at numeric joint angles."""
    substitution = {s: float(v) for s, v in zip(theta_syms, q)}
    # SymPy typing: cast result to Matrix for mypy.
    return cast(
        sp.Matrix,
        sp.N(T_sym.subs(substitution), 15),  # type: ignore[no-untyped-call]
    )


def spT_to_np(T: sp.Matrix, unit_scale: float = 1.0) -> np.ndarray:
    """Convert a SymPy matrix to NumPy, scaling translation components."""
    matrix = np.array(T.tolist(), dtype=np.float64)  # type: ignore[no-untyped-call]
    matrix[:3, 3] = matrix[:3, 3] * float(unit_scale)
    return matrix


def load_q_from_csv(path: str, deg: bool) -> list[list[float]]:
    """Load joint rows from a CSV/space-separated text file."""
    data: list[list[float]] = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            values = [float(x) for x in stripped.replace(",", " ").split()]
            if len(values) != 6:
                raise ValueError("CSV lines must have 6 values")
            row = [math.radians(x) for x in values] if deg else values
            data.append(row)
    return data


def configure_verify_fk_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Attach CLI options shared with the CLI entry point."""
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
    parser.add_argument("--mode", choices=["position", "target"], default="position")
    parser.add_argument("--sleep", type=float, default=0.05)
    parser.add_argument("--unit-scale", type=float, default=0.001)
    parser.add_argument("--tol-pos", type=float, default=1e-3)
    parser.add_argument("--tol-rot", type=float, default=0.5)
    parser.add_argument("--deg", action="store_true")
    parser.add_argument("--q", nargs=6, action="append", type=float, default=None)
    parser.add_argument("--csv", default=None)
    return parser

def _collect_joint_sets(args: argparse.Namespace) -> list[list[float]]:
    if args.csv:
        return load_q_from_csv(args.csv, args.deg)
    if args.q:
        return [
            [math.radians(v) for v in row] if args.deg else list(row)
            for row in args.q
        ]
    return [[math.radians(v) for v in row] for row in PRESETS_DEG]


def run_verify_fk(args: argparse.Namespace) -> int:
    joint_sets = _collect_joint_sets(args)
    T06_sym, theta_syms = build_T06_symbolic()
    # for some reason, we can not call .close() on client. so just ignore that
    _, sim = connect_coppelia(args.host, args.port)

    joint_names = list(args.joints) if args.joints is not None else list(DEFAULT_JOINT_NAMES)
    if len(joint_names) != 6:
        raise SystemExit(f"Expected 6 joints, got {len(joint_names)}")
    joints = [get_object_handle(sim, name) for name in joint_names]
    tip = get_object_handle(sim, args.tip)
    base = get_object_handle(sim, args.base) if args.base else None

    print("Verifying (pos m, rot deg)…")
    num_pass = 0
    for index, q in enumerate(joint_sets, start=1):
        set_joint_positions(sim, joints, q, mode=args.mode)
        time.sleep(max(args.sleep, 0.0))

        matrix_sim = get_matrix4(sim, tip, base)
        T_fk = eval_T_numeric(T06_sym, theta_syms, q)
        matrix_fk = spT_to_np(T_fk, unit_scale=args.unit_scale)

        pos_err = translation_error(matrix_fk, matrix_sim)
        rot_err = rotation_angle_deg(matrix_fk[:3, :3], matrix_sim[:3, :3])
        passed = (pos_err <= args.tol_pos) and (rot_err <= args.tol_rot)

        print(f"\nCase {index}: {'PASS' if passed else 'FAIL'}")
        print("  q (rad):", [round(value, 4) for value in q])
        print(f"  pos_err: {pos_err * 1000:.3f} mm ({pos_err:.6f} m)")
        print(f"  rot_err: {rot_err:.3f} deg")
        if passed:
            num_pass += 1
        np.set_printoptions(precision=4, suppress=True)
        print("  M_fk:\n", matrix_fk)
        print("  M_sim:\n", matrix_sim)

    print(
        f"\nSummary: {num_pass}/{len(joint_sets)} passed "
        f"(tol_pos={args.tol_pos} m, tol_rot={args.tol_rot} deg)"
    )
    return 0
