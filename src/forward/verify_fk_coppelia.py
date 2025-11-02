from __future__ import annotations

import argparse
import math
import time
from collections.abc import Sequence
from typing import Any, cast

import numpy as np
import sympy as sp
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from forward.dh_params import demo_standard_6R


def build_T06_symbolic() -> tuple[sp.Matrix, list[sp.Symbol]]:
    T06, th_syms, _ = demo_standard_6R()
    return T06, th_syms


def eval_T_numeric(
    T_sym: sp.Matrix, theta_syms: Sequence[sp.Symbol], q: Sequence[float]
) -> sp.Matrix:
    # SymPy typing: cast result to Matrix for mypy.
    return cast(
        sp.Matrix,
        sp.N(  # type: ignore[no-untyped-call]
            T_sym.subs({s: float(v) for s, v in zip(theta_syms, q)}), 15  # type: ignore[no-untyped-call]
        ),
    )


def spT_to_np(T: sp.Matrix, unit_scale: float = 1.0) -> np.ndarray:
    lst = T.tolist()  # type: ignore[no-untyped-call]
    M = np.array(lst, dtype=np.float64)
    M[:3, 3] = M[:3, 3] * float(unit_scale)
    return M


def rotation_angle_error_deg(RA: np.ndarray, RB: np.ndarray) -> float:
    R = RA.T @ RB
    tr = float(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0))
    return math.degrees(math.acos(tr))


def pos_error(MA: np.ndarray, MB: np.ndarray) -> float:
    return float(np.linalg.norm(MA[:3, 3] - MB[:3, 3]))


def connect_coppelia(host: str, port: int) -> tuple[RemoteAPIClient, Any]:
    if RemoteAPIClient is None:
        raise RuntimeError("Install coppeliasim-zmqremoteapi (or zmqRemoteApi)")
    client = RemoteAPIClient(host=host, port=port)
    sim_get = getattr(client, "getObject", None)
    sim = client.getObject("sim") if callable(sim_get) else client.require("sim")
    return client, sim


def get_handle(sim: Any, name: str) -> int:
    return int(sim.getObject(name))


def get_matrix4(sim: Any, obj: int, rel: int | None = None) -> np.ndarray:
    relh = rel if rel is not None else -1
    m = sim.getObjectMatrix(obj, relh)
    M = np.eye(4, dtype=float)
    M[:3, :4] = np.array(m, dtype=float).reshape(3, 4)
    return M


def set_joint_positions(
    sim: Any, joints: list[int], q: Sequence[float], mode: str = "position"
) -> None:
    if mode not in ("position", "target"):
        raise ValueError("mode must be position|target")
    for h, v in zip(joints, q):
        (sim.setJointPosition if mode == "position" else sim.setJointTargetPosition)(
            h, float(v)
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Verify FK against CoppeliaSim via ZMQ")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=23000)
    p.add_argument(
        "--joint",
        dest="joints",
        action="append",
        default=[f"/Robot/Joint{i}" for i in range(1, 7)],
    )
    p.add_argument("--tip", default="/Robot/SuctionCup/SuctionCup_connect")
    p.add_argument("--base", default=None)
    p.add_argument("--mode", choices=["position", "target"], default="position")
    p.add_argument("--sleep", type=float, default=0.05)
    p.add_argument("--unit-scale", type=float, default=0.001)
    p.add_argument("--tol-pos", type=float, default=1e-3)
    p.add_argument("--tol-rot", type=float, default=0.5)
    p.add_argument("--deg", action="store_true")
    p.add_argument("--q", nargs=6, action="append", type=float, default=None)
    p.add_argument("--csv", default=None)
    return p.parse_args()


def load_q_from_csv(path: str, deg: bool) -> list[list[float]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [float(x) for x in line.replace(",", " ").split()]
            if len(parts) != 6:
                raise ValueError("CSV lines must have 6 values")
            data.append([math.radians(x) for x in parts] if deg else parts)
    return data


def main() -> int:
    args = parse_args()
    Q = []
    if args.csv:
        Q = load_q_from_csv(args.csv, args.deg)
    elif args.q:
        for row in args.q:
            Q.append([math.radians(v) for v in row] if args.deg else list(row))
    else:
        # Five test sets from the assignment (angles in degrees):
        # 1) (π/6,   0,   π/2, π/3,   0,    0)
        # 2) (-π/6,  0,   π/3, π/3,  -π/6,  0)
        # 3) (π/6,   0,   π/3, π/3,  π/3,   0)
        # 4) (-π/6,  0,   π/3, π/3,  π/12,  0)
        # 5) (π/12, π/12, π/12, π/12, π/12, π/12)
        defaults_deg = [
            [30, 0, 90, 60, 0, 0],
            [-30, 0, 60, 60, -30, 0],
            [30, 0, 60, 60, 60, 0],
            [-30, 0, 60, 60, 15, 0],
            [15, 15, 15, 15, 15, 15],
        ]
        Q = [[math.radians(v) for v in r] for r in defaults_deg]

    T06_sym, th_syms = build_T06_symbolic()
    _, sim = connect_coppelia(args.host, args.port)
    if len(args.joints) != 6:
        raise SystemExit(f"Expected 6 joints, got {len(args.joints)}")
    joints = [get_handle(sim, n) for n in args.joints]
    tip = get_handle(sim, args.tip)
    base = get_handle(sim, args.base) if args.base else None

    print("Verifying (pos m, rot deg)…")
    n_pass = 0
    for i, q in enumerate(Q):
        set_joint_positions(sim, joints, q, mode=args.mode)
        time.sleep(max(args.sleep, 0.0))
        M_sim = get_matrix4(sim, tip, base)
        T_fk = eval_T_numeric(T06_sym, th_syms, q)
        M_fk = spT_to_np(T_fk, unit_scale=args.unit_scale)
        pe = pos_error(M_fk, M_sim)
        re = rotation_angle_error_deg(M_fk[:3, :3], M_sim[:3, :3])
        ok = (pe <= args.tol_pos) and (re <= args.tol_rot)
        print(f"\nCase {i + 1}: {'PASS' if ok else 'FAIL'}")
        print("  q (rad):", [round(v, 4) for v in q])
        print(f"  pos_err: {pe * 1000:.3f} mm ({pe:.6f} m)")
        print(f"  rot_err: {re:.3f} deg")
        if ok:
            n_pass += 1
        np.set_printoptions(precision=4, suppress=True)
        print("  M_fk:\n", M_fk)
        print("  M_sim:\n", M_sim)
    print(
        f"\nSummary: {n_pass}/{len(Q)} passed (tol_pos={args.tol_pos} m, tol_rot={args.tol_rot} deg)"
    )
    return 0
