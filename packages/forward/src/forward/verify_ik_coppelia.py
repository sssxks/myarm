from __future__ import annotations

import argparse
import math
import time
from typing import Any, List, Sequence, Tuple, cast

import numpy as np

from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from forward.ik_solver import IKOptions, fk_numeric, solve_ik


def connect_coppelia(host: str, port: int) -> tuple[RemoteAPIClient, Any]:
    if RemoteAPIClient is None:
        raise RuntimeError("Install coppeliasim-zmqremoteapi-client")
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


def get_joint_positions(sim: Any, joints: Sequence[int]) -> List[float]:
    return [float(sim.getJointPosition(h)) for h in joints]


def set_joint_positions(sim: Any, joints: Sequence[int], q: Sequence[float]) -> None:
    for h, v in zip(joints, q):
        sim.setJointPosition(h, float(v))


def rotation_angle_deg(RA: np.ndarray, RB: np.ndarray) -> float:
    R = RA.T @ RB
    tr = float(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0))
    return math.degrees(math.acos(tr))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Verify IK against CoppeliaSim via ZMQ")
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
    p.add_argument("--unit-scale", type=float, default=0.001, help="FK(mm)↔Sim(m)")
    p.add_argument("--tol-pos-mm", type=float, default=1e-2)
    p.add_argument("--tol-rot-deg", type=float, default=0.2)
    p.add_argument("--max-iter", type=int, default=200)
    p.add_argument("--lmbda", type=float, default=1e-3)
    p.add_argument("--sleep", type=float, default=0.05)
    p.add_argument("--apply", action="store_true", help="apply best IK q to sim")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    _, sim = connect_coppelia(args.host, args.port)
    if len(args.joints) != 6:
        raise SystemExit(f"Expected 6 joints, got {len(args.joints)}")
    joints = [get_handle(sim, n) for n in args.joints]
    tip = get_handle(sim, args.tip)
    base = get_handle(sim, args.base) if args.base else None

    # Read current sim pose and (optionally) current q as a good seed
    M_sim_m = get_matrix4(sim, tip, base)
    M_des = M_sim_m.copy()
    M_des[:3, 3] = M_des[:3, 3] / float(args.unit_scale)  # meters→millimeters
    q_sim = get_joint_positions(sim, joints)

    opts = IKOptions(
        max_iter=int(args.max_iter),
        lambda_dls=float(args.lmbda),
        tol_pos=float(args.tol_pos_mm),
        tol_rot=math.radians(float(args.tol_rot_deg)),
    )

    results = solve_ik(M_des, seeds=[q_sim], opts=opts)
    if not results:
        print("No solution found.")
        return 1

    # Re‑evaluate and pick best by errors; show a few
    print("Target M (meters from sim; mm internally for IK):")
    np.set_printoptions(precision=4, suppress=True)
    print(M_sim_m)

    best = None
    for i, (q, pe, re, it) in enumerate(results, 1):
        T_fk = fk_numeric(q)
        pe2 = float(np.linalg.norm(T_fk[:3, 3] - M_des[:3, 3]))
        re2 = rotation_angle_deg(T_fk[:3, :3], M_des[:3, :3])
        print(
            f"\nSol {i}: iters={it}, pos_err={pe2:.3e} mm, rot_err={re2:.3f} deg\n  q(rad)="
            f"{[round(float(v), 6) for v in q]}\n  q(deg)="
            f"{[round(math.degrees(float(v)), 3) for v in q]}"
        )
        if best is None or (pe2, re2) < (best[1], best[2]):
            best = (q, pe2, re2)

    assert best is not None
    ok = best[1] <= args.tol_pos_mm and best[2] <= args.tol_rot_deg
    print(
        f"\nBest: pos_err={best[1]:.3e} mm, rot_err={best[2]:.3f} deg → {'PASS' if ok else 'FAIL'}"
    )

    if args.apply:
        set_joint_positions(sim, joints, cast(Sequence[float], best[0]))
        time.sleep(max(0.0, float(args.sleep)))
        M_after = get_matrix4(sim, tip, base)
        print("\nApplied best solution. Sim tip now:")
        print(M_after)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
