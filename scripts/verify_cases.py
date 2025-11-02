from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from myarm.ik_solver import pose_from_xyz_euler, rotation_angle
from myarm.verify_ik_coppelia import connect_coppelia, get_handle, get_matrix4, set_joint_positions


@dataclass(frozen=True)
class Scenario:
    name: str
    xyz_m: tuple[float, float, float]
    euler: tuple[float, float, float]
    q_rad: tuple[float, float, float, float, float, float]


SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        name="Case 1",
        xyz_m=(0.117, 0.334, 0.499),
        euler=(-2.019, -0.058, -2.190),
        q_rad=(1.046916, 0.543234, 0.531454, -0.551512, 0.523909, 0.698541),
    ),
    Scenario(
        name="Case 2",
        xyz_m=(-0.066, 0.339, 0.444),
        euler=(-2.618, -0.524, -3.141),
        q_rad=(1.571526, 0.460517, 0.660834, -0.074512, 0.523635, 0.001322),
    ),
    Scenario(
        name="Case 3",
        xyz_m=(0.300, 0.250, 0.260),
        euler=(-2.640, 0.590, -2.350),
        q_rad=(-2.364871, -0.787570, -1.345606, 1.310991, 3.036375, 0.111535),
    ),
    Scenario(
        name="Case 4",
        xyz_m=(0.420, 0.000, 0.360),
        euler=(3.140, 1.000, -1.570),
        q_rad=(-0.065918, 0.827356, 0.824529, -1.080132, 0.054597, -0.036195),
    ),
    Scenario(
        name="Case 5",
        xyz_m=(0.320, -0.250, 0.160),
        euler=(3.000, 0.265, -0.840),
        q_rad=(-0.735652, 1.102509, 1.053207, -0.875362, 0.074856, -0.012805),
    ),
)

# CoppeliaSim defaults for this project
HOST = "127.0.0.1"
PORT = 23000
JOINT_PATHS = tuple(f"/Robot/Joint{i}" for i in range(1, 7))
TIP_PATH = "/Robot/SuctionCup/SuctionCup_connect"
BASE_PATH = None
UNIT_SCALE = 0.001  # mm <-> m

np.set_printoptions(precision=6, suppress=True)


def main() -> None:
    client, sim = connect_coppelia(HOST, PORT)
    joints = [get_handle(sim, path) for path in JOINT_PATHS]
    tip = get_handle(sim, TIP_PATH)
    base = get_handle(sim, BASE_PATH) if BASE_PATH else None

    print("Connected to CoppeliaSim via ZMQ.")
    for scenario in SCENARIOS:
        set_joint_positions(sim, joints, scenario.q_rad)
        time.sleep(0.3)

        M_sim = get_matrix4(sim, tip, base)
        M_mm = M_sim.copy()
        M_mm[:3, 3] = M_mm[:3, 3] / UNIT_SCALE

        target_mm = pose_from_xyz_euler(
            *(axis * 1000.0 for axis in scenario.xyz_m),
            *scenario.euler,
        )

        pos_err = float(np.linalg.norm(M_mm[:3, 3] - target_mm[:3, 3]))
        rot_err_deg = math.degrees(rotation_angle(M_mm[:3, :3], target_mm[:3, :3]))

        xyz_target = [round(float(target_mm[i, 3]), 3) for i in range(3)]
        xyz_sim = [round(float(M_mm[i, 3]), 3) for i in range(3)]

        print(f"\n{scenario.name}:")
        print("  Target XYZ (mm):", xyz_target)
        print("  Sim XYZ (mm):", xyz_sim)
        print(f"  Position error (mm): {pos_err:.6f}")
        print(f"  Rotation error (deg): {rot_err_deg:.6f}")

    print("\nVerification complete.")
    if hasattr(client, "close"):
        client.close()


if __name__ == "__main__":
    main()
