"""
Common helpers for interacting with CoppeliaSim via the ZMQ remote API.

These utilities are intentionally lightweight so that both FK and IK
verification scripts can reuse a single, well-documented set of helpers.
"""

from __future__ import annotations

from typing import Any
from collections.abc import Iterable, Sequence

import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

# Default object names used across the project CLI utilities.
DEFAULT_JOINT_NAMES = [f"/Robot/Joint{i}" for i in range(1, 7)]
DEFAULT_TIP_NAME = "/Robot/SuctionCup/SuctionCup_connect"


def connect_coppelia(host: str, port: int) -> tuple[RemoteAPIClient, Any]:
    """Connect to CoppeliaSim and return (client, sim module)."""
    if RemoteAPIClient is None:  # pragma: no cover - defensive guard
        raise RuntimeError("Install coppeliasim-zmqremoteapi-client")
    client = RemoteAPIClient(host=host, port=port)
    sim_get = getattr(client, "getObject", None)
    sim = client.getObject("sim") if callable(sim_get) else client.require("sim")
    return client, sim


def get_object_handle(sim: Any, name: str) -> int:
    """Return an integer handle for the given object name."""
    return int(sim.getObject(name))


def get_matrix4(sim: Any, obj: int, rel: int | None = None) -> np.ndarray:
    """Fetch a 4x4 transform matrix for `obj` relative to `rel` (-1 means world)."""
    relh = rel if rel is not None else -1
    matrix = sim.getObjectMatrix(obj, relh)
    M = np.eye(4, dtype=float)
    M[:3, :4] = np.array(matrix, dtype=float).reshape(3, 4)
    return M


def get_joint_positions(sim: Any, joints: Sequence[int]) -> list[float]:
    """Return the current joint angles for each handle in `joints`."""
    return [float(sim.getJointPosition(h)) for h in joints]


def set_joint_positions(
    sim: Any,
    joints: Iterable[int],
    q: Sequence[float],
    *,
    mode: str = "position",
) -> None:
    """Set joint angles in either position or target mode."""
    if mode not in {"position", "target"}:
        raise ValueError("mode must be 'position' or 'target'")
    setter = (
        sim.setJointPosition if mode == "position" else sim.setJointTargetPosition
    )
    for handle, value in zip(joints, q):
        setter(handle, float(value))


def rotation_angle_deg(RA: np.ndarray, RB: np.ndarray) -> float:
    """Return the rotation error in degrees between two 3×3 rotation blocks."""
    R = RA.T @ RB
    trace_term = float(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(trace_term)))


def rotation_angle_rad(RA: np.ndarray, RB: np.ndarray) -> float:
    """Return the rotation error in radians between two 3×3 rotation blocks."""
    R = RA.T @ RB
    trace_term = float(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.arccos(trace_term))


def translation_error(MA: np.ndarray, MB: np.ndarray) -> float:
    """Return positional error (Euclidean distance of translation components)."""
    return float(np.linalg.norm(MA[:3, 3] - MB[:3, 3]))
