"""Trajectory generators for the velocity-control demos."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from myarm.core.orientation import rotation_xy_dash_z_numeric

Vector3 = NDArray[np.float64]
Matrix33 = NDArray[np.float64]

__all__ = [
    "TrajState",
    "square_traj",
    "circle_traj",
    "cone_traj",
    "build_traj_fn",
]


@dataclass
class TrajState:
    pos_mm: Vector3
    rot_des: Matrix33
    vel_mm_s: Vector3
    omega_rad_s: Vector3


def _orthonormal_basis_from_axis(axis: Vector3) -> tuple[Vector3, Vector3, Vector3]:
    """Return (n_hat, u, v) where u, v are orthonormal and ⟂ n_hat."""
    n_norm = float(np.linalg.norm(axis))
    if n_norm <= 1e-9:
        raise ValueError("spin axis must be non-zero")
    n_hat = axis / n_norm
    helper = np.array([0.0, 0.0, 1.0], dtype=float)
    if abs(np.dot(helper, n_hat)) > 0.99:
        helper = np.array([0.0, 1.0, 0.0], dtype=float)
    u = np.cross(helper, n_hat).astype(np.float64)
    u /= np.linalg.norm(u)
    v = np.cross(n_hat, u).astype(np.float64)
    return n_hat, u, v


def square_traj(center_mm: Vector3, side_mm: float, speed_m_s: float, t: float) -> TrajState:
    """Axis-aligned square in X-Y, fixed Z."""
    side = side_mm
    v = speed_m_s * 1000.0  # convert to mm/s
    if v <= 1e-6:
        raise ValueError("speed too small")

    t_edge = side / v
    total_t = 4 * t_edge
    tau = t % total_t
    edge = int(tau // t_edge)
    local_t = tau - edge * t_edge

    ramp_frac = 0.12
    ramp_t = ramp_frac * t_edge
    if local_t < ramp_t:
        speed_scale = local_t / ramp_t
    elif local_t > t_edge - ramp_t:
        speed_scale = (t_edge - local_t) / ramp_t
    else:
        speed_scale = 1.0
    speed_scale = max(0.1, speed_scale)

    start_xy = np.array(
        [
            [-0.5, -0.5],
            [0.5, -0.5],
            [0.5, 0.5],
            [-0.5, 0.5],
        ]
    ) * side
    dir_xy = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
            [0.0, -1.0],
        ]
    )
    p_xy = start_xy[edge] + dir_xy[edge] * v * local_t
    pos = center_mm.copy()
    pos[0] += p_xy[0]
    pos[1] += p_xy[1]
    vel = np.zeros(3, dtype=float)
    vel[0:2] = dir_xy[edge] * v * speed_scale
    rot = rotation_xy_dash_z_numeric(math.pi, 0.0, 0.0)  # tool facing down
    return TrajState(pos_mm=pos, rot_des=rot, vel_mm_s=vel, omega_rad_s=np.zeros(3, dtype=float))


def circle_traj(center_mm: Vector3, radius_mm: float, speed_m_s: float, t: float) -> TrajState:
    r = radius_mm
    v = speed_m_s * 1000.0
    if r <= 1e-6 or v <= 1e-6:
        raise ValueError("radius/speed too small")
    omega = v / r  # rad/s
    theta = omega * t
    pos = center_mm.copy()
    pos[0] += r * math.cos(theta)
    pos[1] += r * math.sin(theta)
    vel = np.array([-v * math.sin(theta), v * math.cos(theta), 0.0], dtype=float)
    rot = rotation_xy_dash_z_numeric(math.pi, 0.0, 0.0)
    return TrajState(pos_mm=pos, rot_des=rot, vel_mm_s=vel, omega_rad_s=np.zeros(3, dtype=float))


def cone_traj(
    apex_mm: Vector3,
    half_angle_deg: float,
    ang_speed: float,
    t: float,
    spin_axis: Vector3 | None = None,
) -> TrajState:
    """End-effector orientation traces a cone about the given world-frame spin axis."""
    theta = math.radians(half_angle_deg)
    phi = ang_speed * t

    n_hat, u, v = _orthonormal_basis_from_axis(spin_axis if spin_axis is not None else np.array([0.0, 0.0, 1.0]))

    z_tool = math.cos(theta) * n_hat + math.sin(theta) * (math.cos(phi) * u + math.sin(phi) * v)

    x_tool = np.cross(z_tool, n_hat)
    norm_x = float(np.linalg.norm(x_tool))
    if norm_x <= 1e-9:
        x_tool = u
    else:
        x_tool /= norm_x
    y_tool = np.cross(z_tool, x_tool)

    rot = np.column_stack((x_tool, y_tool, z_tool))
    omega = ang_speed * n_hat
    pos = apex_mm
    vel = np.zeros(3, dtype=float)
    return TrajState(pos_mm=pos, rot_des=rot, vel_mm_s=vel, omega_rad_s=omega)


def build_traj_fn(args: argparse.Namespace, center_mm: Vector3) -> Callable[[float], TrajState]:
    """Choose a trajectory Callable based on CLI arguments."""
    if args.traj == "square":
        side_mm = args.side * 1000.0
        return lambda t: square_traj(center_mm.copy(), side_mm, args.speed, t)
    if args.traj == "circle":
        radius_mm = args.radius * 1000.0
        return lambda t: circle_traj(center_mm.copy(), radius_mm, args.speed, t)
    half_angle = args.cone_angle / 2.0
    spin_axis = np.array(args.axis, dtype=float)
    return lambda t: cone_traj(center_mm.copy(), half_angle, args.ang_speed, t, spin_axis=spin_axis)
