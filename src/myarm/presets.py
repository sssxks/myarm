"""Shared joint angle presets used across CLI helpers."""

from __future__ import annotations

JointPreset = list[float]

# Stored in degrees to match assignment handout; convert to radians where needed.
PRESETS_DEG: list[JointPreset] = [
    [30, 0, 90, 60, 0, 0],
    [-30, 0, 60, 60, -30, 0],
    [30, 0, 60, 60, 60, 0],
    [-30, 0, 60, 60, 15, 0],
    [15, 15, 15, 15, 15, 15],
]
