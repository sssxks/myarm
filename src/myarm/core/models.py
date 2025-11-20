"""Lightweight data models shared by CLI modules."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

Matrix33 = NDArray[np.float64]
Matrix44 = NDArray[np.float64]
Vector3 = NDArray[np.float64]


@dataclass(frozen=True)
class PoseTarget:
    matrix: Matrix44

    @property
    def position_mm(self) -> Vector3:
        return self.matrix[:3, 3]

    @property
    def rotation(self) -> Matrix33:
        return self.matrix[:3, :3]


@dataclass(frozen=True)
class JointAngles:
    radians: tuple[float, ...]

    @classmethod
    def from_sequence(cls, values: Sequence[float]) -> "JointAngles":
        if len(values) != 6:
            raise ValueError("Expected 6 joint angles")
        return cls(tuple(float(value) for value in values))

    def as_list(self) -> list[float]:
        return list(self.radians)

    def as_degrees(self) -> tuple[float, ...]:
        return tuple(math.degrees(value) for value in self.radians)
