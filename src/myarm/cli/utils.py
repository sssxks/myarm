"""Shared helpers for CLI modules."""

from __future__ import annotations

import sympy as sp


def pprint_matrix(matrix: sp.Matrix) -> None:
    sp.pprint(matrix, use_unicode=True)  # type: ignore[operator]
