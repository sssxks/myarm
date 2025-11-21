"""Symbolic homogeneous transform primitives for standard DH chains."""

from __future__ import annotations

import sympy as sp

from myarm.core.type_utils import Num


def Rx(alpha: Num) -> sp.Matrix:
    ca = sp.cos(alpha)
    sa = sp.sin(alpha)
    return sp.Matrix([[1, 0, 0, 0], [0, ca, -sa, 0], [0, sa, ca, 0], [0, 0, 0, 1]])


def Ry(beta: Num) -> sp.Matrix:
    cb = sp.cos(beta)
    sb = sp.sin(beta)
    return sp.Matrix([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])


def Rz(theta: Num) -> sp.Matrix:
    ct = sp.cos(theta)
    st = sp.sin(theta)
    return sp.Matrix([[ct, -st, 0, 0], [st, ct, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])


def Tx(a: Num) -> sp.Matrix:
    return sp.Matrix([[1, 0, 0, a], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])


def Tz(d: Num) -> sp.Matrix:
    return sp.Matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, d], [0, 0, 0, 1]])


__all__ = ["Rx", "Ry", "Rz", "Tx", "Tz"]
