import math
import numpy as np
from .type_utils import HALF_PI, Num
from .solver import fk_standard
import sympy as sp
from typing import NamedTuple

class DHParams(NamedTuple):
    T: sp.Matrix
    thetas: list[sp.Symbol]
    params: dict[str, list[Num]]

class DHParamsNum(NamedTuple):
    a: np.ndarray
    alpha: np.ndarray
    d: np.ndarray
    theta_offset: np.ndarray

def demo_standard_6R() -> DHParams:
    """Convenience constructor for the ZJU‑I 6‑DoF arm used in this project.

    Returns
    -------
    T06 : sympy.Matrix (4x4)
        Homogeneous transform from base to tool.
    theta_syms : list[sympy.Symbol]
        [th1..th6] joint symbols.
    params : dict
        Dict with keys ``a``, ``alpha``, ``d``, ``theta`` (the DH lists).
    """
    th1, th2, th3, th4, th5, th6 = sp.symbols("th1 th2 th3 th4 th5 th6", real=True)
    
    a: list[Num] = [0, 185, 170, 0, 0, 0]
    alpha: list[Num] = [-HALF_PI, 0, 0, HALF_PI, HALF_PI, 0]
    d: list[Num] = [230, -54, 0, 77, 77, 85.5]
    theta: list[Num] = [th1, th2-HALF_PI,th3,th4+HALF_PI,th5+HALF_PI,th6]
    
    T06 = fk_standard(a, alpha, d, theta)
    
    return DHParams(
        T06,
        [th1, th2, th3, th4, th5, th6],
        {"a": a, "alpha": alpha, "d": d, "theta": theta},
    )

def demo_standard_6R_num():
    # Units: millimeters for a/d, radians for angles

    a = np.array([0.0, 185.0, 170.0, 0.0, 0.0, 0.0], dtype=float)
    alpha = np.array(
        [-math.pi / 2, 0.0, 0.0, math.pi / 2, math.pi / 2, 0.0], dtype=float
    )
    d = np.array([230.0, -54.0, 0.0, 77.0, 77.0, 85.5], dtype=float)
    theta_offset = np.array([0.0, -math.pi / 2, 0.0, math.pi / 2, math.pi / 2, 0.0])

    return DHParamsNum(a=a, alpha=alpha, d=d, theta_offset=theta_offset)
    
