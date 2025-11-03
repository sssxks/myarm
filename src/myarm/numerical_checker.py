from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, cast

import numpy as np
import sympy as sp

from myarm.orientation import rotation_xy_dash_z_symbolic
from myarm.fk_solver import T_to_euler_xy_dash_z


def R3(M: sp.Matrix) -> sp.Matrix:
    """Ensure 3x3 rotation block."""
    return cast(sp.Matrix, M[:3, :3]) if M.shape == (4, 4) else M


def mat_to_np(M: sp.Matrix) -> np.ndarray:
    lst = M.tolist()  # type: ignore[no-untyped-call]
    return np.array(lst, dtype=np.float64)


def check_numeric_once(T06: sp.Matrix, subs_map: Mapping[sp.Symbol, float]) -> tuple[float, float]:
    # 1) rotation block from T06
    R = R3(T06)
    R_num = sp.N(R.subs(subs_map), 15)  # type: ignore[no-untyped-call]

    # 2) XY'Z' euler using numeric-safe path (handles gimbal lock)
    T_num = sp.N(T06.subs(subs_map), 15)  # type: ignore[no-untyped-call]
    a_num, b_num, g_num = T_to_euler_xy_dash_z(T_num, safe=True)

    # 3) reconstruct Rrec numerically (consistent 3x3)
    Rrec_num = sp.N(rotation_xy_dash_z_symbolic(a_num, b_num, g_num), 15)  # type: ignore[no-untyped-call]

    # 4) error metrics
    Delta = sp.N(R_num - Rrec_num, 15)  # type: ignore[no-untyped-call]

    # Safely chop tiny numerical noise while avoiding NaN comparisons
    def _chop_if_small(x: Any) -> Any:
        try:
            xf = float(x)
            if math.isfinite(xf) and abs(xf) < 1e-10:
                return 0.0
        except Exception:
            pass
        return x

    Delta_chopped = Delta.applyfunc(_chop_if_small)

    # Frobenius/∞-norm
    d_np = mat_to_np(Delta)
    err_F = float(np.linalg.norm(d_np, ord="fro"))
    err_inf = float(np.max(np.abs(d_np)))

    print("XY'Z' (rad):", float(a_num), float(b_num), float(g_num))
    print(f"||R-Rrec||_F = {err_F:.3e},  ||R-Rrec||_inf = {err_inf:.3e}")
    sp.pprint(Delta_chopped)  # type: ignore[operator]
    return err_F, err_inf
