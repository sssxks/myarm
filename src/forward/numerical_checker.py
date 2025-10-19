import numpy as np
import sympy as sp
import math

from forward.solver import Rx, Ry, Rz, T_to_euler_xy_dash_z

def R3(M):
    """Ensure 3x3 rotation block."""
    return M[:3, :3] if M.shape == (4, 4) else M

def mat_to_np(M):
    return np.array(M.tolist(), dtype=float)

def check_numeric_once(T06, subs_map):
    # 1) rotation block from T06
    R = R3(T06)
    R_num = sp.N(R.subs(subs_map), 15)

    # 2) XY'Z' euler using numeric-safe path (handles gimbal lock)
    T_num = sp.N(T06.subs(subs_map), 15)
    a_num, b_num, g_num = T_to_euler_xy_dash_z(T_num, safe=True)

    # 3) reconstruct Rrec numerically (consistent 3x3)
    Rrec_num = sp.N(R3(Rx(a_num)) * R3(Ry(b_num)) * R3(Rz(g_num)), 15)

    # 4) error metrics
    Delta = sp.N(R_num - Rrec_num, 15)
    # Safely chop tiny numerical noise while avoiding NaN comparisons
    def _chop_if_small(x):
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
    err_F = float(np.linalg.norm(d_np, ord='fro'))
    err_inf = float(np.max(np.abs(d_np)))

    print("XY'Z' (rad):", float(a_num), float(b_num), float(g_num))
    print(f"||R-Rrec||_F = {err_F:.3e},  ||R-Rrec||_inf = {err_inf:.3e}")
    sp.pprint(Delta_chopped)
    return err_F, err_inf