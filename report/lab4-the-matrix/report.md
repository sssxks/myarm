
实验 4：几何雅可比

1. 公式梳理
   - 标准 DH 链式变换
     $$
       T_i = R_z(\theta_i)\,T_z(d_i)\,T_x(a_i)\,R_x(\alpha_i),\qquad
       T_{0n} = \prod_{i=1}^n T_i .
     $$
     通过
     $$
       \mathbf{o}_i = T_{0i}\begin{bmatrix}0\\0\\0\\1\end{bmatrix},\qquad
       \mathbf{z}_i = T_{0i}\begin{bmatrix}0\\0\\1\\0\end{bmatrix}
     $$
     得到每个关节的原点与旋转轴。
   - 第 $i$ 个旋转关节的线速度和角速度列向量
     $$
       \mathbf{J}_{v,i} = \mathbf{z}_i \times (\mathbf{o}_n - \mathbf{o}_i),\qquad
       \mathbf{J}_{\omega,i} = \mathbf{z}_i .
     $$
     几何雅可比矩阵
     $$
       J =
       \begin{bmatrix}
         \mathbf{J}_{v,1} & \cdots & \mathbf{J}_{v,6} \\
         \mathbf{J}_{\omega,1} & \cdots & \mathbf{J}_{\omega,6}
       \end{bmatrix} \in \mathbb{R}^{6\times6}.
     $$
   - 线速度块的奇异指标可用 $\det J_v$；整体退化满足 $\operatorname{rank}(J)<6$，常出现在腕部共点构型附近。

2. CAS 计算代码（SymPy）

jacobian.py
```python
import sympy as sp

from myarm.model.dh_params import demo_standard_6R
from myarm.solvers.fk_solver import Rx, Tx, Tz, Rz


def _forward_chain_symbolic(dh):
    transforms = [sp.eye(4)]
    T = transforms[0]
    for ai, alpha_i, di, theta_i in zip(dh.params["a"], dh.params["alpha"], dh.params["d"], dh.params["theta"]):
        Ti = Rz(theta_i) * Tz(di) * Tx(ai) * Rx(alpha_i)
        T = T * Ti
        transforms.append(T)
    return transforms


def symbolic_geometric_jacobian():
    dh = demo_standard_6R()
    chain = _forward_chain_symbolic(dh)
    origins = [sp.Matrix(T[:3, 3]) for T in chain]
    z_axes = [sp.Matrix(T[:3, 2]) for T in chain]

    on = origins[-1]
    Jv_cols, Jw_cols = [], []
    for oi, zi in zip(origins[:-1], z_axes[:-1]):
        Jv_cols.append(zi.cross(on - oi))
        Jw_cols.append(zi)

    Jv = sp.Matrix.hstack(*Jv_cols)
    Jw = sp.Matrix.hstack(*Jw_cols)
    return sp.Matrix.vstack(Jv, Jw)
```

3. 数值验证脚本

```python
from math import radians
import numpy as np
from myarm.model.jacobian import evaluate_numeric_geometric_jacobian

q_deg = [0, -20, 35, 0, 45, 0]
J = evaluate_numeric_geometric_jacobian([radians(v) for v in q_deg])
np.set_printoptions(precision=5, suppress=True)
print(J)                   # 与 sympy 数值化结果应当一致
```

4. CLI 运行结果（joint angles = `[0, -20, 35, 0, 45, 0]`，单位 °）

```bash
> uv.exe run myarm -- jacobian symbolic --q 0 -20 35 0 45 0 --deg --digits 5
Full geometric Jacobian (substituted)
⎡-83.458  396.78   222.94   58.729   -58.398     0    ⎤
⎢                                                     ⎥
⎢59.052      0        0        0     60.458      0    ⎥
⎢                                                     ⎥
⎢   0     -59.052  -122.33  -78.327  15.648      0    ⎥
⎢                                                     ⎥
⎢   0        0        0        0     0.25882  0.68301 ⎥
⎢                                                     ⎥
⎢   0       1.0      1.0      1.0       0     0.70711 ⎥
⎢                                                     ⎥
⎣  1.0       0        0        0     0.96593  -0.18301⎦
```

```bash
> uv.exe run myarm -- jacobian numeric --q 0 -20 35 0 45 0 --deg
Full geometric Jacobian
[[ -83.45763  396.77923  222.93609   58.7287   -58.39759    0.     ]
 [  59.05216    0.         0.         0.        60.45763   -0.     ]
 [   0.       -59.05216 -122.32589  -78.32665   15.64759    0.     ]
 [   0.         0.         0.         0.         0.25882    0.68301]
 [   0.         1.         1.         1.         0.         0.70711]
 [   1.         0.         0.         0.         0.96593   -0.18301]]
```

复现/使用方法见`README.md`， `uv.exe run myarm jacobian symbolic` 不带参数可以输出公式
