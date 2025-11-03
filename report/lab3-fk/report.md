# 实验3：ZJU‑I 机械臂正运动学（FK）报告

作者：项科深 3230102841

日期：2025‑10‑19

## 1. 实验目标

1) 写出 ZJU‑I 型桌面机械臂的标准 DH 参数；
2) 建立符号化的正运动学模型，姿态采用内禀 XY′Z′ 欧拉角表示；
3) 将 5 组关节角代入，给出末端齐次变换矩阵与 XY′Z′ 欧拉角；
4) 通过 CoppeliaSim 仿真比对 FK 的位置/姿态误差；
5) 给出可复现实验的代码与命令。

## 2. DH 参数与建模方法

采用标准 DH（Denavit–Hartenberg）参数，角度以弧度计：

| Joint | a (mm) | alpha (rad) | d (mm) | theta |
|:-----:|:------:|:-----------:|:------:|:-----|
| 1 | 0   | -π/2 | 230   | th1 |
| 2 | 185 | 0    | -54   | th2 - π/2 |
| 3 | 170 | 0    | 0     | th3 |
| 4 | 0   | π/2  | 77    | th4 + π/2 |
| 5 | 0   | π/2  | 77    | th5 + π/2 |
| 6 | 0   | 0    | 85.5  | th6 |

齐次变换采用标准 DH 链式相乘：`T = Π (Rz(theta)·Tz(d)·Tx(a)·Rx(alpha))`。

## 3. 姿态表示（XY′Z′ 欧拉角）

使用内禀序列 X → Y′ → Z′（等价于外禀 Z‑Y‑X）。若 `R = Rx(α)·Ry(β)·Rz(γ)`，在非奇异 `|cos(β)|>0` 下：

- `β = asin(r13)`
- `α = atan2(-r23, r33)`
- `γ = atan2(-r12, r11)`

数值路径在 `β = ±π/2` 的万向节锁附近采取稳健处理（置 `γ=0` 并将“偏航”折入 `α`）。实现见 `T_to_euler_xy_dash_z(..., safe=True)`。

计算结果见附录

## 4. 计算结果（5 组关节）

输入的 5 组角度（度）：

1) (30, 0, 90, 60, 0, 0)
2) (-30, 0, 60, 60, -30, 0)
3) (30, 0, 60, 60, 60, 0)
4) (-30, 0, 60, 60, 15, 0)
5) (15, 15, 15, 15, 15, 15)

以组 1 为例，`T06`（单位：mm）：

```
[[-0.5      0.4330  -0.75   104.9413]
 [ 0.8660   0.25    -0.4330  87.1460]
 [ 0.      -0.8660  -0.5    305.5660]
 [ 0.       0.       0.       1.    ]]
```

对应 XY′Z′：`α, β, γ ≈ (139.1066°, -48.5904°, -139.1066°)`。

其余 4 组的结果（省略矩阵，仅列欧拉角，度）：

- 组 2：`(163.8979, -38.6822, -73.8979)`
- 组 3：`(-124.7150, -40.5054, -80.5377)`
- 组 4：`(-150.8986, -16.7890, -51.5722)`
- 组 5：`(-148.0010, 36.3526, -106.9990)`

完整矩阵与角度可通过 CLI 复现（见第 6 节）。

## 5. 仿真比对（CoppeliaSim）

通过 ZMQ Remote API 连接 CoppeliaSim，将关节角输入仿真，读取末端位姿与 FK 结果对比。记录见 `report/verification_results.log`，摘要：

- 5/5 组通过；位置误差 ≤ 0.020 mm；姿态误差 ≈ 0.000°；
- 采用 `--unit-scale 0.001`（FK 毫米 → 仿真米）。

## 6. 复现实验

推荐使用 `uv`：

```
uv sync
uv run forward -- symbolic --euler
uv run forward -- eval --preset 1 2 3 4 5 --deg
uv run forward -- random --count 5
```

若进行仿真比对，先启动 CoppeliaSim 的 ZMQ 服务器（默认 `127.0.0.1:23000`），再执行：

```
uv run verify_fk

# 或自定义角度：
uv run verify_fk --q 30 0 90 60 0 0 --deg
```

## 7. 结论与讨论

1) 以标准 DH 建模的符号 FK 在 5 组测试上与仿真对齐良好（位置 ~20µm 量级，姿态 ~0°）。
2) XY′Z′ 在 `β≈±π/2` 存在奇异性，数值实现已通过 `safe=True` 路径规避；实际应用建议远离奇异位形或切换姿态参数化。
3) 代码已封装为 CLI，便于课堂复现与批改。

## 8. 附录

代码路径：

- FK 与欧拉角：`src/forward/solver.py`
- CLI：`src/forward/main.py`（控制台脚本名：`forward`）
- 数值重构与误差：`src/forward/numerical_checker.py`
- 仿真比对：`src/forward/verify_fk_coppelia.py`（控制台脚本名：`verify_fk`）

完整正运动学解（可用`uv run forward symbolic`得到）

```
(forward) PS C:\Users\86151\Desktop\s\l4\forward> uv run forward symbolic
Symbolic T06:
⎡-(sin(th₁)⋅cos(th₅) + sin(th₅)⋅cos(th₁)⋅cos(th₂ + th₃ + th₄))⋅cos(th₆) + sin(th₆)⋅sin(th₂ + th₃ + th₄)⋅cos(th₁)  (sin(th₁)⋅cos(th₅) + sin(th₅)⋅cos(th₁)⋅cos(th₂ + th₃ + th₄))⋅sin( ↪
⎢                                                                                                                                                                                   ↪
⎢-(sin(th₁)⋅sin(th₅)⋅cos(th₂ + th₃ + th₄) - cos(th₁)⋅cos(th₅))⋅cos(th₆) + sin(th₁)⋅sin(th₆)⋅sin(th₂ + th₃ + th₄)  (sin(th₁)⋅sin(th₅)⋅cos(th₂ + th₃ + th₄) - cos(th₁)⋅cos(th₅))⋅sin( ↪
⎢                                                                                                                                                                                   ↪
⎢                    sin(th₅)⋅sin(th₂ + th₃ + th₄)⋅cos(th₆) + sin(th₆)⋅cos(th₂ + th₃ + th₄)                                          -sin(th₅)⋅sin(th₆)⋅sin(th₂ + th₃ + th₄) + cos( ↪
⎢                                                                                                                                                                                   ↪
⎣                                                       0                                                                                                               0           ↪

↪ th₆) + sin(th₂ + th₃ + th₄)⋅cos(th₁)⋅cos(th₆)  -sin(th₁)⋅sin(th₅) + cos(th₁)⋅cos(th₅)⋅cos(th₂ + th₃ + th₄)  -85.5⋅sin(th₁)⋅sin(th₅) - 23.0⋅sin(th₁) + 185.0⋅sin(th₂)⋅cos(th₁) + 1 ↪
↪                                                                                                                                                                                   ↪
↪ th₆) + sin(th₁)⋅sin(th₂ + th₃ + th₄)⋅cos(th₆)  sin(th₁)⋅cos(th₅)⋅cos(th₂ + th₃ + th₄) + sin(th₅)⋅cos(th₁)   185.0⋅sin(th₁)⋅sin(th₂) + 170.0⋅sin(th₁)⋅sin(th₂ + th₃) + 77.0⋅sin(th ↪
↪                                                                                                                                                                                   ↪
↪ th₆)⋅cos(th₂ + th₃ + th₄)                                    -sin(th₂ + th₃ + th₄)⋅cos(th₅)                                                   -85.5⋅sin(th₂ + th₃ + th₄)⋅cos(th₅) ↪
↪                                                                                                                                                                                   ↪
↪                                                                             0                                                                                                     ↪

↪ 70.0⋅sin(th₂ + th₃)⋅cos(th₁) + 77.0⋅sin(th₂ + th₃ + th₄)⋅cos(th₁) + 85.5⋅cos(th₁)⋅cos(th₅)⋅cos(th₂ + th₃ + th₄)⎤
↪                                                                                                                ⎥
↪ ₁)⋅sin(th₂ + th₃ + th₄) + 85.5⋅sin(th₁)⋅cos(th₅)⋅cos(th₂ + th₃ + th₄) + 85.5⋅sin(th₅)⋅cos(th₁) + 23.0⋅cos(th₁) ⎥
↪                                                                                                                ⎥
↪  + 185.0⋅cos(th₂) + 170.0⋅cos(th₂ + th₃) + 77.0⋅cos(th₂ + th₃ + th₄) + 230.0                                   ⎥
↪                                                                                                                ⎥
↪                     1                                                                                          ⎦

T06 at rest (rad=0):
⎡ 0    0   1.0  85.5 ⎤
⎢                    ⎥
⎢1.0   0    0   23.0 ⎥
⎢                    ⎥
⎢ 0   1.0   0   662.0⎥
⎢                    ⎥
⎣ 0    0    0    1.0 ⎦
```