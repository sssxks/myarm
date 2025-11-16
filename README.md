# myarm Manipulator Toolkit

This project provides symbolic and numeric tooling for a 6‑DoF desktop arm (ZJU‑I) including forward and inverse kinematics, Euler angle helpers, and CoppeliaSim verification hooks. The command-line interface has been consolidated under a single `myarm` entry point.

## Quick Start

Prerequisites:
- Python ≥ 3.13
- [uv](https://github.com/astral-sh/uv) (recommended for reproducible environments). On Windows use `uv.exe`.

Create the environment and explore the CLI:

```ps1
uv.exe sync
uv.exe run myarm -- fk symbolic                              # show symbolic T06
uv.exe run myarm -- fk eval --preset 1 2 3 4 5 --deg         # evaluate presets (degrees)
uv.exe run myarm -- fk random --count 3                      # random numeric checks
uv.exe run myarm -- ik euler --target 0.117 0.334 0.499 -2.019 -0.058 -2.190 --pos-unit m
uv.exe run myarm -- ik solve --from-q 0 0 1.57 0 0 0
uv.exe run myarm -- jacobian symbolic --block linear          # print symbolic Jv
uv.exe run myarm -- jacobian numeric --q 0 0 0 0 0 0         # evaluate J(q)
```

### CLI Overview

All tooling is exposed through `uv(.exe) run myarm -- <group> <command>`. Key groups:

- `fk` – forward kinematics utilities
  - `symbolic`  Show symbolic `T06`; add `--steps` for `T01..T06`, `--euler` for symbolic XY'Z' Euler angles, `--no-eval` to skip evaluation at the rest pose.
  - `eval`      Evaluate `T06` and XY'Z' for provided joint sets via `--preset 1..5` or `--q q1..q6` (combine with `--deg`).
  - `random`    Run random numeric checks that reconstruct `R` from XY'Z' while avoiding gimbal lock.
  - `dh`        Print the DH parameter lists.
- `ik` – inverse kinematics helpers
  - `solve`     Damped-least-squares IK solver. Provide a target as `--T` (16 row-major values) or `--from-q` (6 joints). Optional seeds via repeatable `--seed`.
  - `euler`     Analytic + numeric IK for XYZ + intrinsic XY'Z' Euler targets. Accepts `--target x y z α β γ` with `--pos-unit m|mm`, `--deg`.
- `jacobian` – geometric Jacobian utilities
  - `symbolic`  Print the symbolic Jacobian. Use `--block full|linear|angular`, optional `--q` (with `--deg`) to substitute a configuration, and `--digits` to control numeric precision.
  - `numeric`   Evaluate the 6×6 Jacobian for a numeric joint vector via `--q`. Supports `--block`, `--deg`, `--digits`, and `--scientific` for formatting.
- `verify` – CoppeliaSim validation utilities
  - `fk`        Compare symbolic FK against the simulator using presets, `--q`, or `--csv` joint sets. Tolerances configurable via `--tol-pos` (m) and `--tol-rot` (deg).
  - `ik`        Capture the live simulator pose, solve IK, and optionally `--apply` the best solution back to CoppeliaSim. Control tolerances with `--tol-pos-mm` and `--tol-rot-deg`.

Run `uv(.exe) run myarm -- <group> <command> --help` for detailed arguments.

## CoppeliaSim Verification

The verification subcommands connect to CoppeliaSim through the ZMQ remote API and compare simulated results with the symbolic / numeric solvers.

1. Launch CoppeliaSim with the remote API server active (default `127.0.0.1:23000`). Ensure joint objects are named `/Robot/Joint1..6` and the tool frame matches `/Robot/SuctionCup/SuctionCup_connect` or pass custom paths.
2. From the repo root, run:

```ps1
uv.exe run myarm -- verify fk
# Provide custom joint sets (degrees)
uv.exe run myarm -- verify fk --q 30 0 90 60 0 0 --deg
# Verify IK using the current simulator pose and apply the best solution
uv.exe run myarm -- verify ik --apply
```

Common flags include `--host/--port`, repeatable `--joint`, `--tip`, `--base`, `--deg`, `--unit-scale`, `--tol-pos`, and `--tol-rot`. IK verification also supports `--tol-pos-mm`, `--tol-rot-deg`, `--max-iter`, and `--lmbda`. FK runs are logged under `report/lab3-fk/verification_results.log`; IK runs under `report/lab3-ik/verification_results.log`.

## Euler Convention (XY'Z')

We use intrinsic XY'Z' Euler angles (equivalently extrinsic Z‑Y‑X). For a rotation block `R`:

```
beta  = asin( r13 )
alpha = atan2(-r23, r33)
gamma = atan2(-r12, r11)
```

When `beta = ±π/2`, the CLI switches to a gimbal-lock-safe reconstruction by clamping `gamma` to zero and folding yaw into `alpha`.

## Project Layout

- `src/myarm/fk_solver.py`             Symbolic & numeric FK helpers (build `T06`, XY'Z' conversions)
- `src/myarm/ik_solver.py`             Damped least-squares IK core plus utilities (`pose_from_xyz_euler`)
- `src/myarm/orientation.py`           XY'Z' Euler helpers shared across modules
- `src/myarm/dh_params.py`             Demo DH definitions (ZJU-I arm)
- `src/myarm/coppelia_utils.py`        Thin wrapper over the CoppeliaSim ZMQ remote API
- `src/myarm/verify_fk_coppelia.py`    FK verification routines (reused by CLI)
- `src/myarm/verify_ik_coppelia.py`    IK verification routines (reused by CLI)
- `src/myarm/numerical_checker.py`     Numeric XY'Z' reconstruction checks
- `src/myarm/presets.py`               Joint angle presets (degrees)
- `src/myarm/main.py`                  Unified CLI entry point (`myarm`)
- `scripts/verify_cases.py`            Batch verification script for canned IK/FK scenarios
- `report/`                            Experiment logs and notes (`lab3-fk/`, `lab3-ik/`)
- `typings/`                           Third-party `.pyi` stubs (added to `mypy_path`)

## Reproducing the Experiment

1. Inspect symbolic FK: `uv.exe run myarm -- fk symbolic --euler`
2. Evaluate the five preset configurations: `uv.exe run myarm -- fk eval --preset 1 2 3 4 5 --deg`
3. Run numeric checks: `uv.exe run myarm -- fk random --count 5`
4. (Optional) Verify in CoppeliaSim: `uv.exe run myarm -- verify fk --deg`

All expected tolerances and outputs are documented in `report/lab3-fk/verification_results.log`.

- Solve lab3 IK poses directly from task-space targets: `uv.exe run myarm -- ik euler --target …`.
- Batch replay the assignment cases against CoppeliaSim: `uv.exe run python scripts/verify_cases.py` (results in `report/lab3-ik/verification_results.log`).
- Outputs and analysis live in `report/lab3-ik/report.md`.

## Development Notes

- Run static checks with `uv.exe run mypy . --strict` or `uv.exe run pyright`.
- DH parameters use millimeters. CoppeliaSim verification defaults to `--unit-scale 0.001` to convert to meters.
- Avoid `|cos(beta)| ≈ 0` for XY'Z' to stay away from singularities during random sampling.
- Recommended VS Code settings:

```json
{
    "python.analysis.typeCheckingMode": "standard",
    "python.analysis.stubPath": "./typings"
}
```
