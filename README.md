# myarm Manipulator Toolkit

This project provides symbolic and numeric tooling for a 6‑DoF desktop arm (ZJU‑I) including forward and inverse kinematics, Euler angle helpers, and CoppeliaSim verification hooks. The command-line interface has been consolidated under a single `myarm` entry point.

## Quick Start

Prerequisites:
- Python ≥ 3.13
- [uv](https://github.com/astral-sh/uv) (recommended for reproducible environments). On Windows use `uv.exe`.

Create the environment and explore the CLI:

```ps1
uv sync
uv run myarm -- fk symbolic           # show symbolic T06
uv run myarm -- fk eval --preset 1 2  # evaluate presets (degrees)
uv run myarm -- fk random --count 3   # random numeric checks
uv run myarm -- ik solve --from-q 0 0 1.57 0 0 0
```

### CLI Overview

All tooling is exposed through `uv run myarm -- <group> <command>`. Key groups:

- `fk` – forward kinematics utilities
  - `symbolic`  Show symbolic `T06`; add `--steps` for `T01..T06`, `--euler` for symbolic XY'Z' Euler angles, `--no-eval` to skip evaluation at the rest pose.
  - `eval`      Evaluate `T06` and XY'Z' for provided joint sets via `--preset 1..5` or `--q q1..q6` (combine with `--deg`).
  - `random`    Run random numeric checks that reconstruct `R` from XY'Z' while avoiding gimbal lock.
  - `dh`        Print the DH parameter lists.
- `ik` – inverse kinematics helpers
  - `solve`     Damped-least-squares IK solver. Provide a target as `--T` (16 row-major values) or `--from-q` (6 joints). Optional seeds via repeatable `--seed`.
- `verify` – CoppeliaSim validation utilities
  - `fk`        Mirror of the previous `verify_fk` script.
  - `ik`        Mirror of the previous `verify_ik` script.

Run `uv run myarm -- <group> <command> --help` for detailed arguments.

## CoppeliaSim Verification

The verification subcommands connect to CoppeliaSim through the ZMQ remote API and compare simulated results with the symbolic / numeric solvers.

1. Launch CoppeliaSim with the remote API server active (default `127.0.0.1:23000`). Ensure joint objects are named `/Robot/Joint1..6` and the tool frame matches `/Robot/SuctionCup/SuctionCup_connect` or pass custom paths.
2. From the repo root, run:

```ps1
uv run myarm -- verify fk
# Provide custom joint sets (degrees)
uv run myarm -- verify fk --q 30 0 90 60 0 0 --deg
# Verify IK using the current simulator pose and apply the best solution
uv run myarm -- verify ik --apply
```

Common flags include `--host/--port`, repeatable `--joint`, `--tip`, `--base`, `--deg`, `--unit-scale`, `--tol-pos`, and `--tol-rot`. IK verification also supports `--tol-pos-mm`, `--tol-rot-deg`, `--max-iter`, and `--lmbda`. Example logs are stored in `report/verification_results.log`.

## Euler Convention (XY'Z')

We use intrinsic XY'Z' Euler angles (equivalently extrinsic Z‑Y‑X). For a rotation block `R`:

```
beta  = asin( r13 )
alpha = atan2(-r23, r33)
gamma = atan2(-r12, r11)
```

When `beta = ±π/2`, the CLI switches to a gimbal-lock-safe reconstruction by clamping `gamma` to zero and folding yaw into `alpha`.

## Project Layout

- `src/myarm/solver.py`                Core FK helpers (`fk_standard`, XY'Z' utilities, demo setup)
- `src/myarm/main.py`                  Unified CLI entry point (`myarm`)
- `src/myarm/numerical_checker.py`     Numeric XY'Z' reconstruction checks
- `src/myarm/ik_solver.py`             Damped least-squares IK implementation
- `src/myarm/verify_fk_coppelia.py`    FK verification helpers (reused by CLI)
- `src/myarm/verify_ik_coppelia.py`    IK verification helpers (reused by CLI)
- `report/`                            Experiment logs and notes
- `typings/`                           Third-party `.pyi` stubs (added to `mypy_path`)

## Reproducing the Experiment

1. Inspect symbolic FK: `uv run myarm -- fk symbolic --euler`
2. Evaluate the five preset configurations: `uv run myarm -- fk eval --preset 1 2 3 4 5 --deg`
3. Run numeric checks: `uv run myarm -- fk random --count 5`
4. (Optional) Verify in CoppeliaSim: `uv run myarm -- verify fk --deg`

All expected tolerances and outputs are documented in `report/verification_results.log`.

## Development Notes

- Run static checks with `uv run mypy . --strict` or `uv run pyright`.
- DH parameters use millimeters. CoppeliaSim verification defaults to `--unit-scale 0.001` to convert to meters.
- Avoid `|cos(beta)| ≈ 0` for XY'Z' to stay away from singularities during random sampling.
- Recommended VS Code settings:

```json
{
    "python.analysis.typeCheckingMode": "strict",
    "python.analysis.stubPath": "./typings"
}
```
