# Forward Kinematics (ZJU‑I Arm)

This repo provides a small, reproducible setup to derive and verify the forward kinematics of a 6‑DoF desktop arm (ZJU‑I). It includes:

- Symbolic FK via standard DH parameters (SymPy)
- A CLI for inspecting symbolic forms and evaluating joint sets
- Numeric reconstruction checks for intrinsic XY'Z' Euler angles
- Optional verification against CoppeliaSim over ZMQ

## Quick Start

Prereqs:
- Python ≥ 3.13
- Recommended: [uv](https://github.com/astral-sh/uv) for fast, reproducible runs

Create the environment and run the CLI (Windows users can replace `uv` with `uv.exe`):

```ps1
uv sync # (optional) creates venv and installs deps
uv run forward symbolic           # show symbolic T06
uv run forward eval --preset 1 2  # evaluate presets (degrees)
uv run forward random --count 3   # random numeric checks
```

Available commands:

- `symbolic`  Show symbolic `T06`. Add `--steps` for `T01..T06`. Add `--euler` to print the symbolic XY'Z' formulas. Use `--no-eval` to avoid substituting the zero pose.
- `eval`      Evaluate `T06` and XY'Z' for provided joint angles. Use `--preset 1..5` (degrees) or `--q q1 q2 q3 q4 q5 q6` with optional `--deg`.
- `random`    Run numeric checks that reconstruct `R` from XY'Z' and report errors.
- `dh`        Print the DH parameter lists used.

## CoppeliaSim Verification (optional)

The script `verify_fk` connects to CoppeliaSim via ZMQ, sets joints, reads the tip pose, and compares it with the symbolic FK.

1) Start CoppeliaSim with the ZMQ remote API server enabled (default: `127.0.0.1:23000`). Ensure your scene exposes joint names similar to `/Robot/Joint1..6` and a tip object like `/Robot/SuctionCup/SuctionCup_connect`.
2) From the repo root, run:

```ps1
uv run verify_fk

# or provide your own sets
uv run verify_fk --q 30 0 90 60 0 0 --deg
```

Options:
- `--host/--port`    ZMQ endpoint
- `--joint`          Repeatable; 6 joint paths (defaults match the provided scene)
- `--tip/--base`     Tip (and optional base) object paths
- `--deg`            Interpret inputs in degrees (else radians)
- `--unit-scale`     Scale factor for translation (e.g., 0.001 if FK is in mm but sim in m)
- `--tol-pos/--tol-rot`  Tolerances for pass/fail

Example output is recorded in `report/verification_results.log`.

## Euler Convention (XY'Z')

We use intrinsic XY'Z' Euler angles (equivalently extrinsic Z‑Y‑X). For a rotation block `R`, the non‑singular mapping is:

```
beta  = asin( r13 )
alpha = atan2(-r23, r33)
gamma = atan2(-r12, r11)
```

Numerically we handle gimbal lock at `beta = ±π/2` by setting `gamma = 0` and folding yaw into `alpha`.

## Repo Layout

- `src/forward/solver.py`              Core FK and Euler utilities (+ `demo_standard_6R()`)
- `src/forward/main.py`                CLI entry (`forward`)
- `src/forward/numerical_checker.py`   Numeric XY'Z' reconstruction and error metrics
- `src/forward/verify_fk_coppelia.py`  CoppeliaSim verification (`verify_fk`)
- `report/`                            Experiment requirements and logs

## Reproducing the Experiment

1) Inspect symbolic FK: `uv run forward symbolic --euler`
2) Evaluate the 5 assignment sets: `uv run forward eval --preset 1 2 3 4 5 --deg`
3) Run numeric checks: `uv run forward random --count 5`
4) If available, verify in CoppeliaSim: `uv run verify_fk --deg`

All results should match within tolerances (see `report/verification_results.log`).

## Development

Recommended to add this to your `.vscode/settings.json` for strict type checking with custom stubs.

```json
{
    "python.analysis.typeCheckingMode": "strict",
    "python.analysis.stubPath": "./type_stubs"
}
``` 

you can also use clis like `uv run mypy . --strict`, `uv run pyright`

## Notes

- Units: DH `a` & `d` are in millimeters. `verify_fk` uses `--unit-scale 0.001` so meters align with CoppeliaSim.
- Singularities: Avoid `|cos(beta)| ≈ 0` for XY'Z' to prevent gimbal lock in numeric paths.
