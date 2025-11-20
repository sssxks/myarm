# Repository Guidelines

## Project Structure & Module Organization
- `src/myarm/` — primary package and CLI code:
  - `fk_solver.py`, `ik_solver.py`, `orientation.py`, `dh_params.py`, `type_utils.py`.
  - `coppelia_utils.py`, `verify_fk_coppelia.py`, `verify_ik_coppelia.py` for simulator hooks.
  - `main.py` (CLI entry point), `numerical_checker.py`, `presets.py`.
- `scripts/` — automation helpers (e.g. `verify_cases.py` for batch CoppeliaSim checks).
- `typings/` — third-party `.pyi` stubs; included in `mypy_path`.
- `report/` — experiment notes and logs (e.g., `verification_results.log`).
- `dist/` — build artifacts (do not commit). See `.gitignore`.

## Build, Test, and Development Commands
- you may or may not in wsl, so prefer `uv.exe` to `uv` to run in windows environment.
- `uv.exe sync` — create/update the environment from `pyproject.toml`/`uv.lock`.
- `uv.exe run myarm -- fk symbolic|eval|random|dh` — FK tooling.
- `uv.exe run myarm -- ik solve|euler` — IK solvers.
- `uv.exe run myarm -- verify fk|ik [flags]` — CoppeliaSim validation.
- `uv.exe run mypy . --strict` and `uv.exe run pyright` — static checks using `typings/`.
- `uv.exe build` — build wheel/sdist into `dist/`.

## Coding Style & Naming Conventions
- Python ≥ 3.13, 4‑space indent, PEP 8 names: modules/functions `snake_case`, classes `PascalCase`, constants `UPPER_CASE`.
- Prefer pure, small functions; add docstrings; keep I/O in CLI modules.
- Type‑hint everything; avoid `Any` when feasible.

## Testing Guidelines
- No pytest suite yet. Validate via:
  - Static checks: `uv.exe run mypy . --strict`, `uv.exe run pyright`.
  - Numeric checks: `uv.exe run myarm -- fk random --count 5` and inspect Euler reconstruction errors.
  - Simulation: `uv.exe run myarm -- verify fk|ik` and review tolerances/summary; capture output in `report/`.
  - Batch verification: `uv.exe run python scripts/verify_cases.py` (logs under `report/`).
- When adding tests, place them under `tests/` as `test_*.py`.

## Commit Guidelines
- Use Conventional Commits (history shows `feat: …`, `chore: …`).
- Update `README.md` if commands or flags change.

## Security & Configuration
- ZMQ server: default `127.0.0.1:23000`. Do not expose publicly; no secrets in repo.
- Ensure caches/artifacts aren’t committed (`__pycache__/`, `dist/`). If present, run `git rm -r --cached <path>`.
