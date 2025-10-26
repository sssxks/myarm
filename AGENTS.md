# Repository Guidelines

## Project Structure & Module Organization
- `src/forward/` — library and CLI code: `solver.py` (FK + XY'Z'), `main.py` (CLI), `numerical_checker.py`, `verify_fk_coppelia.py`.
- `type_stubs/` — extra stubs; included in `mypy_path`.
- `report/` — experiment notes and logs (e.g., `verification_results.log`).
- `dist/` — build artifacts (do not commit). See `.gitignore`.

## Build, Test, and Development Commands
- Note `.venv` is currently generated in Windows. so always use `uv.exe` to run commands.
- `uv.exe sync` — create venv and install deps from `pyproject.toml`/`uv.lock`.
- `uv.exe run forward symbolic|eval|random|dh` — run CLI tasks (see `README.md`).
- `uv.exe run verify_fk [--deg]` — check FK in CoppeliaSim over ZMQ.
- `uv.exe run mypy . --strict` and `uv run pyright` — type checking using `type_stubs/`.
- `uv.exe build` — build wheel/sdist into `dist/`.

## Coding Style & Naming Conventions
- Python ≥ 3.13, 4‑space indent, PEP 8 names: modules/functions `snake_case`, classes `PascalCase`, constants `UPPER_CASE`.
- Prefer pure, small functions; add docstrings; keep I/O in CLI modules.
- Type‑hint everything; avoid `Any` when feasible; use `typing.cast` for SymPy where needed (see `solver.py`).

## Testing Guidelines
- No pytest suite yet. Validate via:
  - Static checks: `uv.exe run mypy . --strict`, `uv.exe run pyright`.
  - Numeric checks: `uv.exe run forward random --count 5` and compare reconstruction errors.
  - Simulation: `uv.exe run verify_fk` and review tolerances/summary; capture output in `report/`.
- When adding tests, place them under `tests/` as `test_*.py`.

## Commit Guidelines
- Use Conventional Commits (history shows `feat: …`, `chore: …`).
- Update `README.md` if commands or flags change.

## Security & Configuration
- ZMQ server: default `127.0.0.1:23000`. Do not expose publicly; no secrets in repo.
- Ensure caches/artifacts aren’t committed (`__pycache__/`, `dist/`). If present, run `git rm -r --cached <path>`.