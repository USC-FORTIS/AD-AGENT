# Modal Sandbox Integration

## What Changed

Replaced bare `subprocess.run()` calls in Reviewer, Evaluator, and Optimizer agents with a unified `execute_code()` function that runs scripts in isolated Modal cloud containers. The orchestration layer (LangGraph, LLM calls, agent logic) stays local; only code execution moves to Modal.

## Created Files

- `sandbox/__init__.py` — Package init, re-exports `execute_code`
- `sandbox/config.py` — Sandbox configuration (`SANDBOX_MODE`, `MODAL_APP_NAME`, `DEFAULT_TIMEOUT`); controlled by `OPENAD_SANDBOX` env var (`"modal"` or `"local"`)
- `sandbox/modal_images.py` — Modal `Image` definitions for PyOD, PyGOD, Darts, and TSLib (with Time-Series-Library cloned into the TSLib image)
- `sandbox/executor.py` — `execute_code()` dispatcher: Modal sandbox or local subprocess fallback. Returns `(stdout, stderr, returncode)`. Handles file uploads, script writing, and sandbox lifecycle

## Modified Files

- `agents/agent_reviewer.py` — Uses `execute_code()` from sandbox instead of `subprocess.run()`
- `agents/agent_evaluator.py` — Uses `sandbox_execute_code()`; added `package_name` and `data_files` parameters
- `agents/agent_optimizer.py` — Uses `sandbox_execute_code()`; added `package_name` threaded through `execute_code()` and `run()`
- `main.py` — Passes `package_name` to evaluator and optimizer calls
- `pipeline_api.py` — Passes `package_name` to evaluator and optimizer calls
- `config/config.py` — Added `SANDBOX_MODE`, `MODAL_APP_NAME`, `SANDBOX_DEFAULT_TIMEOUT`
- `requirements.txt` — Added `modal`
- `.gitignore` — Added `data/` to ignore data files

## Architecture

```
Local (unchanged)                    Modal Sandbox (new)
────────────────                     ──────────────────
LangGraph DAG
  → Processor (local)
  → Selector (local)          ┌───→ PyOD image: run script
  → InfoMiner (local)         │      (pyod, numpy, scikit-learn)
  → CodeGenerator (local)     │
  → Reviewer ─── send script ─┤───→ PyGOD image: run script
  → Evaluator ── send script ─┤      (pygod, torch-geometric)
  → Optimizer ── send script ─┤
                              ├───→ Darts image: run script
                              │      (darts, torch)
                              └───→ TSLib image: run script
                                     (torch, custom deps)
```

## Usage

- **Modal mode** (default): `export OPENAD_SANDBOX=modal`
- **Local mode** (fallback): `export OPENAD_SANDBOX=local` — uses `subprocess.run()` as before
