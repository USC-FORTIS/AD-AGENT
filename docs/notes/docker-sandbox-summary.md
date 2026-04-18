# AD-AGENT: Dual Sandbox Execution Architecture

## Overview

AD-AGENT adopts a **two-layer architecture**: a lightweight agent environment on the host machine, and an isolated execution environment (Docker or Modal) for running generated ML code. This cleanly separates the "brain" (LLM agents) from the "hands" (code execution), so users never need to install heavy ML dependencies locally.

```
┌─────────────────────────────────────────────────────────┐
│                    Host Machine                         │
│                                                         │
│  ┌───────────────────────────────────┐                  │
│  │      Agent Environment            │                  │
│  │  (pip install ad-agent)           │                  │
│  │                                   │                  │
│  │  LangGraph / LangChain            │                  │
│  │  OpenAI SDK / Bedrock SDK         │                  │
│  │  Config & Routing Logic           │                  │
│  └──────────────┬────────────────────┘                  │
│                                                         │ 
        subprocess.run()                                  |
│          execute_code()                                 │
│           │            │                                │
│     ┌─────┘            └──────┐                         │
│     ▼                         ▼                         │
│  ┌──────────────┐   ┌─────────────────┐                 │
│  │ Docker Mode  │   │  Modal Mode     │                 │
│  │ (local)      │   │  (cloud)        │                 │
│  │              │   │                 │                 │
│  │ pyod/torch/  │   │  Same deps,     │                 │
│  │ darts/etc.   │   │  remote infra   │                 │
│  │              │   │                 │                 │
│  │ Free, offline│   │  Pay-per-use    │                 │
│  │ Your hardware│   │  Cloud GPU/CPU  │                 │
│  └──────────────┘   └─────────────────┘
|                                                         |
└─────────────────────────────────────────────────────────┘
```

## Two Execution Modes

| | Docker | Modal |
|---|---|---|
| **Where** | Local Docker container | Modal cloud sandbox |
| **Cost** | Free | Pay-per-use |
| **Network** | Offline OK | Requires internet |
| **GPU** | Uses local GPU if available | Cloud GPU |
| **Best for** | Development, testing | Production, large-scale runs |
| **Isolation** | Container-level | Container-level (remote) |

Both modes use **identical dependency sets** — the Dockerfiles mirror the Modal image definitions exactly, so results are reproducible across modes.

## How It Works

### Execution Flow

```
User runs: python main.py --sandbox docker
                │
                ▼
    main.py parses --sandbox flag
    sets OPENAD_SANDBOX=docker env var
    (before any agent imports)
                │
                ▼
    Agent pipeline runs (Processor → Selector → InfoMiner
    → CodeGenerator → Reviewer → Evaluator → Optimizer)
                │
                ▼
    When code needs to run, executor.execute_code() is called
                │
                ▼
    sandbox/config.py routes to the correct backend
                │
        ┌───────┴────────┐
        ▼                ▼
    _execute_docker()  _execute_modal()
```

### Docker Mode Details

```
_execute_docker(code, algorithm_name, package_name, data_files, timeout)
        │
        ├─ 1. ensure_image(package_name)
        │      docker image inspect openad-pyod:latest
        │        ├─ exists → use cached image (instant)
        │        └─ missing → docker build from Dockerfile (first time only)
        │
        ├─ 2. Write generated code to temp file
        │
        ├─ 3. docker run --rm --memory=4g --cpus=2
        │        -v script.py:/workspace/script.py:ro
        │        -v data_file:/path/in/container:ro
        │        openad-pyod:latest python /workspace/script.py
        │
        ├─ 4. Capture stdout / stderr / returncode
        │
        └─ 5. Cleanup temp file (in finally block)
```

Key design choices:
- **`--rm`**: Container auto-deletes after execution, no leftover state
- **`:ro` mounts**: Code and data are read-only inside the container
- **Resource limits**: `--memory=4g --cpus=2` prevents runaway processes
- **Docker CLI via subprocess**: No `docker` Python SDK needed, zero extra dependencies
- **Build-once cache**: `docker image inspect` check before building

### Configuration Priority

```
--sandbox CLI flag          (highest priority)
        ↓
config/settings.yaml        (persistent config)
  system.sandbox_mode
        ↓
default: "modal"            (lowest priority)
```

## Files Changed

### New Files

| File | Purpose |
|---|---|
| `sandbox/docker_images.py` | `DOCKER_IMAGE_MAP` + `ensure_image()` — manages Docker image lifecycle |
| `sandbox/dockerfiles/Dockerfile.pyod` | pyod, numpy, scikit-learn, pandas |
| `sandbox/dockerfiles/Dockerfile.pygod` | pygod, torch, torch-geometric, numpy |
| `sandbox/dockerfiles/Dockerfile.darts` | darts, torch, numpy, pandas |
| `sandbox/dockerfiles/Dockerfile.tslib` | torch, numpy, pandas, scikit-learn + Time-Series-Library repo |

### Modified Files

| File | Change |
|---|---|
| `sandbox/config.py` | Added `_load_sandbox_mode()` with env var → settings.yaml → default fallback chain, validation |
| `sandbox/executor.py` | Added `_execute_docker()` function, removed `_execute_locally()` (replaced by Docker) |
| `config/settings.yaml` | Added `sandbox_mode: "modal"` under `system:` |
| `main.py` | Added early `--sandbox` argparse before agent imports to set env var in time |


### Removed

- **Local subprocess mode** (`_execute_locally`) — Docker fully replaces it with proper isolation. No reason to run ML code in the host Python environment.

## Usage

```bash
# Docker mode (local, free, offline)
python main.py --sandbox docker

# Modal mode (cloud, pay-per-use) — default
python main.py --sandbox modal

# Use default from settings.yaml
python main.py
```

Or set persistently in `config/settings.yaml`:
```yaml
system:
  sandbox_mode: "docker"
```

## Test Results

Tested on Docker v28.3.3 (macOS):

```
# 1. Image build — auto-builds on first call
>>> ensure_image('pyod')
[Docker] Building image openad-pyod:latest ...
[Docker] Image openad-pyod:latest built successfully.
'openad-pyod:latest'

# 2. Image cache — skips build on subsequent calls
>>> ensure_image('pyod')
'openad-pyod:latest'          # no build, instant return

# 3. Code execution — runs in container, returns output
>>> execute_code('print(42)', 'test', 'pyod')
('42\n', '', 0)               # stdout, stderr, returncode
```

## Next Steps

- [ ] Test remaining images (pygod, darts, tslib)
- [ ] Test with real dataset through full pipeline (`--sandbox docker`)
- [ ] Test Modal mode still works as before
- [ ] Package as `ad-agent` Python package (pyproject.toml + CLI entrypoint)
