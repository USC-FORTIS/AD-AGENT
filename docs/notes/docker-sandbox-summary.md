# AD-AGENT Sandbox Configuration

This note describes the current sandbox implementation in the refactored codebase under `src/sandbox/`.

The important distinction is:

- The agent workflow runs in the local Python process.
- Only generated model scripts are executed inside a sandbox backend.

That means `processor`, `selector`, `info_miner`, `code_generator`, `reviewer`, and `evaluator` logs are printed by the host process, while the generated script's `stdout` and `stderr` are streamed from Docker or Modal into the same terminal.

## Relevant Files

- `main.py`
- `src/sandbox/config.py`
- `src/sandbox/executor.py`
- `src/sandbox/docker_images.py`
- `src/sandbox/modal_images.py`
- `src/sandbox/dockerfiles/`

## Supported Backends

AD-AGENT currently supports two sandbox backends:

- `modal`
- `docker`

The backend is selected once at process startup and then used by `src/sandbox/executor.py`.

## Configuration Resolution

Sandbox mode is resolved in this order:

1. `--sandbox` CLI flag passed to `main.py`
2. `ADAGENT_SANDBOX` environment variable
3. legacy `OPENAD_SANDBOX` environment variable
4. `src/config/settings.yaml` if present
5. default: `modal`

The CLI flag works by setting `ADAGENT_SANDBOX` before sandbox-aware imports happen in `main.py`, so it is effectively the highest-priority override for normal CLI usage.

Valid values are:

- `modal`
- `docker`

Example:

```bash
ADAGENT_SANDBOX=modal .venv/bin/python -u main.py --sandbox modal
ADAGENT_SANDBOX=docker .venv/bin/python -u main.py --sandbox docker
```

If you want a persistent local default, create `src/config/settings.yaml` with:

```yaml
system:
  sandbox_mode: docker
```

At the moment, no `settings.yaml` is committed in the repo, so the fallback path is optional rather than required.

## Debug Mode

Modal has an additional debug flag:

```bash
ADAGENT_SANDBOX_DEBUG=1
```

When enabled:

- Modal sandboxes get longer timeout and idle-timeout values.
- Completed or failed sandboxes are retained instead of terminated immediately.
- The executor prints sandbox inspection hints such as `modal shell <sandbox-id>`.

This is implemented in `src/sandbox/executor.py`.

There is no equivalent retained-container debug mode for Docker right now. Docker runs use `--rm`, so containers are removed after execution.

The code still reads legacy `OPENAD_SANDBOX_DEBUG` for backward compatibility, but new usage should prefer `ADAGENT_SANDBOX_DEBUG`.

## Modal Prerequisite

If you use the `modal` backend, you need the Modal CLI installed and authenticated at least once on the host machine.

Typical setup:

```bash
pip install modal
modal setup
```

After that, `main.py --sandbox modal` can create the app, volume, and sandbox objects it needs. If your team uses token-based auth instead of browser login, configure Modal the same way you normally do for the CLI before running AD-AGENT.

## Modal Execution

Modal execution is implemented in `_execute_modal()` in `src/sandbox/executor.py`.

Current behavior:

- Looks up or creates Modal app `adagent-sandbox`
- Looks up or creates Modal volume `adagent-data`
- Uploads data files destined for `/data`
- Creates symlinks for non-`/data` paths when needed
- Writes generated code to `/workspace/<algorithm>_run.py`
- Executes `python /workspace/<algorithm>_run.py`
- Streams process `stdout` and `stderr` directly into the local terminal

Important paths inside Modal:

- workdir: `/workspace`
- persistent volume mount: `/data`

Current Modal-specific configuration constants:

- `MODAL_APP_NAME = "adagent-sandbox"`
- `MODAL_VOLUME_NAME = "adagent-data"`
- `DEFAULT_TIMEOUT = 120`

## Docker Execution

Docker execution is implemented in `_execute_docker()` in `src/sandbox/executor.py`.

Current behavior:

- Resolves an image tag via `src/sandbox/docker_images.py`
- Builds the image on demand if it is not available locally
- Writes generated code to a local temp file
- Mounts that file read-only at `/workspace/script.py`
- Mounts any provided dataset files read-only at their requested in-container paths
- Runs the container with resource limits
- Captures and returns `stdout`, `stderr`, and `returncode`

Current Docker run settings:

- `--rm`
- `--memory=4g`
- `--cpus=2`

## Package-to-Image Mapping

Both backends support the same package names:

- `pyod`
- `pygod`
- `darts`
- `tslib`
- `tsb_ad`

### Docker

Defined in `src/sandbox/docker_images.py`.

Current image tags:

- `adagent-pyod:latest`
- `adagent-pygod:latest`
- `adagent-darts:latest`
- `adagent-tslib:latest`
- `adagent-tsb-ad:latest`

Dockerfiles live under `src/sandbox/dockerfiles/`.

### Modal

Defined in `src/sandbox/modal_images.py`.

Each package has a dedicated `modal.Image` definition, for example:

- `PYOD_IMAGE`
- `PYGOD_IMAGE`
- `DARTS_IMAGE`
- `TSLIB_IMAGE`
- `TSB_AD_IMAGE`

`pygod` is special because it installs PyG wheel dependencies explicitly:

- `pyg_lib`
- `torch_scatter`
- `torch_sparse`
- `torch_cluster`
- `torch-geometric`

This is necessary for graph workflows that depend on PyG samplers and sparse ops.

## Logging Behavior

Current logging behavior is intentional:

- workflow stage logs such as `[main]`, `[selector]`, `[reviewer]` are printed by the host process
- sandbox script output is streamed into the same terminal
- Modal app logs are not the canonical place to read full workflow output

In practice:

- watch the local terminal while the workflow is running
- use `ADAGENT_SANDBOX_DEBUG=1` only when you need to inspect a retained Modal sandbox afterward

## Practical Commands

Run with Modal:

```bash
ADAGENT_SANDBOX=modal PYTHONUNBUFFERED=1 .venv/bin/python -u main.py --sandbox modal
```

Run with Docker:

```bash
ADAGENT_SANDBOX=docker PYTHONUNBUFFERED=1 .venv/bin/python -u main.py --sandbox docker
```

Run with Modal debug retention:

```bash
ADAGENT_SANDBOX=modal ADAGENT_SANDBOX_DEBUG=1 PYTHONUNBUFFERED=1 .venv/bin/python -u main.py --sandbox modal
```

## Current Caveats

- `main.py` currently injects `src/` into `sys.path` as a transition measure.
- The host process still owns workflow orchestration; sandboxes only execute generated scripts.
- `tslib` support exists in the sandbox layer, but it is not the current priority path.
- Runtime artifacts should live under `runtime/`; legacy `generated_scripts/` should be treated as old leftovers.

## Recommendation

If you update sandbox behavior, update this file alongside:

- backend selection logic in `src/sandbox/config.py`
- execution behavior in `src/sandbox/executor.py`
- image definitions in `src/sandbox/docker_images.py` and `src/sandbox/modal_images.py`

This file should stay aligned with code, not with old branch history.
