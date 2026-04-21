# AD-AGENT Sandbox

AD-AGENT runs its agent workflow on the host machine and executes generated model scripts inside an isolated sandbox backend. This keeps the orchestration layer lightweight while moving package-heavy model execution into Docker containers or Modal sandboxes.

In practice:

- the host process runs `processor`, `selector`, `info_miner`, `code_generator`, `reviewer`, and `evaluator`
- the sandbox backend runs the generated Python script for the selected package

If you are using AD-AGENT as a library or CLI tool, this is the document to read first.

## Backends

AD-AGENT currently supports two sandbox backends:

- `docker`
- `modal`

Use `docker` when you want local execution and already have Docker available. Use `modal` when you want remote execution and a managed cloud sandbox.

## Quick Start

Run with Docker:

```bash
ADAGENT_SANDBOX=docker PYTHONUNBUFFERED=1 .venv/bin/python -u main.py --sandbox docker
```

Run with Modal:

```bash
ADAGENT_SANDBOX=modal PYTHONUNBUFFERED=1 .venv/bin/python -u main.py --sandbox modal
```

Run with Modal debug retention:

```bash
ADAGENT_SANDBOX=modal ADAGENT_SANDBOX_DEBUG=1 PYTHONUNBUFFERED=1 .venv/bin/python -u main.py --sandbox modal
```

## Modal Prerequisite

If you use the `modal` backend, install the Modal CLI and authenticate once on the host machine:

```bash
pip install modal
modal setup
```

After that, AD-AGENT can create the Modal app, volume, and sandbox resources it needs.

## Configuration

Sandbox mode is resolved in this order:

1. `--sandbox` passed to `main.py`
2. `ADAGENT_SANDBOX`
3. legacy `OPENAD_SANDBOX`
4. `src/config/settings.yaml` if present
5. default: `modal`

Supported values:

- `modal`
- `docker`

Debug retention is controlled by:

- `ADAGENT_SANDBOX_DEBUG`

For backward compatibility, the code still reads legacy `OPENAD_SANDBOX_DEBUG`.

Modal resource names can also be overridden with environment variables:

- `ADAGENT_MODAL_APP_NAME`
- `ADAGENT_MODAL_VOLUME_NAME`

Default values are:

- app: `adagent-sandbox`
- volume: `adagent-data`

## What Gets Executed Where

The workflow logic stays local. Only generated model scripts are sandboxed.

That means:

- workflow logs such as `[main]`, `[selector]`, `[reviewer]` come from the host process
- generated script output is streamed back from the sandbox into the same terminal

This is why sandbox logs and workflow logs appear together during a run.

## Data and Runtime Layout

### Modal

Important paths inside Modal:

- `/workspace`: generated script location
- `/data`: persistent volume mount for uploaded datasets

Current behavior:

- the executor looks up or creates the Modal app
- looks up or creates the Modal volume
- uploads dataset files that should live under `/data`
- creates symlinks when a generated script expects a different path
- writes the generated script to `/workspace/<algorithm>_run.py`
- runs that script and streams `stdout` and `stderr`

### Docker

Current behavior:

- the executor resolves an image tag for the selected package
- builds the image on demand if needed
- writes the generated script to a temporary local file
- mounts that script read-only into the container
- mounts any dataset files read-only at the requested in-container paths
- runs the container with resource limits

Current Docker limits:

- `--rm`
- `--memory=4g`
- `--cpus=2`

## Supported Package Images

Both backends support these package keys:

- `pyod`
- `pygod`
- `darts`
- `tslib`
- `tsb_ad`

### Docker image tags

- `adagent-pyod:latest`
- `adagent-pygod:latest`
- `adagent-darts:latest`
- `adagent-tslib:latest`
- `adagent-tsb-ad:latest`

### Modal image definitions

Defined in `modal_images.py`:

- `PYOD_IMAGE`
- `PYGOD_IMAGE`
- `DARTS_IMAGE`
- `TSLIB_IMAGE`
- `TSB_AD_IMAGE`

`pygod` is the most dependency-sensitive package because it needs PyG wheel dependencies such as `pyg_lib`, `torch_sparse`, and `torch_scatter`.

## Logs and Debugging

The local terminal is the primary place to watch a run.

This is the recommended mental model:

- watch the local terminal for normal execution
- use Modal debug retention only when you need post-run inspection

When `ADAGENT_SANDBOX_DEBUG=1` is enabled for Modal:

- sandboxes are retained instead of terminated immediately
- the executor prints a sandbox id
- you can inspect the retained sandbox with commands such as `modal shell <sandbox-id>`

Modal app logs are not the canonical source for full workflow output.

## Current Caveats

- `main.py` currently injects `src/` into `sys.path` as a transition measure
- the host process still owns workflow orchestration
- sandboxes only execute generated scripts
- `tslib` support exists in the sandbox layer, but it is not the current priority path
- runtime artifacts should live under `runtime/`; legacy `generated_scripts/` should be treated as old leftovers

## Developer Notes

If you are changing sandbox behavior, start with these files:

- `config.py`
- `executor.py`
- `docker_images.py`
- `modal_images.py`
- `dockerfiles/`

There is also an implementation-oriented note at:

- `../../docs/notes/docker-sandbox-summary.md`
