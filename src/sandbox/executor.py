import os
import re
import subprocess
import sys
import tempfile
import threading
import uuid

from sandbox.config import (
    DEFAULT_TIMEOUT,
    MODAL_APP_NAME,
    MODAL_VOLUME_NAME,
    SANDBOX_MODE,
)

MODAL_VOLUME_MOUNT = "/data"
REMOTE_WORKDIR = "/workspace"

DEBUG_SANDBOX_TIMEOUT_SECONDS = 1800
DEBUG_SANDBOX_IDLE_TIMEOUT_SECONDS = 600


def execute_code(
    code: str,
    algorithm_name: str,
    package_name: str,
    data_files: dict[str, str] | None = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> tuple[str, str, int]:
    """
    Execute a Python script in a Modal Sandbox or Docker container.
    """
    if SANDBOX_MODE == "modal":
        return _execute_modal(code, algorithm_name, package_name, data_files, timeout)
    return _execute_docker(code, algorithm_name, package_name, data_files, timeout)


def _normalize_remote_path(remote_path: str) -> str:
    if os.path.isabs(remote_path):
        return remote_path
    return os.path.join(REMOTE_WORKDIR, remote_path.lstrip("./"))


def _upload_to_volume(volume, data_files: dict[str, str]):
    """Upload local files into the Modal volume, skipping files already present."""
    existing = set()
    try:
        for entry in volume.listdir("/", recursive=True):
            existing.add("/" + entry.path)
    except Exception:
        pass

    to_upload = {}
    for remote_path, local_path in data_files.items():
        normalized_remote_path = _normalize_remote_path(remote_path)
        if not normalized_remote_path.startswith(MODAL_VOLUME_MOUNT):
            continue

        volume_path = normalized_remote_path[len(MODAL_VOLUME_MOUNT):] or "/"
        if volume_path in existing:
            print(f"[sandbox] Volume already has {volume_path}, skipping upload")
            continue
        if not os.path.exists(local_path):
            print(f"[sandbox] Warning: local file {local_path} not found, skipping")
            continue
        to_upload[volume_path] = local_path

    if not to_upload:
        return

    with volume.batch_upload() as batch:
        for volume_path, local_path in to_upload.items():
            batch.put_file(local_path, volume_path)
            size = os.path.getsize(local_path)
            print(f"[sandbox] Uploaded {local_path} -> volume:{volume_path} ({size} bytes)")


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _sanitize_name_component(value: str) -> str:
    sanitized = re.sub(r"[^a-z0-9-]+", "-", value.lower())
    sanitized = re.sub(r"-{2,}", "-", sanitized).strip("-")
    return sanitized or "run"


def _build_sandbox_name(package_name: str, algorithm_name: str) -> str:
    pkg = _sanitize_name_component(package_name)
    algo = _sanitize_name_component(algorithm_name)
    suffix = uuid.uuid4().hex[:8]
    return f"openad-{pkg}-{algo}-{suffix}"[:63]


def _print_debug_retention_hint(
    app_id: str | None,
    sandbox_name: str,
    sandbox_id: str,
    returncode: int | None = None,
) -> None:
    status = "failed" if returncode not in (None, 0) else "completed"
    print()
    print(f"[sandbox] Debug mode is enabled; keeping {status} sandbox alive for inspection")
    print(f"[sandbox] Retained sandbox name: {sandbox_name}")
    print(f"[sandbox] Retained sandbox id: {sandbox_id}")
    if returncode is not None:
        print(f"[sandbox] Sandbox return code: {returncode}")
    if app_id:
        print(f"[sandbox] Modal app id: {app_id}")
    print(f"[sandbox] Open an interactive shell with: modal shell {sandbox_id}")
    print("[sandbox] To find the live container, run: modal container list")
    print("[sandbox] Then inspect it with: modal container logs <container-id>")
    print("[sandbox] Or execute commands inside it with: modal container exec <container-id> -- /bin/bash")
    print("[sandbox] You can also reattach in Python with:")
    print("python3 - <<'PY'")
    print("import modal")
    print(f"sb = modal.Sandbox.from_id('{sandbox_id}')")
    print("proc = sb.exec('/bin/bash', '-lc', 'pwd && ls -la /workspace && ls -la /data | head')")
    print("print(proc.stdout.read())")
    print("print(proc.stderr.read(), file=__import__('sys').stderr)")
    print("PY")


def _execute_modal(
    code: str,
    algorithm_name: str,
    package_name: str,
    data_files: dict[str, str] | None,
    timeout: int,
) -> tuple[str, str, int]:
    """Execute code inside a Modal Sandbox with a persistent volume mounted at /data."""
    import base64
    import modal
    from sandbox.modal_images import IMAGE_MAP

    image = IMAGE_MAP.get(package_name)
    if image is None:
        return "", f"[ERROR] Unknown package: {package_name}", 1

    debug_enabled = _env_flag("OPENAD_SANDBOX_DEBUG")
    sandbox_timeout = max(timeout, DEBUG_SANDBOX_TIMEOUT_SECONDS) if debug_enabled else timeout
    idle_timeout = DEBUG_SANDBOX_IDLE_TIMEOUT_SECONDS if debug_enabled else None
    sandbox_name = _build_sandbox_name(package_name, algorithm_name)

    with modal.enable_output():
        app = modal.App.lookup(MODAL_APP_NAME, create_if_missing=True)
        volume = modal.Volume.from_name(MODAL_VOLUME_NAME, create_if_missing=True)

        volume_files = {}
        symlinks = {}
        if data_files:
            for remote_path, local_path in data_files.items():
                normalized = _normalize_remote_path(remote_path)
                if normalized.startswith(MODAL_VOLUME_MOUNT):
                    volume_files[normalized] = local_path
                else:
                    # Upload via volume and create a symlink from the expected path
                    basename = os.path.basename(local_path)
                    parent = os.path.basename(os.path.dirname(local_path))
                    volume_key = f"{MODAL_VOLUME_MOUNT}/{parent}/{basename}"
                    volume_files[volume_key] = local_path
                    symlinks[normalized] = volume_key

        if volume_files:
            _upload_to_volume(volume, volume_files)

        app_id = getattr(app, "app_id", None)
        if app_id:
            print()
            print(f"[sandbox] Using Modal app '{MODAL_APP_NAME}' ({app_id})")
        else:
            print()
            print(f"[sandbox] Using Modal app '{MODAL_APP_NAME}'")
        print("[sandbox] Streaming sandbox stdout/stderr to this terminal")
        if debug_enabled:
            print(
                f"[sandbox] Debug mode enabled; sandbox timeout={sandbox_timeout}s "
                f"idle_timeout={idle_timeout}s"
            )

        print(f"[sandbox] Creating Modal sandbox for package '{package_name}'...")
        sandbox = modal.Sandbox.create(
            app=app,
            name=sandbox_name,
            image=image,
            timeout=sandbox_timeout,
            idle_timeout=idle_timeout,
            volumes={MODAL_VOLUME_MOUNT: volume},
            verbose=debug_enabled,
        )
        sandbox.set_tags(
            {
                "source": "openad",
                "package": package_name,
                "algorithm": algorithm_name,
            }
        )
        print(f"[sandbox] Modal sandbox created: {sandbox.object_id}")

        preserve_sandbox = False
        final_returncode: int | None = None
        try:
            # Create symlinks so generated code paths resolve to volume files
            for link_path, target_path in symlinks.items():
                link_dir = os.path.dirname(link_path)
                if link_dir:
                    sandbox.exec("mkdir", "-p", link_dir).wait()
                sandbox.exec("ln", "-sf", target_path, link_path).wait()

            sandbox.exec("mkdir", "-p", REMOTE_WORKDIR).wait()
            script_path = f"{REMOTE_WORKDIR}/{algorithm_name}_run.py"
            encoded_code = base64.b64encode(code.encode()).decode()
            process = sandbox.exec(
                "bash",
                "-lc",
                f"echo {encoded_code} | base64 -d > {script_path} && cd {REMOTE_WORKDIR} && python {script_path}",
            )
            stdout, stderr, returncode = _stream_process_output(process)
            final_returncode = returncode
            preserve_sandbox = debug_enabled
            return stdout, stderr, returncode
        except Exception:
            preserve_sandbox = debug_enabled
            raise
        finally:
            if preserve_sandbox:
                _print_debug_retention_hint(
                    app_id,
                    sandbox_name,
                    sandbox.object_id,
                    returncode=final_returncode,
                )
            else:
                sandbox.terminate()


def _stream_process_output(process) -> tuple[str, str, int]:
    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []

    def _drain(reader, sink, chunks: list[str]) -> None:
        if reader is None:
            return
        try:
            for chunk in reader:
                if chunk is None:
                    continue
                text = chunk.decode() if isinstance(chunk, bytes) else str(chunk)
                chunks.append(text)
                sink.write(text)
                sink.flush()
        except Exception as exc:
            msg = f"\n[sandbox] log streaming error: {exc}\n"
            chunks.append(msg)
            sys.stderr.write(msg)
            sys.stderr.flush()

    stdout_thread = threading.Thread(
        target=_drain,
        args=(process.stdout, sys.stdout, stdout_chunks),
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=_drain,
        args=(process.stderr, sys.stderr, stderr_chunks),
        daemon=True,
    )
    stdout_thread.start()
    stderr_thread.start()

    returncode = process.wait()
    stdout_thread.join()
    stderr_thread.join()
    return "".join(stdout_chunks), "".join(stderr_chunks), returncode


def _execute_docker(
    code: str,
    algorithm_name: str,
    package_name: str,
    data_files: dict[str, str] | None,
    timeout: int,
) -> tuple[str, str, int]:
    """Execute code inside a Docker container."""
    from sandbox.docker_images import ensure_image

    try:
        image_tag = ensure_image(package_name)
    except (ValueError, RuntimeError) as exc:
        return "", f"[ERROR] {exc}", 1

    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        prefix=f"{algorithm_name}_",
        delete=False,
    )
    try:
        tmp.write(code)
        tmp.close()

        cmd = [
            "docker",
            "run",
            "--rm",
            "--memory=4g",
            "--cpus=2",
            "-v",
            f"{tmp.name}:{REMOTE_WORKDIR}/script.py:ro",
        ]

        if data_files:
            for remote_path, local_path in data_files.items():
                abs_local = os.path.abspath(local_path)
                if os.path.exists(abs_local):
                    cmd.extend(["-v", f"{abs_local}:{_normalize_remote_path(remote_path)}:ro"])

        cmd.extend([image_tag, "python", f"{REMOTE_WORKDIR}/script.py"])

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired:
        return "", "[ERROR] Docker execution timed out.", 1
    finally:
        os.unlink(tmp.name)
