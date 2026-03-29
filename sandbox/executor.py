import os
import subprocess
import tempfile

from sandbox.config import (
    DEFAULT_TIMEOUT,
    MODAL_APP_NAME,
    MODAL_VOLUME_NAME,
    SANDBOX_MODE,
)

MODAL_VOLUME_MOUNT = "/data"
REMOTE_WORKDIR = "/workspace"


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

    app = modal.App.lookup(MODAL_APP_NAME, create_if_missing=True)
    volume = modal.Volume.from_name(MODAL_VOLUME_NAME, create_if_missing=True)

    volume_files = {}
    direct_files = {}
    if data_files:
        for remote_path, local_path in data_files.items():
            normalized = _normalize_remote_path(remote_path)
            if normalized.startswith(MODAL_VOLUME_MOUNT):
                volume_files[normalized] = local_path
            else:
                direct_files[normalized] = local_path

    if volume_files:
        _upload_to_volume(volume, volume_files)

    print(f"[sandbox] Creating Modal sandbox for package '{package_name}'...")
    sandbox = modal.Sandbox.create(
        app=app,
        image=image,
        timeout=timeout,
        volumes={MODAL_VOLUME_MOUNT: volume},
    )
    print(f"[sandbox] Modal sandbox created: {sandbox.object_id}")

    try:
        if direct_files:
            for remote_path, local_path in direct_files.items():
                if not os.path.exists(local_path):
                    continue
                with open(local_path, "rb") as file_obj:
                    data = file_obj.read()
                remote_dir = os.path.dirname(remote_path)
                if remote_dir:
                    sandbox.exec("mkdir", "-p", remote_dir).wait()
                proc = sandbox.exec("bash", "-c", f"cat > {remote_path}")
                proc.stdin.write(data)
                proc.stdin.write_eof()
                proc.wait()

        sandbox.exec("mkdir", "-p", REMOTE_WORKDIR).wait()
        script_path = f"{REMOTE_WORKDIR}/{algorithm_name}_run.py"
        encoded_code = base64.b64encode(code.encode()).decode()
        process = sandbox.exec(
            "bash",
            "-lc",
            f"echo {encoded_code} | base64 -d > {script_path} && cd {REMOTE_WORKDIR} && python {script_path}",
        )
        process.wait()

        stdout = process.stdout.read()
        stderr = process.stderr.read()
        returncode = process.returncode
        return stdout, stderr, returncode
    finally:
        sandbox.terminate()


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
