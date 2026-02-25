import os
import subprocess
import tempfile

from sandbox.config import SANDBOX_MODE, MODAL_APP_NAME, DEFAULT_TIMEOUT


def execute_code(
    code: str,
    algorithm_name: str,
    package_name: str,
    data_files: dict[str, str] | None = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> tuple[str, str, int]:
    """
    Execute a Python script in a Modal Sandbox, Docker container, or locally.

    Args:
        code: Python source code to execute.
        algorithm_name: Name of the algorithm (used for script naming).
        package_name: Package name to select the right image.
        data_files: Optional dict mapping {remote_path: local_path} for files to
                    upload into the sandbox before execution.
        timeout: Max execution time in seconds.

    Returns:
        (stdout, stderr, returncode)
    """
    if SANDBOX_MODE == "modal":
        return _execute_modal(code, algorithm_name, package_name, data_files, timeout)
    else:
        return _execute_docker(code, algorithm_name, package_name, data_files, timeout)


def _execute_modal(
    code: str,
    algorithm_name: str,
    package_name: str,
    data_files: dict[str, str] | None,
    timeout: int,
) -> tuple[str, str, int]:
    """Execute code inside a Modal Sandbox."""
    import modal
    from sandbox.modal_images import IMAGE_MAP

    image = IMAGE_MAP.get(package_name)
    if image is None:
        return "", f"[ERROR] Unknown package: {package_name}", 1

    app = modal.App.lookup(MODAL_APP_NAME, create_if_missing=True)

    sandbox = modal.Sandbox.create(
        app=app,
        image=image,
        timeout=timeout,
    )

    try:
        # Upload data files if provided
        if data_files:
            for remote_path, local_path in data_files.items():
                if os.path.exists(local_path):
                    with open(local_path, "rb") as f:
                        data = f.read()
                    # Ensure remote directory exists, then write file
                    remote_dir = os.path.dirname(remote_path)
                    if remote_dir:
                        sandbox.exec("mkdir", "-p", remote_dir).wait()
                    p = sandbox.exec("bash", "-c", f"cat > {remote_path}")
                    p.stdin.write(data)
                    p.stdin.write_eof()
                    p.wait()

        # Write the script and execute it
        script_path = f"/tmp/{algorithm_name}_run.py"
        p = sandbox.exec("bash", "-c", f"cat > {script_path}")
        p.stdin.write(code.encode())
        p.stdin.write_eof()
        p.wait()

        process = sandbox.exec("python", script_path)
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
    except (ValueError, RuntimeError) as e:
        return "", f"[ERROR] {e}", 1

    # Write code to a temp file
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", prefix=f"{algorithm_name}_", delete=False
    )
    try:
        tmp.write(code)
        tmp.close()

        cmd = [
            "docker", "run", "--rm",
            "--memory=4g", "--cpus=2",
            "-v", f"{tmp.name}:/workspace/script.py:ro",
        ]

        # Mount data files read-only
        if data_files:
            for remote_path, local_path in data_files.items():
                abs_local = os.path.abspath(local_path)
                if os.path.exists(abs_local):
                    cmd.extend(["-v", f"{abs_local}:{remote_path}:ro"])

        cmd.extend([image_tag, "python", "/workspace/script.py"])

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


