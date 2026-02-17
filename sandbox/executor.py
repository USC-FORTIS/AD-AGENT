import os
import subprocess

from sandbox.config import SANDBOX_MODE, MODAL_APP_NAME, DEFAULT_TIMEOUT


def execute_code(
    code: str,
    algorithm_name: str,
    package_name: str,
    data_files: dict[str, str] | None = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> tuple[str, str, int]:
    """
    Execute a Python script in either a Modal Sandbox or locally via subprocess.

    Args:
        code: Python source code to execute.
        algorithm_name: Name of the algorithm (used for script naming in local mode).
        package_name: Package name to select the right Modal image.
        data_files: Optional dict mapping {remote_path: local_path} for files to
                    upload into the sandbox before execution.
        timeout: Max execution time in seconds.

    Returns:
        (stdout, stderr, returncode)
    """
    if SANDBOX_MODE == "modal":
        return _execute_modal(code, algorithm_name, package_name, data_files, timeout)
    else:
        return _execute_locally(code, algorithm_name, package_name, timeout)


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


def _execute_locally(
    code: str,
    algorithm_name: str,
    package_name: str,
    timeout: int,
) -> tuple[str, str, int]:
    """Execute code locally using subprocess (fallback / development mode)."""
    folder = "./generated_scripts"
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{algorithm_name}.py")

    with open(path, "w", encoding="utf-8") as f:
        f.write(code)

    try:
        result = subprocess.run(
            ["python", path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired:
        return "", "[ERROR] Execution timed out.", 1
