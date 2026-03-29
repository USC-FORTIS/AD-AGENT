import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


TSLIB_REPO_NAME = "Time-Series-Library"
TSLIB_REPO_URL = "https://github.com/thuml/Time-Series-Library.git"
_STAMP_FILE = ".ad_agent_tslib_install.json"


def prepare_tslib_repo(project_root: str | None = None) -> Path:
    root = Path(project_root or os.getcwd()).resolve()
    repo_dir = root / TSLIB_REPO_NAME

    if repo_dir.exists():
        _sync_existing_repo(repo_dir)
    else:
        _clone_repo(root, repo_dir)

    _install_requirements_if_needed(repo_dir)
    return repo_dir


def _sync_existing_repo(repo_dir: Path) -> None:
    _run(
        ["git", "-C", str(repo_dir), "pull", "--ff-only"],
        cwd=str(repo_dir.parent),
        error_prefix=f"Failed to update {TSLIB_REPO_NAME}",
    )


def _clone_repo(root: Path, repo_dir: Path) -> None:
    _run(
        ["git", "clone", TSLIB_REPO_URL, str(repo_dir)],
        cwd=str(root),
        error_prefix=f"Failed to clone {TSLIB_REPO_NAME}",
    )


def _install_requirements_if_needed(repo_dir: Path) -> None:
    requirements_path = repo_dir / "requirements.txt"
    if not requirements_path.exists():
        raise FileNotFoundError(f"Missing requirements file at {requirements_path}")

    head = _git_head(repo_dir)
    requirements_hash = _file_hash(requirements_path)
    stamp_path = repo_dir / _STAMP_FILE
    stamp = _read_stamp(stamp_path)

    if (
        stamp.get("head") == head
        and stamp.get("requirements_hash") == requirements_hash
    ):
        return

    _run(
        [sys.executable, "-m", "pip", "install", "-r", str(requirements_path)],
        cwd=str(repo_dir),
        error_prefix=f"Failed to install dependencies for {TSLIB_REPO_NAME}",
    )

    stamp_path.write_text(
        json.dumps(
            {"head": head, "requirements_hash": requirements_hash},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _git_head(repo_dir: Path) -> str:
    result = _run(
        ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
        cwd=str(repo_dir.parent),
        error_prefix=f"Failed to read HEAD for {TSLIB_REPO_NAME}",
        capture_output=True,
    )
    return result.stdout.strip()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _read_stamp(stamp_path: Path) -> dict:
    if not stamp_path.exists():
        return {}
    try:
        return json.loads(stamp_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _run(
    cmd: list[str],
    cwd: str,
    error_prefix: str,
    capture_output: bool = False,
) -> subprocess.CompletedProcess:
    result = subprocess.run(
        cmd,
        cwd=cwd,
        check=False,
        text=True,
        capture_output=capture_output,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        suffix = f": {detail}" if detail else ""
        raise RuntimeError(f"{error_prefix}{suffix}")
    return result
