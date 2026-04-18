import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import sys

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils import tslib_setup


class _Result:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class TestTSLibSetup(unittest.TestCase):
    def test_prepare_tslib_repo_clones_and_installs_when_missing(self):
        calls = []

        def fake_run(cmd, cwd, check, text, capture_output):
            calls.append((cmd, cwd, capture_output))
            if cmd[:2] == ["git", "clone"]:
                repo_dir = Path(cmd[-1])
                repo_dir.mkdir(parents=True, exist_ok=True)
                (repo_dir / "requirements.txt").write_text("numpy\n", encoding="utf-8")
                return _Result()
            if cmd[:4] == ["git", "-C", cmd[2], "rev-parse"]:
                return _Result(stdout="abc123\n")
            if cmd[1:3] == ["-m", "pip"]:
                return _Result()
            raise AssertionError(f"Unexpected command: {cmd}")

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("utils.tslib_setup.subprocess.run", side_effect=fake_run):
                repo_dir = tslib_setup.prepare_tslib_repo(project_root=tmpdir)

            self.assertTrue((repo_dir / ".ad_agent_tslib_install.json").exists())
            self.assertEqual(calls[0][0][:2], ["git", "clone"])
            self.assertEqual(calls[1][0][3:], ["rev-parse", "HEAD"])
            self.assertEqual(calls[2][0][1:3], ["-m", "pip"])

    def test_prepare_tslib_repo_pulls_and_skips_install_when_stamp_matches(self):
        calls = []

        def fake_run(cmd, cwd, check, text, capture_output):
            calls.append((cmd, cwd, capture_output))
            if cmd[:4] == ["git", "-C", cmd[2], "pull"]:
                return _Result()
            if cmd[:4] == ["git", "-C", cmd[2], "rev-parse"]:
                return _Result(stdout="same-head\n")
            raise AssertionError(f"Unexpected command: {cmd}")

        with tempfile.TemporaryDirectory() as tmpdir:
            repo_dir = Path(tmpdir) / tslib_setup.TSLIB_REPO_NAME
            repo_dir.mkdir(parents=True, exist_ok=True)
            requirements = repo_dir / "requirements.txt"
            requirements.write_text("numpy\n", encoding="utf-8")
            stamp = {
                "head": "same-head",
                "requirements_hash": tslib_setup._file_hash(requirements),
            }
            (repo_dir / ".ad_agent_tslib_install.json").write_text(
                json.dumps(stamp),
                encoding="utf-8",
            )

            with patch("utils.tslib_setup.subprocess.run", side_effect=fake_run):
                tslib_setup.prepare_tslib_repo(project_root=tmpdir)

        self.assertEqual(calls[0][0][3:], ["pull", "--ff-only"])
        self.assertEqual(calls[1][0][3:], ["rev-parse", "HEAD"])
        self.assertEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()
