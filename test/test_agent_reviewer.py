import types
import unittest
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np

from agent_test_stubs import install_common_stubs, load_real_agent_module

install_common_stubs()
agent_reviewer_mod = load_real_agent_module("agents.agent_reviewer")
AgentReviewer = agent_reviewer_mod.AgentReviewer


class TestAgentReviewer(unittest.TestCase):
    def setUp(self):
        self._print_patcher = patch("builtins.print")
        self._print_patcher.start()
        self.addCleanup(self._print_patcher.stop)

    def test_test_code_success(self):
        reviewer = AgentReviewer()
        with patch.object(
            agent_reviewer_mod.llm,
            "invoke",
            return_value=types.SimpleNamespace(content="```python\nprint('ok')\n```"),
        ), patch.object(
            reviewer,
            "_execute_test_script",
            return_value=types.SimpleNamespace(returncode=0, stdout="ok", stderr=""),
        ):
            err = reviewer.test_code("print('base')", "IForest", "pyod")
        self.assertEqual(err, "")

    def test_extract_tslib_root_path_from_cmd(self):
        reviewer = AgentReviewer()
        code = (
            'import subprocess\n'
            'cmd = ["python", "-u", "run.py", "--task_name", "anomaly_detection", '
            '"--is_training", "1", "--root_path", "./data/MSL", "--model_id", "MSL", '
            '"--model", "LightTS", "--data", "MSL", "--features", "M", "--seq_len", "100", '
            '"--pred_len", "0", "--d_model", "128", "--d_ff", "128", "--e_layers", "3", '
            '"--enc_in", "55", "--c_out", "55", "--batch_size", "128", "--train_epochs", "10"]\n'
            'subprocess.run(cmd)\n'
        )

        root = reviewer._extract_tslib_root_path(code)
        self.assertEqual(root, Path("./data/MSL"))

    def test_generate_tslib_synthetic_data_uses_prompt_output(self):
        reviewer = AgentReviewer()
        with tempfile.TemporaryDirectory() as td:
            tmp_dir = Path(td)
            train_path = tmp_dir / "MSL_train.npy"
            np.save(train_path, np.zeros((4, 55), dtype=np.float32))
            output_root = (tmp_dir / "generated_scripts" / "test_msl_root").resolve()
            script = f"""
import os
import numpy as np
root = r"{output_root}"
os.makedirs(root, exist_ok=True)
np.save(os.path.join(root, "MSL_train.npy"), np.zeros((4, 55), dtype=np.float32))
np.save(os.path.join(root, "MSL_test.npy"), np.zeros((3, 55), dtype=np.float32))
np.save(os.path.join(root, "MSL_test_label.npy"), np.zeros((3,), dtype=np.float32))
"""
            with patch.object(
                agent_reviewer_mod.llm,
                "invoke",
                return_value=types.SimpleNamespace(content=script),
            ):
                root = reviewer._generate_tslib_synthetic_data(
                    code=f'cmd = ["python", "-u", "run.py", "--data", "MSL", "--root_path", r"{output_root}"]',
                    algorithm_name="LightTS",
                    train_dataset=str(train_path),
                )

            self.assertEqual(root, output_root)
            self.assertTrue((output_root / "MSL_train.npy").exists())
            self.assertTrue((output_root / "MSL_test.npy").exists())
            self.assertTrue((output_root / "MSL_test_label.npy").exists())

    def test_test_code_detects_nested_failure_from_stdout(self):
        reviewer = AgentReviewer()
        with patch.object(
            agent_reviewer_mod.llm,
            "invoke",
            return_value=types.SimpleNamespace(content="```python\nprint('ok')\n```"),
        ), patch.object(
            reviewer,
            "_execute_test_script",
            return_value=types.SimpleNamespace(
                returncode=0,
                stdout="Traceback (most recent call last):\nboom",
                stderr="",
            ),
        ):
            err = reviewer.test_code("print('base')", "IForest", "pyod")
        self.assertIn("Traceback", err)


if __name__ == "__main__":
    unittest.main()
