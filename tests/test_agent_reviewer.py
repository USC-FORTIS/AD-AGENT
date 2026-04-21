import unittest
import types
from unittest.mock import patch

import numpy as np

from utils.script_paths import reviewer_script_path
from agent_test_stubs import install_common_stubs, load_real_agent_module

install_common_stubs()
agent_reviewer_mod = load_real_agent_module("agents.agent_reviewer")
AgentReviewer = agent_reviewer_mod.AgentReviewer


class TestAgentReviewer(unittest.TestCase):
    def setUp(self):
        self._print_patcher = patch("builtins.print")
        self._print_patcher.start()
        self.addCleanup(self._print_patcher.stop)

    def test_test_code_passes_feature_dim_to_prompt(self):
        reviewer = AgentReviewer()
        captured = {}

        def fake_invoke(prompt_value):
            captured["prompt"] = prompt_value.to_string()
            return types.SimpleNamespace(content="```python\nprint('ok')\n```")

        with patch.object(agent_reviewer_mod.llm, "invoke", side_effect=fake_invoke), patch.object(
            reviewer,
            "_execute_test_script",
            return_value=types.SimpleNamespace(returncode=0, stdout="ok", stderr=""),
        ):
            err = reviewer.test_code(
                "print('base')",
                "SAND",
                "tsb_ad",
                feature_dim=7,
                train_dataset="./data/swat_train.csv",
                dataset_metadata={"feature_dim": 7},
            )

        self.assertEqual(err, "")
        self.assertIn("feature_dim=7", captured["prompt"])
        self.assertIn("train_length >= 1500", captured["prompt"])
        self.assertIn("periodic base signal", captured["prompt"])
        self.assertIn("base = base[:, None]", captured["prompt"])
        self.assertIn("Do not invent fallback kwargs such as `window_size`, `threshold`, `normalize`, or `verbose`", captured["prompt"])

    def test_test_code_pygod_prompt_requests_small_graphs_and_short_training(self):
        reviewer = AgentReviewer()
        captured = {}

        def fake_invoke(prompt_value):
            captured["prompt"] = prompt_value.to_string()
            return types.SimpleNamespace(content="```python\nprint('ok')\n```")

        with patch.object(agent_reviewer_mod.llm, "invoke", side_effect=fake_invoke), patch.object(
            reviewer,
            "_execute_test_script",
            return_value=types.SimpleNamespace(returncode=0, stdout="ok", stderr=""),
        ):
            reviewer.test_code(
                "print('base')",
                "AnomalyDAE",
                "pygod",
                feature_dim=32,
                train_dataset="./data/inj_cora_train.pt",
                dataset_metadata={"num_features": 32},
            )

        self.assertIn("num_nodes <= 400", captured["prompt"])
        self.assertIn("num_edges <= 1600", captured["prompt"])
        self.assertIn("epoch", captured["prompt"])

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

    def test_run_script_for_validation_passes_package_name(self):
        reviewer = AgentReviewer()
        script_path = str(reviewer_script_path("IForest"))
        with patch.object(
            reviewer,
            "_execute_test_script",
            return_value=types.SimpleNamespace(returncode=0, stdout="ok", stderr=""),
        ) as mock_exec:
            err = reviewer._run_script_for_validation(script_path, "IForest", "pyod")

        self.assertEqual(err, "")
        mock_exec.assert_called_once_with(script_path, "IForest", "pyod")

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
