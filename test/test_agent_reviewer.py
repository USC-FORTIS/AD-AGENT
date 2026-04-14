import unittest
import types
from unittest.mock import patch

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
