import types
import unittest
from unittest.mock import patch

from agent_test_stubs import install_common_stubs, load_real_agent_module

install_common_stubs()
agent_evaluator_mod = load_real_agent_module("agents.agent_evaluator")
AgentEvaluator = agent_evaluator_mod.AgentEvaluator


class TestAgentEvaluator(unittest.TestCase):
    def setUp(self):
        self._print_patcher = patch("builtins.print")
        self._print_patcher.start()
        self.addCleanup(self._print_patcher.stop)

    def test_execute_code_parses_metrics(self):
        evaluator = AgentEvaluator()
        out = "AUROC: 0.91\nAUPRC: 0.82\nFailed prediction at point [1,2] with true label 1"
        with patch(
            "subprocess.run",
            return_value=types.SimpleNamespace(returncode=0, stdout=out, stderr=""),
        ):
            cq = evaluator.execute_code("print('x')", "IForest")

        self.assertAlmostEqual(cq.auroc, 0.91)
        self.assertAlmostEqual(cq.auprc, 0.82)
        self.assertEqual(len(cq.error_points), 1)


if __name__ == "__main__":
    unittest.main()
