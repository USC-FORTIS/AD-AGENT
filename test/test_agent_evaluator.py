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

    def test_execute_code_detects_nested_failure_in_stdout(self):
        evaluator = AgentEvaluator()
        out = "Traceback (most recent call last):\nNotImplementedError: boom"
        with patch(
            "subprocess.run",
            return_value=types.SimpleNamespace(returncode=0, stdout=out, stderr=""),
        ):
            cq = evaluator.execute_code("print('x')", "IForest")

        self.assertEqual(cq.error_message, "NotImplementedError: boom")
        self.assertEqual(cq.auroc, -1)
        self.assertEqual(cq.auprc, -1)

    def test_execute_code_parses_tslib_metrics(self):
        evaluator = AgentEvaluator()
        out = (
            "Threshold : 9.5\n"
            "Accuracy : 0.9647, Precision : 0.8956, Recall : 0.7531, F-score : 0.8182 \n"
        )
        with patch(
            "subprocess.run",
            return_value=types.SimpleNamespace(returncode=0, stdout=out, stderr=""),
        ):
            cq = evaluator.execute_code("print('x')", "TimesNet", "tslib")

        self.assertAlmostEqual(cq.accuracy, 0.9647)
        self.assertAlmostEqual(cq.specificity, 0.8956)
        self.assertAlmostEqual(cq.sensitivity, 0.7531)
        self.assertAlmostEqual(cq.f1, 0.8182)
        self.assertEqual(cq.auroc, -1)
        self.assertEqual(cq.auprc, -1)


if __name__ == "__main__":
    unittest.main()
