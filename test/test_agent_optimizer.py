import unittest
from unittest.mock import patch

from agent_test_stubs import install_common_stubs, load_real_agent_module
from entity.code_quality import CodeQuality

install_common_stubs()
agent_optimizer_mod = load_real_agent_module("agents.agent_optimizer")
AgentOptimizer = agent_optimizer_mod.AgentOptimizer


class TestAgentOptimizer(unittest.TestCase):
    def test_extract_param_dict(self):
        text = "Thought: t\nAction: execute_code({'a': 1, 'b': 2})"
        parsed = AgentOptimizer._extract_param_dict(text)
        self.assertEqual(parsed, {"a": 1, "b": 2})

    def test_execute_code_returns_error_when_model_line_missing(self):
        out = AgentOptimizer.execute_code({"a": 1}, "print('no model')", "IForest")
        self.assertIn("Model instantiation line not found", out)

    def test_run_parses_tslib_metrics(self):
        optimizer = AgentOptimizer()

        class DummyLLM:
            def invoke(self, _messages):
                class _Resp:
                    content = "Final: done"
                return _Resp()

        quality = CodeQuality(
            code="model = TimesNet()",
            algorithm="TimesNet",
            parameters={},
            std_output="",
            error_message="",
            auroc=-1,
            auprc=-1,
            error_points=[],
            review_count=0,
        )

        dummy_prompt = type("DummyPrompt", (), {"format": staticmethod(lambda **kwargs: "system prompt")})()

        with patch.object(
            agent_optimizer_mod,
            "SYSTEM_PROMPT_TMPL",
            dummy_prompt,
        ), patch.object(
            AgentOptimizer,
            "execute_code",
            return_value="Accuracy : 0.9647, Precision : 0.8956, Recall : 0.7531, F-score : 0.8182",
        ):
            result = optimizer.run(
                llm=DummyLLM(),
                quality=quality,
                algorithm_doc="doc",
                package_name="tslib",
                max_steps=1,
            )

        self.assertAlmostEqual(result.accuracy, 0.9647)
        self.assertAlmostEqual(result.specificity, 0.8956)
        self.assertAlmostEqual(result.sensitivity, 0.7531)
        self.assertAlmostEqual(result.f1, 0.8182)


if __name__ == "__main__":
    unittest.main()
