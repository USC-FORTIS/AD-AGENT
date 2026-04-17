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



if __name__ == "__main__":
    unittest.main()
