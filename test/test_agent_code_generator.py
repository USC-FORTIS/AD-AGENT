import types
import unittest
from unittest.mock import patch

from agent_test_stubs import install_common_stubs, load_real_agent_module

install_common_stubs()
agent_code_generator_mod = load_real_agent_module("agents.agent_code_generator")
AgentCodeGenerator = agent_code_generator_mod.AgentCodeGenerator


class TestAgentCodeGenerator(unittest.TestCase):
    def test_extract_init_params_dict(self):
        text = """```python\n{'contamination': 0.1, 'n_neighbors': 5}\n```"""
        parsed = AgentCodeGenerator._extract_init_params_dict(text)
        self.assertEqual(parsed, {"contamination": 0.1, "n_neighbors": 5})

    def test_generate_code_returns_cleaned_content(self):
        agent = AgentCodeGenerator()
        with patch.object(
            agent_code_generator_mod.llm,
            "invoke",
            return_value=types.SimpleNamespace(content="```python\nprint('x')\n```"),
        ):
            code = agent.generate_code(
                algorithm="IForest",
                data_path_train="./data/glass_train.mat",
                data_path_test="./data/glass_test.mat",
                algorithm_doc="doc",
                input_parameters={},
                package_name="pyod",
            )
        self.assertEqual(code, "print('x')")


if __name__ == "__main__":
    unittest.main()
