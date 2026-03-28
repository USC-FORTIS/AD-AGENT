import unittest
from unittest.mock import patch

from agent_test_stubs import install_common_stubs, load_real_agent_module

install_common_stubs()
agent_selector_mod = load_real_agent_module("agents.agent_selector")
AgentSelector = agent_selector_mod.AgentSelector


class TestAgentSelector(unittest.TestCase):
    def setUp(self):
        self._print_patcher = patch("builtins.print")
        self._print_patcher.start()
        self.addCleanup(self._print_patcher.stop)

    def test_selector_init_pyod_with_algorithm(self):
        user_input = {
            "algorithm": ["IForest"],
            "dataset_train": "./data/glass_train.mat",
            "dataset_test": "",
            "parameters": {},
        }
        with patch("os.path.exists", return_value=False):
            selector = AgentSelector(user_input)

        self.assertEqual(selector.package_name, "pyod")
        self.assertEqual(selector.tools, ["IForest"])

    def test_generate_tools_all_pyod(self):
        selector = AgentSelector.__new__(AgentSelector)
        selector.package_name = "pyod"
        tools = selector.generate_tools(["all"])
        self.assertIn("IForest", tools)

    def test_resolve_time_series_package_prefers_tsbad_for_non_darts_algo(self):
        selector = AgentSelector.__new__(AgentSelector)
        selector.user_input = {"algorithm": ["IForest"]}

        package_name = selector._resolve_time_series_package()

        self.assertEqual(package_name, "tsbad")


if __name__ == "__main__":
    unittest.main()
