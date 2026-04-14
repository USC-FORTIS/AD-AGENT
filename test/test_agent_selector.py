import unittest
from unittest.mock import patch
import numpy as np

from agent_test_stubs import install_common_stubs, load_real_agent_module

install_common_stubs()
agent_selector_mod = load_real_agent_module("agents.agent_selector")
AgentSelector = agent_selector_mod.AgentSelector
algorithm_registry = load_real_agent_module("utils.algorithm_registry")


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
        self.assertEqual(tools, algorithm_registry.PYOD_ALGORITHMS)
        self.assertIn("IForest", tools)

    def test_generate_tools_all_pygod(self):
        selector = AgentSelector.__new__(AgentSelector)
        selector.package_name = "pygod"
        tools = selector.generate_tools(["all"])
        self.assertEqual(tools, algorithm_registry.PYGOD_ALGORITHMS)
        self.assertIn("DOMINANT", tools)

    def test_generate_tools_all_tsb_ad_uses_available_algorithms(self):
        selector = AgentSelector.__new__(AgentSelector)
        selector.package_name = "tsb_ad"

        tools = selector.generate_tools(["all"])

        self.assertIn("IForest", tools)
        self.assertIn("TimesFM", tools)
        self.assertNotIn("TimesNet", tools)
        self.assertNotIn("TranAD", tools)

    def test_selector_init_tsb_ad_for_time_series_csv_algorithm(self):
        user_input = {
            "algorithm": ["IForest"],
            "dataset_train": "./data/yahoo_train.csv",
            "dataset_test": "./data/yahoo_test.csv",
            "parameters": {},
        }

        def fake_load_data(self, split_data=False):
            return np.zeros((10, 3)), "time-series"

        with patch.object(agent_selector_mod.DataLoader, "load_data", fake_load_data), patch("os.path.exists", return_value=False):
            selector = AgentSelector(user_input)

        self.assertEqual(selector.package_name, "tsb_ad")
        self.assertEqual(selector.tools, ["IForest"])

if __name__ == "__main__":
    unittest.main()
