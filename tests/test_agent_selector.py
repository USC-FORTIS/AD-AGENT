import json
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from agent_test_stubs import install_common_stubs, load_real_agent_module

install_common_stubs()

# Stub sandbox.executor before loading agent_selector
import types, sys
sandbox_pkg = types.ModuleType("sandbox")
sandbox_executor_mod = types.ModuleType("sandbox.executor")
sandbox_pkg.executor = sandbox_executor_mod
sys.modules["sandbox"] = sandbox_pkg
sys.modules["sandbox.executor"] = sandbox_executor_mod

# Default mock: return pyod-style metadata
_META_PYOD = json.dumps({"format": ".mat", "num_samples": 100, "feature_dim": 5, "has_labels": True})
sandbox_executor_mod.execute_code = MagicMock(return_value=("META:" + _META_PYOD, "", 0))

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
        self.assertEqual(selector.feature_dim, 5)
        self.assertEqual(selector.metadata["num_samples"], 100)

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
            "dataset_test": "",
            "parameters": {},
        }
        meta = json.dumps({"format": ".csv", "num_samples": 10, "feature_dim": 3, "has_labels": False})
        with patch.object(agent_selector_mod.sandbox_executor_mod if hasattr(agent_selector_mod, "sandbox_executor_mod") else sandbox_executor_mod, "execute_code", return_value=("META:" + meta, "", 0)):
            selector = AgentSelector(user_input)

        self.assertEqual(selector.package_name, "tsb_ad")
        self.assertEqual(selector.tools, ["IForest"])

    def test_selector_rejects_test_dataset_for_tsb_ad_unsupervised_algorithm(self):
        user_input = {
            "algorithm": ["IForest"],
            "dataset_train": "./data/yahoo_train.csv",
            "dataset_test": "./data/yahoo_test.csv",
            "parameters": {},
        }

        with self.assertRaises(ValueError) as ctx:
            AgentSelector(user_input)

        self.assertIn("TSB-AD unsupervised algorithms accept only a training dataset", str(ctx.exception))

    def test_selector_allows_test_dataset_for_pyod_unsupervised_algorithm(self):
        user_input = {
            "algorithm": ["IForest"],
            "dataset_train": "./data/glass_train.mat",
            "dataset_test": "./data/glass_test.mat",
            "parameters": {},
        }

        selector = AgentSelector(user_input)

        self.assertEqual(selector.package_name, "pyod")
        self.assertEqual(selector.data_path_test, "./data/glass_test.mat")

    def test_selector_autofills_test_dataset_for_tsb_ad_semisupervised_algorithm(self):
        user_input = {
            "algorithm": ["SAND"],
            "dataset_train": "./data/yahoo_train.csv",
            "dataset_test": "",
            "parameters": {},
        }

        selector = AgentSelector(user_input)

        self.assertEqual(selector.data_path_test, "./data/yahoo_train.csv")
        self.assertEqual(selector.user_input["dataset_test"], "./data/yahoo_train.csv")

    def test_selector_requires_distinct_test_dataset_for_tsb_ad_semisupervised_algorithm(self):
        user_input = {
            "algorithm": ["SAND"],
            "dataset_train": "./data/yahoo_train.csv",
            "dataset_test": "./data/yahoo_train.csv",
            "parameters": {},
        }

        with self.assertRaises(ValueError) as ctx:
            AgentSelector(user_input)

        self.assertIn("dataset_test to be different from dataset_train", str(ctx.exception))

if __name__ == "__main__":
    unittest.main()
