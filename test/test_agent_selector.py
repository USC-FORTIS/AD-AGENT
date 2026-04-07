import json
import unittest
from unittest.mock import patch, MagicMock

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
        self.assertIn("IForest", tools)

    def test_detect_package_from_extension(self):
        self.assertEqual(AgentSelector._detect_package("data/train.mat"), "pyod")
        self.assertEqual(AgentSelector._detect_package("data/train.csv"), "pyod")
        self.assertEqual(AgentSelector._detect_package("data/train.pt"), "pygod")
        self.assertEqual(AgentSelector._detect_package("data/train.npy"), "tslib")
        self.assertEqual(AgentSelector._detect_package("data/MSL"), "tslib")
        self.assertEqual(AgentSelector._detect_package("data/PSM/"), "tslib")

if __name__ == "__main__":
    unittest.main()
