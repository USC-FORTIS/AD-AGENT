import sys
import tempfile
import types
import unittest
from unittest.mock import patch


def _install_stubs():
    class _DummyAgent:
        pass

    class _DummyConfig:
        OPENAI_API_KEY = "test-key"

    def _noop(state):
        return state

    config_pkg = types.ModuleType("config")
    config_mod = types.ModuleType("config.config")
    config_mod.Config = _DummyConfig

    agents_pkg = types.ModuleType("agents")
    agent_processor_mod = types.ModuleType("agents.agent_processor")
    agent_processor_mod.AgentProcessor = _DummyAgent
    agent_info_miner_mod = types.ModuleType("agents.agent_info_miner")
    agent_info_miner_mod.AgentInfoMiner = _DummyAgent
    agent_code_generator_mod = types.ModuleType("agents.agent_code_generator")
    agent_code_generator_mod.AgentCodeGenerator = _DummyAgent
    agent_reviewer_mod = types.ModuleType("agents.agent_reviewer")
    agent_reviewer_mod.AgentReviewer = _DummyAgent
    agent_evaluator_mod = types.ModuleType("agents.agent_evaluator")
    agent_evaluator_mod.AgentEvaluator = _DummyAgent
    agent_optimizer_mod = types.ModuleType("agents.agent_optimizer")
    agent_optimizer_mod.AgentOptimizer = _DummyAgent

    main_mod = types.ModuleType("main")
    main_mod.call_processor = _noop
    main_mod.call_selector = _noop
    main_mod.call_info_miner = _noop
    main_mod.call_code_generator_for_single_tool = _noop
    main_mod.call_reviewer_for_single_tool = _noop
    main_mod.call_evaluator_for_single_tool = _noop
    main_mod.call_optimizer_for_single_tool = _noop

    sys.modules.setdefault("config", config_pkg)
    sys.modules["config.config"] = config_mod
    sys.modules.setdefault("agents", agents_pkg)
    sys.modules["agents.agent_processor"] = agent_processor_mod
    sys.modules["agents.agent_info_miner"] = agent_info_miner_mod
    sys.modules["agents.agent_code_generator"] = agent_code_generator_mod
    sys.modules["agents.agent_reviewer"] = agent_reviewer_mod
    sys.modules["agents.agent_evaluator"] = agent_evaluator_mod
    sys.modules["agents.agent_optimizer"] = agent_optimizer_mod
    sys.modules["main"] = main_mod


_install_stubs()
import pipeline_api


class TestPipelineAPI(unittest.TestCase):
    def test_check_dataset_exists_train_missing(self):
        with self.assertRaises(FileNotFoundError):
            pipeline_api.check_dataset_exists("./not_exist_train.mat")

    def test_check_dataset_exists_passes_for_existing_files(self):
        with tempfile.NamedTemporaryFile() as train, tempfile.NamedTemporaryFile() as test:
            pipeline_api.check_dataset_exists(train.name, test.name)

    @patch("pipeline_api.call_selector")
    @patch("pipeline_api.check_dataset_exists")
    def test_run_selector_builds_experiment_config(self, mock_check_dataset, mock_call_selector):
        state = pipeline_api.run_selector(
            algorithm=["IForest"],
            dataset_train="./data/glass_train.mat",
            dataset_test="./data/glass_test.mat",
            parameters={"contamination": 0.1},
        )

        self.assertEqual(state["experiment_config"]["algorithm"], ["IForest"])
        self.assertEqual(state["experiment_config"]["dataset_train"], "./data/glass_train.mat")
        self.assertEqual(state["experiment_config"]["dataset_test"], "./data/glass_test.mat")
        self.assertEqual(state["experiment_config"]["parameters"], {"contamination": 0.1})
        mock_check_dataset.assert_called_once_with("./data/glass_train.mat", "./data/glass_test.mat")
        mock_call_selector.assert_called_once()

    def test_run_info_miner_requires_algorithm_and_package(self):
        with self.assertRaises(ValueError):
            pipeline_api.run_info_miner("", "pyod")


if __name__ == "__main__":
    unittest.main()
