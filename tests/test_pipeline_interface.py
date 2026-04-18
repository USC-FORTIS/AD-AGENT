import os
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _install_stubs():
    class _DummyAgent:
        pass

    class _DummyConfig:
        OPENAI_API_KEY = "test-key"

    config_pkg = types.ModuleType("config")
    config_mod = types.ModuleType("config.config")
    config_mod.Config = _DummyConfig

    agents_pkg = types.ModuleType("agents")
    agents_pkg.__path__ = []
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
    agent_selector_mod = types.ModuleType("agents.agent_selector")
    agent_selector_mod.AgentSelector = _DummyAgent

    entity_pkg = types.ModuleType("models")
    entity_pkg.__path__ = []
    entity_code_quality_mod = types.ModuleType("models.code_quality")

    class _DummyCodeQuality:
        pass

    entity_code_quality_mod.CodeQuality = _DummyCodeQuality
    entity_pkg.code_quality = entity_code_quality_mod

    langchain_openai_mod = types.ModuleType("langchain_openai")

    class _DummyChatOpenAI:
        pass

    langchain_openai_mod.ChatOpenAI = _DummyChatOpenAI

    sys.modules.setdefault("config", config_pkg)
    sys.modules["config.config"] = config_mod
    sys.modules.setdefault("agents", agents_pkg)
    sys.modules["agents.agent_processor"] = agent_processor_mod
    sys.modules["agents.agent_info_miner"] = agent_info_miner_mod
    sys.modules["agents.agent_code_generator"] = agent_code_generator_mod
    sys.modules["agents.agent_reviewer"] = agent_reviewer_mod
    sys.modules["agents.agent_evaluator"] = agent_evaluator_mod
    sys.modules["agents.agent_optimizer"] = agent_optimizer_mod
    sys.modules["agents.agent_selector"] = agent_selector_mod
    sys.modules["models"] = entity_pkg
    sys.modules["models.code_quality"] = entity_code_quality_mod
    sys.modules["langchain_openai"] = langchain_openai_mod


_install_stubs()
from api import pipeline as pipeline_interface


class FakeCodeQuality:
    def __init__(
        self,
        code="print(1)",
        algorithm="IForest",
        parameters=None,
        std_output="",
        error_message="",
        auroc=0.0,
        auprc=0.0,
        error_points=None,
        review_count=0,
        accuracy=-1.0,
        f1=-1.0,
        specificity=-1.0,
        sensitivity=-1.0,
    ):
        self.code = code
        self.algorithm = algorithm
        self.parameters = parameters or {}
        self.std_output = std_output
        self.error_message = error_message
        self.auroc = auroc
        self.auprc = auprc
        self.error_points = error_points or []
        self.review_count = review_count
        self.accuracy = accuracy
        self.f1 = f1
        self.specificity = specificity
        self.sensitivity = sensitivity


class TestPipelineInterface(unittest.TestCase):
    def setUp(self):
        self._print_patcher = patch("builtins.print")
        self._print_patcher.start()
        self.addCleanup(self._print_patcher.stop)

    def test_check_dataset_exists_train_missing(self):
        with self.assertRaises(FileNotFoundError):
            pipeline_interface.check_dataset_exists("./not_exist_train.mat")

    def test_check_dataset_exists_passes_for_existing_files(self):
        with tempfile.NamedTemporaryFile() as train, tempfile.NamedTemporaryFile() as test:
            pipeline_interface.check_dataset_exists(train.name, test.name)

    @patch("api.pipeline.call_selector")
    def test_run_selector_with_state_uses_experiment_config(self, mock_call_selector):
        state = pipeline_interface.build_state()
        state["experiment_config"] = {
            "algorithm": ["IForest"],
            "dataset_train": "./data/glass_train.mat",
            "dataset_test": "./data/glass_test.mat",
            "parameters": {"contamination": 0.1},
        }

        result = pipeline_interface.run_selector(state=state)

        self.assertIs(result, state)
        mock_call_selector.assert_called_once_with(state)

    def test_run_selector_missing_train_raises(self):
        state = pipeline_interface.build_state()
        state["experiment_config"] = {"dataset_train": None, "dataset_test": None}

        with self.assertRaises(ValueError):
            pipeline_interface.run_selector(state=state)

    @patch("api.pipeline.call_info_miner", return_value={"algorithm_doc": "doc"})
    def test_run_info_miner_returns_doc_dict(self, mock_call_info_miner):
        state = pipeline_interface.build_state()
        state["current_tool"] = "IForest"
        state["package_name"] = "pyod"

        result = pipeline_interface.run_info_miner(state=state)

        self.assertEqual(result, {"algorithm_doc": "doc"})
        mock_call_info_miner.assert_called_once_with(state)

    def test_call_info_miner_returns_doc(self):
        class DummyInfoMiner:
            def query_docs(self, algorithm, package_name):
                return "doc"

            def query_metadata(self, algorithm, package_name):
                return {"dataset_root_arg": "root_path"}

        state = pipeline_interface.build_state()
        state["agent_info_miner"] = DummyInfoMiner()
        state["current_tool"] = "IForest"
        state["package_name"] = "pyod"

        result = pipeline_interface.call_info_miner(state)

        self.assertEqual(result, {"algorithm_doc": "doc"})

    def test_run_info_miner_requires_algorithm_and_package_without_state(self):
        with self.assertRaises(ValueError):
            pipeline_interface.run_info_miner("", "pyod")

    @patch("api.pipeline.call_code_generator_for_single_tool", return_value={"code_quality": "cq"})
    def test_run_code_generator_state_path(self, mock_call_codegen):
        state = pipeline_interface.build_state()
        result = pipeline_interface.run_code_generator(state=state)

        self.assertEqual(result, {"code_quality": "cq"})
        mock_call_codegen.assert_called_once_with(state)

    def test_run_code_generator_requires_inputs_without_state(self):
        with self.assertRaises(ValueError):
            pipeline_interface.run_code_generator(tool="IForest")

    @patch("api.pipeline.call_reviewer_for_single_tool", return_value={"code_quality": "reviewed"})
    def test_run_reviewer_state_path(self, mock_call_reviewer):
        state = pipeline_interface.build_state()
        result = pipeline_interface.run_reviewer(state=state)

        self.assertEqual(result, {"code_quality": "reviewed"})
        mock_call_reviewer.assert_called_once_with(state)

    def test_run_reviewer_requires_inputs_without_state(self):
        with self.assertRaises(ValueError):
            pipeline_interface.run_reviewer()

    @patch("api.pipeline.call_evaluator_for_single_tool", return_value={"code_quality": "evaluated"})
    def test_run_evaluator_state_path(self, mock_call_evaluator):
        state = pipeline_interface.build_state()
        result = pipeline_interface.run_evaluator(state=state)

        self.assertEqual(result, {"code_quality": "evaluated"})
        mock_call_evaluator.assert_called_once_with(state)

    @patch("api.pipeline.call_optimizer_for_single_tool", return_value={"code_quality": "optimized"})
    def test_run_optimizer_state_path(self, mock_call_optimizer):
        state = pipeline_interface.build_state()
        result = pipeline_interface.run_optimizer(state=state)

        self.assertEqual(result, {"code_quality": "optimized"})
        mock_call_optimizer.assert_called_once_with(state)

    def test_run_optimizer_requires_inputs_without_state(self):
        with self.assertRaises(ValueError):
            pipeline_interface.run_optimizer()

    @patch("api.pipeline.CodeQuality", FakeCodeQuality)
    def test_call_reviewer_increments_review_count_on_error(self):
        class DummyReviewer:
            def test_code(self, code, tool, package_name, feature_dim=None, train_dataset=None, dataset_metadata=None):
                self.metadata = dataset_metadata
                return "error"

        state = pipeline_interface.build_state()
        state["agent_reviewer"] = DummyReviewer()
        state["current_tool"] = "IForest"
        state["package_name"] = "pyod"
        state["metadata"] = {"dataset": {"train": {"n_features": 7}}}
        state["code_quality"] = FakeCodeQuality(review_count=0)

        result = pipeline_interface.call_reviewer_for_single_tool(state)

        self.assertEqual(result["code_quality"].review_count, 1)
        self.assertEqual(result["code_quality"].error_message, "error")
        self.assertEqual(state["agent_reviewer"].metadata, {"dataset": {"train": {"n_features": 7}}})

    @patch("api.pipeline.run_reviewer")
    @patch("api.pipeline.run_code_generator")
    def test_run_codegenerator_reviewer_loop_stops_on_success(self, mock_codegen, mock_reviewer):
        first = FakeCodeQuality(error_message="error", review_count=1)
        second = FakeCodeQuality(error_message="", review_count=1)
        mock_codegen.return_value = {"code_quality": first}
        mock_reviewer.return_value = {"code_quality": second}

        result = pipeline_interface.run_codegenerator_reviewer_loop(
            tool="IForest",
            data_path_train="./data/glass_train.mat",
            algorithm_doc="doc",
            package_name="pyod",
            data_path_test="./data/glass_test.mat",
            max_reviews=2,
        )

        self.assertEqual(result["code_quality"].error_message, "")
        self.assertEqual(mock_codegen.call_count, 1)
        self.assertEqual(mock_reviewer.call_count, 1)

    @patch("api.pipeline.run_reviewer")
    @patch("api.pipeline.run_code_generator")
    def test_run_codegenerator_reviewer_loop_stops_on_max_reviews(self, mock_codegen, mock_reviewer):
        cq = FakeCodeQuality(error_message="still_error", review_count=2)
        mock_codegen.return_value = {"code_quality": cq}
        mock_reviewer.return_value = {"code_quality": cq}

        result = pipeline_interface.run_codegenerator_reviewer_loop(
            tool="IForest",
            data_path_train="./data/glass_train.mat",
            algorithm_doc="doc",
            package_name="pyod",
            max_reviews=2,
        )

        self.assertEqual(result["code_quality"].review_count, 2)
        self.assertEqual(mock_codegen.call_count, 1)
        self.assertEqual(mock_reviewer.call_count, 1)

    @patch("api.pipeline.run_optimizer")
    @patch("api.pipeline.run_evaluator")
    def test_run_evaluator_optimizer_loop_returns_on_initial_error(self, mock_evaluator, mock_optimizer):
        err_cq = FakeCodeQuality(error_message="eval_error")
        mock_evaluator.return_value = {"code_quality": err_cq}

        result = pipeline_interface.run_evaluator_optimizer_loop(
            cq=FakeCodeQuality(),
            tool="IForest",
            algorithm_doc="doc",
            optimizer_cycles=2,
        )

        self.assertEqual(result["code_quality"].error_message, "eval_error")
        mock_optimizer.assert_not_called()

    @patch("api.pipeline.run_optimizer")
    @patch("api.pipeline.run_evaluator")
    def test_run_evaluator_optimizer_loop_runs_cycles(self, mock_evaluator, mock_optimizer):
        eval_ok_1 = {"code_quality": FakeCodeQuality(error_message="", auroc=0.7)}
        eval_ok_2 = {"code_quality": FakeCodeQuality(error_message="", auroc=0.8)}
        opt_ok = {"code_quality": FakeCodeQuality(error_message="")}

        mock_evaluator.side_effect = [eval_ok_1, eval_ok_2]
        mock_optimizer.return_value = opt_ok

        result = pipeline_interface.run_evaluator_optimizer_loop(
            cq=FakeCodeQuality(),
            tool="IForest",
            algorithm_doc="doc",
            optimizer_cycles=1,
        )

        self.assertEqual(result["code_quality"].auroc, 0.8)
        self.assertEqual(mock_optimizer.call_count, 1)
        self.assertEqual(mock_evaluator.call_count, 2)


if __name__ == "__main__":
    unittest.main()
