import unittest
from unittest.mock import patch
import sys
import types


openai_stub = types.ModuleType("openai")
openai_stub.chat = types.SimpleNamespace(
    completions=types.SimpleNamespace(create=lambda **kwargs: None)
)
sys.modules.setdefault("openai", openai_stub)

from agents.agent_processor import AgentProcessor


class TestAgentProcessorRunChatbot(unittest.TestCase):
    def setUp(self):
        self._print_patcher = patch("builtins.print")
        self._print_patcher.start()
        self.addCleanup(self._print_patcher.stop)

    def test_run_chatbot_single_valid_input(self):
        # ECOD is a pyod-only algorithm, not in Unsupervise_AD_Pool or Semisupervise_AD_Pool,
        # so both train and test datasets must be preserved unchanged.
        processor = AgentProcessor()

        with patch("builtins.input", side_effect=["Run ECOD on train and test"]), \
             patch.object(processor, "get_chatgpt_response", return_value="ok"), \
             patch.object(
                 processor,
                 "extract_config",
                 return_value={
                     "algorithm": ["ECOD"],
                     "dataset_train": "./data/glass_train.mat",
                     "dataset_test": "./data/glass_test.mat",
                     "parameters": {"contamination": 0.1},
                 },
             ), \
             patch("os.path.exists", side_effect=lambda p: p in {"./data/glass_train.mat", "./data/glass_test.mat"}):
            processor.run_chatbot()

        self.assertEqual(processor.experiment_config["algorithm"], ["ECOD"])
        self.assertEqual(processor.experiment_config["dataset_train"], "./data/glass_train.mat")
        self.assertEqual(processor.experiment_config["dataset_test"], "./data/glass_test.mat")
        self.assertEqual(processor.experiment_config["parameters"], {"contamination": 0.1})

    def test_run_chatbot_unsupervised_keeps_test_dataset_for_selector(self):
        # Processor should preserve user input and let selector enforce tsb_ad-specific constraints.
        processor = AgentProcessor()

        with patch("builtins.input", side_effect=["Run IForest on train and test"]), \
             patch.object(processor, "get_chatgpt_response", return_value="ok"), \
             patch.object(
                 processor,
                 "extract_config",
                 return_value={
                     "algorithm": ["IForest"],
                     "dataset_train": "./data/glass_train.mat",
                     "dataset_test": "./data/glass_test.mat",
                     "parameters": {},
                 },
            ), \
             patch("os.path.exists", side_effect=lambda p: p in {"./data/glass_train.mat", "./data/glass_test.mat"}):
            processor.run_chatbot()

        self.assertEqual(processor.experiment_config["dataset_train"], "./data/glass_train.mat")
        self.assertEqual(processor.experiment_config["dataset_test"], "./data/glass_test.mat")

    def test_run_chatbot_skips_empty_input(self):
        processor = AgentProcessor()

        with patch("builtins.input", side_effect=["", "Run IForest"]), \
             patch.object(processor, "get_chatgpt_response", return_value="ok") as mock_llm, \
             patch.object(
                 processor,
                 "extract_config",
                 return_value={
                     "algorithm": ["IForest"],
                     "dataset_train": "./data/glass_train.mat",
                     "dataset_test": "./data/glass_test.mat",
                     "parameters": {},
                 },
             ), \
             patch("os.path.exists", side_effect=lambda p: p in {"./data/glass_train.mat", "./data/glass_test.mat"}):
            processor.run_chatbot()

        self.assertEqual(mock_llm.call_count, 1)

    def test_run_chatbot_retries_when_train_path_invalid(self):
        processor = AgentProcessor()

        extract_side_effect = [
            {
                "algorithm": ["IForest"],
                "dataset_train": "./missing_train.mat",
                "dataset_test": "",
                "parameters": {},
            },
            {
                "algorithm": ["IForest"],
                "dataset_train": "./data/glass_train.mat",
                "dataset_test": "./data/glass_test.mat",
                "parameters": {},
            },
        ]

        with patch("builtins.input", side_effect=["first", "second"]), \
             patch.object(processor, "get_chatgpt_response", return_value="ok"), \
             patch.object(processor, "extract_config", side_effect=extract_side_effect), \
             patch("os.path.exists", side_effect=lambda p: p in {"./data/glass_train.mat", "./data/glass_test.mat"}), \
             patch("builtins.print") as mock_print:
            processor.run_chatbot()

        printed_text = "\n".join(str(call.args[0]) for call in mock_print.call_args_list if call.args)
        self.assertIn("Please provide a valid training dataset location", printed_text)
        self.assertEqual(processor.experiment_config["dataset_train"], "./data/glass_train.mat")


if __name__ == "__main__":
    unittest.main()
