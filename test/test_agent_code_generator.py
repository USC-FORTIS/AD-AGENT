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
                dataset_metadata={"dataset": {"train": {"file_type": "mat"}}},
            )
        self.assertEqual(code, "print('x')")

    def test_generate_code_tsb_ad_uses_python_api_wrapper(self):
        agent = AgentCodeGenerator()
        with patch.object(
            agent_code_generator_mod.llm,
            "invoke",
            return_value=types.SimpleNamespace(content="```python\nprint('tsb_ad')\n```"),
        ):
            code = agent.generate_code(
                algorithm="IForest",
                data_path_train="./data/yahoo_train.csv",
                data_path_test="./data/yahoo_test.csv",
                algorithm_doc="doc",
                input_parameters={"window_size": 32},
                package_name="tsb_ad",
                dataset_metadata={"dataset": {"train": {"file_type": "csv"}}},
            )

        self.assertEqual(code, "print('tsb_ad')")

    # ------------------------------------------------------------------ #
    # Template selection: unsupervised vs semisupervised                  #
    # ------------------------------------------------------------------ #

    def test_tsb_ad_unsupervised_algorithm_uses_unlabeled_template(self):
        """Unsupervised algorithms (Unsupervise_AD_Pool) must use the unlabeled template."""
        agent = AgentCodeGenerator()
        captured = {}

        def fake_invoke(prompt):
            captured["prompt"] = prompt.to_string()
            return types.SimpleNamespace(content="print('unlabeled')")

        with patch.object(agent_code_generator_mod.llm, "invoke", side_effect=fake_invoke):
            agent.generate_code(
                algorithm="IForest",   # IForest is in Unsupervise_AD_Pool
                data_path_train="./data/train.npy",
                data_path_test="./data/train.npy",  # same file (set by selector)
                algorithm_doc="doc",
                input_parameters={},
                package_name="tsb_ad",
            )

        # Unlabeled template loads only from train path, no X_test loading
        self.assertIn("./data/train.npy", captured["prompt"])
        self.assertNotIn("X_test", captured["prompt"])

    def test_tsb_ad_semisupervised_algorithm_uses_labeled_template(self):
        """Semisupervised algorithms (Semisupervise_AD_Pool) must use the labeled template."""
        agent = AgentCodeGenerator()
        captured = {}

        def fake_invoke(prompt):
            captured["prompt"] = prompt.to_string()
            return types.SimpleNamespace(content="print('labeled')")

        with patch.object(agent_code_generator_mod.llm, "invoke", side_effect=fake_invoke):
            agent.generate_code(
                algorithm="SAND",   # SAND is in Semisupervise_AD_Pool
                data_path_train="./data/train.npy",
                data_path_test="./data/train.npy",
                algorithm_doc="doc",
                input_parameters={},
                package_name="tsb_ad",
            )

        # Labeled template loads both X_train and X_test
        self.assertIn("./data/train.npy", captured["prompt"])
        self.assertIn("X_train", captured["prompt"])
        self.assertIn("X_test", captured["prompt"])

    # ------------------------------------------------------------------ #
    # template_tsb_ad_labeled content checks                              #
    # ------------------------------------------------------------------ #

    def test_tsb_ad_labeled_prompt_single_call_with_train_and_test(self):
        """Labeled template must instruct a single model_runner(X_train, X_test) call."""
        rendered = agent_code_generator_mod.template_tsb_ad_labeled.invoke(
            {
                "algorithm": "SAND",
                "data_path_train": "./data/MSL_train.npy",
                "data_path_test": "./data/MSL_train.npy",
                "algorithm_doc": "official doc",
                "parameters": "{}",
                "dataset_metadata": "{}",
            }
        ).to_string()

        self.assertIn("model_runner(X_train, X_test, **run_kwargs)", rendered)
        self.assertIn("Do NOT call the wrapper twice", rendered)

    def test_tsb_ad_labeled_prompt_univariate_reduction_hint(self):
        """Labeled template must include the 1D reduction hint for univariate-only algorithms."""
        rendered = agent_code_generator_mod.template_tsb_ad_labeled.invoke(
            {
                "algorithm": "SAND",
                "data_path_train": "./data/MSL_train.npy",
                "data_path_test": "./data/MSL_train.npy",
                "algorithm_doc": "official doc",
                "parameters": "{}",
                "dataset_metadata": "{}",
            }
        ).to_string()

        self.assertIn("X_train[:, 0]", rendered)
        self.assertIn("X_test[:, 0]", rendered)

    def test_tsb_ad_labeled_prompt_mentions_wrapper_and_supported_algorithms(self):
        rendered = agent_code_generator_mod.template_tsb_ad_labeled.invoke(
            {
                "algorithm": "SAND",
                "data_path_train": "./data/MSL_train.npy",
                "data_path_test": "./data/MSL_train.npy",
                "algorithm_doc": "official doc",
                "parameters": "{}",
                "dataset_metadata": "{}",
            }
        ).to_string()

        self.assertIn("run_SAND", rendered)
        self.assertIn("Do not use `run_Semisupervise_AD`", rendered)
        self.assertIn("inspect.signature(model_runner).parameters", rendered)
        self.assertIn("Do not reference `run_SAND` unless it is explicitly imported", rendered)
        self.assertIn("./data/MSL_train.npy", rendered)

    # ------------------------------------------------------------------ #
    # template_tsb_ad_unlabeled content checks                            #
    # ------------------------------------------------------------------ #

    def test_tsb_ad_unlabeled_prompt_prints_outlier_indices(self):
        rendered = agent_code_generator_mod.template_tsb_ad_unlabeled.invoke(
            {
                "algorithm": "RobustPCA",
                "data_path_train": "./data/demo_train.npy",
                "data_path_test": "",
                "algorithm_doc": "official doc",
                "parameters": "{}",
                "dataset_metadata": "{}",
            }
        ).to_string()

        self.assertIn("95th percentile", rendered)
        self.assertIn("Detected outlier at point [0, 5, 12]", rendered)
        self.assertIn("outliers.tolist()", rendered)
        self.assertIn("np.nan_to_num", rendered)
        self.assertIn("np.linalg.norm", rendered)
        self.assertIn("final score length does not equal", rendered)

    # ------------------------------------------------------------------ #
    # template_fix content checks                                         #
    # ------------------------------------------------------------------ #

    def test_fix_prompt_contains_range_zero_rule(self):
        """template_fix must include rule 15 for the range() arg 3 must not be zero error."""
        rendered = agent_code_generator_mod.template_fix.invoke(
            {
                "code": "scores = model_runner(X_train, X_test)",
                "error_message": "ValueError: range() arg 3 must not be zero",
                "algorithm": "SAND",
                "algorithm_doc": "official doc",
            }
        ).to_string()

        self.assertIn("range() arg 3 must not be zero", rendered)
        self.assertIn("X_train[:, 0]", rendered)
        self.assertIn("X_test[:, 0]", rendered)
        self.assertIn("Do NOT remove the `data_test` argument", rendered)


if __name__ == "__main__":
    unittest.main()
