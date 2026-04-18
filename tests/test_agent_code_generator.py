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
                metadata={"dataset": {"train": {"file_type": "mat"}}},
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
                metadata={"dataset": {"train": {"file_type": "csv"}}},
            )

        self.assertEqual(code, "print('tsb_ad')")

    def test_pygod_prompt_prefers_small_epoch_when_user_did_not_provide_one(self):
        rendered = agent_code_generator_mod.template_pygod_labeled.invoke(
            {
                "algorithm": "AnomalyDAE",
                "data_path_train": "./data/inj_cora_train.pt",
                "data_path_test": "./data/inj_cora_test.pt",
                "algorithm_doc": "official doc",
                "parameters": "{}",
                "dataset_metadata": "{}",
            }
        ).to_string()

        self.assertIn("epoch=5", rendered)

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
        self.assertIn("Never set `periodicity` to `0`", rendered)
        self.assertIn("Do not substitute official demo paths", rendered)

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

    def test_fix_prompt_contains_population_sampling_rule(self):
        rendered = agent_code_generator_mod.template_fix.invoke(
            {
                "code": "scores = model_runner(X_train_in, X_test_in, **run_kwargs)",
                "error_message": "ValueError: Cannot take a larger sample than population when 'replace=False'",
                "algorithm": "SAND",
                "algorithm_doc": "official doc",
            }
        ).to_string()

        self.assertIn("Cannot take a larger sample than population", rendered)
        self.assertIn("do not set `periodicity` to `0`", rendered)
        self.assertIn("Do not replace the dataset with demo files", rendered)
        self.assertIn("loads the same dataset path for both `X_train` and `X_test`", rendered)
        self.assertIn("replace only the mistaken test path with the `_test.npy` companion file", rendered)

    def test_fix_prompt_forbids_readding_unsupported_kwargs_after_filtering(self):
        rendered = agent_code_generator_mod.template_fix.invoke(
            {
                "code": "run_kwargs = {k: v for k, v in user_params.items() if k in inspect.signature(model_runner).parameters}\nrun_kwargs.update({'window_size': 100})",
                "error_message": "TypeError: run_Left_STAMPi() got an unexpected keyword argument 'window_size'",
                "algorithm": "Left_STAMPi",
                "algorithm_doc": "official doc",
            }
        ).to_string()

        self.assertIn("do not add extra kwargs afterward with `run_kwargs.update(...)`", rendered)
        self.assertIn("Do not introduce guessed defaults such as `window_size`", rendered)

    def test_tsb_ad_labeled_prompt_keeps_2d_for_non_univariate_wrappers(self):
        rendered = agent_code_generator_mod.template_tsb_ad_labeled.invoke(
            {
                "algorithm": "Left_STAMPi",
                "data_path_train": "./data/train.npy",
                "data_path_test": "./data/test.npy",
                "algorithm_doc": "official doc",
                "parameters": "{}",
                "dataset_metadata": "{}",
            }
        ).to_string()

        self.assertIn("pass both datasets positionally", rendered)
        self.assertIn("do not place `data` or `data_train` inside `run_kwargs`", rendered)
        self.assertIn("`run_kwargs` should contain only filtered user parameters", rendered)
        self.assertIn("Only reduce multivariate inputs to 1D when the documentation or a concrete runtime error proves the wrapper is univariate-only", rendered)
        self.assertIn("Otherwise keep `X_train` and `X_test` as 2D arrays", rendered)
        self.assertIn("dtype=np.float64", rendered)

    def test_fix_prompt_handles_left_stampi_call_shape_errors(self):
        rendered = agent_code_generator_mod.template_fix.invoke(
            {
                "code": "scores = model_runner(X_train_in, **run_kwargs)",
                "error_message": "TypeError: run_Left_STAMPi() missing 1 required positional argument: 'data'",
                "algorithm": "Left_STAMPi",
                "algorithm_doc": "official doc",
            }
        ).to_string()

        self.assertIn("missing 1 required positional argument: 'data'", rendered)
        self.assertIn("Pass both arrays positionally", rendered)
        self.assertIn("Do not drop the second dataset argument", rendered)

    def test_fix_prompt_handles_float64_dtype_requirement(self):
        rendered = agent_code_generator_mod.template_fix.invoke(
            {
                "code": "scores = model_runner(X_train, X_test, **run_kwargs)",
                "error_message": "TypeError: <class 'numpy.float64'> dtype expected but found float32 in input array",
                "algorithm": "Left_STAMPi",
                "algorithm_doc": "official doc",
            }
        ).to_string()

        self.assertIn("expects `numpy.float64` but found `float32`", rendered)
        self.assertIn("dtype=np.float64", rendered)

    def test_sanitize_tsb_ad_code_re_filters_run_kwargs_before_call(self):
        code = """import inspect
from TSB_AD.model_wrapper import run_Left_STAMPi
model_runner = run_Left_STAMPi
run_kwargs = {"normalize": True, "verbose": False}
scores = model_runner(X_train, X_test, **run_kwargs)
"""
        sanitized = AgentCodeGenerator._sanitize_tsb_ad_code(code)

        self.assertIn('valid_params = inspect.signature(model_runner).parameters', sanitized)
        self.assertIn("if 'user_params' in locals() and isinstance(user_params, dict):", sanitized)
        self.assertIn('k not in {"data", "data_train"}', sanitized)
        self.assertLess(
            sanitized.index('valid_params = inspect.signature(model_runner).parameters'),
            sanitized.index('scores = model_runner(X_train, X_test, **run_kwargs)'),
        )

    def test_fix_prompt_contains_small_window_rule(self):
        rendered = agent_code_generator_mod.template_fix.invoke(
            {
                "code": "run_kwargs['periodicity'] = 0\nscores = model_runner(X_train_in, X_test_in, **run_kwargs)",
                "error_message": "ValueError: ('All window sizes must be greater than or equal to three', '...')",
                "algorithm": "SAND",
                "algorithm_doc": "official doc",
            }
        ).to_string()

        self.assertIn("All window sizes must be greater than or equal to three", rendered)
        self.assertIn("Remove the manual `periodicity=0` override", rendered)
        self.assertIn("use a valid integer `>= 1`", rendered)


if __name__ == "__main__":
    unittest.main()
