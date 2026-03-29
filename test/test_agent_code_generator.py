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

    def test_generate_code_tslib_uses_official_script_params(self):
        agent = AgentCodeGenerator()
        with patch.object(
            agent_code_generator_mod.llm,
            "invoke",
            return_value=types.SimpleNamespace(
                content=(
                    "```python\n"
                    "import os\nimport subprocess\n"
                    "cmd = [\"python\", \"-u\", \"./Time-Series-Library/run.py\", "
                    "\"--task_name\", \"anomaly_detection\", \"--is_training\", \"1\", "
                    "\"--root_path\", \"./dataset/MSL/\", \"--model\", \"LightTS\", "
                    "\"--model_id\", \"./dataset/MSL/\", \"--use_gpu\", \"False\"]\n"
                    "subprocess.run(cmd)\n```"
                )
            ),
        ):
            code = agent.generate_code(
                algorithm="LightTS",
                data_path_train="./data/MSL_train.npy",
                data_path_test="./data/MSL_test.npy",
                algorithm_doc="doc",
                input_parameters={},
                package_name="tslib",
            )

        self.assertIn('"./Time-Series-Library/run.py"', code)
        self.assertIn('"./dataset/MSL/"', code)
        self.assertIn('"--model_id"', code)
        self.assertIn('"--use_gpu", "False"', code)

    def test_tslib_prompt_mentions_official_doc_as_baseline(self):
        rendered = agent_code_generator_mod.template_tslib_labeled.invoke(
            {
                "algorithm": "LightTS",
                "data_path_train": "./data/MSL_train.npy",
                "data_path_test": "./data/MSL_test.npy",
                "algorithm_doc": "official doc",
                "parameters": "{}",
            }
        ).to_string()

        self.assertIn("Use those official arguments as the starting point", rendered)
        self.assertIn('Apply user parameters from', rendered)
        self.assertIn("train file", rendered)

    def test_sanitize_tslib_args_removes_unsupported_and_normalizes_gpu_flag(self):
        code = (
            'cmd = ["python", "-u", "./Time-Series-Library/run.py", '
            '"--model", "TimesNet", "--fc_dropout", "0.1", "--head_dropout", "0", '
            '"--stride", "8", "--file_name", "MSL", "--use_gpu", "False"]'
        )
        out = AgentCodeGenerator._sanitize_tslib_args(code)

        self.assertNotIn("--fc_dropout", out)
        self.assertNotIn("--head_dropout", out)
        self.assertNotIn("--stride", out)
        self.assertNotIn("--file_name", out)
        self.assertNotIn('"--use_gpu", "False"', out)
        self.assertIn("--no_use_gpu", out)

    def test_sanitize_tslib_args_rewrites_run_path_and_adds_cwd(self):
        code = (
            'import subprocess\n'
            'cmd = ["python", "-u", "./Time-Series-Library/run.py", "--model", "TimesNet"]\n'
            'subprocess.run(cmd)\n'
        )
        out = AgentCodeGenerator._sanitize_tslib_args(code)

        self.assertIn('"run.py"', out)
        self.assertNotIn('"./Time-Series-Library/run.py"', out)
        self.assertIn('subprocess.run(cmd, cwd="./Time-Series-Library")', out)

    def test_sanitize_tslib_args_keeps_official_args_and_normalizes_gpu(self):
        code = (
            'import subprocess\n'
            'cmd = ["python", "-u", "run.py", "--root_path", "./dataset/MSL_train.npy", '
            '"--model", "LightTS", "--model_id", "./dataset/MSL_train.npy", "--use_gpu", "False"]\n'
            'subprocess.run(cmd)\n'
        )

        out = AgentCodeGenerator._sanitize_tslib_args(code)

        self.assertIn('"./dataset/MSL_train.npy"', out)
        self.assertIn('"--model_id"', out)
        self.assertNotIn('"--use_gpu", "False"', out)
        self.assertIn('"--no_use_gpu"', out)


if __name__ == "__main__":
    unittest.main()
