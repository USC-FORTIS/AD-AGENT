import json
import os
import tempfile
import unittest
import types
from unittest.mock import patch

from agent_test_stubs import install_common_stubs, load_real_agent_module

install_common_stubs()
agent_info_miner_mod = load_real_agent_module("agents.agent_info_miner")
AgentInfoMiner = agent_info_miner_mod.AgentInfoMiner
agent_reviewer_mod = load_real_agent_module("agents.agent_reviewer")


class TestAgentInfoMiner(unittest.TestCase):
    def setUp(self):
        self._print_patcher = patch("builtins.print")
        self._print_patcher.start()
        self.addCleanup(self._print_patcher.stop)

    def test_query_docs_cache_hit(self):
        agent = AgentInfoMiner()
        with tempfile.TemporaryDirectory() as td:
            cache_path = os.path.join(td, "cache.json")
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        AgentInfoMiner._cache_key("IForest", "pyod"): {
                            "query_datetime": "2099-01-01T00:00:00",
                            "document": "cached_doc",
                        }
                    },
                    f,
                )

            doc = agent.query_docs("IForest", "pyod", cache_path=cache_path)
            self.assertEqual(doc, "cached_doc")

    def test_query_docs_cache_miss_writes_cache(self):
        agent = AgentInfoMiner()
        with tempfile.TemporaryDirectory() as td:
            cache_path = os.path.join(td, "cache.json")
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump({}, f)

            fake_client = types.SimpleNamespace(
                responses=types.SimpleNamespace(
                    create=lambda **kwargs: types.SimpleNamespace(output_text="new_doc")
                )
            )
            with patch.object(agent_info_miner_mod, "OpenAI", return_value=fake_client):
                doc = agent.query_docs("IForest", "pyod", cache_path=cache_path)

            self.assertEqual(doc, "new_doc")
            with open(cache_path, "r", encoding="utf-8") as f:
                cache = json.load(f)
            cache_key = AgentInfoMiner._cache_key("IForest", "pyod")
            self.assertIn(cache_key, cache)
            self.assertEqual(cache[cache_key]["document"], "new_doc")

    def test_query_docs_cache_expired_requeries(self):
        agent = AgentInfoMiner()
        with tempfile.TemporaryDirectory() as td:
            cache_path = os.path.join(td, "cache.json")
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        AgentInfoMiner._cache_key("IForest", "pyod"): {
                            "query_datetime": "2000-01-01T00:00:00",
                            "document": "old_doc",
                        }
                    },
                    f,
                )

            fake_client = types.SimpleNamespace(
                responses=types.SimpleNamespace(
                    create=lambda **kwargs: types.SimpleNamespace(output_text="refreshed_doc")
                )
            )
            with patch.object(agent_info_miner_mod, "OpenAI", return_value=fake_client):
                doc = agent.query_docs("IForest", "pyod", cache_path=cache_path)

            self.assertEqual(doc, "refreshed_doc")

    def test_query_docs_corrupted_cache_recovers(self):
        agent = AgentInfoMiner()
        with tempfile.TemporaryDirectory() as td:
            cache_path = os.path.join(td, "cache.json")
            with open(cache_path, "w", encoding="utf-8") as f:
                f.write("{bad_json")

            fake_client = types.SimpleNamespace(
                responses=types.SimpleNamespace(
                    create=lambda **kwargs: types.SimpleNamespace(output_text="doc_after_recover")
                )
            )
            with patch.object(agent_info_miner_mod, "OpenAI", return_value=fake_client):
                doc = agent.query_docs("IForest", "pyod", cache_path=cache_path)

            self.assertEqual(doc, "doc_after_recover")

    def test_tslib_prompt_formats_with_algorithm_name(self):
        rendered = agent_info_miner_mod.web_search_prompt_tslib.invoke(
            {"algorithm_name": "LightTS"}
        ).to_string()

        self.assertIn("LightTS.sh", rendered)
        self.assertIn('"task_name": "anomaly_detection"', rendered)

    def test_query_docs_tslib_uses_local_script(self):
        agent = AgentInfoMiner()
        with tempfile.TemporaryDirectory() as td:
            cache_path = os.path.join(td, "cache.json")
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump({}, f)
            fake_agents_dir = os.path.join(td, "agents")
            fake_repo_dir = os.path.join(td, "Time-Series-Library")
            scripts_dir = os.path.join(fake_repo_dir, "scripts", "anomaly_detection", "MSL")
            os.makedirs(scripts_dir, exist_ok=True)

            with open(os.path.join(fake_repo_dir, "run.py"), "w", encoding="utf-8") as f:
                f.write(
                    "import argparse\n"
                    "parser = argparse.ArgumentParser()\n"
                    "parser.add_argument('--task_name', type=str, required=True)\n"
                    "parser.add_argument('--is_training', type=int, required=True)\n"
                    "parser.add_argument('--root_path', type=str, default='./data/ETT/')\n"
                    "parser.add_argument('--model_id', type=str, required=True)\n"
                    "parser.add_argument('--model', type=str, required=True)\n"
                    "parser.add_argument('--data', type=str, required=True)\n"
                    "parser.add_argument('--features', type=str, default='M')\n"
                    "parser.add_argument('--seq_len', type=int, default=96)\n"
                    "parser.add_argument('--pred_len', type=int, default=96)\n"
                    "parser.add_argument('--use_gpu', action='store_true', default=True)\n"
                    "parser.add_argument('--no_use_gpu', action='store_false', dest='use_gpu')\n"
                    "parser.add_argument('--gpu_type', type=str, default='cuda')\n"
                )

            with open(os.path.join(scripts_dir, "LightTS.sh"), "w", encoding="utf-8") as f:
                f.write(
                    "python -u run.py \\\n"
                    "  --task_name anomaly_detection \\\n"
                    "  --is_training 1 \\\n"
                    "  --root_path ./dataset/MSL \\\n"
                    "  --model_id MSL \\\n"
                    "  --model LightTS \\\n"
                    "  --data MSL \\\n"
                    "  --features M \\\n"
                    "  --seq_len 100 \\\n"
                    "  --pred_len 0\n"
                )

            with patch.object(agent_info_miner_mod.os.path, "dirname", return_value=fake_agents_dir):
                doc = agent.query_docs("LightTS", "tslib", cache_path=cache_path)

        self.assertIn("Official script path:", doc)
        self.assertIn('"data": "MSL"', doc)
        self.assertIn('"pred_len": 0', doc)
        self.assertIn("CPU-safe runtime overrides", doc)
        self.assertIn('"gpu_type": "cpu"', doc)
        self.assertIn('"no_use_gpu": true', doc)

    def test_tsb_ad_prompt_formats_with_algorithm_name(self):
        rendered = agent_info_miner_mod.web_search_prompt_tsb_ad.invoke(
            {"algorithm_name": "IForest"}
        ).to_string()

        self.assertIn("TSB-AD", rendered)
        self.assertIn("IForest", rendered)
        self.assertIn("run_IForest", rendered)

    def test_reviewer_prompt_formats_with_train_dataset(self):
        rendered = agent_reviewer_mod.test_prompt.invoke(
            {
                "code": "print('x')",
                "algorithm_name": "LightTS",
                "package_name": "tslib",
                "train_dataset": "./data/MSL",
                "feature_dim": 55,
                "dataset_metadata": "{}",
            }
        ).to_string()

        self.assertIn("./data/MSL", rendered)


if __name__ == "__main__":
    unittest.main()
