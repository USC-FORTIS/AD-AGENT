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
            cache_key = AgentInfoMiner._cache_key("IForest", "pyod")
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        cache_key: {
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
            cache_key = AgentInfoMiner._cache_key("IForest", "pyod")
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        cache_key: {
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

    def test_query_docs_cache_isolated_by_package_name(self):
        agent = AgentInfoMiner()
        with tempfile.TemporaryDirectory() as td:
            cache_path = os.path.join(td, "cache.json")
            pyod_key = AgentInfoMiner._cache_key("SharedAlgo", "pyod")
            pygod_key = AgentInfoMiner._cache_key("SharedAlgo", "pygod")
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        pyod_key: {
                            "query_datetime": "2099-01-01T00:00:00",
                            "document": "pyod_doc",
                        },
                        pygod_key: {
                            "query_datetime": "2099-01-01T00:00:00",
                            "document": "pygod_doc",
                        },
                    },
                    f,
                )

            pyod_doc = agent.query_docs("SharedAlgo", "pyod", cache_path=cache_path)
            pygod_doc = agent.query_docs("SharedAlgo", "pygod", cache_path=cache_path)

        self.assertEqual(pyod_doc, "pyod_doc")
        self.assertEqual(pygod_doc, "pygod_doc")

    def test_query_docs_tslib_uses_prompt_lookup(self):
        agent = AgentInfoMiner()
        with tempfile.TemporaryDirectory() as td:
            cache_path = os.path.join(td, "cache.json")
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump({}, f)
            fake_client = types.SimpleNamespace(
                responses=types.SimpleNamespace(
                    create=lambda **kwargs: types.SimpleNamespace(output_text="tslib_doc")
                )
            )
            with patch.object(agent_info_miner_mod, "OpenAI", return_value=fake_client):
                doc = agent.query_docs("LightTS", "tslib", cache_path=cache_path)

        self.assertEqual(doc, "tslib_doc")

    def test_tsb_ad_prompt_formats_with_algorithm_name(self):
        rendered = agent_info_miner_mod.web_search_prompt_tsb_ad.invoke(
            {"algorithm_name": "IForest"}
        ).to_string()

        self.assertIn("TSB_AD.model_wrapper.run_IForest", rendered)
        self.assertIn("IForest", rendered)

    def test_query_docs_tsb_ad_uses_prompt_lookup(self):
        agent = AgentInfoMiner()
        with tempfile.TemporaryDirectory() as td:
            cache_path = os.path.join(td, "cache.json")
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump({}, f)

            fake_client = types.SimpleNamespace(
                responses=types.SimpleNamespace(
                    create=lambda **kwargs: types.SimpleNamespace(output_text="tsb_ad_doc")
                )
            )
            with patch.object(agent_info_miner_mod, "OpenAI", return_value=fake_client):
                doc = agent.query_docs("IForest", "tsb_ad", cache_path=cache_path)

        self.assertEqual(doc, "tsb_ad_doc")

    def test_reviewer_prompt_formats_with_train_dataset(self):
        rendered = agent_reviewer_mod.test_prompt.invoke(
            {
                "code": "np.load('./data/MSL_train.npy')",
                "package_name": "tslib",
                "dataset_metadata": "{}",
                "feature_count": 55,
            }
        ).to_string()

        self.assertIn("./data/MSL", rendered)


if __name__ == "__main__":
    unittest.main()
