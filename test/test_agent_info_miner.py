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
                        "IForest": {
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
            self.assertIn("IForest", cache)
            self.assertEqual(cache["IForest"]["document"], "new_doc")

    def test_query_docs_cache_expired_requeries(self):
        agent = AgentInfoMiner()
        with tempfile.TemporaryDirectory() as td:
            cache_path = os.path.join(td, "cache.json")
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "IForest": {
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


if __name__ == "__main__":
    unittest.main()
