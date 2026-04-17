import asyncio
import importlib
import sys
import types
import unittest
from unittest.mock import patch

_SNAPSHOT_KEYS = [
    "config",
    "config.config",
    "langchain_core.messages",
    "langgraph.graph",
    "pipeline_interface",
    "agents",
    "agents.agent_processor",
    "agents.agent_selector",
    "agents.agent_info_miner",
    "agents.agent_code_generator",
    "agents.agent_reviewer",
    "agents.agent_evaluator",
    "agents.agent_optimizer",
    "entity",
    "entity.code_quality",
]
_MODULE_SNAPSHOT = {k: sys.modules.get(k) for k in _SNAPSHOT_KEYS}


def _install_main_stubs():
    config_pkg = types.ModuleType("config")
    config_mod = types.ModuleType("config.config")
    config_mod.Config = type("Config", (), {"OPENAI_API_KEY": "test-key"})

    langchain_core_messages = types.ModuleType("langchain_core.messages")
    langchain_core_messages.BaseMessage = type("BaseMessage", (), {})

    langgraph_graph = types.ModuleType("langgraph.graph")

    class _DummyCompiled:
        def invoke(self, state, config=None):
            return state

    class _DummyStateGraph:
        def __init__(self, *_args, **_kwargs):
            pass

        def add_node(self, *_args, **_kwargs):
            return None

        def set_entry_point(self, *_args, **_kwargs):
            return None

        def add_edge(self, *_args, **_kwargs):
            return None

        def add_conditional_edges(self, *_args, **_kwargs):
            return None

        def compile(self):
            return _DummyCompiled()

    langgraph_graph.StateGraph = _DummyStateGraph
    langgraph_graph.END = object()

    pipeline_interface_mod = types.ModuleType("pipeline_interface")
    pipeline_interface_mod.FullToolState = dict
    pipeline_interface_mod.run_processor = lambda state: state
    pipeline_interface_mod.run_selector = lambda state=None, **kwargs: state
    pipeline_interface_mod.run_code_generator = lambda state=None, **kwargs: {"code_quality": None}
    pipeline_interface_mod.run_info_miner = lambda state=None, **kwargs: {"algorithm_doc": "doc"}
    pipeline_interface_mod.run_reviewer = lambda state=None, **kwargs: {"code_quality": None}
    pipeline_interface_mod.run_evaluator = lambda state=None, **kwargs: {"code_quality": None}
    pipeline_interface_mod.run_optimizer = lambda state=None, **kwargs: {"code_quality": None}

    agents_pkg = types.ModuleType("agents")
    for mod_name, cls_name in [
        ("agent_processor", "AgentProcessor"),
        ("agent_selector", "AgentSelector"),
        ("agent_info_miner", "AgentInfoMiner"),
        ("agent_code_generator", "AgentCodeGenerator"),
        ("agent_reviewer", "AgentReviewer"),
        ("agent_evaluator", "AgentEvaluator"),
        ("agent_optimizer", "AgentOptimizer"),
    ]:
        m = types.ModuleType(f"agents.{mod_name}")
        setattr(m, cls_name, type(cls_name, (), {}))
        sys.modules[f"agents.{mod_name}"] = m

    entity_pkg = types.ModuleType("entity")
    entity_code_quality = types.ModuleType("entity.code_quality")
    entity_code_quality.CodeQuality = type("CodeQuality", (), {})

    sys.modules["config"] = config_pkg
    sys.modules["config.config"] = config_mod
    sys.modules["langchain_core.messages"] = langchain_core_messages
    sys.modules["langgraph.graph"] = langgraph_graph
    sys.modules["pipeline_interface"] = pipeline_interface_mod
    sys.modules.setdefault("agents", agents_pkg)
    sys.modules["entity"] = entity_pkg
    sys.modules["entity.code_quality"] = entity_code_quality


_install_main_stubs()
if "main" in sys.modules:
    del sys.modules["main"]
main = importlib.import_module("main")

# Restore global module table so this test module does not pollute others.
for _k, _v in _MODULE_SNAPSHOT.items():
    if _v is None:
        sys.modules.pop(_k, None)
    else:
        sys.modules[_k] = _v


class _FakeCompiledSingleTool:
    def invoke(self, state, config=None):
        return {"code_quality": f"ok-{state['current_tool']}"}


async def _fake_to_thread(func, *args, **kwargs):
    return func(*args, **kwargs)


class TestMainOrchestration(unittest.TestCase):
    def test_main_returns_nonzero_and_prints_clean_error_on_value_error(self):
        with patch.object(main, "compiled_full_graph", types.SimpleNamespace(invoke=lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("selector rejected input")))), \
             patch.object(main.asyncio, "to_thread", new=_fake_to_thread), \
             patch("builtins.print") as mock_print:
            result = asyncio.run(main.main())

        self.assertEqual(result, 1)
        printed = "\n".join(str(call.args[0]) for call in mock_print.call_args_list if call.args)
        self.assertIn("Pipeline stopped", printed)
        self.assertIn("selector rejected input", printed)

    def test_route_selector_falls_back_to_code_quality_when_route_missing(self):
        failing_cq = types.SimpleNamespace(error_message="docker failed", review_count=1)
        success_cq = types.SimpleNamespace(error_message="", review_count=1)

        self.assertEqual(main.route_selector({"code_quality": failing_cq}), "code_generator")
        self.assertEqual(main.route_selector({"code_quality": success_cq}), "evaluator")

    def test_process_all_tools_requires_selector(self):
        with self.assertRaises(ValueError):
            main.process_all_tools({"agent_selector": None})

    def test_process_all_tools_empty_tools(self):
        state = {
            "agent_selector": types.SimpleNamespace(tools=[]),
            "results": None,
        }
        result = main.process_all_tools(state)
        self.assertEqual(result["results"], [])

    def test_process_all_tools_sequential(self):
        state = {
            "agent_selector": types.SimpleNamespace(tools=["A", "B"]),
            "results": None,
            "current_tool": "",
            "code_quality": None,
            "should_rerun": False,
        }
        with patch.object(main, "compiled_single_tool", _FakeCompiledSingleTool()), \
             patch.object(main.asyncio, "to_thread", new=_fake_to_thread), \
             patch.object(main.sys, "argv", ["main.py"]):
            result = main.process_all_tools(state)

        self.assertEqual(len(result["results"]), 2)
        self.assertEqual(result["results"][0][0], "A")
        self.assertEqual(result["results"][1][0], "B")

    def test_process_all_tools_parallel(self):
        state = {
            "agent_selector": types.SimpleNamespace(tools=["A", "B"]),
            "results": None,
            "current_tool": "",
            "code_quality": None,
            "should_rerun": False,
        }
        with patch.object(main, "compiled_single_tool", _FakeCompiledSingleTool()), \
             patch.object(main.asyncio, "to_thread", new=_fake_to_thread), \
             patch.object(main.sys, "argv", ["main.py", "-p"]):
            result = main.process_all_tools(state)

        self.assertEqual(len(result["results"]), 2)


if __name__ == "__main__":
    unittest.main()
