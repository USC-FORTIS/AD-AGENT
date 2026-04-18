import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Parse sandbox mode before importing any sandbox-aware modules.
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--sandbox", choices=["modal", "docker"], default=None)
_known, _ = _parser.parse_known_args()
if _known.sandbox:
    os.environ["OPENAD_SANDBOX"] = _known.sandbox

from config.config import Config
try:
    from sandbox.config import SANDBOX_MODE
except ModuleNotFoundError:
    SANDBOX_MODE = os.environ.get("OPENAD_SANDBOX", "modal")

api_key = os.getenv("OPENAI_API_KEY") or Config.OPENAI_API_KEY
if not api_key:
    raise ValueError(
        "OPENAI_API_KEY is missing. Set environment variable OPENAI_API_KEY "
        "or set Config.OPENAI_API_KEY in config/config.py."
    )
os.environ["OPENAI_API_KEY"] = api_key
logging.basicConfig(stream=sys.stdout, level=logging.ERROR)

from langgraph.graph import END, StateGraph

from api.pipeline import (
    FullToolState,
    run_code_generator,
    run_evaluator,
    run_info_miner,
    run_optimizer,
    run_processor,
    run_reviewer,
    run_selector,
)

from agents.agent_code_generator import AgentCodeGenerator
from agents.agent_evaluator import AgentEvaluator
from agents.agent_info_miner import AgentInfoMiner
from agents.agent_optimizer import AgentOptimizer
from agents.agent_processor import AgentProcessor
from agents.agent_reviewer import AgentReviewer
from models.code_quality import CodeQuality
try:
    from api.pipeline import log_local
except ImportError:
    _LAST_LOG_CONTEXT: tuple[str, str | None] | None = None

    def log_local(stage: str, message: str, tool: str | None = None) -> None:
        global _LAST_LOG_CONTEXT
        context = (stage, tool)
        if _LAST_LOG_CONTEXT is not None and _LAST_LOG_CONTEXT != context:
            print()
        prefix = f"[{stage}]"
        if tool:
            prefix += f"[{tool}]"
        print(f"{prefix} {message}")
        _LAST_LOG_CONTEXT = context


def decide_next(state: FullToolState) -> dict:
    cq = state["code_quality"]
    need_rerun = bool(cq.error_message) and cq.review_count < 4
    return {"route": "code_generator" if need_rerun else "evaluator"}


def route_selector(state: FullToolState):
    route = state.get("route")
    if route:
        return route

    cq = state.get("code_quality")
    if cq is None:
        return "evaluator"

    need_rerun = bool(getattr(cq, "error_message", "")) and getattr(cq, "review_count", 0) < 4
    return "code_generator" if need_rerun else "evaluator"


single_tool_graph = StateGraph(FullToolState)
single_tool_graph.add_node("info_miner", lambda state: run_info_miner(state=state))
single_tool_graph.add_node("code_generator", lambda state: run_code_generator(state=state))
single_tool_graph.add_node("reviewer", lambda state: run_reviewer(state=state))
single_tool_graph.add_node("decider", decide_next)
single_tool_graph.add_node("evaluator", lambda state: run_evaluator(state=state))
single_tool_graph.add_node("optimizer", lambda state: run_optimizer(state=state))

single_tool_graph.set_entry_point("info_miner")
single_tool_graph.add_edge("info_miner", "code_generator")
single_tool_graph.add_edge("code_generator", "reviewer")
single_tool_graph.add_edge("reviewer", "decider")
single_tool_graph.add_conditional_edges(
    "decider",
    route_selector,
    {"code_generator": "code_generator", "evaluator": "evaluator"},
)
single_tool_graph.add_edge("evaluator", "optimizer")
single_tool_graph.add_edge("optimizer", END)

compiled_single_tool = single_tool_graph.compile()


def process_all_tools(state: FullToolState) -> dict:
    if not state["agent_selector"]:
        raise ValueError("agent_selector is not set!")
    tools = state["agent_selector"].tools
    if not tools:
        state["results"] = []
        return state

    async def run_tool(tool):
        tool_state = state.copy()
        tool_state.update(
            current_tool=tool,
            code_quality=None,
            should_rerun=False,
        )
        log_local("main", "Starting tool pipeline", tool)
        result = await asyncio.to_thread(
            compiled_single_tool.invoke,
            tool_state,
            config={"recursion_limit": 20},
        )
        cq: CodeQuality | None = result.get("code_quality")
        error_message = getattr(cq, "error_message", "") if cq is not None else ""
        auroc = getattr(cq, "auroc", None) if cq is not None else None
        auprc = getattr(cq, "auprc", None) if cq is not None else None
        if cq and not error_message and auroc is not None and auprc is not None:
            log_local(
                "main",
                f"Finished tool pipeline: AUROC={auroc:.4f}, AUPRC={auprc:.4f}",
                tool,
            )
        elif cq and not error_message:
            log_local("main", "Finished tool pipeline", tool)
        else:
            log_local(
                "main",
                f"Finished tool pipeline with error: {error_message if cq else 'Unknown'}",
                tool,
            )
        return tool, result

    results = []
    log_local(
        "main",
        (
            f"Dispatching {len(tools)} tool(s) "
            f"({'parallel' if '-p' in sys.argv else 'sequential'}): {', '.join(tools)}"
        ),
    )
    if "-p" in sys.argv:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        results = loop.run_until_complete(asyncio.gather(*(run_tool(t) for t in tools)))
        loop.close()
    else:
        for tool in tools:
            results.append(asyncio.run(run_tool(tool)))

    state["results"] = results
    return state


full_graph = StateGraph(FullToolState)
full_graph.add_node("processor", lambda state: run_processor(state))
full_graph.add_node("selector", lambda state: run_selector(state=state))
full_graph.add_node("process_all_tools", lambda state: process_all_tools(state))

full_graph.set_entry_point("processor")
full_graph.add_edge("processor", "selector")
full_graph.add_edge("selector", "process_all_tools")

compiled_full_graph = full_graph.compile()


async def main():
    state: FullToolState = {
        "messages": [],
        "current_tool": "",
        "input_parameters": {},
        "data_path_train": "",
        "data_path_test": "",
        "package_name": "",
        "agent_info_miner": AgentInfoMiner(),
        "agent_code_generator": AgentCodeGenerator(),
        "agent_reviewer": AgentReviewer(),
        "agent_evaluator": AgentEvaluator(),
        "agent_optimizer": AgentOptimizer(),
        "vectorstore": None,
        "code_quality": None,
        "should_rerun": False,
        "agent_processor": AgentProcessor(),
        "agent_selector": None,
        "experiment_config": None,
        "results"         : None,
        "algorithm_doc"   : None,
        "feature_dim"     : None,
        "metadata"        : None,
    }

    log_local(
        "main",
        (
            f"Starting full pipeline "
            f"(sandbox={SANDBOX_MODE}, parallel={'-p' in sys.argv}, optimizer={'-o' in sys.argv})"
        ),
    )
    try:
        final_state = await asyncio.to_thread(
            compiled_full_graph.invoke,
            state,
            config={"recursion_limit": 20},
        )
    except ValueError as exc:
        log_local("main", f"Pipeline stopped: {exc}")
        return 1

    log_local("main", "Pipeline finished")

    for tool, tstate in final_state.get("results", []):
        cq: CodeQuality | None = tstate.get("code_quality")
        error_message = getattr(cq, "error_message", "") if cq is not None else ""
        auroc = getattr(cq, "auroc", None) if cq is not None else None
        auprc = getattr(cq, "auprc", None) if cq is not None else None
        parameters = getattr(cq, "parameters", None) if cq is not None else None
        if cq and not error_message and auroc is not None and auprc is not None:
            log_local(
                "result",
                f"AUROC={auroc:.4f} AUPRC={auprc:.4f} parameters={parameters}",
                tool,
            )
        elif cq and not error_message:
            log_local("result", "completed", tool)
        else:
            log_local("result", f"error={error_message if cq else 'Unknown'}", tool)

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
