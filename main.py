import logging, sys, asyncio, os

from config.config import Config
api_key = os.getenv("OPENAI_API_KEY") or Config.OPENAI_API_KEY
if not api_key:
    raise ValueError(
        "OPENAI_API_KEY is missing. Set environment variable OPENAI_API_KEY "
        "or set Config.OPENAI_API_KEY in config/config.py."
    )
os.environ["OPENAI_API_KEY"] = api_key
logging.basicConfig(stream=sys.stdout, level=logging.ERROR)

# ========== langgraph ==========
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, END

# ========== pipeline API (tool‑specific calls) ==========
from pipeline_interface import (
    FullToolState,
    run_processor, 
    run_selector, 
    run_code_generator,
    run_info_miner,
    run_reviewer,
    run_evaluator,
    run_optimizer
)

# ========== business agents ==========
from agents.agent_processor import AgentProcessor
from agents.agent_selector     import AgentSelector
from agents.agent_info_miner    import AgentInfoMiner
from agents.agent_code_generator        import AgentCodeGenerator
from agents.agent_reviewer     import AgentReviewer
from agents.agent_evaluator    import AgentEvaluator
from agents.agent_optimizer    import AgentOptimizer        # ★ new
from entity.code_quality       import CodeQuality


# ------------------------------------------------------------------
# Node: Decider  (branch: rerun | evaluate)
# ------------------------------------------------------------------

def decide_next(state: FullToolState) -> dict:
    cq = state["code_quality"]
    need_rerun = bool(cq.error_message) and cq.review_count < 4
    return {"route": "code_generator" if need_rerun else "evaluator"}

def route_selector(state: FullToolState):
    return state["route"]


# ------------------------------------------------------------------
# Build single‑tool StateGraph
# ------------------------------------------------------------------

single_tool_graph = StateGraph(FullToolState)

single_tool_graph.add_node("info_miner", lambda state: run_info_miner(state=state))
single_tool_graph.add_node("code_generator", lambda state: run_code_generator(state=state))
single_tool_graph.add_node("reviewer", lambda state: run_reviewer(state=state))
single_tool_graph.add_node("decider",    decide_next)
single_tool_graph.add_node("evaluator", lambda state: run_evaluator(state=state))
single_tool_graph.add_node("optimizer", lambda state: run_optimizer(state=state))      # ★ new

single_tool_graph.set_entry_point("info_miner")
single_tool_graph.add_edge("info_miner", "code_generator")
single_tool_graph.add_edge("code_generator",      "reviewer")
single_tool_graph.add_edge("reviewer",   "decider")
single_tool_graph.add_conditional_edges(
    "decider", route_selector,
    {"code_generator": "code_generator", "evaluator": "evaluator"}
)
single_tool_graph.add_edge("evaluator",  "optimizer")   # ★ changed
single_tool_graph.add_edge("optimizer",  END)            # ★ new

compiled_single_tool = single_tool_graph.compile()

# ------------------------------------------------------------------
# process_all_tools
# ------------------------------------------------------------------

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
            current_tool = tool,
            code_quality = None,
            should_rerun = False
        )
        return tool, await asyncio.to_thread(
            compiled_single_tool.invoke,
            tool_state,
            config={"recursion_limit": 20}
        )

    results = []
    if "-p" in sys.argv:          # parallel
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        results = loop.run_until_complete(asyncio.gather(
            *(run_tool(t) for t in tools)
        ))
        loop.close()
    else:                         # sequential
        for t in tools:
            results.append(asyncio.run(run_tool(t)))

    state["results"] = results
    return state

# ------------------------------------------------------------------
# Build full process graph
# ------------------------------------------------------------------

full_graph = StateGraph(FullToolState)
full_graph.add_node("processor", lambda state: run_processor(state))
full_graph.add_node("selector", lambda state: run_selector(state=state))
full_graph.add_node("process_all_tools", lambda state: process_all_tools(state))

full_graph.set_entry_point("processor")
full_graph.add_edge("processor",     "selector")
full_graph.add_edge("selector",         "process_all_tools")

compiled_full_graph = full_graph.compile()

# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------

async def main():
    # clean loader scripts
    for f in ("train_data_loader.py","test_data_loader.py",
              "head_train_data_loader.py","head_test_data_loader.py"):
        if os.path.exists(f): os.remove(f)

    state: FullToolState = {
        "messages"        : [],
        "current_tool"    : "",
        "input_parameters": {},
        "data_path_train" : "",
        "data_path_test"  : "",
        "package_name"    : "",
        "agent_info_miner" : AgentInfoMiner(),
        "agent_code_generator"     : AgentCodeGenerator(),
        "agent_reviewer"  : AgentReviewer(),
        "agent_evaluator" : AgentEvaluator(),     # original
        "agent_optimizer" : AgentOptimizer(),     # ★ new
        "vectorstore"     : None,
        "code_quality"    : None,
        "should_rerun"    : False,
        "agent_processor": AgentProcessor(),
        "agent_selector"  : None,
        "experiment_config": None,
        "results"         : None,
        "algorithm_doc"   : None,
        "n_features"     : None,
    }

    print("\n=== [Main] Starting full pipeline ===")
    final_state = await asyncio.to_thread(
        compiled_full_graph.invoke,
        state,
        config={"recursion_limit": 20}
    )
    print("\n=== [Main] Pipeline finished ===")

    # ---------- output results ----------
    for tool, tstate in final_state.get("results", []):
        cq: CodeQuality | None = tstate.get("code_quality")
        if cq and not cq.error_message:
            print(f"[{tool}] AUROC: {cq.auroc:.4f}  AUPRC: {cq.auprc:.4f}  Parameters: {cq.parameters}")
        else:
            print(f"[{tool}] Error: {cq.error_message if cq else 'Unknown'}")

if __name__ == "__main__":
    asyncio.run(main())
