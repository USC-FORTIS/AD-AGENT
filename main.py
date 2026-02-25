import logging, sys, operator, asyncio, os, argparse
from typing import TypedDict, Annotated, Sequence, List, Tuple, Any

# --- Parse --sandbox BEFORE any agent/sandbox imports (they read env at import time) ---
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--sandbox", choices=["modal", "docker"], default=None)
_known, _ = _parser.parse_known_args()
if _known.sandbox:
    os.environ["OPENAD_SANDBOX"] = _known.sandbox

from config.config import Config
os.environ["OPENAI_API_KEY"] = Config.OPENAI_API_KEY
logging.basicConfig(stream=sys.stdout, level=logging.ERROR)

# ========== langgraph ==========
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, END
from langchain_openai          import ChatOpenAI

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
# Full state
# ------------------------------------------------------------------
class FullToolState(TypedDict):
    messages        : Annotated[Sequence[Any], operator.add]
    current_tool    : str
    input_parameters: dict
    data_path_train : str
    data_path_test  : str
    package_name    : str
    agent_info_miner : Any
    agent_code_generator     : Any
    agent_reviewer  : Any
    agent_evaluator : Any
    agent_optimizer : Any                                 # ★ new
    vectorstore     : Any
    code_quality    : Any | None
    should_rerun    : bool
    agent_processor: Any
    agent_selector  : Any | None
    experiment_config: dict | None
    results         : List[Tuple[str, Any]] | None
    algorithm_doc   : str | None
    feature_dim     : int | None

# ------------------------------------------------------------------
# Node: processor
# ------------------------------------------------------------------
def call_processor(state: FullToolState) -> dict:
    processor = state["agent_processor"]
    print("\n=== [Processor] Processing user input ===")
    # Interact
    processor.run_chatbot()
    state["experiment_config"] = processor.experiment_config
    print("\n=== [Processor] User input processing complete ===")
    return state

# ------------------------------------------------------------------
# Node: Selector
# ------------------------------------------------------------------
def call_selector(state: FullToolState) -> dict:
    print("\n=== [Selector] Processing user input ===")
    if state["experiment_config"] is None:
        raise ValueError("experiment_config not set, run processor first!")
    print("\n=== [Selector] Selecting package & algorithm ===")
    selector = AgentSelector(state["experiment_config"])
    state.update(
        agent_selector = selector,
        input_parameters = selector.parameters,
        data_path_train = selector.data_path_train,
        data_path_test  = selector.data_path_test,
        package_name    = selector.package_name,
        feature_dim     = selector.feature_dim,
        # vectorstore     = selector.vectorstore
    )
    print("\n=== [Selector] Selection complete ===")
    return state

# ------------------------------------------------------------------
# Node: info_miner
# ------------------------------------------------------------------
def call_info_miner(state: FullToolState) -> dict:
    print(f"\n=== [Info_miner] Querying documentation for {state['current_tool']} ===")
    info_miner = state["agent_info_miner"]
    doc = info_miner.query_docs(
        state["current_tool"],
        state["package_name"]
    )
    print(f"\n=== [Info_miner] Documentation retrieved for {state['current_tool']} ===")
    return {"algorithm_doc": doc}

# ------------------------------------------------------------------
# Node: code_generator  (generate / revise, **no execution**)
# ------------------------------------------------------------------
def call_code_generator_for_single_tool(state: FullToolState) -> dict:
    code_generator = state["agent_code_generator"]
    tool  = state["current_tool"]

    # generate code || revise code
    if state["code_quality"] is None:
        print(f"\n=== [code_generator] Generating code for {tool} ===")
        code = code_generator.generate_code(
            algorithm       = tool,
            data_path_train = state["data_path_train"],
            data_path_test  = state["data_path_test"],
            algorithm_doc   = state["algorithm_doc"],
            input_parameters= state["input_parameters"],
            package_name    = state["package_name"]
        )
        parameters = code_generator._extract_init_params_dict(state["algorithm_doc"])
        cq = CodeQuality(code=code, algorithm=tool, parameters=parameters, std_output="",
                         error_message="", auroc=-1, auprc=-1,
                         error_points=[], review_count=0)
    else:
        print( f"\n=== [code_generator] Revising code for {tool} ===")
        cq = state["code_quality"]
        code = code_generator.revise_code(cq, state["algorithm_doc"])
        cq.code = code                                 # cover new code

    return {"code_quality": cq}

# ------------------------------------------------------------------
# Node: Reviewer  (synthetic‑data test)
# ------------------------------------------------------------------

def call_reviewer_for_single_tool(state: FullToolState) -> dict:
    reviewer = state["agent_reviewer"]
    cq       = state["code_quality"]
    tool     = state["current_tool"]

    print(f"\n=== [Reviewer] Running validation for {tool} ===")
    cq.error_message = reviewer.test_code(
        cq.code,
        tool,
        state["package_name"],
        feature_dim=state.get("feature_dim"),
    )

    if cq.error_message:
        cq.review_count += 1
    print(f"\n=== [Reviewer] Validation completed for {tool} ===")
    return {"code_quality": cq}

# ------------------------------------------------------------------
# Node: Decider  (branch: rerun | evaluate)
# ------------------------------------------------------------------

def decide_next(state: FullToolState) -> dict:
    cq = state["code_quality"]
    need_rerun = bool(cq.error_message) and cq.review_count < 2
    return {"route": "code_generator" if need_rerun else "evaluator"}

def route_selector(state: FullToolState):
    return state["route"]

# ------------------------------------------------------------------
# Node: Evaluator  (real‑data execution & metrics)
# ------------------------------------------------------------------

def call_evaluator_for_single_tool(state: FullToolState) -> dict:
    evaluator = state["agent_evaluator"]
    cq        = state["code_quality"]
    tool      = state["current_tool"]

    print(f"\n=== [Evaluator] Real‑data run for {tool} ===")
    final_cq = evaluator.execute_code(cq.code, tool, package_name=state["package_name"])

    # keep review_count & parameters
    final_cq.review_count = cq.review_count
    final_cq.parameters = cq.parameters
    return {"code_quality": final_cq}

# ------------------------------------------------------------------
# Node: Optimizer  (LLM‑driven parameter tuning)
# ------------------------------------------------------------------

def call_optimizer_for_single_tool(state: FullToolState) -> dict:
    optimizer = state["agent_optimizer"]
    cq        = state["code_quality"]
    doc       = state["algorithm_doc"]
    if "-o" not in sys.argv:
        return {"code_quality": cq}  # skip if not in optimizer mode
    if cq is None:
        raise ValueError("code_quality is None before optimizer")
    if cq.error_message:
        return {"code_quality": cq}

    print(f"\n=== [Optimizer] Parameter tuning for {state['current_tool']} ===")
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    tuned_cq = optimizer.run(llm=llm,
                             quality=cq,
                             algorithm_doc=doc,
                             max_steps=8,
                             package_name=state["package_name"])
    print(f"\n=== [Optimizer] Tuning finished for {state['current_tool']} ===")
    return {"code_quality": tuned_cq}

# ------------------------------------------------------------------
# Build single‑tool StateGraph
# ------------------------------------------------------------------

single_tool_graph = StateGraph(FullToolState)

single_tool_graph.add_node("info_miner", call_info_miner)
single_tool_graph.add_node("code_generator",      call_code_generator_for_single_tool)
single_tool_graph.add_node("reviewer",   call_reviewer_for_single_tool)
single_tool_graph.add_node("decider",    decide_next)
single_tool_graph.add_node("evaluator",  call_evaluator_for_single_tool)
single_tool_graph.add_node("optimizer",  call_optimizer_for_single_tool)      # ★ new

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
full_graph.add_node("processor",     call_processor)
full_graph.add_node("selector",         call_selector)
full_graph.add_node("process_all_tools",process_all_tools)

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
        "feature_dim"     : None,
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
