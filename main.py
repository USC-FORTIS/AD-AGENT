import logging
import sys
import operator
import asyncio
import os
from config.config import Config
from typing import TypedDict, Annotated, Sequence, List, Tuple, Any

os.environ['OPENAI_API_KEY'] = Config.OPENAI_API_KEY
logging.basicConfig(stream=sys.stdout, level=logging.ERROR)

# ===== langgraph related =====
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, END

# ===== Business logic related =====
from agents.agent_preprocessor import AgentPreprocessor
from agents.agent_planner import AgentPlanner
from agents.agent_instructor import AgentInstructor
from agents.agent_reviewer import AgentReviewer
from entity.code_quality import CodeQuality

# =================================================================
# Define full state, adding `results` to store the processing results of all tools
# =================================================================
class FullToolState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    current_tool: str
    input_parameters: dict
    data_path_train: str
    data_path_test: str
    agent_instructor: AgentInstructor
    agent_reviewer: AgentReviewer
    vectorstore: Any
    code_quality: CodeQuality | None
    should_rerun: bool
    agent_preprocessor: AgentPreprocessor
    agent_planner: AgentPlanner | None
    experiment_config: dict | None
    results: List[Tuple[str, Any]] | None

# =================================================================
# 1. Node: Preprocessor
#    Calls preprocessor.run_chatbot() (synchronous), writes experiment_config into state
# =================================================================
def call_preprocessor(state: FullToolState) -> dict:
    preprocessor = state["agent_preprocessor"]
    print("\n=== [Preprocessor] Starting to process user input ===")
    preprocessor.run_chatbot()
    state["experiment_config"] = preprocessor.experiment_config
    print("\n=== [Preprocessor] User input processing complete ===")
    return state

# =================================================================
# 2. Node: Planner
#    Instantiates AgentPlanner based on experiment_config and writes tool list, parameters, data path, and vectorstore
# =================================================================
def call_planner(state: FullToolState) -> dict:
    if state["experiment_config"] is None:
        raise ValueError("experiment_config not set, please run the preprocessor first!")
    print("\n=== [Planner] Starting to generate idea space ===")
    planner_instance = AgentPlanner(state["experiment_config"])
    state["agent_planner"] = planner_instance
    state["input_parameters"] = planner_instance.parameters
    state["data_path_train"] = planner_instance.data_path_train
    state["data_path_test"] = planner_instance.data_path_test
    state["vectorstore"] = planner_instance.vectorstore
    print("\n=== [Planner] Idea space generation complete ===")
    return state

# =================================================================
# 3. Nodes for single-tool processing: Instructor, Reviewer, Decider
# =================================================================
def call_instructor_for_single_tool(state: FullToolState) -> dict:
    instructor = state["agent_instructor"]
    tool = state["current_tool"]
    vectorstore = state["vectorstore"]
    input_parameters = state["input_parameters"]
    data_path_train = state["data_path_train"]
    data_path_test = state["data_path_test"]

    if not state["code_quality"]:
        print(f"\n=== [Instructor] Processing {tool} (first execution) ===")
        code = instructor.generate_code(
            algorithm=tool,
            data_path_train=data_path_train,
            data_path_test=data_path_test,
            vectorstore=vectorstore,
            input_parameters=input_parameters
        )
    else:
        print(f"\n=== [Instructor] Re-executing updated code for {tool} ===")
        code = state["code_quality"].code

    new_code_quality = instructor.execute_generated_code(code, tool)
    if state["code_quality"]:
        new_code_quality.review_count = state["code_quality"].review_count
        new_code_quality.algorithm = state["code_quality"].algorithm
    if not new_code_quality.algorithm:
        new_code_quality.algorithm = tool
    return {"code_quality": new_code_quality}

def call_reviewer_for_single_tool(state: FullToolState) -> dict:
    reviewer = state["agent_reviewer"]
    code_quality = state["code_quality"]
    tool = state["current_tool"]

    print(f"\n=== [Reviewer] Reviewing code for {tool} ===")
    if code_quality.error_message:
        code_quality.review_count += 1
        revised_code = reviewer.review_code(
            code_quality=code_quality,
            vectorstore=state["vectorstore"]
        )
        code_quality.code = revised_code
    return {"code_quality": code_quality}

def decide_reviewer_result(state: FullToolState) -> dict:
    code_quality = state["code_quality"]
    if code_quality.error_message and code_quality.review_count < 2:
        return {"should_rerun": True}
    else:
        return {"should_rerun": False}

def check_if_need_rerun(state: FullToolState):
    return "need_rerun" if state["should_rerun"] else "done"

# Construct single-tool processing graph (instructor → reviewer → decider)
single_tool_graph = StateGraph(FullToolState)
single_tool_graph.add_node("instructor", call_instructor_for_single_tool)
single_tool_graph.add_node("reviewer", call_reviewer_for_single_tool)
single_tool_graph.add_node("decider", decide_reviewer_result)
single_tool_graph.set_entry_point("instructor")
single_tool_graph.add_edge("instructor", "reviewer")
single_tool_graph.add_edge("reviewer", "decider")
single_tool_graph.add_conditional_edges("decider", check_if_need_rerun, {
    "need_rerun": "instructor",
    "done": END,
})
compiled_single_tool = single_tool_graph.compile()

# =================================================================
# 4. Node: Process all tools (parallel or sequential)
#    Get tool list from agent_planner, build per-tool state,
#    Call single-tool processing graph for each tool, store results in state["results"]
# =================================================================
def process_all_tools(state: FullToolState) -> dict:
    if not state["agent_planner"]:
        raise ValueError("agent_planner is not set!")
    tools = state["agent_planner"].tools
    if not tools:
        print("No tool list generated. Exiting.")
        state["results"] = []
        return state

    async def process_tool_async(tool, state):
        tool_state = state.copy()
        tool_state["current_tool"] = tool
        tool_state["code_quality"] = None
        tool_state["should_rerun"] = False
        final_state = await asyncio.to_thread(
            compiled_single_tool.invoke,
            tool_state,
            config={"recursion_limit": 20}
        )
        return tool, final_state

    results = []
    use_parallel = "-p" in sys.argv
    if use_parallel:
        # Create new event loop and process in parallel
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        tasks = [process_tool_async(tool, state) for tool in tools]
        results = loop.run_until_complete(asyncio.gather(*tasks))
        loop.close()
    else:
        # Process sequentially
        for tool in tools:
            results.append(asyncio.run(process_tool_async(tool, state)))
    state["results"] = results
    return state

# =================================================================
# 5. Build full process graph: preprocessor → planner → process_all_tools
#    Combine preprocessing, planning, and tool processing
# =================================================================
full_graph = StateGraph(FullToolState)
full_graph.add_node("preprocessor", call_preprocessor)
full_graph.add_node("planner", call_planner)
full_graph.add_node("process_all_tools", process_all_tools)
full_graph.set_entry_point("preprocessor")
full_graph.add_edge("preprocessor", "planner")
full_graph.add_edge("planner", "process_all_tools")
compiled_full_graph = full_graph.compile()

# =================================================================
# 6. Main function
#    Initialize agents and initial state, call the full process graph,
#    then print the results of each tool
# =================================================================
async def main():
    if os.path.exists("train_data_loader.py"):
        os.remove("train_data_loader.py")
    if os.path.exists("test_data_loader.py"):
        os.remove("test_data_loader.py")
    instructor_instance = AgentInstructor()
    reviewer_instance = AgentReviewer()
    preprocessor_instance = AgentPreprocessor()
    
    initial_state: FullToolState = {
        "messages": [],
        "current_tool": "",
        "input_parameters": {},
        "data_path_train": "",
        "data_path_test": "",
        "agent_instructor": instructor_instance,
        "agent_reviewer": reviewer_instance,
        "vectorstore": None,
        "code_quality": None,
        "should_rerun": False,
        "agent_preprocessor": preprocessor_instance,
        "agent_planner": None,
        "experiment_config": None,
        "results": None,
    }
    
    print("\n=== [Main] Starting full process graph ===")
    final_state = await asyncio.to_thread(
        compiled_full_graph.invoke,
        initial_state,
        config={"recursion_limit": 20}
    )
    print("\n=== [Main] Full process graph completed ===")
    
    # Print results for each tool
    results = final_state.get("results", [])
    for tool, tool_state in results:
        code_quality = tool_state.get("code_quality")
        if code_quality and not code_quality.error_message:
            if code_quality.detected_anomalies >= 0:
                print(f"[{tool}] Success, AUROC: {code_quality.auroc:.4f}, AUPRC: {code_quality.auprc:.4f}, Error Points: {code_quality.error_points}")
            else:
                print(f"[{tool}] Failed, data type not suitable for {tool}")
        else:
            err_msg = code_quality.error_message if code_quality else "Unknown"
            print(f"[{tool}] Failed, error message: {err_msg}")

if __name__ == "__main__":
    asyncio.run(main())
