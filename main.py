import logging
import sys
import operator
import asyncio
import os
from config.config import Config
from typing import TypedDict, Annotated, Sequence

os.environ['OPENAI_API_KEY'] = Config.OPENAI_API_KEY
logging.basicConfig(stream=sys.stdout, level=logging.ERROR)

# ===== langgraph related =====
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, END

# ===== Business-related =====
from agents.agent_preprocessor import AgentPreprocessor
from agents.agent_planner import AgentPlanner
from agents.agent_instructor import AgentInstructor
from agents.agent_reviewer import AgentReviewer
from entity.code_quality import CodeQuality

# =================================================================
# 1) Define SingleToolState, representing the state of a "single tool" process in langgraph
# =================================================================
class SingleToolState(TypedDict):
    # Optional, stores conversation messages
    messages: Annotated[Sequence[BaseMessage], operator.add]
    # The name of the PyOD algorithm to be processed
    current_tool: str
    input_parameters: dict
    data_path: str
    # Dependency injection
    agent_instructor: AgentInstructor
    agent_reviewer: AgentReviewer
    vectorstore: object  # Can be changed to FAISS or another specific type
    # Code quality information, including error_message, review_count, code, etc.
    code_quality: CodeQuality | None
    # Boolean for decision node: determines whether to return to the instructor for re-execution
    should_rerun: bool

# =================================================================
# 2) Define node functions for processing a single tool (synchronous implementation)
# =================================================================
def call_instructor_for_single_tool(state: SingleToolState) -> dict:
    """
    Node: Instructor generates/executes code.
    If it is the first execution (no code_quality), generate_code + execute_code;
    If the Reviewer has just fixed the code (has code_quality), execute the revised code again.
    """
    instructor = state["agent_instructor"]
    tool = state["current_tool"]
    vectorstore = state["vectorstore"]
    input_parameters = state["input_parameters"]
    data_path = state["data_path"]

    if not state["code_quality"]:
        print(f"\n=== [Instructor] Start processing {tool} (first run) ===")
        code = instructor.generate_code(
            algorithm=tool,
            data_path=data_path,
            vectorstore=vectorstore,
            input_parameters=input_parameters
        )
    else:
        print(f"\n=== [Instructor] Re-run updated code for {tool} ===")
        code = state["code_quality"].code

    new_code_quality = instructor.execute_generated_code(code,tool)

    if state["code_quality"]:
        new_code_quality.review_count = state["code_quality"].review_count
        new_code_quality.algorithm = state["code_quality"].algorithm

    if not new_code_quality.algorithm:
        new_code_quality.algorithm = tool

    return {"code_quality": new_code_quality}


def call_reviewer_for_single_tool(state: SingleToolState) -> dict:
    """
    Node: Reviewer reviews the code.
    If there are errors and review_count < 2, revised code is generated and written back to code_quality.code.
    """
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


def decide_reviewer_result(state: SingleToolState) -> dict:
    """
    Decision node: Determines whether to return to the Instructor for re-execution.
    Condition: error_message is not empty and review_count < 2
    """
    code_quality = state["code_quality"]
    if code_quality.error_message and code_quality.review_count < 2:
        return {"should_rerun": True}
    else:
        return {"should_rerun": False}


def check_if_need_rerun(state: SingleToolState):
    """
    Returns a string for conditional transition in 'decider' → (instructor or END)
    """
    return "need_rerun" if state["should_rerun"] else "done"
    # return "done"

# =================================================================
# 3) Construct SingleTool Graph (subgraph for a single tool)
# =================================================================
single_tool_graph = StateGraph(SingleToolState)

single_tool_graph.add_node("instructor", call_instructor_for_single_tool)
single_tool_graph.add_node("reviewer", call_reviewer_for_single_tool)
single_tool_graph.add_node("decider", decide_reviewer_result)

single_tool_graph.set_entry_point("instructor")

single_tool_graph.add_edge("instructor", "reviewer")
single_tool_graph.add_edge("reviewer", "decider")

single_tool_graph.add_conditional_edges(
    "decider",
    check_if_need_rerun,
    {
        "need_rerun": "instructor",
        "done": END,
    }
)

single_tool_compiled = single_tool_graph.compile()

# =================================================================
# 4) Process multiple tools in parallel in main() (using asyncio coroutines)
# =================================================================
async def process_single_tool(tool, instructor_instance, reviewer_instance, vectorstore, input_parameters, data_path):
    """
    Asynchronously execute the process of a single tool.
    Wrap the synchronous single_tool_compiled.invoke() call using asyncio.to_thread() for async execution.
    """
    initial_state_for_this_tool = {
        "messages": [],
        "current_tool": tool,
        "input_parameters": input_parameters,  # No input parameters for now
        "data_path": data_path,
        "agent_instructor": instructor_instance,
        "agent_reviewer": reviewer_instance,
        "vectorstore": vectorstore,
        "code_quality": None,  # Initially, no code_quality
        "should_rerun": False,
    }
    final_state = await asyncio.to_thread(
        single_tool_compiled.invoke,
        initial_state_for_this_tool,
        config={"recursion_limit": 20}
    )
    print(f"[Main] Tool={tool} done, final_state={final_state}")
    return tool, final_state


async def main():
    # Remove generated files
    if os.path.exists("generated_data_loader.py"):
        os.remove("generated_data_loader.py")
    # Initialize agents
    print(f"\n=== [Preprocessor] Start processing user input ===")
    preprocessor_instance = AgentPreprocessor()
    print(f"\n=== [Planner] Start generate idea space ===")
    planner_instance = AgentPlanner(preprocessor_instance.experiment_config)
    instructor_instance = AgentInstructor()
    reviewer_instance = AgentReviewer()

    use_parallel = "-p" in sys.argv
    # Obtain the tool list and vectorstore from Planner
    tools = planner_instance.tools  # e.g., ['ECOD', 'ABOD', 'COPOD', ...]
    parameters = planner_instance.parameters # e.g., {'contamination': 0.1}
    data_path = planner_instance.data_path
    vectorstore = planner_instance.vectorstore 

    # Create async tasks for each tool
    results = []
    if use_parallel:
        tasks = [
            process_single_tool(tool, instructor_instance, reviewer_instance, vectorstore, parameters, data_path)
            for tool in tools
        ]
        results = await asyncio.gather(*tasks)
    else:
        for tool in tools:
            result = await process_single_tool(tool, instructor_instance, reviewer_instance, vectorstore, parameters, data_path)
            results.append(result)

    print("\n=== All parallel subtasks are processed ===")
    for tool, final_state in results:
        code_quality = final_state["code_quality"]
        if code_quality and not code_quality.error_message:
            if code_quality.detected_anomalies >= 0:
                print(f"【{tool}】Execution successful, detected anomaly number: {code_quality.detected_anomalies}, "
                      f"True anomaly number: {code_quality.true_anomalies}")
            else:
                print(f"【{tool}】Execution failed, as data type is not suitable for {tool}")
        else:
            print(f"【{tool}】Execution failed, error message: {code_quality.error_message if code_quality else 'Unknown'}")

if __name__ == "__main__":
    asyncio.run(main())