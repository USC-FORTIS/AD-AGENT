import operator
import os
import sys
from typing import Annotated, Any, Dict, List, Optional, Sequence, Tuple, TypedDict

from agents.agent_selector import AgentSelector
from config.config import Config
from agents.agent_processor import AgentProcessor
from agents.agent_info_miner import AgentInfoMiner
from agents.agent_code_generator import AgentCodeGenerator
from agents.agent_reviewer import AgentReviewer
from agents.agent_evaluator import AgentEvaluator
from agents.agent_optimizer import AgentOptimizer
from entity.code_quality import CodeQuality
from utils.tslib_setup import prepare_tslib_repo

from langchain_openai          import ChatOpenAI

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

# Ensure API key is available
os.environ.setdefault("OPENAI_API_KEY", Config.OPENAI_API_KEY)

# Public API helpers for running the pipeline step-by-step.
def build_state() -> Dict[str, Any]:
    """
    Create a default state dictionary for pipeline APIs.

    Returns
    -------
    state : dict
        The initialized pipeline state dictionary with default values for all keys.
    """

    return {
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
        "results": None,
        "algorithm_doc": None,
    }

# ------------------------------------------------------------------
# Node: processor
# ------------------------------------------------------------------

def run_processor(state: FullToolState) -> Dict[str, Any]:
    """
    Run the processor to collect user input and build experiment config.

    Parameters
    ----------
    state : FullToolState, optional
        Existing pipeline state to reuse. If None, a new default state is created.

    Returns
    -------
    state : dict
        The updated pipeline state after processing user input.
    """
    if state is None:
        state = build_state()
    return call_processor(state)


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
def run_selector(
    algorithm: list[str] = None, 
    dataset_train: str = None, 
    dataset_test: Optional[str] = None, 
    parameters: Optional[dict] = None,
    state: FullToolState = None
) -> Dict[str, Any]:
    """
    Run the selector to determine package and tools based on experiment config.

    Parameters
    ----------
    algorithm : list[str], optional
        Algorithm name or list to use for selection. Can be a specific algorithm, "all", or None to let the agent choose.
    dataset_train : str, optional
        Path to training dataset.
    dataset_test : str, optional
        Path to testing dataset.
    parameters : dict, optional
        Additional parameters for selection.
    state : FullToolState, optional
        Existing pipeline state to reuse. If None, a new state is created and
        `experiment_config` is built from the function arguments.

    Returns
    -------
    state : dict
        The updated pipeline state after selection.
    """
    if state is None:
        state = build_state()
        state["experiment_config"] = {
            "algorithm": algorithm,
            "dataset_train": dataset_train,
            "dataset_test": dataset_test,
            "parameters": parameters,
        }
        train_path = dataset_train
        test_path = dataset_test
    else:
        exp = state.get("experiment_config") or {}
        train_path = exp.get("dataset_train")
        test_path = exp.get("dataset_test")

    if not train_path:
        raise ValueError("dataset_train is missing in selector input/state.")
    # check_dataset_exists(train_path, test_path)
    call_selector(state)
    return state

    
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
        # vectorstore     = selector.vectorstore
    )
    print("\n=== [Selector] Selection complete ===")
    return state



# ------------------------------------------------------------------
# Node: info_miner
# ------------------------------------------------------------------
def _prepare_tslib_repo_if_needed(package_name: Optional[str]) -> None:
    if package_name == "tslib":
        prepare_tslib_repo(project_root=os.path.dirname(os.path.abspath(__file__)))


def run_info_miner(
    algorithm: Optional[str] = None,
    package_name: Optional[str] = None,
    state: FullToolState = None
) -> Dict[str, Any]:
    """
    Query documentation for a given tool and return the algorithm doc.

    Parameters
    ----------
    algorithm : str
        Algorithm name to query.
    package_name : str
        Package name to query.
    state : FullToolState, optional
        Existing pipeline state to reuse. If None, a new state is created from
        `algorithm` and `package_name`.

    Returns
    -------
    state : dict
        The updated pipeline state after code generation, containing the new CodeQuality.
    """

    if state is not None:
        result = call_info_miner(state)
        return result

    if not algorithm or not package_name:
        raise ValueError("Algorithm and package name must be provided for info mining.")
    state = build_state()
    state["current_tool"] = algorithm
    state["package_name"] = package_name
    _prepare_tslib_repo_if_needed(package_name)
    result = call_info_miner(state)
    return result

def call_info_miner(state: FullToolState) -> dict:
    _prepare_tslib_repo_if_needed(state["package_name"])
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
def run_code_generator(
    tool: Optional[str] = None,
    data_path_train: Optional[str] = None,
    algorithm_doc: str = None,
    package_name: str = None,
    data_path_test: Optional[str] = None,
    input_parameters: Optional[dict] = None,
    code_quality: Optional[CodeQuality] = None,
    state: FullToolState = None
) -> Dict[str, Any]:
    """
    Generate or revise code for a tool. Returns CodeQuality.

    Parameters
    ----------
    tool : str
        Algorithm name.
    data_path_train : str
        Path to training dataset.
    algorithm_doc : str, optional
        Documentation for the algorithm.
    package_name : str, optional
        Package name.
    data_path_test : str, optional
        Path to testing dataset.
    input_parameters : dict, optional
        Additional parameters for code generation.
    code_quality : CodeQuality, optional
        Existing CodeQuality object for revision.
    state : FullToolState, optional`
        Existing pipeline state to reuse. If None, a new state is created from
        the function arguments.

    Returns
    -------
    state : dict
        The updated pipeline state after code generation, containing the new CodeQuality.
    """
    if state is not None:
        result = call_code_generator_for_single_tool(state)
        return result

    state = build_state()
    print(f"Running code generator for tool: {package_name}")
    if tool is None or data_path_train is None or algorithm_doc is None or package_name is None:
        raise ValueError("Tool, training data path, algorithm documentation, and package name must be provided for code generation.")
    check_dataset_exists(data_path_train, data_path_test)
    state["current_tool"] = tool
    state["data_path_train"] = data_path_train
    state["data_path_test"] = data_path_test
    state["algorithm_doc"] = algorithm_doc
    state["input_parameters"] = input_parameters
    state["package_name"] = package_name
    state["code_quality"] = code_quality

    result = call_code_generator_for_single_tool(state)
    return result

def call_code_generator_for_single_tool(state: FullToolState) -> dict:
    code_generator = state["agent_code_generator"]
    tool  = state["current_tool"]
    _prepare_tslib_repo_if_needed(state["package_name"])

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
def run_reviewer(
    code_quality: Optional[CodeQuality] = None,
    tool: Optional[str] = None,
    state: FullToolState = None
) -> Dict[str, Any]:
    """
    Run reviewer once and return updated CodeQuality.

    Parameters
    ----------
    code_quality : CodeQuality
        CodeQuality object to review.
    tool : str
        Algorithm/tool name.
        Feature dimension for synthetic data (default is 2), which 
        should be the dimension of the training data.
    state : FullToolState, optional
        Existing pipeline state to reuse. If None, a new state is created from
        `code_quality` and `tool`.

    Returns
    -------
    state : dict
        The updated pipeline state after code generation, containing the new CodeQuality.
    """
    if state is not None:
        result = call_reviewer_for_single_tool(state)
        return result

    if code_quality is None or tool is None:
        raise ValueError("Code quality and tool must be provided for review.")
    state = build_state()
    state["current_tool"] = tool
    state["code_quality"] = code_quality
    result = call_reviewer_for_single_tool(state)
    
    # print(f"Reviewer result for {tool}: AUROC={result_cq.auroc}, AUPRC={result_cq.auprc}, Error Points={result_cq.error_points}, Error Message={result_cq.error_message}")
    return result
    

def call_reviewer_for_single_tool(state: FullToolState) -> dict:
    reviewer = state["agent_reviewer"]
    cq       = state["code_quality"]
    tool     = state["current_tool"]

    print(f"\n=== [Reviewer] Running validation for {tool} ===")
    cq.error_message = reviewer.test_code(
        cq.code,
        tool,
        state["package_name"],
        train_dataset=state["data_path_train"]
    )

    if cq.error_message:
        cq.review_count += 1
    print(f"\n=== [Reviewer] Validation completed for {tool} ===")
    return {"code_quality": cq}


def run_codegenerator_reviewer_loop(
    tool: str,
    data_path_train: str,
    algorithm_doc: str = None,
    package_name: str = None,
    data_path_test: Optional[str] = None,
    input_parameters: Optional[dict] = None,
    max_reviews: int = 2,
) -> Dict[str, Any]:
    """
    Loop reviewer + code generation until success or max reviews.

    Parameters
    ----------
    tool : str
        Algorithm name.
    data_path_train : str
        Path to training dataset.
    algorithm_doc : str, optional
        Documentation for the algorithm.
    package_name : str, optional
        Package name.
    data_path_test : str, optional
        Path to testing dataset.
    input_parameters : dict, optional
        Additional parameters for code generation.
    max_reviews : int, optional
        Maximum number of review cycles (default is 2).

    Returns
    -------
    state : dict
        The updated pipeline state after code generation, containing the new CodeQuality.
    """

    cq: Optional[CodeQuality] = None
    while True:
        cq = run_code_generator(tool, data_path_train, algorithm_doc, package_name, data_path_test, input_parameters, cq)
        cq = run_reviewer(cq, tool)
        if not cq["code_quality"].error_message or cq["code_quality"].review_count >= max_reviews:
            break
    print(f"Final code quality after {cq['code_quality'].review_count} reviews: AUROC={cq['code_quality'].auroc}, AUPRC={cq['code_quality'].auprc}, Error Points={cq['code_quality'].error_points}")
    return cq



# ------------------------------------------------------------------
# Node: Evaluator  (real‑data execution & metrics)
# ------------------------------------------------------------------
def run_evaluator(
    code_quality: Optional[CodeQuality] = None, 
    tool: Optional[str] = None,
    state: FullToolState = None
) -> Dict[str, Any]:
    """
    Run evaluator once on real data and return updated CodeQuality.

    Parameters
    ----------
    code_quality : CodeQuality
        CodeQuality object to evaluate.
    tool : str
        Algorithm name.
    state : FullToolState, optional
        Existing pipeline state to reuse. If None, a new state is created from
        `code_quality` and `tool`.

    Returns
    -------
    state : dict
        The updated pipeline state after evaluation, containing the new CodeQuality.
    """
    if state is not None:
        result = call_evaluator_for_single_tool(state)
        return result

    if code_quality is None or tool is None:
        raise ValueError("Code quality and tool must be provided for evaluation.")
    state = build_state()
    state["current_tool"] = tool
    state["code_quality"] = code_quality
    result = call_evaluator_for_single_tool(state) 
    return result

def call_evaluator_for_single_tool(state: FullToolState) -> dict:
    evaluator = state["agent_evaluator"]
    cq        = state["code_quality"]
    tool      = state["current_tool"]

    print(f"\n=== [Evaluator] Real‑data run for {tool} ===")
    final_cq = evaluator.execute_code(cq.code, tool)

    # keep review_count & parameters
    final_cq.review_count = cq.review_count
    final_cq.parameters = cq.parameters
    return {"code_quality": final_cq}


# ------------------------------------------------------------------
# Node: Optimizer  (LLM‑driven parameter tuning)
# ------------------------------------------------------------------
def run_optimizer(
    code_quality: Optional[CodeQuality] = None,
    algorithm_doc: Optional[str] = None,
    state: FullToolState = None
) -> Dict[str, Any]:
    """
    Run optimizer once and return updated CodeQuality.

    Parameters
    ----------
    code_quality : CodeQuality
        CodeQuality object to optimize.
    algorithm_doc : str
        Documentation for the algorithm.
    state : FullToolState, optional
        Existing pipeline state to reuse. If None, a new state is created from
        `code_quality` and `algorithm_doc`.

    Returns
    -------
    state : dict
        The updated pipeline state after optimization, containing the new CodeQuality.
    """
    if state is not None:
        result = call_optimizer_for_single_tool(state)
        return result

    if code_quality is None or algorithm_doc is None:
        raise ValueError("Code quality and algorithm documentation must be provided for optimization.")
    state = build_state()
    state["code_quality"] = code_quality
    state["algorithm_doc"] = algorithm_doc
    result = call_optimizer_for_single_tool(state)
    return result

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
                             max_steps=8)
    print(f"\n=== [Optimizer] Tuning finished for {state['current_tool']} ===")
    return {"code_quality": tuned_cq}


def run_evaluator_optimizer_loop(
    cq: CodeQuality,
    tool: str,
    algorithm_doc: str = None,
    optimizer_cycles: int = 1,
) -> Dict[str, Any]:
    """
    Run an initial evaluator pass, then alternate optimizer and evaluator.

    Parameters
    ----------
    cq : CodeQuality
        Initial CodeQuality object.
    tool : str
        Algorithm name.
    algorithm_doc : str, optional
        Documentation for the algorithm.
    optimizer_cycles : int, optional
        Number of optimization cycles (default is 1).

    Returns
    -------
    state : dict
        The updated pipeline state after evaluation and optimization loop, containing the new CodeQuality.
    """
    
    if cq is None or tool is None or algorithm_doc is None:
        raise ValueError("Code quality, tool, and algorithm documentation must be provided for evaluator/optimizer loop.")
    
    # Initial evaluation
    best_cq = run_evaluator(cq, tool)
    if best_cq["code_quality"].error_message:
        return best_cq

    for _ in range(max(0, optimizer_cycles)):
        tuned_cq = run_optimizer(best_cq, algorithm_doc)
        if tuned_cq["code_quality"].error_message:
            return tuned_cq
        best_cq = run_evaluator(tuned_cq, tool)
        if best_cq["code_quality"].error_message:
            return best_cq
    return best_cq


def check_dataset_exists(
    dataset_train: str, 
    dataset_test: Optional[str] = None
) -> None:
    """
    Check if training and testing dataset files exist.

    Parameters
    ----------
    dataset_train : str
        Path to training dataset.
    dataset_test : str, optional
        Path to testing dataset.

    Raises
    ------
    FileNotFoundError
        If either dataset file does not exist.
    """
    if not os.path.exists(dataset_train):
        raise FileNotFoundError(f"Training dataset not found at {dataset_train}")
    if dataset_test and not os.path.exists(dataset_test):
        raise FileNotFoundError(f"Testing dataset not found at {dataset_test}")
