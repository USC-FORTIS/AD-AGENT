import os
from typing import Any, Dict, Optional

from config.config import Config
from agents.agent_processor import AgentProcessor
from agents.agent_info_miner import AgentInfoMiner
from agents.agent_code_generator import AgentCodeGenerator
from agents.agent_reviewer import AgentReviewer
from agents.agent_evaluator import AgentEvaluator
from agents.agent_optimizer import AgentOptimizer
from entity.code_quality import CodeQuality

from main import call_code_generator_for_single_tool, call_evaluator_for_single_tool, call_info_miner, call_optimizer_for_single_tool, call_processor, call_selector, call_selector, call_reviewer_for_single_tool

# Ensure API key is available
os.environ.setdefault("OPENAI_API_KEY", Config.OPENAI_API_KEY)

# optional states

# main -> api


# processor: deal with api usage, arrange task
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
        "n_features": None,
    }


def run_processor() -> Dict[str, Any]:
    """
    Run the processor to collect user input and build experiment config.

    Returns
    -------
    state : dict
        The updated pipeline state after processing user input.
    """
   
    state = build_state()
    return call_processor(state)


def run_selector(
    algorithm: list[str] = None, 
    dataset_train: str = None, 
    dataset_test: Optional[str] = None, 
    parameters: Optional[dict] = None
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

    Returns
    -------
    state : dict
        The updated pipeline state after selection.
    """
    
    state = build_state()
    state["experiment_config"] = {
        "algorithm": algorithm,
        "dataset_train": dataset_train,
        "dataset_test": dataset_test,
        "parameters": parameters,
    }
    check_dataset_exists(dataset_train, dataset_test)
    call_selector(state)
    return state


def run_info_miner(algorithm: str, package_name: str) -> str:
    """
    Query documentation for a given tool and return the algorithm doc.

    Parameters
    ----------
    algorithm : str
        Algorithm name to query.
    package_name : str
        Package name to query.

    Returns
    -------
    algorithm_doc : str
        Documentation string for the algorithm.
    """

    if not algorithm or not package_name:
        raise ValueError("Algorithm and package name must be provided for info mining.")
    state = build_state()
    state["current_tool"] = algorithm
    state["package_name"] = package_name
    call_info_miner(state)
    return state["algorithm_doc"]


def run_code_generator(
    tool: str,
    data_path_train: str,
    algorithm_doc: str = None,
    package_name: str = None,
    data_path_test: Optional[str] = None,
    input_parameters: Optional[dict] = None,
    code_quality: Optional[CodeQuality] = None,
) -> CodeQuality:
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

    Returns
    -------
    code_quality : CodeQuality
        The generated or revised CodeQuality object.
    """
   
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
    result_cq = result["code_quality"]
    print(f"Code generator result for {tool}: AUROC={result_cq.auroc}, AUPRC={result_cq.auprc}, Error Points={result_cq.error_points}, Error Message={result_cq.error_message}")
    return result_cq


def run_reviewer(
    code_quality: CodeQuality,
    tool: str,
    n_features: int = 2,
) -> CodeQuality:
    """
    Run reviewer once and return updated CodeQuality.

    Parameters
    ----------
    code_quality : CodeQuality
        CodeQuality object to review.
    tool : str
        Algorithm/tool name.
    n_features : int, optional
        Feature dimension for synthetic data (default is 2), which 
        should be the dimension of the training data.

    Returns
    -------
    code_quality : CodeQuality
        Updated CodeQuality object after review.
    """

    if code_quality is None or tool is None:
        raise ValueError("Code quality and tool must be provided for review.")
    state = build_state()
    state["current_tool"] = tool
    state["code_quality"] = code_quality
    state["n_features"] = n_features
    result = call_reviewer_for_single_tool(state)
    result_cq = result["code_quality"]
    print(f"Reviewer result for {tool}: AUROC={result_cq.auroc}, AUPRC={result_cq.auprc}, Error Points={result_cq.error_points}, Error Message={result_cq.error_message}")
    return result_cq
    


def run_codegenerator_reviewer_loop(
    tool: str,
    data_path_train: str,
    algorithm_doc: str = None,
    package_name: str = None,
    data_path_test: Optional[str] = None,
    input_parameters: Optional[dict] = None,
    max_reviews: int = 2,
    n_features: int = 2,
) -> CodeQuality:
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
    n_features : int, optional
        Feature dimension for synthetic data (default is 2), which 
        should be the dimension of the training data.

    Returns
    -------
    code_quality : CodeQuality
        Final CodeQuality object after review loop.
    """

    cq: Optional[CodeQuality] = None
    while True:
        cq = run_code_generator(tool, data_path_train, algorithm_doc, package_name, data_path_test, input_parameters, cq)
        cq = run_reviewer(cq, tool, n_features)
        if not cq.error_message or cq.review_count >= max_reviews:
            break
    print(f"Final code quality after {cq.review_count} reviews: AUROC={cq.auroc}, AUPRC={cq.auprc}, Error Points={cq.error_points}")
    return cq


def run_evaluator(
    code_quality: CodeQuality, 
    tool: str
) -> CodeQuality:
    """
    Run evaluator once on real data and return updated CodeQuality.

    Parameters
    ----------
    code_quality : CodeQuality
        CodeQuality object to evaluate.
    tool : str
        Algorithm name.

    Returns
    -------
    code_quality : CodeQuality
        Updated CodeQuality object after evaluation.
    """
    if code_quality is None or tool is None:
        raise ValueError("Code quality and tool must be provided for evaluation.")
    state = build_state()
    state["current_tool"] = tool
    state["code_quality"] = code_quality

    result = call_evaluator_for_single_tool(state) 
    return result["code_quality"]


def run_optimizer(
    code_quality: CodeQuality,
    algorithm_doc: str,
) -> CodeQuality:
    """
    Run optimizer once and return updated CodeQuality.

    Parameters
    ----------
    code_quality : CodeQuality
        CodeQuality object to optimize.
    algorithm_doc : str
        Documentation for the algorithm.

    Returns
    -------
    code_quality : CodeQuality
        Updated CodeQuality object after optimization.
    """

    if code_quality is None or algorithm_doc is None:
        raise ValueError("Code quality and algorithm documentation must be provided for optimization.")
    state = build_state()
    state["code_quality"] = code_quality
    state["algorithm_doc"] = algorithm_doc
    result = call_optimizer_for_single_tool(state)
    return result["code_quality"]


def run_evaluator_optimizer_loop(
    cq: CodeQuality,
    tool: str,
    algorithm_doc: str = None,
    optimizer_cycles: int = 1,
) -> CodeQuality:
    """
    Run reviewer loop, then alternate evaluator and optimizer for a number of cycles.

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
    best_cq : CodeQuality
        Final CodeQuality object after evaluation and optimization loop.
    """
    
    if cq is None or tool is None or algorithm_doc is None:
        raise ValueError("Code quality, tool, and algorithm documentation must be provided for evaluator/optimizer loop.")
    
    # Initial evaluation
    best_cq = run_evaluator(cq, tool)
    if best_cq.error_message:
        return best_cq

    for _ in range(max(0, optimizer_cycles)):
        tuned_cq = run_optimizer(best_cq, algorithm_doc)
        if tuned_cq.error_message:
            return tuned_cq
        best_cq = run_evaluator(tuned_cq, tool)
        if best_cq.error_message:
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