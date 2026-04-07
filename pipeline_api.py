import os
from typing import Any, Dict, Optional

from config.config import Config
from agents.agent_processor import AgentProcessor
from agents.agent_selector import AgentSelector
from agents.agent_info_miner import AgentInfoMiner
from agents.agent_code_generator import AgentCodeGenerator
from agents.agent_reviewer import AgentReviewer
from agents.agent_evaluator import AgentEvaluator
from agents.agent_optimizer import AgentOptimizer
from entity.code_quality import CodeQuality
from langchain_openai import ChatOpenAI

# Ensure API key is available
os.environ.setdefault("OPENAI_API_KEY", Config.OPENAI_API_KEY)



def build_state(experiment_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Create a default state dictionary for pipeline APIs."""
    return {
        "messages": [],
        "current_tool": "",
        "input_parameters": {},
        "data_path_train": "",
        "data_path_test": "",
        "package_name": "",
        "feature_dim": None,
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
        "experiment_config": experiment_config,
        "results": None,
        "algorithm_doc": None,
        "metadata": None,
    }


def run_processor(state: Dict[str, Any]) -> Dict[str, Any]:
    """Run the processor to collect user input and build experiment config."""
    processor = state["agent_processor"]
    processor.run_chatbot()
    state["experiment_config"] = processor.experiment_config
    return state


def run_selector(state: Dict[str, Any]) -> Dict[str, Any]:
    """Run the selector to determine package and tools based on experiment config."""
    if state.get("experiment_config") is None:
        raise ValueError("experiment_config not set, run processor or pass config first")
    selector = AgentSelector(state["experiment_config"])
    state.update(
        agent_selector=selector,
        input_parameters=selector.parameters,
        data_path_train=selector.data_path_train,
        data_path_test=selector.data_path_test,
        package_name=selector.package_name,
    )
    # feature_dim is extracted via sandbox metadata inspection in the selector
    if getattr(selector, "feature_dim", None):
        state["feature_dim"] = int(selector.feature_dim)
    state["metadata"] = getattr(selector, "metadata", None)
    return state


def run_info_miner(state: Dict[str, Any], tool: str) -> str:
    """Query documentation for a given tool and return the algorithm doc."""
    info_miner = state["agent_info_miner"]
    doc = info_miner.query_docs(
        tool,
        state.get("vectorstore"),
        state["package_name"],
    )
    state["algorithm_doc"] = doc
    return doc


def run_code_generator(
    state: Dict[str, Any],
    tool: str,
    algorithm_doc: str,
    code_quality: Optional[CodeQuality] = None,
) -> CodeQuality:
    """Generate or revise code for a tool. Returns CodeQuality."""
    code_generator = state["agent_code_generator"]
    if code_quality is None:
        code = code_generator.generate_code(
            algorithm=tool,
            data_path_train=state["data_path_train"],
            data_path_test=state["data_path_test"],
            algorithm_doc=algorithm_doc,
            input_parameters=state["input_parameters"],
            package_name=state["package_name"],
            metadata=state.get("metadata"),
        )
        parameters = code_generator._extract_init_params_dict(algorithm_doc)
        cq = CodeQuality(
            code=code,
            algorithm=tool,
            parameters=parameters,
            std_output="",
            error_message="",
            auroc=-1,
            auprc=-1,
            error_points=[],
            review_count=0,
        )
    else:
        cq = code_quality
        cq.code = code_generator.revise_code(cq, algorithm_doc)
    state["code_quality"] = cq
    return cq


def run_reviewer(
    state: Dict[str, Any],
    code_quality: CodeQuality,
    tool: str,
    feature_dim: Optional[int] = None,
) -> CodeQuality:
    """Run reviewer once and return updated CodeQuality."""
    reviewer = state["agent_reviewer"]
    if feature_dim is None:
        feature_dim = state.get("feature_dim")
    code_quality.error_message = reviewer.test_code(
        code_quality.code,
        tool,
        state["package_name"],
        feature_dim=feature_dim,
    )
    if code_quality.error_message:
        code_quality.review_count += 1
    return code_quality


def run_evaluator(state: Dict[str, Any], code_quality: CodeQuality, tool: str) -> CodeQuality:
    """Run evaluator once on real data and return updated CodeQuality."""
    evaluator = state["agent_evaluator"]
    final_cq = evaluator.execute_code(code_quality.code, tool, package_name=state["package_name"])
    final_cq.review_count = code_quality.review_count
    final_cq.parameters = code_quality.parameters
    return final_cq


def run_optimizer(
    state: Dict[str, Any],
    code_quality: CodeQuality,
    algorithm_doc: str,
    max_steps: int = 8,
) -> CodeQuality:
    """Run optimizer once and return updated CodeQuality."""
    if code_quality.error_message:
        return code_quality
    optimizer = state["agent_optimizer"]
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    tuned_cq = optimizer.run(
        llm=llm,
        quality=code_quality,
        algorithm_doc=algorithm_doc,
        max_steps=max_steps,
        package_name=state["package_name"],
    )
    return tuned_cq


def run_codegenerator_reviewer_loop(
    state: Dict[str, Any],
    tool: str,
    algorithm_doc: Optional[str] = None,
    max_reviews: int = 2,
    feature_dim: Optional[int] = None,
) -> CodeQuality:
    """Loop reviewer + code generation until success or max reviews."""
    if algorithm_doc is None:
        algorithm_doc = state.get("algorithm_doc") or run_info_miner(state, tool)

    cq: Optional[CodeQuality] = None
    while True:
        cq = run_code_generator(state, tool, algorithm_doc, code_quality=cq)
        cq = run_reviewer(state, cq, tool, feature_dim=feature_dim)
        if not cq.error_message or cq.review_count >= max_reviews:
            break
    state["code_quality"] = cq
    return cq


def run_reviewer_loop(
    state: Dict[str, Any],
    tool: str,
    algorithm_doc: Optional[str] = None,
    max_reviews: int = 2,
    feature_dim: Optional[int] = None,
) -> CodeQuality:
    """Alias for reviewer loop (generate code + review)."""
    return run_codegenerator_reviewer_loop(
        state,
        tool,
        algorithm_doc=algorithm_doc,
        max_reviews=max_reviews,
        feature_dim=feature_dim,
    )


def run_evaluator_loop(
    state: Dict[str, Any],
    tool: str,
    algorithm_doc: Optional[str] = None,
    max_reviews: int = 2,
    feature_dim: Optional[int] = None,
) -> CodeQuality:
    """Run reviewer loop first, then evaluator if review passes."""
    cq = run_codegenerator_reviewer_loop(
        state,
        tool,
        algorithm_doc=algorithm_doc,
        max_reviews=max_reviews,
        feature_dim=feature_dim,
    )
    if cq.error_message:
        return cq
    final_cq = run_evaluator(state, cq, tool)
    state["code_quality"] = final_cq
    return final_cq




def run_evaluator_optimizer_loop(
    state: Dict[str, Any],
    tool: str,
    algorithm_doc: Optional[str] = None,
    max_reviews: int = 2,
    optimizer_steps: int = 8,
    optimizer_cycles: int = 1,
    feature_dim: Optional[int] = None,
) -> CodeQuality:
    """Run reviewer loop, then alternate evaluator and optimizer for a number of cycles."""
    if algorithm_doc is None:
        algorithm_doc = state.get("algorithm_doc") or run_info_miner(state, tool)

    cq = run_codegenerator_reviewer_loop(
        state,
        tool,
        algorithm_doc=algorithm_doc,
        max_reviews=max_reviews,
        feature_dim=feature_dim,
    )
    if cq.error_message:
        return cq

    # Initial evaluation
    best_cq = run_evaluator(state, cq, tool)
    if best_cq.error_message:
        return best_cq

    for _ in range(max(0, optimizer_cycles)):
        tuned_cq = run_optimizer(state, best_cq, algorithm_doc, max_steps=optimizer_steps)
        if tuned_cq.error_message:
            return tuned_cq
        best_cq = run_evaluator(state, tuned_cq, tool)
        if best_cq.error_message:
            return best_cq

    state["code_quality"] = best_cq
    return best_cq
