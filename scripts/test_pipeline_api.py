import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pipeline_api import (
    build_state,
    run_evaluator,
    run_selector,
    run_info_miner,
    run_codegenerator_reviewer_loop,
    run_evaluator_optimizer_loop,
)


def main():
    # Example experiment config (PyOD)
    state = build_state({
        "algorithm": ["IForest"],
        "dataset_train": "./data/glass_train.mat",
        "dataset_test": "./data/glass_test.mat",
        "parameters": {},
    })

    # Run selector (skip processor)
    run_selector(state)

    tool = "IForest"

    # Run info miner
    doc = run_info_miner(state, tool)

    # Reviewer loop (synthetic data)
    cq = run_codegenerator_reviewer_loop(state, tool, algorithm_doc=doc)
    if cq.error_message:
        print("Reviewer failed:", cq.error_message)
        return
    print("Reviewer passed")

    # Evaluator loop (real data)
    final_cq = run_evaluator(state, cq, tool)
    if final_cq.error_message:
        print("Evaluator failed:", final_cq.error_message)
        return

    print(f"AUROC: {final_cq.auroc:.4f}  AUPRC: {final_cq.auprc:.4f}")
    print("Parameters:", final_cq.parameters)

    # Evaluator + Optimizer loop (optional)
    tuned_cq = run_evaluator_optimizer_loop(
        state,
        tool,
        algorithm_doc=doc,
        optimizer_steps=4,
        optimizer_cycles=1,
    )
    if tuned_cq.error_message:
        print("Evaluator/Optimizer failed:", tuned_cq.error_message)
        return
    print(f"Tuned AUROC: {tuned_cq.auroc:.4f}  Tuned AUPRC: {tuned_cq.auprc:.4f}")


if __name__ == "__main__":
    # Ensure API key is set before running
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set. Set it before running.")
    main()
