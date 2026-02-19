import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pipeline_api import (
    run_processor,
    run_selector,
    run_info_miner,
    run_code_generator,
    run_reviewer,
    run_evaluator,
    run_optimizer,
    run_codegenerator_reviewer_loop,
    run_evaluator_optimizer_loop,
)


def main():
    # Example experiment config (PyOD)

    # Run Processor to process user input
    run_processor()

    # Run selector
    algorithm = ["IForest"]  # Can be several algorithms or "all" or let the agent choose
    dataset_train = "./data/glass_train.mat"
    dataset_test = "./data/glass_test.mat"
    parameters = {"contamination": 0.1}
    run_selector(algorithm=algorithm, dataset_train=dataset_train, dataset_test=dataset_test, parameters=parameters)

    # Run info miner
    algorithm = "IForest"
    package_name = "PyOD" 
    run_info_miner(algorithm, package_name)

    # Code Generator
    tool = "IForest"
    data_path_train = "./data/glass_train.mat"
    data_path_test = "./data/glass_test.mat"
    with open("cache.json", "r") as f:
        cache = json.load(f)
    algorithm_doc = cache.get(tool, "No doc found in cache")
    algorithm_doc = algorithm_doc.get('document', str(algorithm_doc))
    package_name = "pyod"
    cq = run_code_generator(tool=tool, data_path_train=data_path_train, data_path_test=data_path_test, algorithm_doc=algorithm_doc, package_name=package_name)

    # # Code Reviewer
    run_reviewer(cq, tool, 8)
    
    # Code Generator + Reviewer loop
    cq = run_codegenerator_reviewer_loop(tool=tool, data_path_train=data_path_train, algorithm_doc=algorithm_doc, package_name=package_name, data_path_test=data_path_test, max_reviews=10, n_features=8)

    # Evaluator
    run_evaluator(code_quality=cq, tool=tool)

    # Optmizer
    run_optimizer(code_quality=cq, algorithm_doc=algorithm_doc)

    # Evaluator + Optimizer loop (optional)
    run_evaluator_optimizer_loop(cq=cq, tool=tool, algorithm_doc=algorithm_doc, optimizer_cycles=2)

if __name__ == "__main__":
    # Ensure API key is set before running
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set. Set it before running.")
    main()
