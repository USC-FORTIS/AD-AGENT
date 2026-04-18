from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

RUNTIME_ROOT = REPO_ROOT / "runtime"
GENERATED_ROOT = RUNTIME_ROOT / "generated"
REVIEWER_SCRIPT_DIR = GENERATED_ROOT / "reviewer"
EVALUATOR_SCRIPT_DIR = GENERATED_ROOT / "evaluator"
BENCHMARK_SCRIPT_DIR = GENERATED_ROOT / "benchmark"
DATA_GEN_SCRIPT_DIR = GENERATED_ROOT / "data_gen"
FIXTURE_SCRIPT_DIR = GENERATED_ROOT / "fixtures"
CACHE_DIR = RUNTIME_ROOT / "cache"
DOC_CACHE_PATH = CACHE_DIR / "cache.json"


def ensure_runtime_dirs() -> None:
    for path in (
        RUNTIME_ROOT,
        GENERATED_ROOT,
        REVIEWER_SCRIPT_DIR,
        EVALUATOR_SCRIPT_DIR,
        BENCHMARK_SCRIPT_DIR,
        DATA_GEN_SCRIPT_DIR,
        FIXTURE_SCRIPT_DIR,
        CACHE_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def reviewer_script_path(algorithm_name: str) -> Path:
    ensure_runtime_dirs()
    return REVIEWER_SCRIPT_DIR / f"{algorithm_name}_test.py"


def evaluator_script_path(algorithm_name: str) -> Path:
    ensure_runtime_dirs()
    return EVALUATOR_SCRIPT_DIR / f"{algorithm_name}.py"


def benchmark_script_path(algorithm_name: str, dataset_name: str) -> Path:
    ensure_runtime_dirs()
    return BENCHMARK_SCRIPT_DIR / f"{algorithm_name}_{dataset_name}.py"
