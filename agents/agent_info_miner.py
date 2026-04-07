from langchain_core.prompts import PromptTemplate
import ast
from datetime import datetime, timedelta
import json
import re
from pathlib import Path
from filelock import FileLock
from openai import OpenAI
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.config import Config
os.environ['OPENAI_API_KEY'] = Config.OPENAI_API_KEY

web_search_prompt_pyod = PromptTemplate.from_template("""
   You are a machine learning expert and will assist me with researching a specific use of a deep learning model in PyOD. Here is the official document you should refer to: https://pyod.readthedocs.io/en/latest/pyod.models.html
   I want to run `{algorithm_name}`. What is the Initialization function, parameters and Attributes? 
   Briefly return realted document content.
   Then, extract **all parameters** of the `__init__` method for the `{algorithm_name}` class, along with their default values if available, and return a valid Python dictionary string in the following format:
    ```python
    {{
        "param1": default_value1,
        "param2": default_value2,
        ...
    }}
   If any default value is an object or function (e.g., MinMaxScaler()), wrap it in quotes to ensure valid Python syntax for ast.literal_eval.
""")
web_search_prompt_pygod = PromptTemplate.from_template("""
   You are a machine learning expert and will assist me with researching a specific use of a deep learning model in PyGOD. Here is the official document you should refer to: https://docs.pygod.org/en/latest/pygod.detector.{algorithm_name}.html
   I want to run `{algorithm_name}`. What is the Initialization function, parameters and Attributes? 
   Briefly return realted document content.
   Then, extract **all parameters** of the `__init__` method for the `{algorithm_name}` class, along with their default values if available, and return a valid Python dictionary string in the following format:
    ```python
    {{
        "param1": default_value1,
        "param2": default_value2,
        ...
    }}
   If any default value is an object or function (e.g., MinMaxScaler()), wrap it in quotes to ensure valid Python syntax for ast.literal_eval.
""")

web_dict = {
    "GlobalNaiveAggregate": "https://unit8co.github.io/darts/generated_api/darts.models.forecasting.global_baseline_models.html",
    "GlobalNaiveDrift": "https://unit8co.github.io/darts/generated_api/darts.models.forecasting.global_baseline_models.html",
    "GlobalNaiveSeasonal": "https://unit8co.github.io/darts/generated_api/darts.models.forecasting.global_baseline_models.html",
    "RNNModel": "https://unit8co.github.io/darts/generated_api/darts.models.forecasting.rnn_model.html",
    "BlockRNNModel": "https://unit8co.github.io/darts/generated_api/darts.models.forecasting.block_rnn_model.html",
    "NBEATSModel": "https://unit8co.github.io/darts/generated_api/darts.models.forecasting.nbeats.html",
    "NHiTSModel": "https://unit8co.github.io/darts/generated_api/darts.models.forecasting.nhits.html",
    "TCNModel": "https://unit8co.github.io/darts/generated_api/darts.models.forecasting.tcn_model.html",
    "TransformerModel": "https://unit8co.github.io/darts/generated_api/darts.models.forecasting.transformer_model.html",
    "TFTModel": "https://unit8co.github.io/darts/generated_api/darts.models.forecasting.tft_model.html",
    "DLinearModel": "https://unit8co.github.io/darts/generated_api/darts.models.forecasting.dlinear.html",
    "NLinearModel": "https://unit8co.github.io/darts/generated_api/darts.models.forecasting.nlinear.html",
    "TiDEModel": "https://unit8co.github.io/darts/generated_api/darts.models.forecasting.tide_model.html",
    "TSMixerModel": "https://unit8co.github.io/darts/generated_api/darts.models.forecasting.tsmixer_model.html",
    "LinearRegressionModel": "https://unit8co.github.io/darts/generated_api/darts.models.forecasting.linear_regression_model.html",
    "RandomForest": "https://unit8co.github.io/darts/generated_api/darts.models.forecasting.random_forest.html",
    "LightGBMModel": "https://unit8co.github.io/darts/generated_api/darts.models.forecasting.lgbm.html",
    "XGBModel": "https://unit8co.github.io/darts/generated_api/darts.models.forecasting.xgboost.html",
    "CatBoostModel": "https://unit8co.github.io/darts/generated_api/darts.models.forecasting.catboost_model.html"
}
web_search_prompt_darts = PromptTemplate.from_template("""
   You are a machine learning expert and will assist me with researching a specific use of a deep learning model in Darts.
                                                                                                    
   I want to run `{algorithm_name}`. What is the Initialization function, parameters and Attributes? 
   Briefly return realted document content.
   Then, extract **all parameters** of the `__init__` method for the `{algorithm_name}` class, along with their default values if available, and return a valid Python dictionary string in the following format:
    ```python
    {{
        "param1": default_value1, (Required)
        "param2": default_value2, (Not Required)
        ...
    }}
   If any default value is an object or function (e.g., MinMaxScaler()), wrap it in quotes to ensure valid Python syntax for ast.literal_eval.
   Here are the official documents you should refer to:
""")

web_search_prompt_tslib = PromptTemplate.from_template("""
You are extracting runnable CLI arguments for Time-Series-Library anomaly detection.

Primary source:
https://github.com/thuml/Time-Series-Library/blob/main/run.py

Target model:
`{algorithm_name}`

Official anomaly detection shell script to inspect:
`{algorithm_name}.sh`

Task:
1. Extract all supported CLI arguments from `run.py` and their default values if specified.
2. Identify which of these arguments are used in the official anomaly detection shell script for `{algorithm_name}`.
3. Use `run.py` only to validate argument names and, when necessary, supply true defaults for missing but execution-critical arguments.
4. Return a single valid Python dictionary only.

What to include:
- Arguments explicitly present in the official shell script.
- A small number of missing arguments only if they are required by `run.py` for a runnable anomaly detection command.
- Values must match the shell script when specified there.

What not to include:
- Any argument not supported by the current `run.py`.
- Any model/class constructor parameters that are not CLI args.
- Any invented values not grounded in the shell script or `run.py`.
- `itr`, `use_amp`, `use_multi_gpu`, unless they are explicitly present in the shell script and supported by `run.py`.

Rules:
- Prefer the shell script over guessed defaults.
- Use `run.py` only as a validator and fallback for required defaults.
- Preserve exact values from the shell script for shared parameters such as `e_layers` and `d_layers`.
- Keep only arguments relevant to anomaly detection.
- Do not include explanatory text.

Output requirements:
- Output exactly one Python dictionary in a fenced Python code block.
- Keys must be CLI argument names without the leading `--`.
- Values must be valid Python literals.

Format:
```python
{{
    "task_name": "anomaly_detection",
    "is_training": 1,
    "model": "{algorithm_name}",
    "data": "...",
    "seq_len": 100
}}
```
""")

web_search_prompt_tsb_ad = PromptTemplate.from_template("""
You are a machine learning expert and will assist me with researching a specific use of a time-series anomaly detection model in TSB-AD.

Official project to refer to:
https://github.com/TheDatumOrg/TSB-AD

Target model:
`{algorithm_name}`

Task:
1. Briefly summarize the relevant official usage for `{algorithm_name}` in TSB-AD.
2. Focus on how `{algorithm_name}` is called through the direct wrapper `TSB_AD.model_wrapper.run_{algorithm_name}`.
3. Extract only the runtime keyword parameters that are actually relevant when calling `run_{algorithm_name}(data, **kwargs)`.
4. Return a valid Python dictionary in a fenced Python code block.

Rules:
- Prefer the official repository and package usage.
- Do not invent class constructor parameters if they are not accepted by `run_{algorithm_name}`.
- If `{algorithm_name}` does not require extra keyword arguments, return an empty dictionary.
- Keep the output grounded in TSB-AD usage rather than generic model papers.

Format:
```python
{{
    "param1": default_value1,
    "param2": default_value2
}}
```
""")

class AgentInfoMiner:
    CACHE_KEY_VERSION = "v2"

    def __init__(self):
        pass

    @classmethod
    def _cache_key(cls, algorithm: str, package_name: str) -> str:
        return f"{cls.CACHE_KEY_VERSION}::{package_name}::{algorithm}"

    def query_metadata(self, algorithm: str, package_name: str) -> dict:
        if package_name != "tslib":
            return {}
        return self._build_tslib_cli_metadata(algorithm)

    def query_docs(self, algorithm, package_name,cache_path = "cache.json"):
        """Searches for relevant documentation with caching, expiration, and thread-safe cache writes."""

        lock_path = cache_path + ".lock"
        lock = FileLock(lock_path)

        # Step 1: Ensure cache file exists
        if not os.path.exists(cache_path):
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump({}, f)

        # Step 2: Use lock to safely read and write to cache
        with lock:
            # Load cache
            with open(cache_path, "r", encoding="utf-8") as f:
                try:
                    cache = json.load(f)
                except json.JSONDecodeError:
                    print("[Cache Error] cache.json is corrupted. Reinitializing...")
                    cache = {}

            # Check cache entry
            cache_key = self._cache_key(algorithm, package_name)
            if cache_key in cache:
                try:
                    cached_time = datetime.fromisoformat(cache[cache_key]["query_datetime"])
                    if datetime.now() - cached_time < timedelta(days=7):
                        print(f"[Cache Hit] Using recent cache for {package_name}:{algorithm}")
                        print(cache[cache_key]["document"])
                        return cache[cache_key]["document"]
                    else:
                        print(f"[Cache Expired] Re-querying {package_name}:{algorithm}")
                except Exception:
                    print(f"[Cache Warning] Datetime parse error for {package_name}:{algorithm}, re-querying.")

        # Step 3: Run actual query outside lock (non-blocking for others)
        client = OpenAI()
        match package_name:
            case "pyod":
                prompt_temp = web_search_prompt_pyod
            case "pygod":
                prompt_temp = web_search_prompt_pygod
            case "tslib":
                prompt_temp = web_search_prompt_tslib
            case "tsb_ad":
                prompt_temp = web_search_prompt_tsb_ad
            case _:
                prompt_temp = web_search_prompt_darts

        prompt = prompt_temp.invoke({"algorithm_name": algorithm}).to_string()
        if package_name == "darts":
            prompt = prompt + "\n\n" + web_dict.get(algorithm, "")

        response = client.responses.create(
            model="gpt-4o",
            tools=[{"type": "web_search_preview"}],
            input=prompt,
            max_output_tokens=2024
        )
        algorithm_doc = response.output_text
        

        # Query using RAG
        #query = ""
        #if package_name == "pyod":
        #    query = f"class pyod.models.{algorithm}.{algorithm}"
        #else:
        #    query = f"class pygod.detector.{algorithm}"
        #doc_list = vectorstore.similarity_search(query, k=3)
        #algorithm_doc = "\n\n".join([doc.page_content for doc in doc_list])

        # if package_name == "tslib":
        #     algorithm_doc = ''

        if not algorithm_doc:
            print("Error in response for " + algorithm)
            print(response)
            return ""
        print(algorithm_doc)

        # Step 4: Re-lock and write updated cache
        with lock:
            with open(cache_path, "r", encoding="utf-8") as f:
                try:
                    cache = json.load(f)
                except json.JSONDecodeError:
                    cache = {}

            cache_key = self._cache_key(algorithm, package_name)
            cache[cache_key] = {
                "query_datetime": datetime.now().isoformat(),
                "document": algorithm_doc
            }

            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)

        print(f"[Cache Updated] Stored new documentation for {package_name}:{algorithm}")
        return algorithm_doc

    @staticmethod
    def _query_tslib_docs(algorithm: str) -> str:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Time-Series-Library"))
        scripts_root = os.path.join(repo_root, "scripts", "anomaly_detection")
        run_py_path = os.path.join(repo_root, "run.py")

        script_path = AgentInfoMiner._find_tslib_anomaly_script(scripts_root, algorithm)

        if script_path is None:
            raise FileNotFoundError(f"Could not find Time-Series-Library anomaly script for {algorithm}")

        with open(script_path, "r", encoding="utf-8") as f:
            script_content = f.read().strip()
        with open(run_py_path, "r", encoding="utf-8") as f:
            run_py_content = f.read()

        parsed_args = AgentInfoMiner._parse_tslib_script_args(script_content)
        supported_args = AgentInfoMiner._parse_run_py_args(run_py_content)
        filtered_args = {k: v for k, v in parsed_args.items() if k in supported_args}
        cpu_overrides = AgentInfoMiner._build_tslib_cpu_overrides(run_py_content)

        return (
            f"Official script path: {script_path}\n\n"
            f"Official script:\n```bash\n{script_content}\n```\n\n"
            "Parsed CLI arguments:\n"
            f"```python\n{json.dumps(filtered_args, ensure_ascii=False, indent=4)}\n```\n"
            "\nExecution modes:\n"
            "- CUDA/GPU mode: the official shell script above may assume GPU execution.\n"
            "- CPU-safe mode: for environments without CUDA, override the official defaults using the CPU-safe runtime settings below.\n\n"
            "CPU-safe runtime overrides for the current environment:\n"
            f"```python\n{json.dumps(cpu_overrides, ensure_ascii=False, indent=4)}\n```\n"
            "\nExecution note for current environment:\n"
            "- The current environment may not support CUDA.\n"
            "- `run.py` defaults to GPU execution unless explicitly overridden.\n"
            "- To run safely on CPU, apply the CPU-safe runtime overrides above.\n"
            "- Environment constraints take priority over the official shell script defaults.\n"
        )

    @staticmethod
    def _find_tslib_anomaly_script(scripts_root: str, algorithm: str) -> str | None:
        root = Path(scripts_root)
        if not root.exists():
            return None

        candidates = sorted(
            path for path in root.glob(f"*/{algorithm}.sh")
            if path.is_file()
        )
        if not candidates:
            return None
        return str(candidates[0])

    @staticmethod
    def _parse_tslib_script_args(script_content: str) -> dict:
        match = re.search(r"python\s+-u\s+run\.py\s+\\\n([\s\S]+)", script_content)
        if not match:
            return {}

        args = {}
        for raw_line in match.group(1).splitlines():
            line = raw_line.strip().rstrip("\\").strip()
            if not line or not line.startswith("--"):
                continue
            parts = line.split(None, 1)
            flag = parts[0][2:]
            if len(parts) == 1:
                args[flag] = True
                continue
            value = parts[1].strip().strip('"').strip("'")
            try:
                args[flag] = int(value)
                continue
            except ValueError:
                pass
            try:
                args[flag] = float(value)
                continue
            except ValueError:
                pass
            args[flag] = value
        return args

    @staticmethod
    def _parse_run_py_args(run_py_content: str) -> set[str]:
        return set(re.findall(r"add_argument\('--([A-Za-z0-9_]+)'", run_py_content))

    @staticmethod
    def _build_tslib_cpu_overrides(run_py_content: str) -> dict:
        supported_args = AgentInfoMiner._parse_run_py_args(run_py_content)
        overrides = {}
        if "gpu_type" in supported_args:
            overrides["gpu_type"] = "cpu"
        if "no_use_gpu" in supported_args:
            overrides["no_use_gpu"] = True
        return overrides

if __name__ == "__main__":
    agent = AgentInfoMiner()
    # Example usage
    algorithm = "RegressionModel"
    vectorstore = None  # Replace with actual vectorstore object
    package_name = "darts"
    doc = agent.query_docs(algorithm, vectorstore, package_name)
