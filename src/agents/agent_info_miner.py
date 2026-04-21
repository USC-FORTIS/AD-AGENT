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
from utils.script_paths import DOC_CACHE_PATH, ensure_runtime_dirs
from config.config import Config
os.environ.setdefault('OPENAI_API_KEY', Config.OPENAI_API_KEY)

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

web_search_prompt_tsb_ad = PromptTemplate.from_template("""
You are a machine learning expert and will assist me with researching a specific use of a time-series anomaly detection model in TSB-AD.

Official project to refer to:
https://github.com/TheDatumOrg/TSB-AD

I want to run `{algorithm_name}`. Summarize the relevant official usage for that model in TSB-AD.
Focus on how `{algorithm_name}` is called through the direct wrapper `TSB_AD.model_wrapper.run_{algorithm_name}`.
Then extract only the runtime keyword parameters that are relevant to calling that wrapper, and return a valid Python dictionary string in a fenced Python code block.
""")
web_search_prompt_tsbad = web_search_prompt_tsb_ad

class AgentInfoMiner:
    CACHE_KEY_VERSION = "v2"

    def __init__(self):
        pass

    @classmethod
    def _cache_key(cls, algorithm: str, package_name: str) -> str:
        return f"{cls.CACHE_KEY_VERSION}::{package_name}::{algorithm}"

    @staticmethod
    def _doc_summary(document: str) -> str:
        first_line = next((line.strip() for line in document.splitlines() if line.strip()), "")
        if len(first_line) > 100:
            first_line = first_line[:97] + "..."
        return f"{len(document)} chars" + (f"; first line: {first_line}" if first_line else "")

    def query_docs(self, algorithm, package_name, cache_path=None):
        """Searches for relevant documentation with caching, expiration, and thread-safe cache writes."""
        ensure_runtime_dirs()
        cache_path = cache_path or str(DOC_CACHE_PATH)

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
                    print(f"[Cache Error] {cache_path} is corrupted. Reinitializing...")
                    cache = {}

            # Check cache entry
            cache_key = self._cache_key(algorithm, package_name)
            legacy_key = algorithm
            if cache_key in cache or legacy_key in cache:
                entry = cache.get(cache_key) or cache.get(legacy_key)
                try:
                    cached_time = datetime.fromisoformat(entry["query_datetime"])
                    if datetime.now() - cached_time < timedelta(days=7):
                        print(
                            f"[info_miner][{algorithm}] Cache hit "
                            f"({self._doc_summary(entry['document'])})"
                        )
                        return entry["document"]
                    else:
                        print(f"[info_miner][{algorithm}] Cache expired; re-querying")
                except Exception:
                    print(f"[info_miner][{algorithm}] Cache datetime parse error; re-querying")

        # Step 3: Run actual query outside lock (non-blocking for others)
        client = OpenAI()
        match package_name:
            case "pyod":
                prompt_temp = web_search_prompt_pyod
            case "pygod":
                prompt_temp = web_search_prompt_pygod
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

        if not algorithm_doc:
            print("Error in response for " + algorithm)
            print(response)
            return ""
        print(f"[info_miner][{algorithm}] Documentation fetched ({self._doc_summary(algorithm_doc)})")

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

        print(f"[info_miner][{algorithm}] Cache updated")
        return algorithm_doc

if __name__ == "__main__":
    agent = AgentInfoMiner()
    # Example usage
    algorithm = "RegressionModel"
    package_name = "darts"
    doc = agent.query_docs(algorithm, package_name)
    print(doc)
