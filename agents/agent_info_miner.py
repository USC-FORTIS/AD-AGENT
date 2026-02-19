from langchain_core.prompts import PromptTemplate
from datetime import datetime, timedelta
import json
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
 You are a machine learning expert and will assist me with researching a specific use of a deep learning model in `Time-Series-Library`. Here is the official document you should refer to: https://github.com/thuml/Time-Series-Library/blob/main/scripts/anomaly_detection/MSL/{algorithm_name}.sh .You only need to read this page and avoid search other related pages.


                                                       
   I want to run `{algorithm_name}`. What is the Initialization function, parameters and Attributes? 
   This is a github sh code. You should read the code and get the parameters and attributes. The script code looks like this:
   ```
export CUDA_VISIBLE_DEVICES=0

python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/MSL \
...
   ```
   There shoulbe be less than 20 parameters. Do not make up any parameters like `itr`, just follow the code in the web page.
   The encoder layer and decoder layer must be equal, eg. `--e_layers 2 --d_layers 2`.
   default values if available, and return a valid Python dictionary string in the following format:
    ```python
    {{
        "param1": default_value1,
        "param2": default_value2,
        ...
    }}
   If any default value is an object or function (e.g., MinMaxScaler()), wrap it in quotes to ensure valid Python syntax for ast.literal_eval.
""")

class AgentInfoMiner:
    def __init__(self):
        pass

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
            if algorithm in cache:
                try:
                    cached_time = datetime.fromisoformat(cache[algorithm]["query_datetime"])
                    if datetime.now() - cached_time < timedelta(days=7):
                        print(f"[Cache Hit] Using recent cache for {algorithm}")
                        print(cache[algorithm]["document"])
                        return cache[algorithm]["document"]
                    else:
                        print(f"[Cache Expired] Re-querying {algorithm}")
                except Exception:
                    print(f"[Cache Warning] Datetime parse error for {algorithm}, re-querying.")

        # Step 3: Run actual query outside lock (non-blocking for others)
        client = OpenAI()
        match package_name:
            case "pyod":
                prompt_temp = web_search_prompt_pyod
            case "pygod":
                prompt_temp = web_search_prompt_pygod
            case "tslib":
                prompt_temp = web_search_prompt_tslib
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

            cache[algorithm] = {
                "query_datetime": datetime.now().isoformat(),
                "document": algorithm_doc
            }

            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)

        print(f"[Cache Updated] Stored new documentation for {algorithm}")
        return algorithm_doc

if __name__ == "__main__":
    agent = AgentInfoMiner()
    # Example usage
    algorithm = "RegressionModel"
    vectorstore = None  # Replace with actual vectorstore object
    package_name = "darts"
    doc = agent.query_docs(algorithm, vectorstore, package_name)