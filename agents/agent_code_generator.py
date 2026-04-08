from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
import ast
import os
import json
import re
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from entity.code_quality import CodeQuality
from utils.tsb_ad_registry import get_installed_tsb_ad_algorithms
import subprocess
from datetime import datetime, timedelta
from config.config import Config
os.environ['OPENAI_API_KEY'] = Config.OPENAI_API_KEY

# Initialize OpenAI LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0)

template_pyod_labeled = PromptTemplate.from_template("""
You are an expert Python developer with deep experience in anomaly detection libraries. Your task is to:

1. Use the provided official documentation content for `{algorithm}` to understand how to use the specified algorithm class, including initialization, training, and prediction methods.
2. Write only executable Python code for anomaly detection using PyOD and do not include any explanations or descriptions.
3. Base your code strictly on the following official documentation excerpt:

--- BEGIN DOCUMENTATION ---
{algorithm_doc}
--- END DOCUMENTATION ---

4. The code should:
   (1) import sys, os and include command `sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))` in the head
   (2) Load data directly from `{data_path_train}` and `{data_path_test}`.
   (3) Use the dataset metadata below to choose the correct loading logic:
       {dataset_metadata}
   (4) Create variables `X_train`, `y_train`, `X_test`, and `y_test` directly from the files. Use standard libraries such as `pandas`, `numpy`, `scipy.io`, or `json` as appropriate for the file type and metadata.
       For CSV files, infer label columns from metadata/head preview and common names like `label`, `Label`, `target`, or `anomaly`; otherwise treat the last clearly target-like column as label.
       For `.mat` files, load with `scipy.io.loadmat` and use metadata/head preview to select feature and label arrays.
       Ensure `X_train` and `X_test` are 2D numeric arrays and `y_train`/`y_test` are 1D numeric arrays.
   (5) Initialize the specified algorithm `{algorithm}` using variable `model`, strictly following the provided documentation and train the model with `X_train`
   (6) Determine whether the following parameters `{parameters}` apply to this initialization function and, if so, add their values ​to the function.
   (7) Use `.decision_scores_` on `X_train` for training outlier scores
       Use `.decision_function(X_test)` for test outlier scores
       Calculate AUROC (Area Under the Receiver Operating Characteristic Curve) and AUPRC (Area Under the Precision-Recall Curve) based on given data
   (8) Using variables to record the AUROC & AUPRC and print them out in following format:
       AUROC:\s*(\d+.\d+)
       AUPRC:\s*(\d+.\d+)
   (9) Using variables to record prediction failed data and print these points out with true label in following format:
       `Failed prediction at point [xx,xx,xx...] with true label xx` Use `.tolist()` to convert point to be an array.
                     

IMPORTANT: 
- Strictly follow steps (2)-(8) to load the data directly from `{data_path_train}` & {data_path_test}.
- Do not use `DataLoader` in the generated script; the selector has already collected dataset metadata for you.
- Do NOT input optional or incorrect parameters.
""")

template_pyod_unlabeled = PromptTemplate.from_template("""
You are an expert Python developer with deep experience in anomaly detection libraries. Your task is to:

1. Use the provided official documentation content for `{algorithm}` to understand how to use the specified algorithm class, including initialization, training, and prediction methods.
2. Write only executable Python code for anomaly detection using PyOD and do not include any explanations or descriptions.
3. Base your code strictly on the following official documentation excerpt:

--- BEGIN DOCUMENTATION ---
{algorithm_doc}
--- END DOCUMENTATION ---

4. The code should:
   (1) Import sys, os and include command `sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))` in the head.
   (2) Load data directly from `{data_path_train}`. Do not import or use `DataLoader`, and do not generate or read `train_data_loader.py`.
   (3) Use the dataset metadata below to choose the correct loading logic:
       {dataset_metadata}
   (4) Extract the feature matrix from the loaded data as `X_train`. Use standard libraries such as `pandas`, `numpy`, `scipy.io`, or `json` as appropriate for the file type and metadata.
       For CSV files, drop obvious label/target columns and non-numeric identifier columns from `X_train`.
       For `.mat` files, load with `scipy.io.loadmat` and use metadata/head preview to select the feature array.
   (5) Initialize the specified algorithm `{algorithm}` using variable `model`, strictly following the provided documentation and train the model with `X_train`
   (6) Determine whether the following parameters `{parameters}` apply to this initialization function and, if so, add their values ​to the function.
   (7) Use `.decision_scores_` on `X_train` for training outlier scores
       Use `.decision_function(X_train)` for test outlier scores
   (8) Print AUROC & AUPRC Using default value `-1`:
       `AUROC: -1`
       `AUPRC: -1`
   (9) Using variables to record outlier data and print these points out with true label in following format:
       `Detected outlier at point [xx,xx,xx...]` Use `.tolist()` to convert point to be an array.
                     

IMPORTANT: 
- Strictly follow steps (2)-(8) to load the data directly from `{data_path_train}`.
- Do not use `DataLoader` in the generated script; the selector has already collected dataset metadata for you.
- Do NOT input optional or incorrect parameters.
""")


template_pygod_labeled = PromptTemplate.from_template("""
You are an expert Python developer with deep experience in anomaly detection libraries. Your task is to:

1. Use the provided official documentation content for `{algorithm}` to understand how to use the specified algorithm class, including initialization, training, and prediction methods.
2. Write only executable Python code for anomaly detection using PyGOD and do not include any explanations or descriptions.
3. Base your code strictly on the following official documentation excerpt:

--- BEGIN DOCUMENTATION ---
{algorithm_doc}
--- END DOCUMENTATION ---

4. The code should:
   (1) Import sys, os, torch, and include the command `sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))`&`from pygod.detector import {algorithm}`
   (2) Load training and test data using `torch.load` with parameter `weights_only=False` from the file paths `{data_path_train}` and `{data_path_test}` respectively.
   (3) Convert labels in the loaded data by executing:
       `train_data.y = (train_data.y != 0).long()`
       `test_data.y = (test_data.y != 0).long()`
   (4) Initialize the specified algorithm `{algorithm}` with the provided parameters `{parameters}`(if parameters applicable) using variable `model`, strictly following the documentation excerpt.
   (5) Train the model using `model.fit(train_data)`.
   (6) Predict on the test data using `pred, score = model.predict(test_data, return_score=True)`.
   (7) Extract the true labels and corresponding scores using the test mask:
       `true_labels = test_data.y[test_data.test_mask]`
       `score = score[test_data.test_mask]`
   (8) Calculate AUROC using `roc_auc_score` and AUPRC using `average_precision_score` from sklearn.metrics.
   (9) Print the AUROC and AUPRC in the following format:
       AUROC:\s*(\d+.\d+)
       AUPRC:\s*(\d+.\d+)

IMPORTANT:
- Strictly follow steps (2)-(9) to load the data from `{data_path_train}` and `{data_path_test}`.
- Do NOT include any additional or incorrect parameters.
""")

template_pygod_unlabeled = PromptTemplate.from_template("""
You are an expert Python developer with deep experience in anomaly detection libraries. Your task is to:

1. Use the provided official documentation content for `{algorithm}` to understand how to use the specified algorithm class, including initialization, training, and prediction methods.
2. Write only executable Python code for anomaly detection using PyGOD and do not include any explanations or descriptions.
3. Base your code strictly on the following official documentation excerpt:

--- BEGIN DOCUMENTATION ---
{algorithm_doc}
--- END DOCUMENTATION ---

4. The code should:
   (1) Import sys, os, torch, and include the command `sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))`&`from pygod.detector import {algorithm}`
   (2) Load training using `torch.load` with parameter `weights_only=False` from the file paths `{data_path_train}`
   (4) Initialize the specified algorithm `{algorithm}` with the provided parameters `{parameters}`(if parameters applicable) using variable `model`, strictly following the documentation excerpt.
   (5) Train the model using `model.fit(train_data)`.
   (6) Predict on the test data using `pred, score = model.predict(train_data, return_score=True)`.
   (7) Compute the total number of predicted anomalies:  
       `num_anomalies = int((pred != 0).sum())`                                                     
   (8) Print AUROC & AUPRC Using default value `-1`:
       `AUROC: -1`
       `AUPRC: -1`
   (9) Using variables to record prediction outlier data and print number of outlier points in following format:
       `Detected outlier number: xx`

IMPORTANT:
- Strictly follow steps (2)-(9) to load the data from `{data_path_train}` and `{data_path_test}`.
- Do NOT include any additional or incorrect parameters.
""")

template_tsb_ad_labeled = PromptTemplate.from_template("""
You are an expert Python developer with deep experience in time-series anomaly detection using TSB-AD.

1. Use the provided official documentation content for `{algorithm}` to understand how to call the model through TSB-AD.
2. Write only executable Python code and do not include any explanations or descriptions.
3. Base your code strictly on the following official documentation excerpt:

--- BEGIN DOCUMENTATION ---
{algorithm_doc}
--- END DOCUMENTATION ---

4. The code should:
   (1) Import `inspect`, `os`, `sys`, `numpy as np`, `pandas as pd`, and sklearn metrics.
   (2) Include `sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))` near the top.
   (3) Import the direct wrapper for the target algorithm using:
       `from TSB_AD.model_wrapper import run_{algorithm}`
       and call that wrapper directly.
       The imported wrapper name, the assigned `model_runner`, and the called function must be exactly the same symbol: `run_{algorithm}`.
       Do not reference `run_{algorithm}` unless it is explicitly imported earlier in the script.
   (4) Load training data from `{data_path_train}` and test data from `{data_path_test}`.
       - Support CSV files with value columns plus one label column such as `Label`, `label`, or `anomaly`.
       - Support `.npy` datasets and legacy dataset roots by resolving companion files like `_train.npy`, `_test.npy`, and `_test_label.npy`.
       - Drop timestamp-like columns from CSV feature matrices.
       - Convert feature matrices to numeric numpy arrays and clean missing values before calling the wrapper:
         `X_train = np.nan_to_num(np.asarray(X_train, dtype=float))`
         `X_test = np.nan_to_num(np.asarray(X_test, dtype=float))`
       - Use this dataset metadata to choose the most accurate loading logic:
         {dataset_metadata}
   (5) Store the algorithm name in variable `ALGORITHM_NAME = "{algorithm}"`.
   (6) Store the callable in variable `model_runner = run_{algorithm}`.
   (7) Apply user parameters from `{parameters}` only if they are valid keyword arguments supported by the direct wrapper signature.
       You must inspect `inspect.signature(model_runner).parameters` and filter `run_kwargs` against that exact signature before the call.
       If a parameter is not in the direct wrapper signature, do not pass it.
   (8) Run `train_scores = model_runner(X_train, **run_kwargs)` and `test_scores = model_runner(X_test, **run_kwargs)`.
   (9) Convert wrapper outputs to sample-level 1D numpy float score arrays and validate lengths.
       Do not blindly flatten 2D outputs. Use a helper such as:
       `def to_sample_scores(raw_scores, n_samples):`
       `    arr = np.asarray(raw_scores, dtype=float)`
       `    if arr.ndim == 2 and arr.shape[0] == n_samples:`
       `        return np.linalg.norm(arr, axis=1)`
       `    arr = arr.reshape(-1)`
       `    if len(arr) != n_samples:`
       `        raise ValueError(f"Score length {{len(arr)}} does not match sample count {{n_samples}}")`
       `    return arr`
       Apply it as `train_scores = to_sample_scores(train_scores, len(X_train))` and `test_scores = to_sample_scores(test_scores, len(X_test))`.
   (10) Calculate AUROC and AUPRC using `roc_auc_score` and `average_precision_score`.
   (11) Print metrics exactly in this format:
       AUROC: 0.1234
       AUPRC: 0.5678
   (12) Threshold test scores using the 95th percentile of training scores and print mismatches exactly as:
       `Failed prediction at point [xx, xx, ...] with true label z`

IMPORTANT:
- Produce only executable Python code.
- Do not use subprocess.
- Do not use `run_Unsupervise_AD`; call the direct wrapper for `{algorithm}` instead.
- Do not invent unsupported arguments or unsupported TSB-AD APIs.
- Keep the script robust to both CSV and `.npy`-style time-series inputs.
""")

template_tsb_ad_unlabeled = PromptTemplate.from_template("""
You are an expert Python developer with deep experience in time-series anomaly detection using TSB-AD.

1. Use the provided official documentation content for `{algorithm}` to understand how to call the model through TSB-AD.
2. Write only executable Python code and do not include any explanations or descriptions.
3. Base your code strictly on the following official documentation excerpt:

--- BEGIN DOCUMENTATION ---
{algorithm_doc}
--- END DOCUMENTATION ---

4. The code should:
   (1) Import `inspect`, `os`, `sys`, `numpy as np`, and `pandas as pd`.
   (2) Include `sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))` near the top.
   (3) Import the direct wrapper for the target algorithm using:
       `from TSB_AD.model_wrapper import run_{algorithm}`
       and call that wrapper directly.
       The imported wrapper name, the assigned `model_runner`, and the called function must be exactly the same symbol: `run_{algorithm}`.
       Do not reference `run_{algorithm}` unless it is explicitly imported earlier in the script.
   (4) Load data from `{data_path_train}`.
       - Support CSV files with value columns and optional label columns.
       - Support `.npy` datasets and legacy dataset roots by resolving companion `_train.npy` files.
       - Drop timestamp-like columns from CSV feature matrices.
       - Convert `X_train` to a numeric numpy array and clean missing values before calling the wrapper:
         `X_train = np.nan_to_num(np.asarray(X_train, dtype=float))`
       - Use this dataset metadata to choose the most accurate loading logic:
         {dataset_metadata}
   (5) Store the algorithm name in variable `ALGORITHM_NAME = "{algorithm}"`.
   (6) Store the callable in variable `model_runner = run_{algorithm}`.
   (7) Apply user parameters from `{parameters}` only if they are valid keyword arguments supported by the direct wrapper signature.
       You must inspect `inspect.signature(model_runner).parameters` and filter `run_kwargs` against that exact signature before the call.
       If a parameter is not in the direct wrapper signature, do not pass it.
   (8) Run `scores = model_runner(X_train, **run_kwargs)`.
   (9) Convert outputs to a sample-level 1D numpy float score array and validate its length.
       If the wrapper returns a tuple/list, choose the element that is a numeric score vector whose length matches `len(X_train)`.
       Do not blindly flatten 2D outputs. If `scores` is a 2D array with `scores.shape[0] == len(X_train)`, convert it with `np.linalg.norm(scores, axis=1)`.
       If the final score length does not equal `len(X_train)`, raise a `ValueError` instead of continuing.
   (10) Print metrics exactly:
       AUROC: -1
       AUPRC: -1
   (11) Detect outliers with a robust threshold such as the 95th percentile of `scores`.
       Print the detected outlier indices as one list exactly as:
       `Detected outlier at point [0, 5, 12]`
       Use `outliers.tolist()` for this unlabeled TSB-AD output format.

IMPORTANT:
- Produce only executable Python code.
- Do not use subprocess.
- Do not use `run_Unsupervise_AD`; call the direct wrapper for `{algorithm}` instead.
- Do not invent unsupported arguments or unsupported TSB-AD APIs.
""")

template_fix = PromptTemplate.from_template("""
You are an expert Python developer fixing an anomaly-detection Python script.

Here is the original code that raised an error:
--- Original Code ---
{code}

Here is the execution error:
--- Error Message ---
{error_message}

Official documentation for `{algorithm}`:
--- BEGIN DOCUMENTATION ---
{algorithm_doc}
--- END DOCUMENTATION ---

Task:
1. Fix the code using the error message and the documentation.
2. Output executable Python only.
3. Make the smallest possible change that fixes the reported error.

Strict rules:
1. Preserve the overall script structure whenever possible.
2. Do not rewrite the script from scratch unless the original code is fundamentally broken.
3. Do not change `cwd` unless the error explicitly proves that the current working directory is wrong.
4. Do not invent unrelated path changes.
5. If the error is about CUDA, GPU availability, or Torch GPU support, fix only the GPU-related argument or device setting.
6. If the error is about a missing file or directory, only change the specific path that is proven to be wrong.
7. Keep all valid existing arguments unless the error indicates one of them is the cause.
8. Do not introduce markdown fences, explanations, or comments.
9. If the script uses a TSB-AD direct wrapper such as `run_IForest` or `run_RobustPCA`, preserve that direct wrapper import and call style.
10. If the error is `unexpected keyword argument`, remove or filter unsupported keyword arguments using the direct wrapper signature instead of inventing replacement parameter names.
11. Keep the script self-contained for sandbox execution. Do not import or use `DataLoader`, and do not depend on generated loader files such as `train_data_loader.py` or `test_data_loader.py`; load dataset files directly instead.
12. For unlabeled TSB-AD anomaly outputs, print detected outlier indices as one list using the format `Detected outlier at point [0, 5, 12]`.
13. For TSB-AD scripts, clean missing feature values with `np.nan_to_num(np.asarray(X, dtype=float))` before calling the model wrapper.
14. For TSB-AD scripts, normalize wrapper outputs to sample-level scores before thresholding: do not blindly flatten 2D arrays; if a 2D score matrix has one row per sample, use `np.linalg.norm(scores, axis=1)`, then require `len(scores) == len(X_train)` or the matching test set length.

Return only executable Python code.
""")

template_darts_labeled = PromptTemplate.from_template("""
You are an expert Python developer with deep knowledge of the **Darts** library for forecasting-based time-series anomaly detection. Your task is to:

1. Carefully study the official documentation excerpt for **`{algorithm}`** provided below so you fully understand how to initialise, fit, and use this class.

--- BEGIN DOCUMENTATION ---
{algorithm_doc}
--- END DOCUMENTATION ---

2. Output **only** executable Python code (no extra text) that performs forecasting-based anomaly detection on two CSV files exactly as specified in the reference implementation.

• Implement the helper function `load_series(path: str) -> tuple[TimeSeries, np.ndarray]` that:  
  – reads the CSV,  
  – converts all `value_…` columns into a multivariate `TimeSeries`,  
  – returns that series plus the `anomaly` column as an `int` numpy array.

• Load the datasets:  
  `series_train, y_train = load_series({data_path_train})`  
  `series_test,  y_test  = load_series({data_path_test})`

• Cast both series to `np.float32` and set the default Torch dtype to `torch.float32`.

• Instantiate the forecasting model:  
  `model = {algorithm}(**{{}})` 
  Do not input any unnecessary parameters
  Add **only** those keys from `{parameters}` that match the class signature.

• Fit the forecasting model with `model.fit(series_train)`.

• Wrap the model in a `ForecastingAnomalyModel` using a `KMeansScorer`:  
  `from darts.ad.anomaly_model import ForecastingAnomalyModel`  
  `from darts.ad.scorers import KMeansScorer`  
  `fa_model = ForecastingAnomalyModel(model=model, scorer=KMeansScorer())`

• Fit the anomaly model with `fa_model.fit(series_train, allow_model_training=False)` and score the test set with `scores = fa_model.score(series_test)`.

• Use `QuantileDetector(high_quantile=0.995)` fitted on `scores` to obtain binary predictions:  
  `from darts.ad.detectors import QuantileDetector`  
  `detector = QuantileDetector(high_quantile=0.995)`  
  `detector.fit(scores)`  
  `y_pred = (detector.detect(scores).values() > 0).any(axis=1).astype(int)`

• Align true labels:  
  `offset = len(y_test) - len(y_pred)`  
  `y_test_aligned = y_test[offset:]`

• Evaluate and **print** metrics exactly as:  
  `AUROC: 0.1234`  
  `AUPRC: 0.5678`  
  (values printed with four decimal places).

• For every mismatch between prediction and true label, print:  
  `Failed prediction at point {{series_test.time_index[offset + i]}} with true label z`.

3. At the very top of the script, add:  
`import sys, os`

IMPORTANT RULES  
• Produce a single runnable Python script following the steps above—no explanations, comments, or additional outputs.  
• Do **not** pass any optional or invalid parameters to `{algorithm}`.  
• Ensure the script works with the CSV paths `{data_path_train}` and `{data_path_test}`.
""")

template_darts_unlabeled = PromptTemplate.from_template("""
You are an expert Python developer with deep knowledge of the **Darts** library for forecasting-based time-series anomaly detection. Your task is to:

1. Carefully study the official documentation excerpt for **`{algorithm}`** provided below so you fully understand how to initialise, fit, and use this class.

--- BEGIN DOCUMENTATION ---
{algorithm_doc}
--- END DOCUMENTATION ---

2. Output **only** executable Python code (no extra text) that performs forecasting-based anomaly detection on two CSV files exactly as specified in the reference implementation.

• Implement the helper function `load_series(path: str) -> TimeSeries` that:  
  reads the CSV,  
  converts all `value_…` columns into a multivariate `TimeSeries`,  
  returns that series.

• Load the datasets:  
  `series_train = load_series({data_path_train})`  
  `series_test = load_series({data_path_test})`

• Cast both series to `np.float32` and set the default Torch dtype to `torch.float32`.

• Instantiate the forecasting model:  
  `model = {algorithm}(**{{}})`  
  Do not input any unnecessary parameters.  
  Add **only** those keys from `{parameters}` that match the class signature.

• Fit the forecasting model with `model.fit(series_train)`.

• Wrap the model in a `ForecastingAnomalyModel` using a `KMeansScorer`:  
  `from darts.ad.anomaly_model import ForecastingAnomalyModel`  
  `from darts.ad.scorers import KMeansScorer`  
  `fa_model = ForecastingAnomalyModel(model=model, scorer=KMeansScorer())`

• Fit the anomaly model with `fa_model.fit(series_train, allow_model_training=False)` and score the test set with:  
  `scores = fa_model.score(series_test)`

• Important: After scoring, do **not** reindex `series_test`.  
  Instead, use `series_test.values()` directly, and slice it from the end to match the length of `scores` and `y_pred`:  
  `series_array = series_test.values()`  
  `series_array = series_array[-len(scores):]`

• Use `QuantileDetector(high_quantile=0.995)` fitted on `scores` to obtain binary predictions:  
  `from darts.ad.detectors import QuantileDetector`  
  `detector = QuantileDetector(high_quantile=0.995)`  
  `detector.fit(scores)`  
  `y_pred = (detector.detect(scores).values() > 0).any(axis=1).astype(int)`

• **print** metrics exactly as:  
  `AUROC: -1`  
  `AUPRC: -1`

• Use `scores.time_index[y_pred == 1]` to obtain the time points of outliers.

• When printing outlier points:  
  - Select outlier points from the sliced `series_array` using `outliers = series_array[y_pred == 1]`.  
  - Iterate through them and print using:  
    `Detected outlier at point [xx, xx, xx...]`  
    Use `.tolist()` to convert each outlier to a Python list.

3. At the very top of the script, add:  
`import sys, os, pandas as pd`

IMPORTANT RULES  
• Produce a single runnable Python script following the steps above—no explanations, comments, or additional outputs.  
• Do **not** pass any optional or invalid parameters to `{algorithm}`.  
• Ensure the script works with the CSV paths `{data_path_train}` and `{data_path_test}`.
""")


# ---------- CLASS ----------
class AgentCodeGenerator:
    """Now responsible for code generation **and** modification."""
    def __init__(self):
        pass

    # -------- generation --------
    def generate_code(
        self,
        algorithm,
        data_path_train,
        data_path_test,
        algorithm_doc,
        input_parameters,
        package_name,
        dataset_metadata=None
    ) -> str:
        tpl = None
       
        if package_name == "pyod":
            tpl = template_pyod_labeled if data_path_test else template_pyod_unlabeled
        elif package_name == "pygod":
            tpl = template_pygod_labeled if data_path_test else template_pygod_unlabeled
        elif package_name == "tsb_ad":
            tpl = template_tsb_ad_labeled if data_path_test else template_tsb_ad_unlabeled
        else:
            tpl = template_darts_labeled if data_path_test else template_darts_unlabeled
        raw = llm.invoke(
            tpl.invoke({
                "algorithm": algorithm,
                "data_path_train": data_path_train,
                "data_path_test": data_path_test,
                "algorithm_doc": algorithm_doc,
                "parameters": str(input_parameters),
                "dataset_metadata": json.dumps(dataset_metadata or {}, default=str, indent=2)
            })
        ).content
        cleaned = self._clean(raw)
        print(f"Generated code: {cleaned}\n")
        return cleaned

    # -------- revision (moved from old Reviewer) --------
    def revise_code(self, code_quality: CodeQuality, algorithm_doc: str) -> str:
        print("Error detected during execution. Attempting to fix the code...")
        print("Error message:", code_quality.error_message)
        fixed = llm.invoke(
            template_fix.invoke({
                "code": code_quality.code,
                "error_message": code_quality.error_message,
                "algorithm": code_quality.algorithm,
                "algorithm_doc": algorithm_doc
            })
        ).content
        # increase review counter here
        code_quality.review_count += 1
        cleaned = self._clean(fixed)
        return cleaned


    # -------- util --------
    @staticmethod
    def _clean(code: str) -> str:
        code = re.sub(r"```(python)?", "", code)
        return re.sub(r"```", "", code).strip()

    @staticmethod
    def _extract_init_params_dict(response_text: str) -> dict:
        """
        Extract the dictionary in the first code block from the string, returning a Python dictionary object.
        """
        # match dictionary in code block
        match = re.search(r"```python\s*({.*?})\s*```", response_text, re.DOTALL)
        if not match:
            return {}
            # raise ValueError("No dictionary found in code block.")
        
        dict_str = match.group(1)
        try:
            return ast.literal_eval(dict_str)
        except Exception as e:
            return {}
            # raise ValueError(f"Failed to parse dictionary: {e}")
        # return {}

if __name__ == "__main__":
  agentCodeGenerator = AgentCodeGenerator()
  from agents.agent_selector import AgentSelector
  from agents.agent_info_miner import AgentInfoMiner
  user_input = {
      "algorithm": ["IForest"],
      "dataset_train": "demo_train.npy",
      "dataset_test": "demo_test.npy",
      "parameters": {}
  }
  agentSelector = AgentSelector(user_input=user_input)# if want to unit test, please import AgentSelector
  agentInfoMiner = AgentInfoMiner()
  algorithm_doc = agentInfoMiner.query_docs(algorithm=agentSelector.tools[0], vectorstore=agentSelector.vectorstore, package_name=agentSelector.package_name)
  # algorithm_doc = ""

  code = agentCodeGenerator.generate_code(
      algorithm=user_input["algorithm"][0],
      data_path_train=user_input["dataset_train"],
      data_path_test=user_input["dataset_test"],
      algorithm_doc=algorithm_doc,
      input_parameters=user_input["parameters"],
      package_name=agentSelector.package_name
  )

  print(code)
