from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
import ast
import json
import os
import re
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from entity.code_quality import CodeQuality
from utils.tsb_ad_registry import Unsupervise_AD_Pool
import subprocess
from datetime import datetime, timedelta
from config.config import Config
os.environ.setdefault('OPENAI_API_KEY', Config.OPENAI_API_KEY)

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
   (6) Using variables to record the AUROC & AUPRC and print them out in following format:
       AUROC:\s*(\d+.\d+)
       AUPRC:\s*(\d+.\d+)
   (7) Using variables to record prediction failed data and print these points out with true label in following format:
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
   (6) Print AUROC & AUPRC Using default value `-1`:
       `AUROC: -1`
       `AUPRC: -1`
   (7) Using variables to record outlier data and print these points out in following format:
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
   (2) Load training and test data from the file paths `{data_path_train}` and `{data_path_test}` respectively.
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

Dataset metadata (use this to understand the graph structure):
{dataset_metadata}

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

Dataset metadata (use this to understand the graph structure):
{dataset_metadata}

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
       - Use this dataset metadata to choose the most accurate loading logic:
         {dataset_metadata}
   (5) Store the algorithm name in variable `ALGORITHM_NAME = "{algorithm}"`.
   (6) Store the callable in variable `model_runner = run_{algorithm}`.
   (7) Apply user parameters from `{parameters}` only if they are valid keyword arguments supported by the direct wrapper signature.
       You must inspect `inspect.signature(model_runner).parameters` and filter `run_kwargs` against that exact signature before the call.
       For semisupervised TSB-AD wrappers, `run_kwargs` should contain only filtered user parameters; do not invent any extra default kwargs.
       If a parameter is not in the direct wrapper signature, do not pass it.
       After filtering `run_kwargs`, do not append unsupported defaults with `run_kwargs.update(...)`.
       If you need to add a fallback default, re-filter the final kwargs against `inspect.signature(model_runner).parameters` before the call.
       Never set `periodicity` to `0`.
       If you explicitly set `periodicity`, it must be an integer `>= 1`.
   (8) The semisupervised wrapper takes both datasets in a single call:
       `scores = model_runner(X_train, X_test, **run_kwargs)`
       This returns a 1D anomaly score array for X_test only (length == len(X_test)).
       Do NOT call the wrapper twice (once for train, once for test).
       Before the wrapper call, convert both arrays to finite `np.float64` values:
       `X_train = np.nan_to_num(np.asarray(X_train, dtype=np.float64))`
       `X_test = np.nan_to_num(np.asarray(X_test, dtype=np.float64))`
       If the direct wrapper signature uses two positional dataset arguments such as `(data_train, data)`, pass both datasets positionally and do not place `data` or `data_train` inside `run_kwargs`.
       Some wrappers (e.g. SAND) only support univariate data internally. Only reduce multivariate inputs to 1D when the documentation or a concrete runtime error proves the wrapper is univariate-only.
       If you reduce to 1D for a univariate-only wrapper, do it like this:
       `X_train_in = X_train[:, 0] if X_train.ndim == 2 else X_train`
       `X_test_in = X_test[:, 0] if X_test.ndim == 2 else X_test`
       and call `scores = model_runner(X_train_in, X_test_in, **run_kwargs)`.
       Otherwise keep `X_train` and `X_test` as 2D arrays and call `scores = model_runner(X_train, X_test, **run_kwargs)`.
   (9) Convert the wrapper output to a sample-level 1D numpy float score array and validate its length.
       Do not blindly flatten 2D outputs. Use a helper such as:
       `def to_sample_scores(raw_scores, n_samples):`
       `    arr = np.asarray(raw_scores, dtype=float)`
       `    if arr.ndim == 2 and arr.shape[0] == n_samples:`
       `        return np.linalg.norm(arr, axis=1)`
       `    arr = arr.reshape(-1)`
       `    if len(arr) != n_samples:`
       `        raise ValueError(f"Score length {{len(arr)}} does not match sample count {{n_samples}}")`
       `    return arr`
       Apply it as `scores = to_sample_scores(scores, len(X_test))`.
   (10) Calculate AUROC and AUPRC using `roc_auc_score` and `average_precision_score` with `scores` vs `y_test`.
   (11) Print metrics exactly in this format:
       AUROC: 0.1234
       AUPRC: 0.5678
   (12) Threshold scores using the 95th percentile and print mismatches exactly as:
       `Failed prediction at point [xx, xx, ...] with true label z`

IMPORTANT:
- Produce only executable Python code.
- Do not use subprocess.
- Do not use `run_Semisupervise_AD`; call the direct wrapper for `{algorithm}` instead.
- Do not invent unsupported arguments or unsupported TSB-AD APIs.
- Keep the script robust to both CSV and `.npy`-style time-series inputs.
- Always use the exact provided dataset paths `{data_path_train}` and `{data_path_test}` or exact companion files derived from those same paths.
- Do not substitute official demo paths such as `./dataset/MSL/...`.
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
       After filtering `run_kwargs`, do not append unsupported defaults with `run_kwargs.update(...)`.
       If you need to add a fallback default, re-filter the final kwargs against `inspect.signature(model_runner).parameters` before the call.
       Never set `periodicity` to `0`.
       If you explicitly set `periodicity`, it must be an integer `>= 1`.
   (8) Run `scores = model_runner(X_train, **run_kwargs)`.
       Before the wrapper call, convert `X_train` to finite `np.float64` values:
       `X_train = np.nan_to_num(np.asarray(X_train, dtype=np.float64))`
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
- Always use the exact provided dataset path `{data_path_train}` or exact companion files derived from it.
- Do not substitute official demo paths such as `./dataset/MSL/...`.
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
    If the script already filtered `run_kwargs` with `inspect.signature(model_runner).parameters`, do not add extra kwargs afterward with `run_kwargs.update(...)` unless you re-filter the final kwargs against the same signature.
    Do not introduce guessed defaults such as `window_size`, `threshold`, `normalize`, or `verbose` unless the wrapper signature explicitly supports them.
11. Keep the script self-contained for sandbox execution. Do not import or use `DataLoader`, and do not depend on generated loader files such as `train_data_loader.py` or `test_data_loader.py`; load dataset files directly instead.
12. For unlabeled TSB-AD anomaly outputs, print detected outlier indices as one list using the format `Detected outlier at point [0, 5, 12]`.
13. For TSB-AD scripts, clean missing feature values with `np.nan_to_num(np.asarray(X, dtype=float))` before calling the model wrapper.
    When the error mentions a dtype mismatch or a dependency such as `stumpy` expects `numpy.float64`, cast the wrapper inputs explicitly with `np.asarray(X, dtype=np.float64)` instead of `float32`.
14. For TSB-AD scripts, normalize wrapper outputs to sample-level scores before thresholding: do not blindly flatten 2D arrays; if a 2D score matrix has one row per sample, use `np.linalg.norm(scores, axis=1)`, then require `len(scores) == len(X_train)` or the matching test set length.
15. If the error is `ValueError: range() arg 3 must not be zero` in a TSB-AD wrapper (e.g. run_SAND), it means the algorithm only supports univariate (1D) data but received multivariate (2D) data. The internal `find_length_rank` returns 0 on 2D arrays, making `overlaping_rate=0`. Fix by extracting the first channel before calling the wrapper:
    `X_train_1d = X_train[:, 0] if X_train.ndim == 2 else X_train`
    `X_test_1d = X_test[:, 0] if X_test.ndim == 2 else X_test`
    Then call `scores = model_runner(X_train_1d, X_test_1d, **run_kwargs)`.
    Keep `X_train` and `X_test` 2D for all other purposes (e.g. `len()`, `shape`). Do NOT remove the `data_test` argument.
16. If the error is `ValueError: Cannot take a larger sample than population when 'replace=False'` in a TSB-AD wrapper that infers window sizes (for example `run_SAND`), keep the same data paths and keep the wrapper call structure, but do not set `periodicity` to `0`.
    If the script added a manual `run_kwargs["periodicity"]` override, remove that override and fall back to the wrapper default, or replace it with a valid integer `>= 1` such as `1`.
    Also inspect how `X_train` and `X_test` are loaded. If a semisupervised TSB-AD script loads the same dataset path for both `X_train` and `X_test`, that is likely the real bug.
    In that case, keep the same loading style but change `X_test` to the matching test dataset path.
    If the script uses a paired naming pattern such as `_train.npy` / `_test.npy`, replace only the mistaken test path with the `_test.npy` companion file.
    Do not replace the dataset with demo files, and do not invent unrelated path changes.
17. If the error is `ValueError: ('All window sizes must be greater than or equal to three', ...)` in a TSB-AD wrapper such as `run_SAND`, it means the script forced an invalid tiny window, often by setting `periodicity=0`.
    Remove the manual `periodicity=0` override.
    If you still need an explicit value, use a valid integer `>= 1`.
    Keep the same dataset paths and same wrapper call shape.
18. If the error is `got multiple values for argument 'data'` in a TSB-AD direct wrapper, the second dataset argument named `data` is being passed both positionally and through kwargs or otherwise duplicated.
    Remove any `data` or `data_train` entry from `run_kwargs`.
    Keep the wrapper call positional, for example `scores = model_runner(X_train, X_test, **run_kwargs)`.
    Do not rewrite the call to use keyword arguments for the dataset arrays unless the wrapper signature explicitly requires that.
19. If the error is `ValueError: not enough values to unpack (expected 2, got 1)` from a TSB-AD model doing `n_samples, n_features = X.shape`, the wrapper received a 1D array but expects a 2D feature matrix.
    Do not reduce `X_train` or `X_test` to one channel in that case.
    Keep the dataset arrays 2D and pass them directly to the wrapper.
20. If the error is `missing 1 required positional argument: 'data'` in a TSB-AD direct wrapper, the wrapper expects both train and test datasets as positional arguments.
    Pass both arrays positionally, for example `scores = model_runner(X_train, X_test, **run_kwargs)`.
    Do not drop the second dataset argument, and do not try to satisfy it by adding a `data` key to `run_kwargs`.
21. If the error says a library expects `numpy.float64` but found `float32` in a TSB-AD script, cast the dataset arrays passed to the wrapper to `np.float64` immediately before the call:
    `X_train = np.nan_to_num(np.asarray(X_train, dtype=np.float64))`
    `X_test = np.nan_to_num(np.asarray(X_test, dtype=np.float64))`
    Keep the same wrapper call shape and do not downcast back to `float32`.

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

Dataset metadata:
{dataset_metadata}
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

Dataset metadata:
{dataset_metadata}
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
        metadata=None,
    ) -> str:
        tpl = None

        if package_name == "pyod":
            tpl = template_pyod_labeled if data_path_test else template_pyod_unlabeled
        elif package_name == "pygod":
            tpl = template_pygod_labeled if data_path_test else template_pygod_unlabeled
        elif package_name == "tsb_ad":
            tpl = template_tsb_ad_unlabeled if algorithm in Unsupervise_AD_Pool else template_tsb_ad_labeled
        else:
            tpl = template_darts_labeled if data_path_test else template_darts_unlabeled

        # Format metadata for prompt context
        metadata_str = (
            json.dumps(metadata or {}, default=str, indent=2)
            if package_name == "tsb_ad"
            else self._format_metadata(metadata, package_name)
        )

        raw = llm.invoke(
            tpl.invoke({
                "algorithm": algorithm,
                "data_path_train": data_path_train,
                "data_path_test": data_path_test,
                "algorithm_doc": algorithm_doc,
                "parameters": str(input_parameters),
                "dataset_metadata": metadata_str,
            })
        ).content
        cleaned = self._clean(raw)
        if package_name == "tsb_ad":
            cleaned = self._sanitize_tsb_ad_code(cleaned)
        line_count = cleaned.count("\n") + 1 if cleaned else 0
        print(f"[code_generator][{algorithm}] Generated candidate ({line_count} lines)")
        return cleaned

    # -------- revision (moved from old Reviewer) --------
    def revise_code(self, code_quality: CodeQuality, algorithm_doc: str) -> str:
        print(f"[code_generator][{code_quality.algorithm}] Review failure detected; revising code")
        print(f"[code_generator][{code_quality.algorithm}] Error: {code_quality.error_message}")
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
        cleaned = self._sanitize_tsb_ad_code(cleaned)
        return cleaned


    # -------- util --------
    @staticmethod
    def _format_metadata(metadata: dict | None, package_name: str) -> str:
        """Format selector metadata into a human-readable string for prompts."""
        if not metadata:
            return "No metadata available."
        parts = []
        fmt = metadata.get("format")
        if fmt:
            parts.append(f"- File format: {fmt}")
        if "num_samples" in metadata and metadata["num_samples"] is not None:
            parts.append(f"- Number of samples: {metadata['num_samples']}")
        if "feature_dim" in metadata and metadata["feature_dim"] is not None:
            parts.append(f"- Feature dimensions: {metadata['feature_dim']}")
        if "has_labels" in metadata:
            parts.append(f"- Has labels: {metadata['has_labels']}")
        if "has_timestamps" in metadata:
            parts.append(f"- Has timestamps: {metadata['has_timestamps']}")
        if "columns" in metadata:
            parts.append(f"- Column names: {metadata['columns']}")
        # pygod-specific
        if "num_nodes" in metadata:
            parts.append(f"- Number of nodes: {metadata['num_nodes']}")
        if "num_edges" in metadata:
            parts.append(f"- Number of edges: {metadata['num_edges']}")
        if "num_features" in metadata:
            parts.append(f"- Number of features: {metadata['num_features']}")
        return "\n".join(parts) if parts else "No metadata available."

    @staticmethod
    def _clean(code: str) -> str:
        code = re.sub(r"```(python)?", "", code)
        return re.sub(r"```", "", code).strip()

    @staticmethod
    def _sanitize_tsb_ad_code(code: str) -> str:
        if "from TSB_AD.model_wrapper import run_" not in code or "**run_kwargs" not in code:
            return code

        filter_block = (
            "valid_params = inspect.signature(model_runner).parameters\n"
            "if 'user_params' in locals() and isinstance(user_params, dict):\n"
            "    run_kwargs = {\n"
            "        k: v for k, v in user_params.items()\n"
            "        if k in valid_params and k not in {\"data\", \"data_train\"}\n"
            "    }\n"
            "else:\n"
            "    run_kwargs = {\n"
            "        k: v for k, v in run_kwargs.items()\n"
            "        if k in valid_params and k not in {\"data\", \"data_train\"}\n"
            "    }\n\n"
        )

        if filter_block in code:
            return code

        score_call_pattern = re.compile(r"^(\s*)scores\s*=\s*model_runner\(", re.MULTILINE)
        match = score_call_pattern.search(code)
        if not match:
            return code

        insert_at = match.start()
        return code[:insert_at] + filter_block + code[insert_at:]

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
