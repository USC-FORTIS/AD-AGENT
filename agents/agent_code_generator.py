from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
import re
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from entity.code_quality import CodeQuality
import subprocess
from datetime import datetime, timedelta
import ast
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
   (2) import DataLoader using following commend `from data_loader.data_loader import DataLoader` after (1)
   (3) Initialize DataLoader using statement `dataloader_train = DataLoader(filepath = {data_path_train}, store_script=True, store_path = 'train_data_loader.py')` & `dataloader_test = DataLoader(filepath = {data_path_test}, store_script=True, store_path = 'test_data_loader.py')`
   (4) Use the statement `X_train, y_train = dataloader_train.load_data(split_data=False)` & `X_test, y_test = dataloader_test.load_data(split_data=False)` to generate variables X_train, y_train, X_test, y_test; 
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
- Strictly follow steps (2)-(8) to load the data from `{data_path_train}` & {data_path_test}.
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
   (1)    
   (2) Load the data from `{data_path_train}`
   (3) Extract the feature matrix `X` from the loaded data as `X_train`
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
- Strictly follow steps (2)-(8) to load the data from `{data_path_train}` & {data_path_test}.
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

template_fix = PromptTemplate.from_template("""
You are an expert Python developer with deep experience in anomaly detection libraries.

Here is the original code that raised an error:
--- Original Code ---
{code}

--- Error Message ---
{error_message}

Official documentation for `{algorithm}`:
--- BEGIN DOCUMENTATION ---
{algorithm_doc}
--- END DOCUMENTATION ---

Task:
1. Analyse the error and fix it strictly according to the doc.
2. Output **executable** Python ONLY, no comments/explanations.
""")

template_tslib_labeled = PromptTemplate.from_template("""
You are an expert Python developer with deep knowledge of the **Tslib** library for forecasting-based anomaly detection. Your task is to:

1. Carefully study the official documentation excerpt for **`{algorithm}`** provided below so you fully understand how to initialise, fit, and use this class.          


--- BEGIN DOCUMENTATION ---
{algorithm_doc}
--- END DOCUMENTATION ---

This is how to use how we use the Tslib library for anomaly detection:
```python
import os
import subprocess

cmd = [
    "python", "-u", "./Time-Series-Library/run.py",
    "--task_name", "anomaly_detection",
    "--is_training", "1",
    "--root_path", "{data_path_train}",
    "--model_id", "{data_path_train}",
    "--model", "{algorithm}",
    "--data", "project name of {data_path_train}", # if the data_path_train is ./data/MSL_train.npy, the project name is MSL

]
1. You must use `{data_path_train}` as root_path. and for `--data` you need choose proper project name according to the `{data_path_train}`, the appdix of `{data_path_train}`
eg.
```
-- root_path ./data/MSL
-- data MSL
```
                                                      

You need to choose proper parameters for the model and add them to the command. Please note that for ETSformer, the encode layers and decode layers must be equal. `--e_layers` and `--d_layers` must be equal. For example, if you set `--e_layers 2`, you must set `--d_layers 2` as well.
Do not add unsupported parameters such as '--mix' nor '--output_attention'. Please follow instruction in [DOCUMENTATION] to add parameters or the example above.
                                                      
Set `--gpu_type` to `cpu`
Set '--use_gpu' to 'False'
                                                      
Avoid adding the following parameters:
--use_amp, --use_multi_gpu, --itr
                                                      

For `--dec_in`, `--enc_in`, and `c_out`, you need to set them according to the dataset name:
- For MSL, set `--dec_in`, `--enc_in`, and `c_out` to 55
- For SMAP, set `--dec_in`, `--enc_in`, and `c_out` to 25
- For PSM, set `--dec_in`, `--enc_in`, and `c_out` to 25
- For SMD, set `--dec_in`, `--enc_in`, and `c_out` to 38
- For SWAT, set `--dec_in`, `--enc_in`, and `c_out` to 51
                                                      
Please set the following parameters to its input value, especially `--dec_in` and `--enc_in`:
{parameters}
You have to set these parameter as required.
                                                      

Output **only** executable Python code (no extra text) that performs forecasting-based anomaly detection.


          
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
        package_name
    ) -> str:
        tpl = None
       
        if package_name == "pyod":
            tpl = template_pyod_labeled if data_path_test else template_pyod_unlabeled
        elif package_name == "pygod":
            tpl = template_pygod_labeled if data_path_test else template_pygod_unlabeled
        elif package_name == "tslib": # tslib only has labeled data
            tpl = template_tslib_labeled 
        else:
            tpl = template_darts_labeled if data_path_test else template_darts_unlabeled
        raw = llm.invoke(
            tpl.invoke({
                "algorithm": algorithm,
                "data_path_train": data_path_train,
                "data_path_test": data_path_test,
                "algorithm_doc": algorithm_doc,
                "parameters": str(input_parameters)
            })
        ).content
        return self._clean(raw)

    # -------- revision (moved from old Reviewer) --------
    def revise_code(self, code_quality: CodeQuality, algorithm_doc: str) -> str:
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
        return self._clean(fixed)

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
      "algorithm": ["Crossformer"],
      "dataset_train": "SMAP_train.npy",
      "dataset_test": "SMAP_test.npy",
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
