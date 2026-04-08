import json
import os
import re
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from entity.code_quality import CodeQuality
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from config.config import Config

# Ensure API key is available for OpenAI client
os.environ.setdefault("OPENAI_API_KEY", Config.OPENAI_API_KEY)

# Initialize the OpenAI LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Prompt template for generating synthetic test data
test_prompt = PromptTemplate.from_template("""
You will receive a Python script for {package_name} that trains an anomaly-detection model with real datasets.

Dataset metadata from selector:
```json
{dataset_metadata}
```

Use `feature_count = {feature_count}` for synthetic feature dimensions unless the original script explicitly requires a different dimension.

--- BEGIN CODE ---
{code}
--- END CODE ---
                                                               
TASK:
1. Replace **all data-loading operations** (DataLoader, torch.load, np.load, pandas.read*, etc.)
   with code that creates SMALL synthetic data directly in the script:
    • For PyOD: generate X_train, y_train, X_test, y_test using `generate_data`; use `feature_count = {feature_count}`.
         `from pyod.utils.data import generate_data`
         Example:
         `feature_count = {feature_count}`
         `X_train, X_test, y_train, y_test = generate_data(n_train=200, n_test=100, contamination=0.1, n_features=feature_count)`
    • For PyGOD: build train and test graph follow instruction below; use `feature_count = {feature_count}`.
     `import torch`
     `from pygod.generator import gen_contextual_outlier, gen_structural_outlier`
     `from torch_geometric.data import Data`
     `num_nodes = 200 
     `feature_count = {feature_count}`
     `x = torch.randn(num_nodes, feature_count)`

     `edge_index = torch.tensor([`  
     `    [i, (i+1) % num_nodes] for i in range(num_nodes)`  
     `], dtype=torch.long).T  # shape: [2, num_edges]`  

     `data = Data(x=x, edge_index=edge_index)`  
     `data, ya = gen_contextual_outlier(data, n=100, k=50)`  
     `data, ys = gen_structural_outlier(data, m=10, n=10)`  
     `data.y = torch.logical_or(ys, ya).long()`  
    • For TSB-AD (`package_name == "tsb_ad"`): create raw numpy arrays, not Darts objects:
      `import numpy as np`
      `rng = np.random.default_rng(42)`
      `feature_count = {feature_count}`
      `X_train = rng.normal(0.0, 1.0, size=(200, feature_count)).astype(np.float32)`
      `anom_idx = rng.choice(len(X_train), max(1, len(X_train) // 20), replace=False)`
      `X_train[anom_idx] += rng.normal(6.0, 1.0, size=(len(anom_idx), feature_count)).astype(np.float32)`
      If the script expects `X_test` or labels, create `X_test`, `y_train`, and `y_test` the same way with numpy arrays.
      Do not import or use `darts.TimeSeries`, `TimeSeries.from_dataframe`, or `torch` for TSB-AD reviewer data.
    • For Darts: use `feature_count = {feature_count}`.

    `import numpy as np`
    `import pandas as pd`
    `from darts import TimeSeries`


    `def load_series(path: str,`
                    `n_samples: int = 500,`
                    `feature_count: int = 1,`
                    `contamination: float = 0.05,`
                    `seed: int = 42):`
        `rng = np.random.default_rng(seed)`

        `dates = pd.date_range("2020-01-01", periods=n_samples, freq="H")`

        `data = rng.normal(loc=0.0, scale=1.0, size=(n_samples, feature_count))`

        `n_anom = int(n_samples * contamination)`
        `anom_idx = rng.choice(n_samples, n_anom, replace=False)`
        `data[anom_idx] += rng.normal(loc=6.0, scale=1.0, size=(n_anom, feature_count))`

        `df = pd.DataFrame(data, columns=[f"value_{{i+1}}" for i in range(feature_count)])`
        `df["timestamp"] = dates`
        `df["anomaly"] = 0`
        `df.loc[anom_idx, "anomaly"] = 1`
        `df.set_index("timestamp", inplace=True)`

        `value_cols = [c for c in df.columns if c.startswith("value_")]`
        `series = TimeSeries.from_dataframe(df, value_cols=value_cols)`
        `labels = df["anomaly"].astype(int).values`
        `return series, labels`

    `series_train, y_train = load_series(None, n_samples=1000, feature_count={feature_count}, seed=0)`
    `series_test,  y_test  = load_series(None, n_samples=300,  feature_count={feature_count}, seed=1)`
    `series_train = series_train.astype(np.float32)`
    `series_test  = series_test.astype(np.float32)`
    `torch.set_default_dtype(torch.float32)`
                                           
    2. Keep the variable names and the rest of the logic unchanged.
       If `{package_name}` is `tsb_ad`, preserve raw numpy variable names such as `X_train` and `X_test`; do not rewrite them to `series_train` or `TimeSeries`.
    3. Output runnable Python **code only** (no explanations, no markdown).
""")

class AgentReviewer:
    """Responsible for executing code and recording metrics only."""
    TEST_SCRIPT_TIMEOUT_SECONDS = 300

    def __init__(self):
        pass

    def test_code(
        self,
        code: str,
        algorithm_name: str,
        package_name: str,
        train_dataset: str | None = None,
        dataset_metadata: dict | None = None,
    ) -> str:
        """
        Generate a test script using synthetic data and execute it.
        Return an empty string on success, or an error message on failure or exception.
        """
        try:
            # 1) Build a test script
            test_script = llm.invoke(
                test_prompt.invoke({
                    "code": code,
                    "algorithm_name": algorithm_name,
                    "package_name": package_name,
                    "train_dataset": train_dataset,
                    "dataset_metadata": json.dumps(dataset_metadata or {}, default=str, indent=2),
                    "feature_count": self._feature_count_from_metadata(dataset_metadata),
                })
            ).content
            test_script = self._clean_markdown(test_script)

            path = self._write_test_script(test_script, algorithm_name)
            return self._run_script_for_validation(path, algorithm_name)
        except Exception as e:
            print(f"[test_code] Exception: {e}")
            return str(e)

    @staticmethod
    def _feature_count_from_metadata(dataset_metadata: dict | None) -> int:
        metadata = dataset_metadata or {}
        candidates = [
            metadata.get("n_features"),
            (metadata.get("train") or {}).get("n_features") if isinstance(metadata.get("train"), dict) else None,
            ((metadata.get("dataset") or {}).get("train") or {}).get("n_features")
            if isinstance(metadata.get("dataset"), dict) and isinstance((metadata.get("dataset") or {}).get("train"), dict)
            else None,
        ]
        for value in candidates:
            try:
                feature_count = int(value)
            except (TypeError, ValueError):
                continue
            if feature_count > 0:
                return feature_count
        return 3

    @staticmethod
    def _clean_markdown(txt: str) -> str:
        """Remove markdown code fences from the script."""
        txt = re.sub(r"```(python)?", "", txt)
        return re.sub(r"```", "", txt).strip()

    @staticmethod
    def _write_test_script(test_script: str, algorithm_name: str) -> str:
        folder = "generated_scripts"
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, f"{algorithm_name}_test.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(test_script)
        return path

    def _run_script_for_validation(self, path: str, algorithm_name: str) -> str:
        print(f"[Reviewer] Test script execution started for {algorithm_name}")
        res = self._execute_test_script(path, algorithm_name)
        print("\n=== Test Execution Output ===\n", res.stdout, res.stderr)
        print(f"[Reviewer] Test script execution finished for {algorithm_name} with return code {res.returncode}")

        if res.returncode != 0:
            return self._subprocess_output_as_error(res.stdout, res.stderr)
        nested_error = self._detect_nested_failure(res.stdout, res.stderr)
        if nested_error:
            return self._subprocess_output_as_error(res.stdout, res.stderr)
        return ""

    def _execute_test_script(self, path: str, algorithm_name: str) -> SimpleNamespace:
        """Stream test output live to avoid long silent runs and surface where execution stalls."""
        cmd = ["python", "-u", path]
        output_lines: list[str] = []
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        start_time = time.monotonic()
        try:
            assert process.stdout is not None
            while True:
                line = process.stdout.readline()
                if line:
                    print(line, end="")
                    output_lines.append(line)
                elif process.poll() is not None:
                    break
                elif time.monotonic() - start_time > self.TEST_SCRIPT_TIMEOUT_SECONDS:
                    process.kill()
                    output_lines.append(
                        f"\n[Reviewer] Test script timed out after {self.TEST_SCRIPT_TIMEOUT_SECONDS} seconds while running {algorithm_name}.\n"
                    )
                    break
        finally:
            if process.stdout is not None:
                process.stdout.close()

        returncode = process.wait()
        stdout = "".join(output_lines)
        stderr = ""
        if returncode is None:
            returncode = 1
        if "timed out after" in stdout and returncode == 0:
            returncode = 1
        return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)

    # -------- helpers --------
    @staticmethod
    def _find(pattern, text, default=-1.0):
        """Find a float number from text using regex pattern."""
        m = re.search(pattern, text)
        return float(m.group(1)) if m else default

    @staticmethod
    def _find_errors(text):
        """Extract failed prediction points and true labels from output logs."""
        pts = []
        for line in text.splitlines():
            if "Failed prediction at point" in line:
                m = re.search(r"\[([^\]]+)] with true label ([\d.]+)", line)
                if m:
                    nums = [float(x.strip()) for x in m.group(1).split(",")]
                    pts.append({"point": nums, "true_label": float(m.group(2))})
        return pts


    @staticmethod
    def _detect_nested_failure(stdout: str, stderr: str) -> str:
        combined = "\n".join(part for part in (stdout, stderr) if part)
        if not combined.strip():
            return ""
        failure_markers = (
            "Traceback (most recent call last):",
            "NotImplementedError:",
            "ModuleNotFoundError:",
            "FileNotFoundError:",
            "can't open file",
            "CalledProcessError:",
        )
        for marker in failure_markers:
            if marker in combined:
                return combined
        return ""

    @staticmethod
    def _subprocess_output_as_error(stdout: str, stderr: str) -> str:
        stdout = (stdout or "").strip()
        stderr = (stderr or "").strip()
        combined = "\n".join(part for part in (stderr, stdout) if part).strip()
        if not combined:
            return "Subprocess failed with empty output."

        lines = [line.strip() for line in combined.splitlines() if line.strip()]
        for line in reversed(lines):
            if re.match(r"^[A-Za-z_][A-Za-z0-9_]*Error: ", line):
                return line
            if re.match(r"^[A-Za-z_][A-Za-z0-9_]*Exception: ", line):
                return line
            

        return combined



if __name__ == "__main__":
    # Simple smoke test for AgentReviewer
    sample_code = """
from pyod.utils.data import generate_data
from pyod.models.iforest import IForest

X_train, X_test, y_train, y_test = generate_data(n_train=200, n_test=100, contamination=0.1)
clf = IForest()
clf.fit(X_train)
preds = clf.predict(X_test)
print("preds:", preds[:10])
"""

    reviewer = AgentReviewer()
    error = reviewer.test_code(code=sample_code, algorithm_name="IForest", package_name="pyod", train_dataset="./data/glass_train.mat")
    if error:
        print("Test failed:", error)
    else:
        print("Test passed")
