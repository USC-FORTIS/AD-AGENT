import ast
import json
import os
import re
import subprocess
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path
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

--- BEGIN CODE ---
{code}
--- END CODE ---
                                                               
TASK:
1. Replace **all data-loading operations** (DataLoader, torch.load, np.load, pandas.read*, etc.)
   with code that creates SMALL synthetic data directly in the script:
    • For PyOD: generate X_train, y_train, X_test, y_test using `generate_data`; You need to make sure that the dimension of the features matches the training dataset {train_dataset}.
         `from pyod.utils.data import generate_data`
         Example:
         `X_train, X_test, y_train, y_test = generate_data(n_train=200, n_test=100, contamination=0.1, n_features=n_features)`
    • For PyGOD: build train and test graph follow instruction below; You need to make sure that the dimension of the features matches the training dataset {train_dataset}.
     `import torch`
     `from pygod.generator import gen_contextual_outlier, gen_structural_outlier`
     `from torch_geometric.data import Data`
     `num_nodes = 200 
     `x = torch.randn(num_nodes, num_features)`  

     `edge_index = torch.tensor([`  
     `    [i, (i+1) % num_nodes] for i in range(num_nodes)`  
     `], dtype=torch.long).T  # shape: [2, num_edges]`  

     `data = Data(x=x, edge_index=edge_index)`  
     `data, ya = gen_contextual_outlier(data, n=100, k=50)`  
     `data, ys = gen_structural_outlier(data, m=10, n=10)`  
     `data.y = torch.logical_or(ys, ya).long()`  
    • For Darts: You need to make sure that the dimension of the features matches the training dataset {train_dataset}.

    `import numpy as np`
    `import pandas as pd`
    `from darts import TimeSeries`


    `def load_series(path: str,`
                    `n_samples: int = 500,`
                    `n_features: int = 1,`
                    `contamination: float = 0.05,`
                    `seed: int = 42):`
        `rng = np.random.default_rng(seed)`

        `dates = pd.date_range("2020-01-01", periods=n_samples, freq="H")`

        `data = rng.normal(loc=0.0, scale=1.0, size=(n_samples, n_features))`

        `n_anom = int(n_samples * contamination)`
        `anom_idx = rng.choice(n_samples, n_anom, replace=False)`
        `data[anom_idx] += rng.normal(loc=6.0, scale=1.0, size=(n_anom, n_features))`

        `df = pd.DataFrame(data, columns=[f"value_{{i+1}}" for i in range(n_features)])`
        `df["timestamp"] = dates`
        `df["anomaly"] = 0`
        `df.loc[anom_idx, "anomaly"] = 1`
        `df.set_index("timestamp", inplace=True)`

        `value_cols = [c for c in df.columns if c.startswith("value_")]`
        `series = TimeSeries.from_dataframe(df, value_cols=value_cols)`
        `labels = df["anomaly"].astype(int).values`
        `return series, labels`

    `series_train, y_train = load_series(None, n_samples=1000, n_features=3, seed=0)`
    `series_test,  y_test  = load_series(None, n_samples=300,  n_features=3, seed=1)`
    `series_train = series_train.astype(np.float32)`
    `series_test  = series_test.astype(np.float32)`
    `torch.set_default_dtype(torch.float32)`
                                           
    2. Keep the variable names and the rest of the logic unchanged.
    3. Output runnable Python **code only** (no explanations, no markdown).
""")

tslib_data_prompt = PromptTemplate.from_template("""
You are generating small synthetic files for a Time-Series-Library anomaly-detection review run.

Use only the following synthetic data specification:
```json
{synthetic_data_spec}
```

Task:
1. Output only executable Python code, with no markdown fences or explanations.
2. Create synthetic train/test/label files exactly according to the specification.
3. Do not invent filenames, directories, columns, or shapes beyond what the specification requires.
4. Ensure the output directory exists before writing files.
5. Use deterministic random generation with a fixed seed.
6. Inject anomalies into the synthetic test data and generate matching binary labels.
7. Do not read the real dataset; use only the provided specification.

Implementation requirements:
- Use `numpy` for generation.
- Use `pandas` when writing CSV files.
- Use `float32` feature values.
- Respect the provided feature count, file format, output filenames, and label placement rules.
""")

tslib_fast_review_prompt = PromptTemplate.from_template("""
You are rewriting a Time-Series-Library anomaly-detection script for fast reviewer validation only.

Original script:
```python
{code}
```

Synthetic training dataset path:
{train_dataset}

Algorithm:
{algorithm_name}

Task:
1. Rewrite the script so it stays in Time-Series-Library and keeps the same algorithm and overall execution path.
2. Make only the smallest possible changes needed to make validation run faster on synthetic data.
3. Prefer lower-cost settings only when they correspond to arguments already present in the script or are clearly standard runtime flags already used by the script.
4. Prefer CPU-safe execution when the script currently assumes GPU.
5. Do not invent a new pipeline, do not replace the library, and do not switch to shell scripts.
6. Return executable Python code only, with no markdown fences or explanations.

Important:
- This rewritten script is only for reviewer validation.
- Preserve the original script structure as much as possible.
- The original code generator result remains the source of truth.
""")


class AgentReviewer:
    """Responsible for executing code and recording metrics only."""
    TEST_SCRIPT_TIMEOUT_SECONDS = 300

    def __init__(self):
        pass

    @staticmethod
    def _import_pandas():
        try:
            import pandas as pd
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "pandas is required only when inferring TSLib synthetic data from CSV inputs."
            ) from exc
        return pd

    def test_code(
        self,
        code: str,
        algorithm_name: str,
        package_name: str,
        train_dataset: str | None = None,
        tslib_cli_metadata: dict | None = None,
    ) -> str:
        """
        Generate a test script using synthetic data and execute it.
        Return an empty string on success, or an error message on failure or exception.
        """
        try:
            # 1) Build a test script
            if package_name == "tslib":
                base_code = code
                print(f"[Reviewer][TSLib] Synthetic data generation started for {algorithm_name}")
                # Use LLM to generate synthetic data script.
                synthetic_root = self._generate_tslib_synthetic_data(
                    code=base_code,
                    algorithm_name=algorithm_name,
                    train_dataset=train_dataset,
                )
                print(f"[Reviewer][TSLib] Synthetic data generation finished for {algorithm_name}")
                # Rewrite the original code's data loading to point to the synthetic data. Use metadata in tslib.
                base_code = self._rewrite_tslib_root_path(base_code, synthetic_root, tslib_cli_metadata)

                review_code = None
                try:
                    # Let LLM attempt a fast rewrite of the original code for reviewer validation. 
                    # This is a best-effort attempt to reduce reviewer runtime.
                    review_code = self._build_fast_tslib_review_code(
                        code=base_code,
                        algorithm_name=algorithm_name,
                        train_dataset=train_dataset,
                    )
                    # Rewrite the LLM-generated review code to point to the synthetic data as well, using the same metadata.
                    review_code = self._rewrite_tslib_root_path(review_code, synthetic_root, tslib_cli_metadata)
                    review_path = self._write_test_script(review_code, f"{algorithm_name}_review")
                    review_error = self._run_script_for_validation(review_path, algorithm_name)
                    if not review_error:
                        return ""
                    print(f"[Reviewer][TSLib] Fast review variant failed for {algorithm_name}; falling back to the original generated code.")
                except Exception as review_exc:
                    print(f"[Reviewer][TSLib] Fast review variant generation/execution failed for {algorithm_name}: {review_exc}")

                base_path = self._write_test_script(base_code, algorithm_name)
                return self._run_script_for_validation(base_path, algorithm_name)
            else:
                test_script = llm.invoke(
                    test_prompt.invoke({
                        "code": code,
                        "algorithm_name": algorithm_name,
                        "package_name": package_name,
                        "train_dataset": train_dataset
                    })
                ).content
                test_script = self._clean_markdown(test_script)

            path = self._write_test_script(test_script, algorithm_name)
            return self._run_script_for_validation(path, algorithm_name)
        except Exception as e:
            print(f"[test_code] Exception: {e}")
            return str(e)

    @staticmethod
    def _clean_markdown(txt: str) -> str:
        """Remove markdown code fences from the script."""
        txt = re.sub(r"```(python)?", "", txt)
        return re.sub(r"```", "", txt).strip()

    def _generate_tslib_synthetic_data(
        self,
        code: str,
        algorithm_name: str,
        train_dataset: str | None,
    ) -> Path:
        spec = self._infer_tslib_synthetic_spec(train_dataset, code)
        output_root = Path(spec["output_root"]).resolve()
        output_root.mkdir(parents=True, exist_ok=True)
        print(f"[Reviewer][TSLib] Synthetic data root: {output_root}")
        print(f"[Reviewer][TSLib] Inferred feature count: {spec['feature_count']}")
        print(f"[Reviewer][TSLib] Synthetic data spec: {json.dumps(spec, ensure_ascii=False)}")

        generation_script = llm.invoke(
            tslib_data_prompt.invoke(
                {
                    "synthetic_data_spec": json.dumps(spec, ensure_ascii=False, indent=2),
                }
            )
        ).content
        generation_script = self._clean_markdown(generation_script)

        script_path = Path("generated_scripts") / f"{algorithm_name}_tslib_data_gen.py"
        script_path.write_text(generation_script, encoding="utf-8")
        print(f"[Reviewer][TSLib] Synthetic data script written to: {script_path}")

        res = subprocess.run(["python", str(script_path)], capture_output=True, text=True)
        print("\n=== TSLib Synthetic Data Generation Output ===\n", res.stdout, res.stderr)
        print(f"[Reviewer][TSLib] Synthetic data script finished with return code {res.returncode}")
        if res.returncode != 0:
            raise RuntimeError(self._subprocess_output_as_error(res.stdout, res.stderr))

        nested_error = self._detect_nested_failure(res.stdout, res.stderr)
        if nested_error:
            raise RuntimeError(self._subprocess_output_as_error(res.stdout, res.stderr))

        return output_root

    def _build_fast_tslib_review_code(
        self,
        code: str,
        algorithm_name: str,
        train_dataset: str | None,
    ) -> str:
        rewritten = llm.invoke(
            tslib_fast_review_prompt.invoke(
                {
                    "code": code,
                    "algorithm_name": algorithm_name,
                    "train_dataset": train_dataset or "",
                }
            )
        ).content
        return self._clean_markdown(rewritten)

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

    @staticmethod
    def _rewrite_tslib_root_path(
        code: str,
        output_root: Path,
        tslib_cli_metadata: dict | None = None,
    ) -> str:
        match = re.search(r"cmd\s*=\s*(\[[\s\S]*?\])", code)
        if not match:
            return code
        try:
            cmd = ast.literal_eval(match.group(1))
        except Exception:
            return code
        if not isinstance(cmd, list):
            return code

        cmd = [str(item) for item in cmd]
        dataset_root_arg = "root_path"
        if tslib_cli_metadata and tslib_cli_metadata.get("dataset_root_arg"):
            dataset_root_arg = str(tslib_cli_metadata["dataset_root_arg"])

        flag = f"--{dataset_root_arg}"
        try:
            idx = cmd.index(flag)
        except ValueError:
            return code

        if idx + 1 < len(cmd) and not str(cmd[idx + 1]).startswith("--"):
            cmd[idx + 1] = str(output_root)
        else:
            cmd.insert(idx + 1, str(output_root))

        rebuilt = repr(cmd)
        return code[:match.start(1)] + rebuilt + code[match.end(1):]

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

    @staticmethod
    def _extract_tslib_root_path(code: str) -> Path | None:
        match = re.search(r"cmd\s*=\s*(\[[\s\S]*?\])", code)
        if not match:
            return None
        try:
            cmd = ast.literal_eval(match.group(1))
        except Exception:
            return None
        if not isinstance(cmd, list):
            return None
        try:
            idx = cmd.index("--root_path")
        except ValueError:
            return None
        if idx + 1 >= len(cmd):
            return None
        return Path(str(cmd[idx + 1]))

    def _infer_tslib_synthetic_spec(self, train_dataset: str | None, code: str) -> dict:
        train_path = Path(train_dataset).resolve() if train_dataset else None
        code_root = self._extract_tslib_root_path(code)
        output_root = code_root.resolve() if code_root else (train_path.parent if train_path else Path("generated_scripts/tslib_prompt_data")).resolve()

        if not train_path or not train_path.exists():
            raise FileNotFoundError("train_dataset is required to infer a tslib synthetic data spec.")

        test_candidate, label_candidate = self._infer_companion_paths(train_path)
        feature_count = self._infer_feature_count(train_path)
        file_format = train_path.suffix.lower().lstrip(".")
        test_has_embedded_label = self._csv_has_label_column(test_candidate) if test_candidate and test_candidate.suffix.lower() == ".csv" else False
        train_has_embedded_label = self._csv_has_label_column(train_path) if train_path.suffix.lower() == ".csv" else False

        return {
            "output_root": str(output_root),
            "file_format": file_format,
            "feature_count": feature_count,
            "train_output_filename": train_path.name,
            "test_output_filename": test_candidate.name if test_candidate else self._replace_train_token(train_path.name, "test"),
            "label_output_filename": label_candidate.name if label_candidate else None,
            "train_has_label_column": train_has_embedded_label,
            "test_has_label_column": test_has_embedded_label,
            "label_values_file": label_candidate.name if label_candidate else None,
        }

    def _infer_companion_paths(self, train_path: Path) -> tuple[Path | None, Path | None]:
        siblings = [p for p in train_path.parent.iterdir() if p.is_file() and p.name != train_path.name]

        def _score(candidate: Path) -> float:
            return SequenceMatcher(None, train_path.stem.lower(), candidate.stem.lower()).ratio()

        label_candidates = [p for p in siblings if "label" in p.stem.lower()]
        label_path = max(label_candidates, key=_score) if label_candidates else None

        test_candidates = [p for p in siblings if p != label_path and p.suffix.lower() == train_path.suffix.lower()]
        if test_candidates:
            explicit_test = [p for p in test_candidates if "test" in p.stem.lower()]
            pool = explicit_test or test_candidates
            test_path = max(pool, key=_score)
        else:
            test_name = self._replace_train_token(train_path.name, "test")
            test_path = train_path.parent / test_name if test_name != train_path.name else None

        return test_path, label_path

    @staticmethod
    def _replace_train_token(filename: str, replacement: str) -> str:
        updated = re.sub(r"train", replacement, filename, count=1, flags=re.IGNORECASE)
        return updated if updated else filename

    def _infer_feature_count(self, train_path: Path) -> int:
        suffix = train_path.suffix.lower()
        if suffix == ".npy":
            arr = np.load(train_path)
            if arr.ndim == 1:
                return 1
            if arr.ndim >= 2:
                return int(arr.shape[1])
        if suffix == ".csv":
            pd = self._import_pandas()
            df = pd.read_csv(train_path, nrows=8)
            return max(1, len(self._infer_feature_columns(df)))
        raise ValueError(f"Unsupported tslib train file format: {train_path.suffix}")

    @staticmethod
    def _infer_feature_columns(df) -> list[str]:
        pd = AgentReviewer._import_pandas()
        lowered = {c.lower(): c for c in df.columns}
        excluded = {
            lowered[c]
            for c in lowered
            if c in {"label", "anomaly", "timestamp", "ts", "date", "time"}
        }
        numeric_cols = [
            c for c in df.columns
            if c not in excluded and pd.api.types.is_numeric_dtype(df[c])
        ]
        return numeric_cols or [c for c in df.columns if c not in excluded]

    @staticmethod
    def _csv_has_label_column(path: Path | None) -> bool:
        if not path or not path.exists() or path.suffix.lower() != ".csv":
            return False
        try:
            pd = AgentReviewer._import_pandas()
            df = pd.read_csv(path, nrows=5)
        except Exception:
            return False
        lowered = {c.lower() for c in df.columns}
        return "label" in lowered or "anomaly" in lowered

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
