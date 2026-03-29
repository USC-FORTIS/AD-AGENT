import ast
import json
import os
import re
import subprocess
import sys
from difflib import SequenceMatcher
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config.config import Config
from entity.code_quality import CodeQuality
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from sandbox.executor import execute_code

os.environ.setdefault("OPENAI_API_KEY", Config.OPENAI_API_KEY)

llm = ChatOpenAI(model="gpt-4o", temperature=0)

test_prompt = PromptTemplate.from_template(
    """
You will receive a Python script for {package_name} that trains an anomaly-detection model with real datasets.

--- BEGIN CODE ---
{code}
--- END CODE ---

TASK:
1. Replace all data-loading operations (DataLoader, torch.load, np.load, pandas.read*, etc.)
   with code that creates SMALL synthetic data directly in the script.
2. Keep the variable names and the rest of the logic unchanged.
3. Output runnable Python code only.

Package-specific rules:
- For PyOD: use `from pyod.utils.data import generate_data`. If `feature_dim={feature_dim}` is provided and greater than 0, set `n_features={feature_dim}` so the synthetic data matches `{train_dataset}`.
- For PyGOD: build a synthetic graph dataset and make sure `num_features` matches the training data from `{train_dataset}`.
- For Darts: create synthetic `TimeSeries` data and make sure `n_features` matches the training data from `{train_dataset}`.
"""
)

tslib_data_prompt = PromptTemplate.from_template(
    """
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
"""
)


class AgentReviewer:
    """Responsible for executing code and recording metrics only."""

    TEST_SCRIPT_TIMEOUT_SECONDS = 300

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
        feature_dim: int | None = None,
        train_dataset: str | None = None,
    ) -> str:
        """
        Generate a test script using synthetic data and execute it.
        Return an empty string on success, or an error message on failure or exception.
        """
        try:
            data_files = None
            test_script = code

            if package_name == "tslib":
                print(f"[Reviewer][TSLib] Synthetic data generation started for {algorithm_name}")
                output_root = self._generate_tslib_synthetic_data(
                    code=code,
                    algorithm_name=algorithm_name,
                    train_dataset=train_dataset,
                )
                spec = self._infer_tslib_synthetic_spec(train_dataset, code)
                remote_root = Path("/data") / "tslib_review" / algorithm_name.lower()
                test_script = self._rewrite_tslib_root_path(code, remote_root)
                data_files = self._build_tslib_data_files(spec, output_root, remote_root)
                print(f"[Reviewer][TSLib] Synthetic data generation finished for {algorithm_name}")
            else:
                test_script = llm.invoke(
                    test_prompt.invoke(
                        {
                            "code": code,
                            "algorithm_name": algorithm_name,
                            "package_name": package_name,
                            "train_dataset": train_dataset,
                            "feature_dim": feature_dim if feature_dim and feature_dim > 0 else 2,
                        }
                    )
                ).content
                test_script = self._clean_markdown(test_script)

            folder = "generated_scripts"
            os.makedirs(folder, exist_ok=True)
            path = os.path.join(folder, f"{algorithm_name}_test.py")
            with open(path, "w", encoding="utf-8") as file_obj:
                file_obj.write(test_script)

            print(f"[Reviewer] Test script execution started for {algorithm_name}")
            res = self._execute_test_script(
                path,
                algorithm_name,
                package_name=package_name,
                data_files=data_files,
            )
            print("\n=== Test Execution Output ===\n", res.stdout, res.stderr)
            print(
                f"[Reviewer] Test script execution finished for {algorithm_name} "
                f"with return code {res.returncode}"
            )

            if res.returncode != 0:
                return self._subprocess_output_as_error(res.stdout, res.stderr)

            nested_error = self._detect_nested_failure(res.stdout, res.stderr)
            if nested_error:
                return self._subprocess_output_as_error(res.stdout, res.stderr)
            return ""
        except Exception as exc:
            print(f"[test_code] Exception: {exc}")
            return str(exc)

    @staticmethod
    def _clean_markdown(txt: str) -> str:
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
        script_path.parent.mkdir(parents=True, exist_ok=True)
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

    def _execute_test_script(
        self,
        path: str,
        algorithm_name: str,
        package_name: str,
        data_files: dict[str, str] | None = None,
    ) -> SimpleNamespace:
        code = Path(path).read_text(encoding="utf-8")
        stdout, stderr, returncode = execute_code(
            code=code,
            algorithm_name=f"{algorithm_name}_test",
            package_name=package_name,
            data_files=data_files,
            timeout=self.TEST_SCRIPT_TIMEOUT_SECONDS,
        )
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
        output_root = (
            code_root.resolve()
            if code_root
            else (
                train_path.parent
                if train_path
                else Path("generated_scripts/tslib_prompt_data")
            ).resolve()
        )

        if not train_path or not train_path.exists():
            raise FileNotFoundError("train_dataset is required to infer a tslib synthetic data spec.")

        test_candidate, label_candidate = self._infer_companion_paths(train_path)
        feature_count = self._infer_feature_count(train_path)
        file_format = train_path.suffix.lower().lstrip(".")
        test_has_embedded_label = (
            self._csv_has_label_column(test_candidate)
            if test_candidate and test_candidate.suffix.lower() == ".csv"
            else False
        )
        train_has_embedded_label = (
            self._csv_has_label_column(train_path)
            if train_path.suffix.lower() == ".csv"
            else False
        )

        return {
            "output_root": str(output_root),
            "file_format": file_format,
            "feature_count": feature_count,
            "train_output_filename": train_path.name,
            "test_output_filename": (
                test_candidate.name
                if test_candidate
                else self._replace_train_token(train_path.name, "test")
            ),
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

        test_candidates = [
            p for p in siblings if p != label_path and p.suffix.lower() == train_path.suffix.lower()
        ]
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
            c
            for c in df.columns
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
    def _rewrite_tslib_root_path(code: str, remote_root: Path) -> str:
        original = AgentReviewer._extract_tslib_root_path(code)
        if original is None:
            return code
        return code.replace(str(original), remote_root.as_posix())

    @staticmethod
    def _build_tslib_data_files(
        spec: dict,
        output_root: Path,
        remote_root: Path,
    ) -> dict[str, str]:
        mapping = {}
        for key in ("train_output_filename", "test_output_filename", "label_output_filename"):
            filename = spec.get(key)
            if not filename:
                continue
            local_path = output_root / filename
            if local_path.exists():
                mapping[(remote_root / filename).as_posix()] = str(local_path)
        return mapping

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

    @staticmethod
    def _find(pattern, text, default=-1.0):
        m = re.search(pattern, text)
        return float(m.group(1)) if m else default

    @staticmethod
    def _find_errors(text):
        pts = []
        for line in text.splitlines():
            if "Failed prediction at point" in line:
                m = re.search(r"\[([^\]]+)] with true label ([\d.]+)", line)
                if m:
                    nums = [float(x.strip()) for x in m.group(1).split(",")]
                    pts.append({"point": nums, "true_label": float(m.group(2))})
        return pts


if __name__ == "__main__":
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
    error = reviewer.test_code(
        code=sample_code,
        algorithm_name="IForest",
        package_name="pyod",
        train_dataset="./data/glass_train.mat",
    )
    if error:
        print("Test failed:", error)
    else:
        print("Test passed")
