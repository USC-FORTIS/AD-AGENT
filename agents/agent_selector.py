import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ad_model_selection.prompts.pygod_ms_prompt import generate_model_selection_prompt_from_pygod
from ad_model_selection.prompts.pyod_ms_prompt import generate_model_selection_prompt_from_pyod
from ad_model_selection.prompts.timeseries_ms_prompt import generate_model_selection_prompt_from_timeseries
from utils.openai_client import query_openai

# Known Time-Series-Library directory-based datasets
_TSLIB_DIR_DATASETS = {"MSL", "PSM", "SMAP", "SMD", "SWaT"}

# ---------------------------------------------------------------------------
# Deterministic metadata extraction scripts (executed in sandbox, no LLM)
# Each prints a single line: META:<json>
# ---------------------------------------------------------------------------

_METADATA_SCRIPT_PYOD = '''\
import json, os, sys
path = "{data_path}"
ext = os.path.splitext(path)[1].lower()
meta = {{"format": ext}}
try:
    if ext == ".mat":
        import scipy.io, numpy as np
        data = scipy.io.loadmat(path)
        keys = [k for k in data if not k.startswith("__")]
        X = data.get("X")
        y = data.get("y")
        if X is None:
            for c in ("data", "features", "input"):
                if c in data:
                    X = data[c]; break
        if y is None:
            for c in ("label", "labels", "target", "targets"):
                if c in data:
                    y = data[c]; break
        if X is None and keys:
            arrays = sorted(
                [(k, data[k]) for k in keys if hasattr(data[k], "shape")],
                key=lambda a: a[1].size, reverse=True,
            )
            if arrays:
                X = arrays[0][1]
                if y is None and len(arrays) > 1:
                    y = arrays[1][1]
        if X is not None:
            X = np.asarray(X)
            meta["num_samples"] = int(X.shape[0])
            meta["feature_dim"] = int(X.shape[1]) if X.ndim > 1 else 1
        meta["has_labels"] = y is not None
    elif ext == ".csv":
        import pandas as pd
        df = pd.read_csv(path)
        cols_lower = {{c.lower(): c for c in df.columns}}
        time_keys = {{"timestamp", "time", "date", "datetime", "ts"}}
        label_keys = {{"label", "anomaly", "target", "y", "class"}}
        excluded = time_keys | label_keys
        feature_cols = [
            c for c in df.columns
            if c.lower() not in excluded and pd.api.types.is_numeric_dtype(df[c])
        ]
        meta["num_samples"] = len(df)
        meta["feature_dim"] = len(feature_cols)
        meta["has_labels"] = bool(set(cols_lower) & label_keys)
        meta["has_timestamps"] = bool(set(cols_lower) & time_keys)
        meta["columns"] = list(df.columns)
    print("META:" + json.dumps(meta))
except Exception as e:
    print("META:" + json.dumps({{"error": str(e)}}))
'''

_METADATA_SCRIPT_PYGOD = '''\
import json, torch
path = "{data_path}"
try:
    data = torch.load(path, weights_only=False)
    meta = {{
        "format": ".pt",
        "num_nodes": int(data.num_nodes),
        "num_edges": int(data.num_edges),
        "num_features": int(data.num_features),
    }}
    print("META:" + json.dumps(meta))
except Exception as e:
    print("META:" + json.dumps({{"error": str(e)}}))
'''

_METADATA_SCRIPT_TSLIB = '''\
import json, numpy as np
path = "{data_path}"
try:
    arr = np.load(path)
    meta = {{
        "format": ".npy",
        "num_samples": int(arr.shape[0]),
        "feature_dim": int(arr.shape[1]) if arr.ndim > 1 else 1,
    }}
    print("META:" + json.dumps(meta))
except Exception as e:
    print("META:" + json.dumps({{"error": str(e)}}))
'''


class AgentSelector:
    def __init__(self, user_input):
        self.parameters = user_input["parameters"]
        self.data_path_train = user_input["dataset_train"]
        self.data_path_test = user_input["dataset_test"]
        self.user_input = user_input
        self.feature_dim = None
        self.algorithm = None

        self._inspect_and_load(self.data_path_train, self.data_path_test)

        self.tools = None
        self.set_tools()

        print(f"Package name: {self.package_name}")
        print(f"Algorithm: {self.algorithm}")
        print(f"Tools: {self.tools}")

    # ------------------------------------------------------------------
    # Step 1: Detect package from file extension / directory name
    # ------------------------------------------------------------------
    @staticmethod
    def _detect_package(train_path: str) -> str:
        basename = os.path.basename(train_path.rstrip("/"))
        if basename in _TSLIB_DIR_DATASETS:
            return "tslib"
        ext = os.path.splitext(train_path)[1].lower()
        if ext == ".npy":
            return "tslib"
        if ext == ".pt":
            return "pygod"
        # .mat and .csv default to pyod; csv may be reclassified after metadata
        return "pyod"

    # ------------------------------------------------------------------
    # Step 2: Run a deterministic metadata-extraction script in sandbox
    # ------------------------------------------------------------------
    # Volume mount point used by the sandbox executor
    _VOLUME_MOUNT = "/data"

    def _inspect_and_load(self, train_path, test_path):
        """Detect package type and extract dataset metadata via sandbox."""
        from sandbox.executor import execute_code

        self.package_name = self._detect_package(train_path)
        self.metadata = {}

        # tslib directory datasets have no single file to inspect
        if os.path.basename(train_path.rstrip("/")) in _TSLIB_DIR_DATASETS:
            self.metadata = {"num_samples": None, "feature_dim": None}
            return

        # Map local files → volume paths so they use reliable volume upload
        data_files, volume_train = self._build_data_files_for_volume(train_path)
        script = self._build_metadata_script(volume_train, None, self.package_name)

        print(f"[Selector] Extracting metadata in sandbox ({self.package_name})...")
        stdout, stderr, rc = execute_code(
            code=script,
            algorithm_name="_metadata_inspect",
            package_name=self.package_name,
            data_files=data_files,
            timeout=60,
        )

        if rc != 0:
            print(f"[Selector] Metadata extraction failed (rc={rc}): {stderr}")
            self.metadata = {}
            return

        # Parse the META: JSON line from stdout
        for line in stdout.splitlines():
            if line.startswith("META:"):
                try:
                    self.metadata = json.loads(line[5:])
                except json.JSONDecodeError:
                    print(f"[Selector] Failed to parse metadata JSON: {line}")
                break

        print(f"[Selector] Metadata: {self.metadata}")

        # Apply metadata to selector state
        self.feature_dim = self.metadata.get("feature_dim")

        # Reclassify csv with timestamps as darts
        if self.package_name == "pyod" and self.metadata.get("has_timestamps"):
            self.package_name = "darts"

        # Set tslib enc_in/c_out from feature_dim
        if self.package_name == "tslib" and self.feature_dim and self.feature_dim > 0:
            self.parameters["enc_in"] = self.feature_dim
            self.parameters["c_out"] = self.feature_dim

    # ------------------------------------------------------------------
    # Metadata extraction scripts (deterministic, no LLM)
    # ------------------------------------------------------------------
    @staticmethod
    def _build_metadata_script(train_path, test_path, package_name):
        """Build a small Python script that inspects the data and prints JSON metadata."""
        if package_name == "pygod":
            return _METADATA_SCRIPT_PYGOD.format(data_path=train_path)
        if package_name == "tslib":
            return _METADATA_SCRIPT_TSLIB.format(data_path=train_path)
        # pyod handles both .mat and .csv
        return _METADATA_SCRIPT_PYOD.format(data_path=train_path)

    @classmethod
    def _build_data_files_for_volume(cls, train_path):
        """Map local train file → volume path for reliable Modal volume upload.

        Returns (data_files_dict, volume_train_path).
        """
        data_files = {}
        abs_path = os.path.abspath(train_path)
        basename = os.path.basename(abs_path)
        volume_path = f"{cls._VOLUME_MOUNT}/{basename}"

        if os.path.isfile(abs_path):
            data_files[volume_path] = abs_path
        elif os.path.isdir(abs_path):
            for root, _dirs, files in os.walk(abs_path):
                for fname in files:
                    local = os.path.join(root, fname)
                    rel = os.path.relpath(local, abs_path)
                    remote = f"{volume_path}/{rel}"
                    data_files[remote] = os.path.abspath(local)

        return (data_files or None), volume_path

    # ------------------------------------------------------------------
    # Step 3: Model selection (uses metadata instead of raw arrays)
    # ------------------------------------------------------------------
    def set_tools(self):
        user_input = self.user_input
        if user_input["algorithm"]:
            self.tools = self.generate_tools(user_input["algorithm"])
            self.algorithm = user_input["algorithm"]
            return

        algorithm = None
        name = os.path.basename(self.data_path_train)
        meta = self.metadata

        if self.package_name == "pyod":
            size = meta.get("num_samples", 0)
            dim = meta.get("feature_dim", 1)
            if size and dim:
                messages = generate_model_selection_prompt_from_pyod(name, size, dim)
                content = query_openai(messages, model="o4-mini")
                algorithm = json.loads(content)["choice"]
                print(f"Algorithm: {algorithm}")
            else:
                algorithm = "IForest"  # safe default when metadata unavailable

        elif self.package_name == "pygod":
            num_node = meta.get("num_nodes", 0)
            num_edge = meta.get("num_edges", 0)
            num_feature = meta.get("num_features", 0)
            avg_degree = num_edge / num_node if num_node else 0
            if num_node:
                print(
                    f"num_node: {num_node}, num_edge: {num_edge}, "
                    f"num_feature: {num_feature}, avg_degree: {avg_degree}"
                )
                messages = generate_model_selection_prompt_from_pygod(
                    name, num_node, num_edge, num_feature, avg_degree,
                )
                content = query_openai(messages, model="o4-mini")
                algorithm = json.loads(content)["choice"]
                print(f"Algorithm: {algorithm}")
            else:
                algorithm = "DOMINANT"

        else:  # tslib / darts
            dim = meta.get("feature_dim")
            num_signals = meta.get("num_samples")
            if dim and num_signals:
                if self.package_name == "tslib" and dim > 0:
                    self.parameters["enc_in"] = dim
                ts_type = "multivariate" if dim > 1 else "univariate"
                messages = generate_model_selection_prompt_from_timeseries(
                    name, num_signals, dim, ts_type,
                )
                content = query_openai(messages, model="o4-mini")
                algorithm = json.loads(content)["choice"]
                print(f"Algorithm: {algorithm}")
            else:
                algorithm = "Autoformer"

        self.algorithm = [algorithm]
        self.tools = [algorithm]
        print("Selector Parameters:", self.parameters)

    def generate_tools(self, algorithm_input):
        """Generates the tools for the agent."""
        if algorithm_input[0].lower() == "all":
            if self.package_name == "pygod":
                return [
                    "SCAN",
                    "GAE",
                    "Radar",
                    "ANOMALOUS",
                    "ONE",
                    "DOMINANT",
                    "DONE",
                    "AdONE",
                    "AnomalyDAE",
                    "GAAN",
                    "DMGD",
                    "OCGNN",
                    "CoLA",
                    "GUIDE",
                    "CONAD",
                    "GADNR",
                    "CARD",
                ]
            if self.package_name == "pyod":
                return [
                    "ECOD",
                    "ABOD",
                    "FastABOD",
                    "COPOD",
                    "MAD",
                    "SOS",
                    "QMCD",
                    "KDE",
                    "Sampling",
                    "GMM",
                    "PCA",
                    "KPCA",
                    "MCD",
                    "CD",
                    "OCSVM",
                    "LMDD",
                    "LOF",
                    "COF",
                    "(Incremental) COF",
                    "CBLOF",
                    "LOCI",
                    "HBOS",
                    "kNN",
                    "AvgKNN",
                    "MedKNN",
                    "SOD",
                    "ROD",
                    "IForest",
                    "INNE",
                    "DIF",
                    "FeatureBagging",
                    "LSCP",
                    "XGBOD",
                    "LODA",
                    "SUOD",
                    "AutoEncoder",
                    "VAE",
                    "Beta-VAE",
                    "SO_GAAL",
                    "MO_GAAL",
                    "DeepSVDD",
                    "AnoGAN",
                    "ALAD",
                    "AE1SVM",
                    "DevNet",
                    "R-Graph",
                    "LUNAR",
                ]
            return [
                "GlobalNaiveAggregate",
                "GlobalNaiveDrift",
                "GlobalNaiveSeasonal",
                "RNNModel",
                "BlockRNNModel",
                "NBEATSModel",
                "NHiTSModel",
                "TCNModel",
                "TransformerModel",
                "TFTModel",
                "DLinearModel",
                "NLinearModel",
                "TiDEModel",
                "TSMixerModel",
                "LinearRegressionModel",
                "RandomForest",
                "LightGBMModel",
                "XGBModel",
                "CatBoostModel",
            ]
        return algorithm_input


if __name__ == "__main__":
    from config.config import Config

    os.environ["OPENAI_API_KEY"] = Config.OPENAI_API_KEY

    user_input = {
        "algorithm": ["TimesNet"],
        "dataset_train": "../data/MSL",
        "dataset_test": "../data/MSL",
        "parameters": {},
    }
    agent_selector = AgentSelector(user_input=user_input)
    print(f"Tools: {agent_selector.tools}")
    print(f"Metadata: {agent_selector.metadata}")
    print("Parameters:", agent_selector.parameters)
