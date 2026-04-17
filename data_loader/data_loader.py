import json
import os
import time

import numpy as np


class DataLoader:
    """
    Lightweight dataset inspector used by AgentSelector.

    This class no longer asks an LLM to generate loader scripts. It only does
    enough deterministic local loading to infer metadata and route the task;
    final experiment scripts should load files directly from code-generator
    prompts.
    """

    LABEL_NAMES = {"label", "labels", "target", "targets", "y", "class", "anomaly", "outlier"}
    TIME_NAMES = {"time", "timestamp", "date", "datetime"}

    def __init__(
        self,
        filepath,
        desc="",
        store_script=False,
        store_path="generated_data_loader.py",
        max_retries=3,
        retry_interval=1.0,
    ):
        self.filepath = filepath
        self.desc = desc
        self.max_retries = max_retries
        self.retry_interval = retry_interval
        self.store_script = store_script
        self.store_path = store_path
        self.X_name = "X"
        self.y_name = "y"

        file_type = os.path.splitext(str(filepath))[1].lstrip(".")
        self.metadata = {
            "filepath": filepath,
            "file_type": file_type,
            "head": None,
            "data_kind": None,
            "is_graph": False,
            "is_time_series": False,
            "is_unsupervised": False,
            "x_shape": None,
            "y_shape": None,
            "n_samples": None,
            "n_features": None,
            "columns": None,
            "feature_columns": None,
            "label_column": None,
            "time_column": None,
            "mat_keys": None,
            "error": None,
        }

        attempt = 0
        while attempt < self.max_retries:
            if os.path.exists(self.filepath):
                break
            attempt += 1
            print(f"File not found (attempt {attempt}/{self.max_retries}): {self.filepath}")
            if attempt < self.max_retries:
                time.sleep(self.retry_interval)
        else:
            raise FileNotFoundError(
                f"File not found after {self.max_retries} attempts: {self.filepath}"
            )

    @staticmethod
    def _shape_tuple(value):
        if hasattr(value, "shape"):
            return tuple(value.shape)
        return None

    @staticmethod
    def _as_2d_array(value):
        arr = np.asarray(value)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        return arr

    def _update_metadata(self, X=None, y=None, data_kind=None, error=None, head=None):
        if head is not None:
            self.metadata["head"] = str(head)
        if error is not None:
            self.metadata["error"] = str(error)
        if data_kind is not None:
            self.metadata["data_kind"] = data_kind

        if X is not None:
            x_shape = self._shape_tuple(X)
            self.metadata["x_shape"] = x_shape
            if x_shape:
                self.metadata["n_samples"] = x_shape[0]
                self.metadata["n_features"] = x_shape[1] if len(x_shape) > 1 else 1

        if y is not None:
            self.metadata["y_shape"] = self._shape_tuple(y)

        if isinstance(y, str):
            self.metadata["is_graph"] = y == "graph"
            self.metadata["is_time_series"] = y == "time-series"
            self.metadata["is_unsupervised"] = y == "Unsupervised"
        elif y is None:
            self.metadata["is_unsupervised"] = True

    def _load_csv(self):
        import pandas as pd

        df = pd.read_csv(self.filepath).dropna()
        columns = list(df.columns)
        lowered = {col: str(col).strip().lower() for col in columns}
        label_col = next((col for col in columns if lowered[col] in self.LABEL_NAMES), None)
        time_col = next(
            (
                col
                for col in columns
                if lowered[col] in self.TIME_NAMES
                or "timestamp" in lowered[col]
                or lowered[col].endswith("_time")
            ),
            None,
        )

        drop_cols = [col for col in (label_col, time_col) if col is not None]
        feature_df = df.drop(columns=drop_cols, errors="ignore").select_dtypes(include=[np.number])
        X = feature_df.to_numpy(dtype=float)
        y = df[label_col].to_numpy().ravel() if label_col else None

        self.metadata.update(
            columns=columns,
            feature_columns=list(feature_df.columns),
            label_column=label_col,
            time_column=time_col,
        )
        self._update_metadata(head=df.head().to_string(), X=X, y=y)
        if time_col:
            self._update_metadata(X, "time-series", data_kind="time-series")
            return X, "time-series"
        if y is None:
            self._update_metadata(X, None, data_kind="unsupervised")
            return X, None
        self._update_metadata(X, y, data_kind="tabular")
        return X, y

    def _load_mat(self):
        import scipy.io

        data = scipy.io.loadmat(self.filepath)
        keys = [key for key in data.keys() if not key.startswith("__")]
        self.metadata["mat_keys"] = keys
        head = {key: self._shape_tuple(data[key]) for key in keys}

        x_key = next((key for key in keys if key.lower() in {"x", "data", "features"}), None)
        y_key = next((key for key in keys if key.lower() in self.LABEL_NAMES), None)

        if x_key is None:
            x_key = next(
                (
                    key
                    for key in keys
                    if hasattr(data[key], "shape")
                    and np.asarray(data[key]).ndim >= 2
                    and np.issubdtype(np.asarray(data[key]).dtype, np.number)
                ),
                None,
            )
        if x_key is None:
            raise ValueError(f"Could not infer feature array from MAT keys: {keys}")

        X = self._as_2d_array(data[x_key])
        y = np.asarray(data[y_key]).ravel() if y_key else None
        self.metadata.update(feature_columns=[x_key], label_column=y_key)
        self._update_metadata(head=head, X=X, y=y)
        if y is None:
            self._update_metadata(X, None, data_kind="unsupervised")
            return X, None
        self._update_metadata(X, y, data_kind="tabular")
        return X, y

    def _load_json(self):
        with open(self.filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            X_raw = data.get("X") or data.get("x") or data.get("data") or data.get("features")
            y_raw = data.get("y") or data.get("label") or data.get("labels") or data.get("target")
            if X_raw is None:
                raise ValueError("Could not infer feature data from JSON keys.")
            X = self._as_2d_array(X_raw)
            y = np.asarray(y_raw).ravel() if y_raw is not None else None
            self._update_metadata(head=list(data.keys()), X=X, y=y)
        else:
            X = self._as_2d_array(data)
            y = None
            self._update_metadata(head=type(data).__name__, X=X, y=None)

        if y is None:
            self._update_metadata(X, None, data_kind="unsupervised")
            return X, None
        self._update_metadata(X, y, data_kind="tabular")
        return X, y

    def _load_npy(self):
        data = np.load(self.filepath, allow_pickle=True)
        if isinstance(data, np.lib.npyio.NpzFile):
            keys = list(data.keys())
            x_key = next((key for key in keys if key.lower() in {"x", "data", "features"}), keys[0])
            y_key = next((key for key in keys if key.lower() in self.LABEL_NAMES), None)
            X = self._as_2d_array(data[x_key])
            y = np.asarray(data[y_key]).ravel() if y_key else None
            self.metadata.update(mat_keys=keys, feature_columns=[x_key], label_column=y_key)
            self._update_metadata(head={key: self._shape_tuple(data[key]) for key in keys}, X=X, y=y)
        else:
            X = self._as_2d_array(data)
            y = None
            self._update_metadata(head={"shape": self._shape_tuple(data)}, X=X, y=None)
        self._update_metadata(X, y, data_kind="time-series")
        return X, "time-series" if y is None else y

    def _load_pt(self):
        import torch

        X = torch.load(self.filepath, weights_only=False)
        self._update_metadata(X, "graph", data_kind="graph", head="graph")
        return X, "graph"

    def load_data(self, split_data=False):
        _, ext = os.path.splitext(str(self.filepath).lower())
        try:
            if ext == ".csv":
                X, y = self._load_csv()
            elif ext == ".mat":
                X, y = self._load_mat()
            elif ext in {".json", ".jsonl"}:
                X, y = self._load_json()
            elif ext in {".npy", ".npz"}:
                X, y = self._load_npy()
            elif ext == ".pt":
                X, y = self._load_pt()
            else:
                raise ValueError(f"Unsupported dataset file type: {ext or '<none>'}")

            if split_data:
                if y is None or isinstance(y, str):
                    return X, None, None, None
                if X.shape[0] != y.shape[0]:
                    print(
                        f"Error: Mismatched samples. X has {X.shape[0]} rows, y has {y.shape[0]} rows."
                    )
                    return None, None, None, None
                import sklearn.model_selection

                X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                self._update_metadata(X_train, y_train, data_kind="tabular_split")
                return X_train, X_test, y_train, y_test
            return X, y
        except Exception as e:
            self._update_metadata(error=e)
            print(f"Error loading dataset metadata from {self.filepath}: {e}")
            return (None, None, None, None) if split_data else (None, None)


if __name__ == "__main__":
    data_loader = DataLoader("data/train.csv", store_script=False)
    X_train, y_train = data_loader.load_data(split_data=False)
    print(X_train)
    print(y_train)
    print(data_loader.metadata)
