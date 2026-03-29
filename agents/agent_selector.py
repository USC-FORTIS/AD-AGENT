# from langchain_community.vectorstores import FAISS
# from langchain_openai import OpenAIEmbeddings
# from langchain.text_splitter import CharacterTextSplitter
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data_loader.data_loader import DataLoader

from ad_model_selection.prompts.pygod_ms_prompt import generate_model_selection_prompt_from_pygod
from ad_model_selection.prompts.pyod_ms_prompt import generate_model_selection_prompt_from_pyod
from ad_model_selection.prompts.timeseries_ms_prompt import generate_model_selection_prompt_from_timeseries
from utils.openai_client import query_openai


class AgentSelector:
    def __init__(self, user_input):
        self.parameters = user_input["parameters"]
        self.data_path_train = user_input["dataset_train"]
        self.data_path_test = user_input["dataset_test"]
        self.user_input = user_input
        self.feature_dim = None
        self.algorithm = None

        self.load_data(self.data_path_train, self.data_path_test)

        self.tools = None
        self.set_tools()

        print(f"Package name: {self.package_name}")
        print(f"Algorithm: {self.algorithm}")
        print(f"Tools: {self.tools}")

    def load_data(self, train_path, test_path):
        train_loader = DataLoader(train_path, store_script=True, store_path="train_data_loader.py")
        x_train, y_train = train_loader.load_data(split_data=False)
        self.X_train = x_train
        self.y_train = y_train
        print(
            f"Loaded training data from {train_path}. "
            f"X_train shape: {getattr(x_train, 'shape', 'N/A')}, "
            f"y_train shape: {getattr(y_train, 'shape', 'N/A')}"
        )
        if hasattr(x_train, "shape") and len(x_train.shape) > 1:
            print(f"Dimension: {x_train.shape[1]}")
            self.feature_dim = x_train.shape[1]

        if test_path and os.path.exists(test_path):
            test_loader = DataLoader(test_path, store_script=True, store_path="test_data_loader.py")
            x_test, y_test = test_loader.load_data(split_data=False)
            self.X_test = x_test
            self.y_test = y_test
        else:
            self.X_test = None
            self.y_test = None

        if isinstance(self.X_train, str) and self.X_train == "tslib":
            self.package_name = "tslib"
        elif train_path.endswith(".npy"):
            self.package_name = "tslib"
            if self.X_train is not None and len(self.X_train.shape) > 1:
                num_features = self.X_train.shape[1]
                self.parameters["enc_in"] = num_features
                self.parameters["c_out"] = num_features
        elif train_path.endswith(".pt") or (isinstance(y_train, str) and y_train == "graph"):
            self.package_name = "pygod"
        elif isinstance(y_train, str) and y_train == "time-series":
            self.package_name = "darts"
        else:
            self.package_name = "pyod"

    def set_tools(self):
        user_input = self.user_input
        if user_input["algorithm"]:
            self.tools = self.generate_tools(user_input["algorithm"])
            self.algorithm = user_input["algorithm"]
            return

        algorithm = None
        name = os.path.basename(self.data_path_train)
        if self.package_name == "pyod":
            size = self.X_train.shape[0]
            dim = self.X_train.shape[1]
            messages = generate_model_selection_prompt_from_pyod(name, size, dim)
            content = query_openai(messages, model="o4-mini")
            algorithm = json.loads(content)["choice"]
            print(f"Algorithm: {algorithm}")
        elif self.package_name == "pygod":
            num_node = self.X_train.num_nodes
            num_edge = self.X_train.num_edges
            num_feature = self.X_train.num_features
            avg_degree = num_edge / num_node
            print(
                f"num_node: {num_node}, num_edge: {num_edge}, "
                f"num_feature: {num_feature}, avg_degree: {avg_degree}"
            )
            messages = generate_model_selection_prompt_from_pygod(
                name,
                num_node,
                num_edge,
                num_feature,
                avg_degree,
            )
            content = query_openai(messages, model="o4-mini")
            algorithm = json.loads(content)["choice"]
            print(f"Algorithm: {algorithm}")
        else:
            if self.X_train is not None and not isinstance(self.X_train, str):
                print("Shape of X_train:", self.X_train.shape)
                dim = 1
                if len(self.X_train.shape) > 1:
                    num_features = self.X_train.shape[1]
                    self.parameters["enc_in"] = num_features
                    dim = num_features
                ts_type = "multivariate" if dim > 1 else "univariate"

                num_signals = len(self.X_train)
                messages = generate_model_selection_prompt_from_timeseries(
                    name,
                    num_signals,
                    dim,
                    ts_type,
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
    for loader_file in (
        "train_data_loader.py",
        "test_data_loader.py",
        "head_train_data_loader.py",
        "head_test_data_loader.py",
    ):
        if os.path.exists(loader_file):
            os.remove(loader_file)

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
    print("Parameters:", agent_selector.parameters)
