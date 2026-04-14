# from langchain_community.vectorstores import FAISS
# from langchain_openai import OpenAIEmbeddings
# from langchain.text_splitter import CharacterTextSplitter
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_loader.data_loader import DataLoader

from ad_model_selection.prompts.pygod_ms_prompt import generate_model_selection_prompt_from_pygod
from ad_model_selection.prompts.pyod_ms_prompt import generate_model_selection_prompt_from_pyod
from ad_model_selection.prompts.tsb_ad_ms_prompt import generate_model_selection_prompt_from_tsb_ad
from utils.openai_client import query_openai
from utils.tsb_ad_registry import TSB_AD_SINGLE_INPUT_ALGORITHMS, Unsupervise_AD_Pool
from utils.algorithm_registry import PYGOD_ALGORITHMS, PYOD_ALGORITHMS
import json

class AgentSelector:
    def __init__(self, user_input):
      self.parameters = user_input['parameters']
      self.data_path_train = user_input['dataset_train']
      self.data_path_test = user_input['dataset_test']
      self.user_input = user_input
      self.algorithm = None
      self.train_metadata = {}
      self.test_metadata = None
      self.dataset_metadata = {"train": self.train_metadata, "test": self.test_metadata}
      self.metadata = {}

      self.load_data(self.data_path_train, self.data_path_test)

      self.tools = None
      self.set_tools()

      print(f"Package name: {self.package_name}")
      print(f"Algorithm: {self.algorithm}")
      print(f"Tools: {self.tools}")

      
      # self.documents = self.load_and_split_documents()
      # self.vectorstore = self.build_vectorstore(self.documents)

    def load_data(self, train_path, test_path):
      train_loader = DataLoader(train_path, store_script=False)
      X_train, y_train = train_loader.load_data(split_data=False)
      self.train_metadata = getattr(train_loader, "metadata", {}) or {}
      self.X_train = X_train
      self.y_train = y_train
      print(f"Loaded training data from {train_path}. X_train shape: {getattr(X_train, 'shape', 'N/A')}, y_train shape: {getattr(y_train, 'shape', 'N/A')}")
      # Only load test data if test_path is provided and not empty
      if test_path and os.path.exists(test_path):
          test_loader = DataLoader(test_path, store_script=False)
          X_test, y_test = test_loader.load_data(split_data=False)
          self.test_metadata = getattr(test_loader, "metadata", {}) or {}
          self.X_test = X_test
          self.y_test = y_test
      else:
          self.test_metadata = None
          self.X_test = None
          self.y_test = None

      if train_path.endswith('.pt') or (isinstance(y_train, str) and y_train == 'graph'):
        self.package_name = "pygod"
      elif train_path.endswith('.npy') or train_path.endswith('.npz') or (isinstance(y_train, str) and y_train == 'time-series'):
        self.package_name = "tsb_ad"
        if self.X_train is not None and hasattr(self.X_train, "shape") and len(self.X_train.shape) > 1:
          num_features = self.X_train.shape[1]
          self.parameters['enc_in'] = num_features
          self.parameters['c_out'] = num_features
      else:
        self.package_name = "pyod"


      self.dataset_metadata = {
        "train": self.train_metadata,
        "test": self.test_metadata,
      }
      self.metadata = {
        "dataset": self.dataset_metadata,
        "package_name": self.package_name,
        "parameters": self.parameters,
      }

    def set_tools(self):
      user_input = self.user_input
      if user_input['algorithm'] or (user_input['algorithm'] and user_input['algorithm'][0].lower() == "all"):
        self.tools = self.generate_tools(user_input['algorithm'])
        self.algorithm = user_input['algorithm']
      else:
        algorithm = None
        name = os.path.basename(self.data_path_train)
        if self.package_name == "pyod":
          size = self.X_train.shape[0]
          dim = self.X_train.shape[1]
          messages = generate_model_selection_prompt_from_pyod(name, size, dim)
          content = query_openai(messages, model="o4-mini")
          algorithm = json.loads(content)["choice"]
          print(f"Algorithm: {algorithm}")
        elif self.package_name == 'pygod':
          num_node = self.X_train.num_nodes
          num_edge = self.X_train.num_edges
          num_feature = self.X_train.num_features
          avg_degree = num_edge / num_node
          print(f"num_node: {num_node}, num_edge: {num_edge}, num_feature: {num_feature}, avg_degree: {avg_degree}")
          messages = generate_model_selection_prompt_from_pygod(name, num_node, num_edge, num_feature, avg_degree)
          content = query_openai(messages, model="o4-mini")
          algorithm = json.loads(content)["choice"]
          print(f"Algorithm: {algorithm}")
        elif self.package_name == "tsb_ad":
          dim = 1
          if self.X_train is not None and hasattr(self.X_train, "shape") and len(self.X_train.shape) > 1:
            dim = self.X_train.shape[1]
          ts_type = "multivariate" if dim > 1 else "univariate"
          num_signals = len(self.X_train) if self.X_train is not None else 0
          has_test = bool(self.data_path_test and self.X_test is not None)
          messages = generate_model_selection_prompt_from_tsb_ad(
            name, num_signals, dim, ts_type, has_test_dataset=has_test
          )
          content = query_openai(messages, model="o4-mini")
          algorithm = json.loads(content)["choice"]
          print(f"Algorithm: {algorithm}")
        else:
          algorithm = 'IForest'
        self.algorithm = [algorithm]
        self.tools = [algorithm]

        print('Selector Parameters:', self.parameters)

      # Apply dataset routing based on algorithm type (tsb_ad only)
      if self.package_name == "tsb_ad" and self.algorithm:
        final_algo = self.algorithm[0] if len(self.algorithm) == 1 else None
        if final_algo and final_algo in Unsupervise_AD_Pool:
          # Unsupervised: test dataset must mirror train dataset
          if self.X_test is None:
            self.X_test = self.X_train
            self.y_test = self.y_train
            self.test_metadata = self.train_metadata
            print(f"Unsupervised algorithm '{final_algo}': test dataset set to train dataset.")
        else:
          # Semisupervised: train and test datasets are set as-is (already loaded)
          pass

      self.metadata.update({
        "algorithm": self.algorithm,
        "tools": self.tools,
      })
        

    # def load_and_split_documents(self,folder_path="../docs"):
    #   """
    #   load ../docs txt doc, divided into small blocks。
    #   """
    #   documents = []
    #   text_splitter = CharacterTextSplitter(separator="\n", chunk_size=700, chunk_overlap=150)

    #   for filename in os.listdir(folder_path):
    #      if filename.startswith(self.package_name):
    #            file_path = os.path.join(folder_path, filename)
    #            with open(file_path, "r", encoding="utf-8") as file:
    #               text = file.read()
    #               chunks = text_splitter.split_text(text)
    #               documents.extend(chunks)

    #   return documents
    
    # def build_vectorstore(self,documents):
    #   """
    #   The segmented document blocks are converted into vectors and stored in the FAISS vector database.
    #   """
    #   embedding = OpenAIEmbeddings()
    #   vectorstore = FAISS.from_texts(documents, embedding)
    #   return vectorstore
    
    def generate_tools(self,algorithm_input):
      """Generates the tools for the agent."""
      if algorithm_input[0].lower() == "all":
        if self.package_name == "pygod":
          return PYGOD_ALGORITHMS
        elif self.package_name == "pyod":
          return PYOD_ALGORITHMS
        elif self.package_name == "tsb_ad":
          return TSB_AD_SINGLE_INPUT_ALGORITHMS
        else:
          # return ['GlobalNaiveAggregate','GlobalNaiveDrift','GlobalNaiveSeasonal']
          return ["GlobalNaiveAggregate","GlobalNaiveDrift","GlobalNaiveSeasonal","RNNModel","BlockRNNModel","NBEATSModel","NHiTSModel","TCNModel","TransformerModel","TFTModel","DLinearModel","NLinearModel","TiDEModel","TSMixerModel","LinearRegressionModel","RandomForest","LightGBMModel","XGBModel","CatBoostModel"]
      return algorithm_input

if __name__ == "__main__":
  import sys
  sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
  from config.config import Config
  os.environ['OPENAI_API_KEY'] = Config.OPENAI_API_KEY

  user_input = {
    "algorithm": ['TimesNet'],
    "dataset_train": "../data/demo_train.npy",
    "dataset_test": "../data/demo_test.npy",
    "parameters": {
    }
  }
  agentSelector = AgentSelector(user_input= user_input)
  print(f"Tools: {agentSelector.tools}")
  print('Parameters:', agentSelector.parameters)
