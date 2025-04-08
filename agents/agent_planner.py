from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
import os
from typing import List


class ToolPlan:
    def __init__(
        self,
        tool: str,
        package_name: str,
        data_path_train: str,
        data_path_test: str,
        parameters: dict,
        documentation: str
    ):
        self.tool = tool
        self.package_name = package_name
        self.data_path_train = data_path_train
        self.data_path_test = data_path_test
        self.parameters = parameters
        self.documentation = documentation

class AgentPlanner:
    def __init__(self, user_input):
      self.parameters = user_input['parameters']
      self.data_path_train = user_input['dataset_train']
      self.data_path_test = user_input['dataset_test']
      self.package_name = "pygod" if user_input['dataset_train'].endswith(".pt") else "pyod"
      self.tools = self.generate_tools(user_input['algorithm'])
      self.documents = self.load_and_split_documents()
      self.vectorstore = self.build_vectorstore(self.documents)
      self.tool_plans: List[ToolPlan] = self.build_tool_plans()

    def query_docs(self, algorithm: str):
      """Searches for relevant documentation based on the query."""
      # Query using RAG
      query = (
            f"class pyod.models.{algorithm}.{algorithm}"
            if self.package_name == "pyod"
            else f"class pygod.detector.{algorithm}"
        )
      docs = self.vectorstore.similarity_search(query, k=3)
      return "\n\n".join([doc.page_content for doc in docs])
      # client = OpenAI()
      # response = client.responses.create(
      #    model="gpt-4o",
      #    tools=[{"type": "web_search_preview"}],
      #    input= web_search_prompt.invoke({"algorithm_name": algorithm}).to_string(),
      #    max_output_tokens=2024
      # )
      # algorithm_doc = response.output_text
      # if not algorithm_doc:
      #    print("Error in response "+ algorithm)
      #    print(response)
      #    
      # return algorithm_doc
    
    def build_tool_plans(self) -> List[ToolPlan]:
      tool_plans = []
      for tool in self.tools:
        documentation = self.query_docs(tool)
        plan = ToolPlan(
            tool=tool,
            package_name=self.package_name,
            data_path_train=self.data_path_train,
            data_path_test=self.data_path_test,
            parameters=self.parameters,
            documentation=documentation
        )
        tool_plans.append(plan)
      return tool_plans
  
    def load_and_split_documents(self,folder_path="./docs"):
      """
      load ./docs txt doc, divided into small blocks。
      """
      documents = []
      text_splitter = CharacterTextSplitter(separator="\n", chunk_size=700, chunk_overlap=150)

      for filename in os.listdir(folder_path):
         if filename.startswith(self.package_name):
               file_path = os.path.join(folder_path, filename)
               with open(file_path, "r", encoding="utf-8") as file:
                  text = file.read()
                  chunks = text_splitter.split_text(text)
                  documents.extend(chunks)

      return documents
    
    def build_vectorstore(self,documents):
      """
      The segmented document blocks are converted into vectors and stored in the FAISS vector database.
      """
      embedding = OpenAIEmbeddings()
      vectorstore = FAISS.from_texts(documents, embedding)
      return vectorstore
    
    def generate_tools(self,algorithm_input):
      """Generates the tools for the agent."""
      if algorithm_input[0].lower() == "all":
        if self.package_name == "pygod":
          return ['SCAN','GAE','Radar','ANOMALOUS','ONE','DOMINANT','DONE','AdONE','AnomalyDAE','GAAN','DMGD','OCGNN','CoLA','GUIDE','CONAD','GADNR','CARD']
        else:
          return ['ECOD', 'SOD', 'ROD']
          #return ['ECOD', 'ABOD', 'FastABOD', 'COPOD', 'MAD', 'SOS', 'QMCD', 'KDE', 'Sampling', 'GMM', 'PCA', 'KPCA', 'MCD', 'CD', 'OCSVM', 'LMDD', 'LOF', 'COF', '(Incremental) COF', 'CBLOF', 'LOCI', 'HBOS', 'kNN', 'AvgKNN', 'MedKNN', 'SOD', 'ROD', 'IForest', 'INNE', 'DIF', 'FeatureBagging', 'LSCP', 'XGBOD', 'LODA', 'SUOD', 'AutoEncoder', 'VAE', 'Beta-VAE', 'SO_GAAL', 'MO_GAAL', 'DeepSVDD', 'AnoGAN', 'ALAD', 'AE1SVM', 'DevNet', 'R-Graph', 'LUNAR']
      return algorithm_input
    
    

if __name__ == "__main__":
  import sys
  sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
  from config.config import Config
  os.environ['OPENAI_API_KEY'] = Config.OPENAI_API_KEY

  user_input = {
        "algorithm": ["IForest"],
        "dataset_train": "./data/glass_train.mat",
        "dataset_test": "./data/glass_test.mat",
        "parameters": {"contamination": 0.1}
    }

    # Instantiate planner
  agent_planner = AgentPlanner(user_input)

    # Debug print: planner results
  print(f"\n Package backend selected: {agent_planner.package_name}")
  print(f" Tools to run: {agent_planner.tools}")
  print(f" Number of document chunks loaded: {len(agent_planner.documents)}")
  print(f" Number of ToolPlans built: {len(agent_planner.tool_plans)}")

    # Print each plan summary
  for i, plan in enumerate(agent_planner.tool_plans):
      print(f"\n=== ToolPlan {i+1}: {plan.tool} ===")
      print(f"  ↪ Package: {plan.package_name}")
      print(f"  ↪ Train Path: {plan.data_path_train}")
      print(f"  ↪ Test Path: {plan.data_path_test}")        
      print(f"  ↪ Parameters: {plan.parameters}")
      print(f"  ↪ Documentation Snippet:\n{plan.documentation[:500]}...")  # print preview