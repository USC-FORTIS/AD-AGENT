from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
import re
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from entity.code_quality import CodeQuality
import subprocess
from openai import OpenAI
import os
from config.config import Config
os.environ['OPENAI_API_KEY'] = Config.OPENAI_API_KEY

# Initialize OpenAI LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0)

template_pyod = PromptTemplate.from_template("""
You are an expert Python developer with deep experience in anomaly detection libraries. Your task is to:

1. Use the provided official documentation content for `{algorithm}` to understand how to use the specified algorithm class, including initialization, training, and prediction methods.
2. Write only executable Python code for anomaly detection using PyOD and do not include any explanations or descriptions.
3. Base your code strictly on the following official documentation excerpt:

--- BEGIN DOCUMENTATION ---
{algorithm_doc}
--- END DOCUMENTATION ---

4. The code should:
   (1) import sys, os and include command `sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))`
   (2) Import DataLoader using following commend `from data_loader.data_loader import DataLoader`
   (3) Initialize DataLoader using statement `dataloader_train = DataLoader(filepath = {data_path_train}, store_script=True, store_path = 'train_data_loader.py')` & `dataloader_test = DataLoader(filepath = {data_path_test}, store_script=True, store_path = 'test_data_loader.py')`
   (4) Use the statement `X_train, y_train = dataloader_train.load_data(split_data=False)` & `X_test, y_test = dataloader_train.load_data(split_data=False)` to generate variables X_train, y_train, X_test, y_test; 
   (5) Initialize the specified algorithm `{algorithm}` strictly following the provided documentation and train the model with `X_train`
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

template_pygod = PromptTemplate.from_template("""
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
   (4) Initialize the specified algorithm `{algorithm}` with the provided parameters `{parameters}` (if applicable) strictly following the documentation excerpt.
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


web_search_prompt = PromptTemplate.from_template("""
   You are a machine learning expert and will assist me with researching a specific use of a deep learning model in PyOD. Here is the official document you should refer to: https://pyod.readthedocs.io/en/latest/pyod.models.html
   I want to run `{algorithm_name}`. What is the Initialization function, parameters and Attributes? 
   Briefly return realted document content.
""")

class AgentInstructor:
   def __init__(self):
      pass
   
   def execute_generated_code(self,code: str,algorithm_name):

      folder_path = "./generated_scripts"
      os.makedirs(folder_path, exist_ok=True)

      file_path = os.path.join(folder_path, f"{algorithm_name}.py")

      with open(file_path, "w") as script_file:
            script_file.write(code)
      try:
         result = subprocess.run(
               ["python", file_path],
               capture_output=True,
               text=True
         )
         print("\n=== Coding Output ===\n")
         print(result.stdout)
         print(result.stderr)

         if result.returncode != 0:
            return CodeQuality(
               code=code,
               algorithm=None,
               error_message= result.stderr,
               auroc=-1,
               auprc=-1,
               error_points=[],
               review_count=0
            )
         else:
            auroc, auprc, error_points = self.extract_eval(result.stdout)
            return CodeQuality(
               code=code,
               algorithm=None,
               error_message="",
               auroc=auroc,
               auprc=auprc,
               error_points=error_points, # list of dicts with point and true_label
               review_count=0
            )
      except Exception as e:
         print(str(e))
         return CodeQuality(
            code=code,
            algorithm=None,
            error_message=str(e),
            auroc=-1,
            auprc=-1,
            error_points=[],
            review_count=0
         )
      
      
   def extract_eval(self,output: str):
      auroc = -1
      auprc = -1
      error_points = []

      for line in output.splitlines():
         if "AUROC:" in line:
            match = re.search(r"AUROC:\s*([\d.]+)", line)
            if match:
               auroc = float(match.group(1))
         elif "AUPRC:" in line:
            match = re.search(r"AUPRC:\s*([\d.]+)", line)
            if match:
               auprc = float(match.group(1))
         elif "Failed prediction at point" in line:
            match = re.search(r"Failed prediction at point \[([^\]]+)\] with true label ([\d\.]+)\.?", line)
            if match:
               numbers_str = match.group(1)
               true_label = float(match.group(2))
               numbers = [float(num.strip()) for num in numbers_str.split(',')]
               error_points.append({
                     "point": numbers,
                     "true_label": true_label
               })
      return auroc, auprc, error_points
   
   def clean_generated_code(self, code):
      """Removes Markdown code block formatting from LLM output."""
      clean_code = re.sub(r"```(python)?", "", code)
      clean_code = re.sub(r"```", "", clean_code)
      return clean_code.strip()

   def generate_code(self, algorithm, data_path_train="./data/glass_train.mat",data_path_test = "./data/glass_test.mat", vectorstore=None, input_parameters = {},package_name = None):
      """Generates Python code for anomaly detection using PyOD, using external documentation."""
      algorithm_doc = self.query_docs(algorithm, vectorstore, package_name)
      print("\n=== Extracted Documentation ===\n")
      print(algorithm_doc)
      generated_code = ""
      if package_name == "pyod":
         generated_code = llm.invoke(
            template_pyod.invoke({
                  "algorithm": algorithm,
                  "data_path_train": data_path_train,
                  "data_path_test": data_path_test,
                  "algorithm_doc": algorithm_doc,
                  "parameters": str(input_parameters)
            })
         ).content
      else:
         generated_code = llm.invoke(
            template_pygod.invoke({
                  "algorithm": algorithm,
                  "data_path_train": data_path_train,
                  "data_path_test": data_path_test,
                  "algorithm_doc": algorithm_doc,
                  "parameters": str(input_parameters)
            })
         ).content


      return self.clean_generated_code(generated_code)
   def query_docs(self, algorithm, vectorstore, package_name):
      """Searches for relevant documentation based on the query."""
      # Query using RAG
      query = ""
      if package_name == "pyod":
         query = f"class pyod.models.{algorithm}.{algorithm}"
      else:
         query = f"class pygod.detector.{algorithm}"
      doc_list = vectorstore.similarity_search(query, k=3)
      algorithm_doc = "\n\n".join([doc.page_content for doc in doc_list])

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
      return algorithm_doc


if __name__ == "__main__":
   agentInstructor = AgentInstructor()
   from agent_planner import AgentPlanner
   user_input = {
      "algorithm": ["CARD"],
      "dataset_train": "./data/inj_cora_train.pt",
      "dataset_test": "./data/inj_cora_test.pt",
      "parameters": {}
   }
   agentPlanner = AgentPlanner(user_input=user_input)# if want to unit test, please import AgentPlanner
   vectorstore = agentPlanner.vectorstore

   code = agentInstructor.generate_code(algorithm=agentPlanner.tools[0], data_path_train = agentPlanner.data_path_train, data_path_test=agentPlanner.data_path_test, vectorstore = vectorstore, input_parameters = agentPlanner.parameters, package_name = agentPlanner.package_name)

   print(agentInstructor.execute_generated_code(code,agentPlanner.tools[0]))
