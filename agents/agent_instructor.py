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

# promt template
# template = PromptTemplate.from_template("""
# You are an expert Python developer with deep experience in anomaly detection libraries. Your task is to:

# 1. Use the provided official documentation content for `{algorithm}` to understand how to use the specified algorithm class, including initialization, training, and prediction methods.
# 2. Write only executable Python code for anomaly detection using PyOD and do not include any explanations or descriptions.
# 3. Base your code strictly on the following official documentation excerpt:

# --- BEGIN DOCUMENTATION ---
# {algorithm_doc}
# --- END DOCUMENTATION ---

# 4. The code should:
#    (1) Notice that some algorithm might not appliable to current data (e.g. algorithm strictly for univariate data) please determin if `{data_path}` suitable for `{algorithm}` from PyOD documentation, if that not suitable please directly print why it is not appliable.
#    (2) Load the data from `{data_path}`
#    (3) Extract the feature matrix `X` and labels `y` from the loaded data
#    (4) Split `X` into `X_train` (80%) and `X_test` (20%), `y` into `y_train` (80%) and `y_test` (20%)
#    (5) Initialize the specified algorithm `{algorithm}` strictly following the provided documentation and train the model with `X_train`
#    (6) Determine whether the following parameters `{parameters}` apply to this initialization function and, if so, add their values ​​to the function.
#    (7) Use `.decision_scores_` on `X_train` for training outlier scores
#        Use `.decision_function(X_test)` for test outlier scores
#    (8) Using variables to record the number of detected and ture anomalies in test data and print them out in following format:
#        Detected anomalies:\s*(\d+)
#        True anomalies:\s*(\d+)
                     

# IMPORTANT: 
# - Strictly follow steps (2)-(4) to load the data from `{data_path}`.
# - Do NOT input optional or incorrect parameters.
# """)
template = PromptTemplate.from_template("""
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
   (6) Determine whether the following parameters `{parameters}` apply to this initialization function and, if so, add their values ​​to the function.
   (7) Use `.decision_scores_` on `X_train` for training outlier scores
       Use `.decision_function(X_test)` for test outlier scores
   (8) Using variables to record the number of detected and ture anomalies in test data and print them out in following format:
       Detected anomalies:\s*(\d+)
       True anomalies:\s*(\d+)
                     

IMPORTANT: 
- Strictly follow steps (2)-(4) to load the data from `{data_path_train}`.
- Do NOT input optional or incorrect parameters.
""")

web_search_prompt_system = "You are a machine learning expert and will assist me with researching a specific use of a deep learning model in PyOD. Here is the official document you should refer to: https://pyod.readthedocs.io/en/latest/pyod.models.html"
web_search_prompt_user = PromptTemplate.from_template("""I want to run `{algorithm_name}`. What is the Initialization function, parameters and Attributes? Please return to the document content directly.""")

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
                  detected_anomalies=-1,
                  true_anomalies=-1,
                  review_count=0
               )
         else:
               detected_anomalies, true_anomalies = self.extract_anomalies(result.stdout)
               return CodeQuality(
                  code=code,
                  algorithm=None,
                  error_message="",
                  detected_anomalies=detected_anomalies,
                  true_anomalies=true_anomalies,
                  review_count=0
               )
      except Exception as e:
          return CodeQuality(
            code=code,
            algorithm=None,
            error_message=str(e),
            detected_anomalies=-1,
            true_anomalies=-1,
            review_count=0
        )
      
   def extract_anomalies(self,output: str):
      detected_anomalies = -1
      true_anomalies = -1

      for line in output.splitlines():
         if "Detected anomalies:" in line:
               match = re.search(r"Detected anomalies:\s*(\d+)", line)
               if match:
                  detected_anomalies = int(match.group(1))

         elif "True anomalies:" in line:
               match = re.search(r"True anomalies:\s*(\d+)", line)
               if match:
                  true_anomalies = int(match.group(1))

      return detected_anomalies, true_anomalies
   
   def clean_generated_code(self, code):
      """Removes Markdown code block formatting from LLM output."""
      clean_code = re.sub(r"```(python)?", "", code)
      clean_code = re.sub(r"```", "", clean_code)
      return clean_code.strip()

   def generate_code(self, algorithm, data_path_train="./data/glass_train.mat",data_path_test = "./data/glass_test.mat", vectorstore=None, input_parameters = {}):
      """Generates Python code for anomaly detection using PyOD, using external documentation."""
      algorithm_doc = self.query_docs(algorithm, vectorstore)
      print("\n=== Extracted Documentation ===\n")
      print(algorithm_doc)

      generated_code = llm.invoke(
         template.invoke({
               "algorithm": algorithm,
               "data_path_train": data_path_train,
               "data_path_test": data_path_test,
               "algorithm_doc": algorithm_doc,
               "parameters": str(input_parameters)
         })
      ).content

      return self.clean_generated_code(generated_code)
   def query_docs(self, algorithm, vectorstore):
      """Searches for relevant documentation based on the query."""
      # Query using RAG
      # query = f"class pyod.models.{algorithm}.{algorithm}"
      # doc_list = vectorstore.similarity_search(query, k=1)
      # algorithm_doc = "\n\n".join([doc.page_content for doc in doc_list])
      client = OpenAI()
      
      response = client.responses.create(
         model="gpt-4o",
         input=[
            {
               "role": "system",
               "content": [
               {
                  "type": "input_text",
                  "text": web_search_prompt_system
               }
               ]
            },
            {
               "role": "user",
               "content": [
               {
                  "type": "input_text",
                  "text": web_search_prompt_user.invoke({"algorithm_name": algorithm}).to_string()
               }
               ]
            }
         ],
         text={
            "format": {
               "type": "text"
            }
         },
         reasoning={},
         tools=[
            {
               "type": "web_search_preview",
               "user_location": {
               "type": "approximate"
               },
               "search_context_size": "medium"
            }
         ],
         temperature=0,
         max_output_tokens=16384,
         top_p=1,
         store=True
         )
      try:
         algorithm_doc = response.output[1].content[0].text
      except (IndexError, AttributeError) as error:
         print(response)
         algorithm_doc = ""
      return algorithm_doc


if __name__ == "__main__":
   agentInstructor = AgentInstructor()
   from agent_planner import AgentPlanner
   user_input = {
      "algorithm": ["IForest"],
      "dataset_train": "./data/glass_train.mat",
      "dataset_test": "./data/glass_test.mat",
      "parameters": {
         "contamination": 0.1
      }
   }
   agentPlanner = AgentPlanner(user_input=user_input)# if want to unit test, please import AgentPlanner
   vectorstore = agentPlanner.vectorstore

   code = agentInstructor.generate_code(algorithm=agentPlanner.tools[0], data_path_train = agentPlanner.data_path_train, data_path_test=agentPlanner.data_path_test, vectorstore = vectorstore, input_parameters = agentPlanner.parameters)

   print(agentInstructor.execute_generated_code(code,agentPlanner.tools[0]))
