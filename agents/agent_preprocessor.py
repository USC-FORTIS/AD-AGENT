# import openai
# import os

# class AgentPreprocessor:
#     def __init__(self, model="gpt-4", temperature=0):
#         self.model = model
#         self.temperature = temperature
#         self.messages = [
#             {
#                 "role": "system",
#                 "content": (
#                     "You are an AI assistant helping users specify algorithm experiments. "
#                     "Ensure they provide the algorithm, dataset, and optional parameters before finalizing the configuration."
#                 )
#             }
#         ]
#         self.experiment_config = {
#             "algorithm": [],
#             "dataset": "",
#             "parameters": {}
#         }

#         # Automatically start the chatbot
#         print("")
#         self.run_chatbot()

#     def get_chatgpt_response(self, messages):
#         """
#         Uses the new OpenAI API (openai>=1.0.0).
#         """
#         response = openai.chat.completions.create(
#             model=self.model,
#             messages=messages,
#             temperature=self.temperature
#         )
#         return response.choices[0].message.content.strip()

#     def run_chatbot(self):
#         while not all([self.experiment_config["algorithm"], self.experiment_config["dataset"], os.path.exists(self.experiment_config["dataset"])]):
#             if len(self.messages) == 1:
#                 print("Enter command (e.g., 'Run IForest on ./data/glass.mat with contamination=0.1'):")
#             user_input = input("User: ")
#             self.messages.append({"role": "user", "content": user_input})

#             response = self.get_chatgpt_response(self.messages)
#             self.messages.append({"role": "assistant", "content": response})
#             # print("Chatbot:", response)

#             # Let GPT return structured information
#             extraction_prompt = [
#                 *self.messages,
#                 {
#                     "role": "system",
#                     "content": (
#                         "Extract the algorithm, dataset, and optional parameters from the above conversation "
#                         "and return them in Python dictionary (JSON) format. "
#                         "If any item is missing, return an empty object."
#                         "User Input follow format `Run XXX on XXX with XXX` where with XXX is optional."
#                         "For example: user might say `Run IForest on ./data/glass.mat with contamination=0.1` you should return `{'algorithm': 'IForest', 'dataset': './data/glass.mat', 'parameters': {'contamination': 0.1}}`"
#                         "If user say `Run IForest on ./data/glass.mat` you should return `{'algorithm': ['IForest'], 'dataset': './data/glass.mat', 'parameters': {}}`"
#                         "If user say `Run IForest` you should return `{'algorithm': ['IForest'], 'dataset': None, 'parameters': {}}`"
#                         "If user say `./data/glass.mat` you should return `{'algorithm': [], 'dataset': './data/glass.mat', 'parameters': {}}`"
#                         "If user say `Run IForest on ./data/glass.mat` you should return `{'algorithm': ['IForest'], 'dataset': './data/glass.mat', 'parameters': {}}`"
#                         "IMPORTANT: DO NOT SUSPECT ALGORITHM NAME OR PARAMETERS NAME."
#                         "IMPORTANT: Algorithm Should Always be an Array."
#                         "IMPORTANT: IF USER WANT RUN ALL ALGORITHM, RETURN 'algorithm' SHOULD BE ['ECOD', 'ABOD', 'FastABOD', 'COPOD', 'MAD', 'SOS', 'QMCD', 'KDE', 'Sampling', 'GMM', 'PCA', 'KPCA', 'MCD', 'CD', 'OCSVM', 'LMDD', 'LOF', 'COF', '(Incremental) COF', 'CBLOF', 'LOCI', 'HBOS', 'kNN', 'AvgKNN', 'MedKNN', 'SOD', 'ROD', 'IForest', 'INNE', 'DIF', 'FeatureBagging', 'LSCP', 'XGBOD', 'LODA', 'SUOD', 'AutoEncoder', 'VAE', 'Beta-VAE', 'SO_GAAL', 'MO_GAAL', 'DeepSVDD', 'AnoGAN', 'ALAD', 'AE1SVM', 'DevNet', 'R-Graph', 'LUNAR']" 
#                     )
#                 }
#             ]
#             structured_response = self.get_chatgpt_response(extraction_prompt)

#             try:
#                 # Attempt to parse GPT's returned dictionary
#                 extracted_info = eval(structured_response)  # Unsafe, for demonstration purposes only
#                 if isinstance(extracted_info, dict):
#                     # Update experiment configuration if new information is extracted
#                     if extracted_info.get("algorithm"):
#                         self.experiment_config["algorithm"] = extracted_info["algorithm"]
#                     if extracted_info.get("dataset"):
#                         self.experiment_config["dataset"] = extracted_info["dataset"]
#                     if extracted_info.get("parameters"):
#                         self.experiment_config["parameters"].update(extracted_info["parameters"])
#             except:
#                 pass

#             if not self.experiment_config["algorithm"]:
#                 print("Chatbot: Please specify which algorithm to run.")
#             if not self.experiment_config["dataset"] or (not os.path.exists(self.experiment_config["dataset"])):
#                 print("Chatbot: Please provide valid dataset location.")

#         print("\nExperiment Configuration:")
#         print(f"Algorithm: {self.experiment_config['algorithm']}")
#         print(f"Dataset: {self.experiment_config['dataset']}")
#         print(f"Parameters: {self.experiment_config['parameters']}")

# if __name__ == "__main__":
#     chatbot_instance = AgentPreprocessor()



import re
import os

class AgentPreprocessor:
    # Predefined list for "all" algorithms
    ALL_ALGORITHMS = [
        'ECOD', 'ABOD', 'FastABOD', 'COPOD', 'MAD', 'SOS', 'QMCD', 'KDE', 'Sampling',
        'GMM', 'PCA', 'KPCA', 'MCD', 'CD', 'OCSVM', 'LMDD', 'LOF', 'COF', '(Incremental) COF',
        'CBLOF', 'LOCI', 'HBOS', 'kNN', 'AvgKNN', 'MedKNN', 'SOD', 'ROD', 'IForest', 'INNE',
        'DIF', 'FeatureBagging', 'LSCP', 'XGBOD', 'LODA', 'SUOD', 'AutoEncoder', 'VAE',
        'Beta-VAE', 'SO_GAAL', 'MO_GAAL', 'DeepSVDD', 'AnoGAN', 'ALAD', 'AE1SVM', 'DevNet',
        'R-Graph', 'LUNAR'
    ]
    
    def __init__(self):
        self.initialized = False
        self.experiment_config = {
            "algorithm": [],
            "dataset": "",
            "parameters": {}
        }
        self.run_chatbot()
    
    def parse_command(self, command: str) -> dict:
        """
        Parse the command string using string matching and regular expressions.
        
        Expected formats:
            Run <algorithm> [on <dataset>] [with <param1>=<value1> [<param2>=<value2> ...]]
        
        If the command does not start with "Run", assume it's a dataset.
        Returns a dictionary with keys:
          - algorithm: always a list (empty if not provided)
          - dataset: string (None if not provided)
          - parameters: dictionary (empty if none provided)
        """
        result = {"algorithm": [], "dataset": None, "parameters": {}}
        command = command.strip()
        
        # If command starts with "Run" (case-insensitive), process it accordingly.
        if re.match(r"^run\s", command, re.IGNORECASE):
            # Remove the "Run" keyword.
            content = re.sub(r"^run\s+", "", command, flags=re.IGNORECASE)
            
            # Split by "on" to separate algorithm part from dataset and parameters.
            parts = re.split(r"\s+on\s+", content, flags=re.IGNORECASE)
            # First part is for the algorithm.
            alg_part = parts[0].strip() if parts[0] else ""
            if alg_part:
                # Allow multiple algorithms separated by commas or whitespace.
                algorithms = re.split(r",\s*|\s+", alg_part)
                algorithms = [a for a in algorithms if a]
                # If any algorithm is "all" (case-insensitive), replace with the full list.
                if any(a.lower() == "all" for a in algorithms):
                    result["algorithm"] = self.ALL_ALGORITHMS
                else:
                    result["algorithm"] = algorithms
            
            # Process the dataset and parameters if available.
            if len(parts) > 1:
                remainder = parts[1].strip()
                # Split by "with" to separate dataset from parameters.
                subparts = re.split(r"\s+with\s+", remainder, flags=re.IGNORECASE)
                dataset = subparts[0].strip() if subparts[0] else None
                result["dataset"] = dataset
                
                # Process parameters if provided.
                if len(subparts) > 1:
                    params_str = subparts[1].strip()
                    # Find key=value pairs.
                    param_matches = re.findall(r"(\w+)\s*=\s*([^\s,]+)", params_str)
                    for key, value in param_matches:
                        # Convert to int or float if possible.
                        try:
                            if '.' in value:
                                conv_value = float(value)
                            else:
                                conv_value = int(value)
                        except ValueError:
                            conv_value = value
                        result["parameters"][key] = conv_value
        else:
            # If the command does not start with "Run", assume it's a dataset path.
            result["dataset"] = command
        
        return result
    
    def run_chatbot(self):
        """
        Runs the chatbot loop until a valid experiment configuration is provided.
        Checks that:
          - algorithm is provided (non-empty list)
          - dataset is provided and exists (os.path.exists)
        """
        while not (self.experiment_config["algorithm"] and 
                   self.experiment_config["dataset"] and 
                   os.path.exists(self.experiment_config["dataset"])):
            if not self.initialized:
                print("Enter command (e.g., 'Run IForest on ./data/glass.mat with contamination=0.1'):")
                self.initialized = True
            user_input = input("User: ")
            
            # Use string matching to parse the command.
            structured_response = self.parse_command(user_input)
            
            if structured_response.get("algorithm"):
                self.experiment_config["algorithm"] = structured_response["algorithm"]
            if structured_response.get("dataset"):
                self.experiment_config["dataset"] = structured_response["dataset"]
            if structured_response.get("parameters"):
                self.experiment_config["parameters"].update(structured_response["parameters"])
            
            if not self.experiment_config["algorithm"]:
                print("Chatbot: Please specify which algorithm to run.")
            if not self.experiment_config["dataset"] or not os.path.exists(self.experiment_config["dataset"]):
                print("Chatbot: Please provide valid dataset location.")
        
        print("\nExperiment Configuration:")
        print(f"Algorithm: {self.experiment_config['algorithm']}")
        print(f"Dataset: {self.experiment_config['dataset']}")
        print(f"Parameters: {self.experiment_config['parameters']}")

if __name__ == "__main__":
    AgentPreprocessor()

