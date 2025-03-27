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


import openai
import os

class AgentPreprocessor:
    def __init__(self, model="gpt-4", temperature=0):
        self.model = model
        self.temperature = temperature
        self.messages = [
            {
                "role": "system",
                "content": (
                    "You are an AI assistant helping users specify algorithm experiments. "
                    "Ensure they provide the algorithm, datasets (both training and testing), "
                    "and optional parameters before finalizing the configuration."
                )
            }
        ]
        self.experiment_config = {
            "algorithm": [],
            "dataset_train": "",
            "dataset_test": "",
            "parameters": {}
        }

        # Automatically start the chatbot
        # self.run_chatbot()

    def get_chatgpt_response(self, messages):
        """
        Uses the new OpenAI API (openai>=1.0.0).
        """
        response = openai.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature
        )
        return response.choices[0].message.content.strip()

    def run_chatbot(self):
        # 修改条件：确保算法以及两个数据集都已指定，并且路径存在
        while not all([
            self.experiment_config["algorithm"],
            self.experiment_config["dataset_train"],
            self.experiment_config["dataset_test"],
            os.path.exists(self.experiment_config["dataset_train"]),
            os.path.exists(self.experiment_config["dataset_test"])
        ]):
            if len(self.messages) == 1:
                print("Enter command (e.g., 'Run IForest on ./data/glass_train.mat and ./data/glass_test.mat with contamination=0.1'):")
            user_input = input("User: ")
            self.messages.append({"role": "user", "content": user_input})

            response = self.get_chatgpt_response(self.messages)
            self.messages.append({"role": "assistant", "content": response})
            # print("Chatbot:", response)

            # 更新提取提示，要求返回训练集和测试集两个数据集信息
            extraction_prompt = [
                *self.messages,
                {
                    "role": "system",
                    "content": (
                        "Extract the algorithm, dataset_train, dataset_test, and optional parameters from the above conversation "
                        "and return them in Python dictionary (JSON) format. "
                        "If any item is missing, return an empty object. "
                        "User input follows format `Run XXX on TRAIN_DATA and TEST_DATA with XXX` where with XXX is optional. "
                        "For example: if the user says `Run IForest on ./data/train.mat and ./data/test.mat with contamination=0.1` "
                        "you should return `{'algorithm': ['IForest'], 'dataset_train': './data/train.mat', 'dataset_test': './data/test.mat', 'parameters': {'contamination': 0.1}}`. "
                        "If user says `Run IForest on ./data/train.mat and ./data/test.mat` you should return `{'algorithm': ['IForest'], 'dataset_train': './data/train.mat', 'dataset_test': './data/test.mat', 'parameters': {}}`. "
                        "If user says `Run IForest` you should return `{'algorithm': ['IForest'], 'dataset_train': None, 'dataset_test': None, 'parameters': {}}`. "
                        "If user says `./data/train.mat and ./data/test.mat` you should return `{'algorithm': [], 'dataset_train': './data/train.mat', 'dataset_test': './data/test.mat', 'parameters': {}}`. "
                        "IMPORTANT: DO NOT ASSUME ALGORITHM NAME OR PARAMETERS NAME. "
                        "IMPORTANT: Algorithm should always be an array. "
                        "IMPORTANT: IF USER WANTS TO RUN ALL ALGORITHMS, return 'algorithm' as ['ECOD', 'ABOD', 'FastABOD', 'COPOD', 'MAD', 'SOS', 'QMCD', 'KDE', 'Sampling', 'GMM', 'PCA', 'KPCA', 'MCD', 'CD', 'OCSVM', 'LMDD', 'LOF', 'COF', '(Incremental) COF', 'CBLOF', 'LOCI', 'HBOS', 'kNN', 'AvgKNN', 'MedKNN', 'SOD', 'ROD', 'IForest', 'INNE', 'DIF', 'FeatureBagging', 'LSCP', 'XGBOD', 'LODA', 'SUOD', 'AutoEncoder', 'VAE', 'Beta-VAE', 'SO_GAAL', 'MO_GAAL', 'DeepSVDD', 'AnoGAN', 'ALAD', 'AE1SVM', 'DevNet', 'R-Graph', 'LUNAR']"
                    )
                }
            ]
            structured_response = self.get_chatgpt_response(extraction_prompt)
            print("Structured Response:", structured_response)

            try:
                # 注意：eval在实际生产环境中存在风险，仅作演示使用
                extracted_info = eval(structured_response)
                if isinstance(extracted_info, dict):
                    if extracted_info.get("algorithm"):
                        self.experiment_config["algorithm"] = extracted_info["algorithm"]
                    if extracted_info.get("dataset_train"):
                        self.experiment_config["dataset_train"] = extracted_info["dataset_train"]
                    if extracted_info.get("dataset_test"):
                        self.experiment_config["dataset_test"] = extracted_info["dataset_test"]
                    if extracted_info.get("parameters"):
                        self.experiment_config["parameters"].update(extracted_info["parameters"])
            except Exception as e:
                # 如果解析失败，继续下一轮
                pass

            if not self.experiment_config["algorithm"]:
                print("Chatbot: Please specify which algorithm to run.")
            if not self.experiment_config["dataset_train"] or (not os.path.exists(self.experiment_config["dataset_train"])):
                print("Chatbot: Please provide a valid training dataset location.")
            if not self.experiment_config["dataset_test"] or (not os.path.exists(self.experiment_config["dataset_test"])):
                print("Chatbot: Please provide a valid testing dataset location.")

        print("\nExperiment Configuration:")
        print(f"Algorithm: {self.experiment_config['algorithm']}")
        print(f"Training Dataset: {self.experiment_config['dataset_train']}")
        print(f"Testing Dataset: {self.experiment_config['dataset_test']}")
        print(f"Parameters: {self.experiment_config['parameters']}")

if __name__ == "__main__":
    import os
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from config.config import Config
    os.environ['OPENAI_API_KEY'] = Config.OPENAI_API_KEY
    chatbot_instance = AgentPreprocessor()
    chatbot_instance.run_chatbot()
