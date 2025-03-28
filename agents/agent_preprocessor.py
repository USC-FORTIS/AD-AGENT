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
        # Condition: Ensure algorithm and both datasets are provided and paths exist
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
                # Warning: Using eval has security risks; used here for demonstration purposes only
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
            except Exception:
                # If parsing fails, move on to the next round
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
