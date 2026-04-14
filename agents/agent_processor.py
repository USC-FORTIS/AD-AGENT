"""
AgentProcessor - Using Few-shot Chain-of-Thought (CoT) Extraction
------------------------------------------------------------------
• Dialogue layer: assists users in specifying algorithm experiment configurations.
• Extraction layer: uses Few-shot CoT for better accuracy.
    - Covers: single algorithm, multiple algorithms, optional parameters, 'all' cases.
    - The model outputs reasoning steps followed by a `FINAL:` line with a JSON object.
"""

import os
import re
import json
import sys
import openai

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.tsb_ad_registry import Unsupervise_AD_Pool, Semisupervise_AD_Pool


class AgentProcessor:
    # ---------- Few-shot Chain-of-Thought Examples ----------
    FEW_SHOT_COT_PROMPT = [
        {
            "role": "system",
            "content": (
                "You are an extraction assistant. "
                "Read the USER_INSTRUCTION and return four fields:\n"
                "• algorithm (array) • dataset_train • dataset_test • parameters (object)\n"
                "First show your reasoning, THEN output one line starting with `FINAL:` "
                "followed by the JSON dictionary. Do NOT invent names."
            ),
        },
        # Example 1: Single algorithm with dataset and parameter
        {
            "role": "user",
            "content": "Run IForest on ./data/train.mat and ./data/test.mat with contamination=0.1",
        },
        {
            "role": "assistant",
            "content": (
                "Step 1 ▶ algorithm → IForest\n"
                "Step 2 ▶ train → ./data/train.mat ; test → ./data/test.mat\n"
                "Step 3 ▶ param → contamination=0.1\n"
                'FINAL: {"algorithm":["IForest"],"dataset_train":"./data/train.mat",'
                '"dataset_test":"./data/test.mat","parameters":{"contamination":0.1}}'
            ),
        },
        # Example 2: Multiple algorithms and multiple parameters
        {
            "role": "user",
            "content": (
                "Please run DeepSVDD and IForest with contamination=0.2, max_iter=300 on "
                "./train.csv for training and ./test.csv for evaluation"
            ),
        },
        {
            "role": "assistant",
            "content": (
                "Step 1 ▶ algorithms → DeepSVDD, IForest\n"
                "Step 2 ▶ train → ./train.csv ; test → ./test.csv\n"
                "Step 3 ▶ params → contamination=0.2, max_iter=300\n"
                'FINAL: {"algorithm":["DeepSVDD","IForest"],"dataset_train":"./train.csv",'
                '"dataset_test":"./test.csv","parameters":{"contamination":0.2,"max_iter":300}}'
            ),
        },
        # Example 3: Algorithm only
        {"role": "user", "content": "Run LOF"},
        {
            "role": "assistant",
            "content": (
                "Step 1 ▶ algorithm → LOF\n"
                "Step 2 ▶ no datasets\n"
                "Step 3 ▶ no parameters\n"
                'FINAL: {"algorithm":["LOF"],"dataset_train":null,"dataset_test":null,"parameters":{}}'
            ),
        },
        # Example 4: Run all algorithms
        {"role": "user", "content": "Run all algorithms on ./d1.txt and ./d2.txt"},
        {
            "role": "assistant",
            "content": (
                "Step 1 ▶ algorithm keyword → all\n"
                "Step 2 ▶ train → ./d1.txt ; test → ./d2.txt\n"
                "Step 3 ▶ no parameters\n"
                'FINAL: {"algorithm":["all"],"dataset_train":"./d1.txt",'
                '"dataset_test":"./d2.txt","parameters":{}}'
            ),
        },
        # Placeholder: user's actual command will replace this
        {"role": "user", "content": "USER_INSTRUCTION:\n<START>\n{user_input}\n<END>"},
    ]

    def __init__(self, model: str = "gpt-4o", temperature: float = 0):
        self.model = model
        self.temperature = temperature

        # Initial conversation history (used for the dialogue experience)
        self.messages = [
            {
                "role": "system",
                "content": (
                    "You are an AI assistant helping users specify algorithm experiments. "
                    "Ensure they provide the algorithm, datasets (both training and testing), "
                    "and optional parameters before finalizing the configuration."
                ),
            }
        ]

        # Final extracted configuration object
        self.experiment_config = {
            "algorithm": [],
            "dataset_train": "",
            "dataset_test": "",
            "parameters": {},
        }

    def get_chatgpt_response(self, messages):
        """
        Call OpenAI chat model and return the assistant's response.
        """
        response = openai.chat.completions.create(
            model=self.model, messages=messages, temperature=self.temperature
        )
        return response.choices[0].message.content.strip()

    def extract_config(self, user_input: str) -> dict:
        """
        Run Few-shot CoT extraction for the given user command.
        Returns a dictionary with algorithm, dataset, and parameters.
        """
        # Clone and format the prompt with the latest user input
        prompt = [dict(p) for p in self.FEW_SHOT_COT_PROMPT]
        prompt[-1]["content"] = prompt[-1]["content"].format(user_input=user_input)

        assistant_text = self.get_chatgpt_response(prompt)

        # Extract JSON object from the FINAL line
        match = re.search(r"^FINAL:\s*(\{.*\})$", assistant_text, re.MULTILINE)
        if not match:
            return {}

        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            return {}

    def run_chatbot(self):
        """
        Main interaction loop that gathers user input and extracts structured config.
        Stops only when valid algorithm and training dataset path are provided.
        """
        while not all(
            [
                # self.experiment_config["algorithm"],
                self.experiment_config["dataset_train"],
                os.path.exists(self.experiment_config["dataset_train"]),
                (
                    not self.experiment_config["dataset_test"]
                    or os.path.exists(self.experiment_config["dataset_test"])
                ),
            ]
        ):
            if len(self.messages) == 1:
                print(
                    "Enter command (e.g., "
                    "'Run IForest on ./data/glass_train.mat and "
                    "./data/glass_test.mat with contamination=0.1'):"
                )

            user_input = input("User: ").strip()
            if not user_input:
                continue

            # Add to conversation history
            self.messages.append({"role": "user", "content": user_input})

            # Get assistant reply for user engagement
            assistant_reply = self.get_chatgpt_response(self.messages)
            self.messages.append({"role": "assistant", "content": assistant_reply})
            # print(f"Chatbot: {assistant_reply}")

            # Extract structured information from user input
            extracted = self.extract_config(user_input)

            if extracted.get("algorithm"):
                self.experiment_config["algorithm"] = extracted["algorithm"]
            if extracted.get("dataset_train"):
                self.experiment_config["dataset_train"] = extracted["dataset_train"]
            if extracted.get("dataset_test"):
                self.experiment_config["dataset_test"] = extracted["dataset_test"]
            else:
                self.experiment_config["dataset_test"] = ""
            if extracted.get("parameters"):
                self.experiment_config["parameters"].update(extracted["parameters"])

            # If all specified algorithms are unsupervised, test dataset is not allowed
            algorithms = self.experiment_config["algorithm"]
            if algorithms and all(a in Unsupervise_AD_Pool for a in algorithms):
                if self.experiment_config["dataset_test"]:
                    print(
                        "Chatbot: The selected algorithm(s) are unsupervised and do not "
                        "accept a test dataset. The test dataset input has been ignored."
                    )
                    self.experiment_config["dataset_test"] = ""

            # If all specified algorithms are semisupervised and only train dataset provided,
            # set test dataset to the same as train dataset
            if algorithms and all(a in Semisupervise_AD_Pool for a in algorithms):
                if not self.experiment_config["dataset_test"]:
                    self.experiment_config["dataset_test"] = self.experiment_config["dataset_train"]
                    print(
                        "Chatbot: The selected algorithm(s) are semisupervised but no test "
                        "dataset was provided. Test dataset has been set to the training dataset."
                    )

            # Missing field guidance
            # if not self.experiment_config["algorithm"]:
            #     print("Chatbot: Please specify which algorithm to run.")
            if (
                not self.experiment_config["dataset_train"]
                or not os.path.exists(self.experiment_config["dataset_train"])
            ):
                print("Chatbot: Please provide a valid training dataset location.")
            if (
                self.experiment_config["dataset_test"]
                and not os.path.exists(self.experiment_config["dataset_test"])
            ):
                print("Chatbot: Please provide a valid testing dataset location (or leave it empty if not applicable).")

        # Final output summary
        print("\nExperiment Configuration")
        print("Algorithm        :", self.experiment_config["algorithm"])
        print("Training Dataset :", self.experiment_config["dataset_train"])
        print("Testing Dataset  :", self.experiment_config["dataset_test"])
        print("Parameters       :", self.experiment_config["parameters"])



if __name__ == "__main__":
    import os
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from config.config import Config
    os.environ['OPENAI_API_KEY'] = Config.OPENAI_API_KEY
    chatbot_instance = AgentProcessor()
    chatbot_instance.run_chatbot()
