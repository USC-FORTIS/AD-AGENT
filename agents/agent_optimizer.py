import ast
import os
import re
from typing import Any, Dict, List, Optional
import sys


from langchain.prompts import PromptTemplate
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
from langchain_openai import ChatOpenAI

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from entity.code_quality import CodeQuality
from config.config import Config
from sandbox.executor import execute_code as sandbox_execute_code
os.environ['OPENAI_API_KEY'] = Config.OPENAI_API_KEY

SYSTEM_PROMPT_TMPL = PromptTemplate.from_template(
    """
You are an expert Python engineer specialising in anomaly‑detection libraries.

Current implementation
----------------------
{code}

Current parameters
------------------
{parameter}

Current output
--------------
{std_output}

Authoritative documentation
---------------------------
{algorithm_doc}

You have access to a single tool:
`execute_code(params: Dict[str, Any]) -> str` which runs the script with the
supplied **new** parameters and returns the console output.


Follow the **ReAct** loop **STRICTLY** – each response must be *Either*:

1. A pair of lines:
   Thought: <reasoning>
   Action: execute_code({{"param": value, ...}})

2. A single line starting with `Final:` when you deternmined the final answer.

IMPORTANT:
1. Do not input `default` in the parameters, use the default values from the code.
"""
)


class AgentOptimizer:
    """ReAct-style parameter tuning agent."""

    _ACTION_RE = re.compile(r"^Action:\s*execute_code\((.*)\)$", re.MULTILINE)
    _THOUGHT_RE = re.compile(r"^Thought:(.*)$", re.MULTILINE)
    _FINAL_RE = re.compile(r"^Final:(.*)$", re.MULTILINE)

    @staticmethod
    def execute_code(
        parameters: Dict[str, Any],
        base_code: str,
        algorithm_name: str,
        package_name: str = "pyod",
    ) -> str:
        """Run modified code with injected parameters."""
        pat = re.compile(r"(model\s*=\s*[A-Za-z_]+\s*\()(.*?)(\))", re.DOTALL)
        match = pat.search(base_code)
        if not match:
            return "[ERROR] Model instantiation line not found."

        new_params = ", ".join(f"{k}={repr(v)}" for k, v in parameters.items())
        new_code = base_code[:match.start()] + match.group(1) + new_params + match.group(3) + base_code[match.end():]

        stdout, stderr, returncode = sandbox_execute_code(
            code=new_code,
            algorithm_name=algorithm_name,
            package_name=package_name,
            timeout=60,
        )
        output = stdout + stderr
        if returncode != 0:
            output += f"\n[ERROR] Return code: {returncode}"
        return output.strip()

    @classmethod
    def _extract_param_dict(cls, text: str) -> Optional[Dict[str, Any]]:
        m = cls._ACTION_RE.search(text)
        if not m:
            return None
        try:
            return ast.literal_eval(m.group(1))
        except Exception:
            return None

    @classmethod
    def _print_thought_and_action(cls, content: str, step: int) -> None:
        print(f"\n--- Step {step} ---")
        thought_match = cls._THOUGHT_RE.search(content)
        action_match = cls._ACTION_RE.search(content)
        final_match = cls._FINAL_RE.search(content)
        if thought_match:
            print("Thought:", thought_match.group(1).strip())
        if action_match:
            print("Action: execute_code(" + action_match.group(1).strip() + ")")
        if final_match:
            print("Final: " + final_match.group(1).strip())
    
    @staticmethod
    def _find_float(pattern: str, text: str, default: float = -1.0) -> float:
        m = re.search(pattern, text)
        return float(m.group(1)) if m else default

    @staticmethod
    def _parse_errors(text: str):
        pts = []
        for line in text.splitlines():
            if "Failed prediction at point" in line:
                m = re.search(r"\[([^\]]+)] with true label ([\d.]+)", line)
                if m:
                    nums = [float(x.strip()) for x in m.group(1).split(",")]
                    pts.append({"point": nums, "true_label": float(m.group(2))})
        return pts
    def run(
        self,
        llm: ChatOpenAI,
        quality: CodeQuality,
        algorithm_doc: str,
        max_steps: int = 8,
        package_name: str = "pyod",
    ) -> CodeQuality:
        """Run the optimization loop using the given inputs and return CodeQuality."""
        code = quality.code
        parameters = quality.parameters
        std_output = quality.std_output
        algorithm_name = quality.algorithm
        system_prompt = SYSTEM_PROMPT_TMPL.format(
            code=code,
            parameter=parameters,
            std_output=std_output,
            algorithm_doc=algorithm_doc,
        )

        messages: List[BaseMessage] = [SystemMessage(content=system_prompt)]
        final_params = parameters 

        for step in range(1, max_steps + 1):
            ai_response: AIMessage = llm.invoke(messages)  # type: ignore[arg-type]
            messages.append(ai_response)

            content = ai_response.content or ""
            self._print_thought_and_action(content, step)

            param_dict = self._extract_param_dict(content)
            if param_dict:
                final_params = param_dict 
            else:
                messages.append(HumanMessage(
                    content="Action line not detected. Please choose parameters and call the tool as instructed."))
                continue

            if "Final:" in content:
                break

            observation = self.execute_code(param_dict, code, algorithm_name, package_name=package_name)
            print(observation)
            messages.append(HumanMessage(content=f"Observation: {observation[:4000]}"))

        final_output = self.execute_code(final_params, code, algorithm_name, package_name=package_name)

        auroc = self._find_float(r"AUROC:\s*([0-9.]+)", final_output, default=quality.auroc)
        auprc = self._find_float(r"AUPRC:\s*([0-9.]+)", final_output, default=quality.auprc)
        error_points = self._parse_errors(final_output)

        return CodeQuality(
            code=code,
            algorithm=algorithm_name,
            parameters=final_params,
            std_output=final_output,
            error_message=quality.error_message,
            auroc=auroc,
            auprc=auprc,
            error_points=error_points,
            review_count=quality.review_count
        )




# -----------------------------------------------------------------------------
# Example usage (can be removed if integrated elsewhere)
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    demo = {
            "code": """
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_loader.data_loader import DataLoader
from pyod.models.abod import ABOD
from sklearn.metrics import roc_auc_score, average_precision_score

# Initialize DataLoader
dataloader_train = DataLoader(filepath='./data/glass_train.mat', store_script=True, store_path='train_data_loader.py')
dataloader_test = DataLoader(filepath='./data/glass_test.mat', store_script=True, store_path='test_data_loader.py')

# Load data
X_train, y_train = dataloader_train.load_data(split_data=False)
X_test, y_test = dataloader_test.load_data(split_data=False)

# Initialize ABOD
model = ABOD()

# Train the model
model.fit(X_train)

# Get training outlier scores
train_scores = model.decision_scores_

# Get test outlier scores
test_scores = model.decision_function(X_test)

# Calculate AUROC and AUPRC
auroc = roc_auc_score(y_test, test_scores)
auprc = average_precision_score(y_test, test_scores)

# Print AUROC and AUPRC
print(f"AUROC: {auroc:.4f}")
print(f"AUPRC: {auprc:.4f}")

# Record and print failed predictions
predictions = model.predict(X_test)
for i, (pred, true_label) in enumerate(zip(predictions, y_test)):
    if pred != true_label:
        print(f"Failed prediction at point {X_test[i].tolist()} with true label {true_label}")
                """,
            "parameters": {"contamination": 0.1, "n_neighbors": 5, "method": "fast"},
            "algorithm_doc": "The `ABOD` (Angle-Based Outlier Detection) class in PyOD is designed to detect outliers by analyzing the variance of angles between data points. It offers two methods: a faster approximation using k-nearest neighbors and the original method that considers all data points, which is computationally intensive.\n\n**Initialization Function and Parameters:**\n\nThe `ABOD` class is initialized with the following parameters:\n\n- **contamination**: A float in the range (0., 0.5), defaulting to 0.1. This parameter specifies the proportion of outliers in the dataset and is used to define the threshold on the decision function.\n\n- **n_neighbors**: An integer, defaulting to 5. It determines the number of neighbors to use for k-neighbors queries.\n\n- **method**: A string, defaulting to 'fast'. It specifies the method to use:\n  - 'fast': Fast ABOD, which considers only `n_neighbors` of training points.\n  - 'default': Original ABOD that considers all training points, which can be slow due to its O(n^3) time complexity.\n\n**Attributes:**\n\nAfter fitting the model, the following attributes are available:\n\n- **decision_scores_**: A numpy array of shape (n_samples,). It contains the outlier scores of the training data, where higher scores indicate more abnormal data points.\n\n- **threshold_**: A float representing the threshold based on the `contamination` parameter. It is calculated as the `n_samples * contamination` most abnormal samples in `decision_scores_`.\n\n- **labels_**: An array of integers (0 or 1). It contains the binary labels of the training data, where 0 stands for inliers and 1 for outliers/anomalies. These labels are generated by applying `threshold_` on `decision_scores_`.\n\n**Parameters Dictionary:**\n\nHere is a Python dictionary representing all parameters of the `__init__` method for the `ABOD` class, along with their default values:\n\n\n```python\n{\n    \"contamination\": 0.1,\n    \"n_neighbors\": 5,\n    \"method\": \"fast\"\n}\n```\n\n\nThis dictionary can be evaluated using `ast.literal_eval` in Python.",
            "std_output": None,
            "algorithm_name": "ABOD",
        }

    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    agent = AgentOptimizer()
    input_quality = CodeQuality(
        code=demo["code"],
        algorithm=demo["algorithm_name"],
        parameters=demo["parameters"],
        std_output=demo["std_output"],
        error_message="",
        auroc=-1,
        auprc=-1,
        error_points=[],
        review_count=0
    )
    final_answer = agent.run(llm=llm,
        quality=input_quality,
        algorithm_doc=demo["algorithm_doc"],
        max_steps=8
    )
    print("\n=== Final Answer ===\n", final_answer.parameters)
