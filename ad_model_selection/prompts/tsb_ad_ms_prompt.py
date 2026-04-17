from utils.tsb_ad_registry import TSB_AD_MODEL_SELECTION_CANDIDATES, TSB_AD_SEMISUPERVISE_ALGORITHMS


def generate_model_selection_prompt_from_tsb_ad(name, size, dim, ts_type, has_test_dataset=True):
    if has_test_dataset:
        candidates = TSB_AD_SEMISUPERVISE_ALGORITHMS
    else:
        candidates = TSB_AD_MODEL_SELECTION_CANDIDATES + TSB_AD_SEMISUPERVISE_ALGORITHMS
    options = ", ".join(f'"{model}"' for model in candidates)
    options_text = "\n".join(f"- {model}" for model in candidates)

    user_message = f"""
You are an expert in model selection for time-series anomaly detection.

## Task:
- Given the information of a dataset and a set of TSB-AD models, select the model you believe will achieve the best performance for detecting anomalies in this dataset. Provide a brief explanation of your choice.

## Dataset Information:
- Dataset Name: {name}
- Dataset Size: {size}
- Data Dimension: {dim}
- Data Type: {ts_type}

## Model Options:
{options_text}

## Rules:
1. Available options include {options}.
2. Treat all models equally and evaluate them based on compatibility with the dataset characteristics and anomaly-detection task.
3. Response Format:
    - Provide responses in strict JSON format with the keys "reason" and "choice."
        - "reason": your explanation.
        - "choice": the selected model.

Response in JSON format:
"""

    return [{"role": "user", "content": user_message}]
