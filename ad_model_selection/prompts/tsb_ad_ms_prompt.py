from utils.tsb_ad_registry import TSB_AD_MODEL_SELECTION_CANDIDATES


def generate_model_selection_prompt_from_tsb_ad(name, size, dim, type):
    model_options = ", ".join(f'"{candidate}"' for candidate in TSB_AD_MODEL_SELECTION_CANDIDATES)

    user_message = f"""
You are an expert in model selection for anomaly detection on time series data.

## Task:
- Given the information of a dataset and a set of TSB-AD compatible models, select the model you believe will achieve the best performance for detecting anomalies in this dataset. Provide a brief explanation of your choice.

## Dataset Information:
- Dataset Name: {name}
- Dataset Size: {size}
- Data Dimension: {dim}
- Data Type: {type}

## Model Options:
- {", ".join(TSB_AD_MODEL_SELECTION_CANDIDATES)}

## Rules:
1. Available options include {model_options}.
2. Prefer models that are plausible for the dataset shape and anomaly-detection task.
3. Respond in strict JSON with keys "reason" and "choice".

Response in JSON format:
"""

    return [{"role": "user", "content": user_message}]
