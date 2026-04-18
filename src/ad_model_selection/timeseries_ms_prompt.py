def generate_model_selection_prompt_from_timeseries(name, size, dim, type):

    user_message = f"""
You are an expert in model selection for anomaly detection on time series data.

## Task:
- Given the information of a dataset and a set of models, select the model you believe will achieve the best performance for detecting anomalies in this dataset. Provide a brief explanation of your choice.

## Dataset Information:
- Dataset Name: {name}
- Dataset Size: {size}
- Data Dimension: {dim}
- Data Type: {type}

## Model Options:
- Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting (Autoformer)
- Are Transformers Effective for Time Series Forecasting? (DLinear)
- Exponential Smoothing Transformers for Time-series Forecasting (ETSformer)
- Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting (FEDformer)
- Beyond Efficient Transformer for Long Sequence Time-Series Forecasting (Informer)
- Less Is More: Fast Multivariate Time Series Forecasting with Light Sampling-oriented MLP Structures (LightTS)
- Low-complexity Pyramidal Attention for Long-range Time Series Modeling and Forecasting (Pyraformer)
- The Efficient Transformer (Reformer)
- Temporal 2D-Variation Modeling for General Time Series Analysis (TimesNet)
- Attention is All You Need (Transformer)

## Rules:
1. Availabel options include "Autoformer", "DLinear", "ETSformer", "FEDformer", "Informer", "LightTS", "Pyraformer", "Reformer", "TimesNet", and "Transformer."
2. Treat all models equally and evaluate them based on their compatibility with the dataset characteristics and the anomaly detection task.
3. Response Format:
    - Provide responses in a strict **JSON** format with the keys "reason" and "choice."
        - "reason": Your explanation of the reasoning.
        - "choice": The model you have selected for anomaly detection in this dataset.

Response in JSON format:
"""
    
    messages = [
        # {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
        # {"role": "assistant", "content": assistant_message}
    ]

    return messages
