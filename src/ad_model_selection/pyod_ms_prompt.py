def generate_model_selection_prompt_from_pyod(name, size, dim):

    user_message = f"""
You are an expert in model selection for anomaly detection on multivariate data.

## Task:
- Given the information of a dataset and a set of models, select the model you believe will achieve the best performance for detecting anomalies in this dataset. Provide a brief explanation of your choice.

## Dataset Information:
- Dataset Name: {name}
- Dataset Size: {size}
- Data Dimension: {dim}

## Model Options:
- Adversarially Learned Anomaly Detection (ALAD)
- Anomaly Detection with Generative Adversarial Networks (AnoGAN)
- AutoEncoder (AE)
- Autoencoder-based One-class Support Vector Machine (AE1SVM)
- Deep One-Class Classification (DeepSVDD)
- Deep Anomaly Detection with Deviation Networks (DevNet)
- Unifying Local Outlier Detection Methods via Graph Neural Networks (LUNAR)
- Multiple-Objective Generative Adversarial Active Learning (MO-GAAL)
- Single-Objective Generative Adversarial Active Learning (SO-GAAL)
- Variational AutoEncoder (VAE)

## Rules:
1. Availabel options include "ALAD", "AnoGAN", "AE", "AE1SVM", "DeepSVDD", "DevNet", "LUNAR", "MO-GAAL", "SO-GAAL", and "VAE."
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
