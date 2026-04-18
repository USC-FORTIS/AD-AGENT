def generate_model_selection_prompt_from_pygod(name, num_node, num_edge, num_feature, avg_degree):

    user_message = f"""
You are an expert in model selection for anomaly detection on graph data.

## Task:
- Given the information of a dataset and a set of models, select the model you believe will achieve the best performance for detecting anomalies in this dataset. Provide a brief explanation of your choice.

## Dataset Information:
- Dataset Name: {name}
- Number of Nodes: {num_node}
- Number of Edges: {num_edge}
- Number of Features: {num_feature}
- Average Degree: {avg_degree}

## Model Options:
- Adversarial Outlier Aware Attributed Network Embedding (AdONE)
- A Joint Modeling Approach for Anomaly Detection on Attributed Networks (ANOMALOUS)
- Dual Autoencoder for Anomaly Detection on Attributed Networks (AnomalyDAE)
- Contrastive Attributed Network Anomaly Detection (CONAD)
- Deep Outlier Aware Attributed Network Embedding (DONE)
- Generative Adversarial Attributed Network Anomaly Detection (GAAN)
- Higher-order Structure based Anomaly Detection on Attributed Networks (GUIDE)
- Residual Analysis for Anomaly Detection in Attributed Networks (Radar)
- Structural Clustering Algorithm for Networks (SCAN)

## Rules:
1. Availabel options include "AdONE", "ANOMALOUS", "AnomalyDAE", "CONAD", "DONE", "GAAN", "GUIDE", "Radar", and "SCAN."
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
