from data_loader import DataLoader
import pandas as pd
import numpy as np
import openai
import os

def proppt_template (X, y):
    X_head = ''
    y_head = ''
    # Ensure X is a DataFrame before calling .head()
    if isinstance(X, pd.DataFrame):
        X_head = X.head()  # Works if X is already a DataFrame
    elif isinstance(X, np.ndarray):
        X_head = pd.DataFrame(X).head()  # Convert NumPy array to DataFrame before calling .head()
    else:
        print("⚠️ Warning: X is neither a NumPy array nor a DataFrame. Type:", type(X))
        return None
    
    if isinstance(y, pd.DataFrame):
        y_head = y.head()  # Works if X is already a DataFrame
    elif isinstance(y, np.ndarray):
        y_head = pd.DataFrame(y).head()  # Convert NumPy array to DataFrame before calling .head()
    else:
        print("⚠️ Warning: X is neither a NumPy array nor a DataFrame. Type:", type(y))
        return None
    
    prompt = f'''
    You are an expert in anomaly detection using the `pyod` library.

    Given the dataset:
    - The first few rows of the feature matrix (X) are:
    {X_head}

    - The first few rows of the target variable (y) are:
    {y_head}

    Your task is to analyze this dataset and **recommend the best anomaly detection model** from the `pyod` library.

    Specifically, consider:
    1. The characteristics of the data (e.g., numerical vs. categorical, distribution, sparsity).
    2. The strengths and weaknesses of various `pyod` models (e.g., Isolation Forest, LOF, AutoEncoder, HBOS, etc.).
    3. The best model based on how anomalies might be structured in this dataset.

    **Output only the best model's name from `pyod` with short explanation.** 
    '''

    system_message = "You are an expert in anomaly detection using the `pyod` library."
    return prompt, system_message

def select_model_using_llm(X, y):
    prompt, system_message = proppt_template(X, y)


    
    # Initialize OpenAI client
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Get response from GPT
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]
    )

    content = response.choices[0].message.content

    print(content)
    return content


    
if __name__ == "__main__":
    # Example usage
    data_loader = DataLoader("data/cardio.mat")
    X, y = data_loader.load_data()
    print("Length of X:", len(X))


    select_model_using_llm(X, y)
