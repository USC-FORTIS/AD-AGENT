## Step 1 Read the input 'mat' file
import scipy.io

# Load the .mat file
mat_data = scipy.io.loadmat('data/cardio.mat')

# Print keys to see what variables are stored in the file
print(mat_data.keys())

# Access the correct variable (example: 'X' and 'y' are present in your file)
X = mat_data['X']
y = mat_data['y']

# Print the data to inspect
print("X:", X)
print("y:", y)

print('length of X:', len(X))
exit()


## Step 2 Ask llm to generate code to discover the anomaly
import openai
import os
import re

# Load API key securely from environment variables
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_anomaly_detection_code():
    """Generates Python code using GPT-4 to detect anomalies with pyod."""
    prompt = """
    Write a Python script that:
    1. Loads a dataset (assumed to be stored in 'X' and 'y' variables).
    2. Uses the `pyod` package, specifically `IForest` (Isolation Forest), to detect anomalies.
    3. Prints the anomalous points detected and the number of anomalies with some space and new lines
    
    Assume that the dataset is already loaded into 'X' and 'y'.
    
    Provide **only the code**, without explanations.
    """

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an expert Python developer."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content

# Call GPT API to generate the code
generated_response = generate_anomaly_detection_code()

# Extract only the Python code using regex
code_match = re.search(r"```python\n(.*?)\n```", generated_response, re.DOTALL)

if code_match:
    extracted_code = code_match.group(1)
else:
    extracted_code = generated_response  # Fallback if no markdown formatting

# Save the extracted code to a file (optional for debugging)
with open("generated_anomaly_detection.py", "w") as f:
    f.write(extracted_code)

# Print extracted code
print("Generated Python Code:\n", extracted_code)

# Execute the extracted code safely
exec(extracted_code)
