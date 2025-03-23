import openai
import os
import re

import sklearn


class DataLoader:
    """
    A class to load various file formats (.mat, .csv, .json, etc.) and extract data.
    """

    def __init__(self, filepath, desc='', store_script = True, store_path = 'generated_data_loader.py'):
        """
        Initialize DataLoader with the path to the file.
        """
        self.filepath = filepath
        self.desc = desc

        self.X_name = 'X'
        self.y_name = 'y'


        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"File not found: {self.filepath}")
        
        self.store_script = store_script
        self.store_path = store_path
        

    def generate_script(self):
        """
        Generates a Python script using GPT-4 to load a data file and extract its content.
        """

        # Ensure self.filepath is correctly formatted
        file_path = self.filepath.replace("\\", "/")  # Normalize for cross-platform compatibility
        file_type = self.filepath.split('.')[-1]  # Extract file extension

        prompt = f"""
    Write a Python script that:
    
    1. **Includes all necessary imports at the beginning** (e.g., `os`, `scipy.io`, `pandas`, `json`).
    2. Checks if the file exists before attempting to load it.
    3. Determines the file type based on the extension (`{file_type}`).
    4. Loads the file using the appropriate Python library:
        - Use `scipy.io.loadmat("{file_path}")` for `.mat` files.
        - Use `pandas.read_csv("{file_path}")` for `.csv` files.
        - Use `json.load(open("{file_path}", 'r'))` for `.json` files.
    5. If the file doesn't exist, print a warning and exit.
    6. Do not print the available keys or columns in the file.
    7. Identify and extract the relevant data attributes:
       - **For `.mat` files:** Assume the primary key for `X` is `{self.X_name}` and for `y` is `{self.y_name}`.
       - **For `.csv` files:** Assume `{self.X_name}` is a feature column (or columns) and `{self.y_name}` is the target column.
       - **For `.json` files:** Assume `X` and `y` are either direct keys or nested under a root key.
       do not need code to handle all type. Just handle the given file type.
    8. Make sure `X` and `y` have the correct shapes after extraction.

    **Important Implementation Details:**
    - Use `os.path.exists("{file_path}")` to check if the file exists.
    - Print **useful error messages** if `X` or `y` are missing.
    - **For `.mat` files**, use `data.keys()` to check available keys and attempt to find `{self.X_name}` and `{self.y_name}` dynamically.
    - **For `.csv` files**, check if `{self.X_name}` and `{self.y_name}` exist in `df.columns` before extracting them.
    - **For `.json` files**, use `data.keys()` to verify `X` and `y` exist.
    - Ensure `X` and `y` are extracted properly and **avoid ambiguous operations** (e.g., `or` on NumPy arrays).
    - **Always print** `X.shape` and `y.shape` (if applicable) after extraction.

    **Example Code Execution:**
    The generated script will be executed like this:
    
        # Execute the generated script safely
        exec(generated_script, {{}}, local_namespace)
        
        # Retrieve X and y from the executed script
        X = local_namespace.get("X")
        y = local_namespace.get("y")

    Ensure the script works with `{file_path}` and does not require manual modifications.

    **Provide only the Python code, without explanations.**
"""


        # Initialize OpenAI client
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Get response from GPT
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert Python developer."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )

        # Extract only the Python code using regex
        code_match = re.search(r"```python\n(.*?)\n```", response.choices[0].message.content, re.DOTALL)

        if code_match:
            extracted_code = code_match.group(1)
        else:
            extracted_code = response.choices[0].message.content  # Fallback

        if self.store_script:
            # Save the generated script for debugging
            with open(self.store_path, "w") as f:
                f.write(extracted_code)

        # Print the generated script
        # print("Generated Script:\n", extracted_code)

        return extracted_code

    def load_data(self, split_data=True):
        """
        Load the data from the specified file using the generated script.
        The script is dynamically generated to include necessary imports and extract 'X' and 'y'.
        """
        if self.store_path and os.path.exists(self.store_path):
            generated_script = open(self.store_path).read()
        else:
            generated_script = self.generate_script()


        # Create a controlled execution namespace
        local_namespace = {}

        try:
            # Execute the generated script safely
            exec(generated_script, local_namespace, local_namespace)
            
            # Retrieve X and y from the executed script
            X = local_namespace.get("X")
            y = local_namespace.get("y")

            # Print the extracted data
            if X is not None:
                # print("✅ Extracted X:\n", X)
                pass
            else:
                print("⚠️ Warning: 'X' not found in the file.")

            if y is not None:
                # print("✅ Extracted y:\n", y)
                pass
            else:
                print("⚠️ Warning: 'y' not found in the file.")

            # Reshape y properly
            if y.shape[0] == 1 and y.shape[1] == X.shape[0]:  # If y is (1, N), reshape to (N, 1)
                y = y.T  # Transpose to (N, 1)

            # Convert y to 1D if required by train_test_split
            y = y.ravel() if y.shape[1] == 1 else y

            # Ensure X and y now have matching samples
            
            # print('split_data:',split_data)
            if split_data:
                # Split the data into training and testing sets
                # print('splitting data...')
                if X.shape[0] != y.shape[0]:
                    # print(f"❌ Error: Mismatched samples. X has {X.shape[0]} rows, y has {y.shape[0]} rows.")
                    return None, None, None, None
                X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2, random_state=42)
                # print("✅ Split data into training and testing sets.")
                return X_train, X_test, y_train, y_test
            else:
                return X, y  # Return extracted data for further processing

        except Exception as e:
            print(f"❌ Error executing the generated script: {e}")
            return None, None




if __name__ == "__main__":
    # Example usage
    data_loader = DataLoader("data/univariate_data.mat", store_script=True)
    X_train, y_train, X_test, y_test = data_loader.load_data(True)
    print("Length of X_train:", len(X_train))