import openai
import os
import re
import time

import sklearn
# from orion.data import load_signal


class DataLoader:
    """
    A class to load various file formats (.mat, .csv, .json, etc.) and extract data.
    """

    def __init__(
        self,
        filepath,
        desc='',
        store_script=False,
        store_path='generated_data_loader.py',
        max_retries=3,
        retry_interval=1.0,
    ):
        """
        Initialize DataLoader with the path to the file.
        """
        self.filepath = filepath
        self.desc = desc
        self.max_retries = max_retries
        self.retry_interval = retry_interval

        self.X_name = 'X'
        self.y_name = 'y'


        attempt = 0
        while attempt < self.max_retries:
            if os.path.exists(self.filepath):
                break
            attempt += 1
            print(f"⚠️ File not found (attempt {attempt}/{self.max_retries}): {self.filepath}")
            if attempt < self.max_retries:
                time.sleep(self.retry_interval)
        else:
            raise FileNotFoundError(
                f"File not found after {self.max_retries} attempts: {self.filepath}"
            )
        
        self.store_script = store_script
        self.store_path = store_path
        
    def generate_script_for_data_head(self):
        file_path = self.filepath.replace("\\", "/")  # Normalize for cross-platform compatibility
        file_type = self.filepath.split('.')[-1]  # Extract file extension

        prompt = f'''
        Write a Python script that:
    
        1. **Includes all necessary imports at the beginning** (e.g., `os`, `scipy.io`, `pandas`, `json`).
        2. Checks if the file exists before attempting to load it.
        3. Determines the file type based on the extension (`{file_type}`).
        4. Loads the file using the appropriate Python library
        5. If the file doesn't exist, print a warning and exit.
        6. store the head or the structure of the data to a variable named `head`.

        Requirement: head is a str which descripbes the structure of the data. It can be the first few rows of the data, or the keys of the data, or any other information that can help understand the data. You can decide how head looks like based on the data type.

        If the file is graph data (e.g., a .pt file containing a PyTorch Geometric Data object), do not try to load or inspect the data. Just generate this line of code:
        head = "graph"  
        Do not include any other imports, file loading logic, or extra code. Only output this one line.
        Do not include any print
        You need to add type(y) is str before this y != "Unsupervised" 
        
        **Example Code Execution:**
        The generated script will be executed like this:
        
            # Execute the generated script safely
            exec(generated_script, {{}}, local_namespace)
            
            # Retrieve X and y from the executed script
            head = local_namespace.get("head")

        Ensure the script works with `{file_path}` and does not require manual modifications.

        **Provide only the Python code, without explanations.**
        '''

         # Initialize OpenAI client
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Get response from GPT
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert Python developer."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
        )

        # Extract only the Python code using regex
        code_match = re.search(r"```python\n(.*?)\n```", response.choices[0].message.content, re.DOTALL)

        if code_match:
            extracted_code = code_match.group(1)
        else:
            extracted_code = response.choices[0].message.content  # Fallback

        if self.store_script:
            # Save the generated script for debugging
            with open('head_' + self.store_path, "w") as f:
                f.write(extracted_code)

        # Print the generated script
        # print("Generated Script:\n", extracted_code)

        return extracted_code


    def generate_script(self):
        """
        Generates a Python script using GPT-4 to load a data file and extract its content.
        """

        # Ensure self.filepath is correctly formatted
        file_path = self.filepath.replace("\\", "/")  # Normalize for cross-platform compatibility
        file_type = self.filepath.split('.')[-1]  # Extract file extension

        prompt = f"""
Write a Python script that:

1. **Includes all necessary imports** (`os`, `scipy.io`, `pandas`, `json`, `numpy`).
2. Checks if the file exists using `os.path.exists("{file_path}")`. If it doesn't, print a clear error and exit.
3. Determines the file type based on the extension: `{file_type}`.
4. Loads the file using the appropriate method:
    - `scipy.io.loadmat("{file_path}")` for `.mat`
    - `pandas.read_csv("{file_path}")` for `.csv`
    - `json.load(open("{file_path}", 'r'))` for `.json`
    - `torch.load("{file_path}", weights_only=False)` for `.pt`
5. Dynamically identify and extract `X` and `y`:
    - If the dataset is **unsupervised**, set `y = "Unsupervised"` and only extract `X`.
    - Use the following head preview to decide which columns or keys represent features (X) and which represent labels (y):
        ------
        {self.head}
        ------
    - For `.csv`:
        - If there is a column clearly acting as a target (like "label", "target", or the last column), extract it as `y`.
        - Otherwise, assume the dataset is **unsupervised** and set `y = "Unsupervised"`.
    - For `.mat`:
        - Use `data.keys()` to explore and guess which keys represent feature and label arrays.
        - If no label key is obvious, set `y = "Unsupervised"`.
    - For `.json`:
        - If both `X` and `y` keys exist, extract them.
        - If only `X` exists, set `y = "Unsupervised"`.

7.
    - Ensure `X` is a 2D NumPy array. If `y` is present and numeric, ensure it is 1D.
    - Avoid ambiguous NumPy conditions. Do not use if X, if y, or similar conditions that rely on the truth value of a NumPy array. Instead, check for X is not None, y is not None, or use X.shape, len(X), or X.size explicitly.
    - If a key (like 'X' or 'y') is missing, do not assign default arrays like np.array([]). Instead, set X = np.empty((0, 0)) or raise an informative warning.
8. Ensure the script runs correctly when executed like:

        exec(generated_script, {{}}, local_namespace)
        X = local_namespace.get("X")
        y = local_namespace.get("y")
    But doe not add this code . 

Do not generate conditional logic to check the file type. The file extension is already known at the time of code generation. Generate code specifically for the given file type without using if or elif statements.

**Return only the Python code.**
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
            temperature=0,
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

    def generate_graph_script(self):
        """
        Generates a Python script using GPT-4 to load a data file and extract its content.
        """

        # Ensure self.filepath is correctly formatted
        file_path = self.filepath.replace("\\", "/")  # Normalize for cross-platform compatibility
        file_type = self.filepath.split('.')[-1]  # Extract file extension

        prompt = f"""
Write a Python script that: 
the file is highly likely a graph data.

1. **Includes all necessary imports** (`os`, `scipy.io`, `pandas`, `json`, `numpy`).
2. Determines the file type based on the extension: `{file_type}`.
3. Load the file using the appropriate method. For example: 
    `torch.load("{file_path}", weights_only=False)` for `.pt`
    for other file, use proper way to load the data
4. store the data in variable call `X`, and set y = "graph"
5. Ensure the script runs correctly when executed like:

        exec(generated_script, {{}}, local_namespace)
        X = local_namespace.get("X")
        y = local_namespace.get("y")

Do not generate if statment code for file type because file type is already given.

**Return only the Python code.**

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
            temperature=0,
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
    
    def load_data(self, split_data=False):
        """
        Load the data from the specified file using the generated script.
        The script is dynamically generated to include necessary imports and extract 'X' and 'y'.
        """

        # It is hard to load tslib data using the generated script, so we need to handle it separately.
        # TODO: research about better way to load the data for tslib
        tslib_data = ['MSL', 'PSM', 'SMAP', 'SMD', 'SWaT']
        if any(self.filepath.endswith(ds) for ds in tslib_data):
            X = 'tslib'
            Y = 'tslib'
            return X, Y


        if self.store_script and  'head_' + self.store_path and os.path.exists('head_' + self.store_path):
            head_script = open('head_' + self.store_path).read()
        else:
            head_script = self.generate_script_for_data_head()

        # print("Head Script:\n", head_script)
        
        head = None
        head_attempt = 0
        while head_attempt < self.max_retries:
            local_namespace = {}
            try:
                exec(head_script, local_namespace)
                head = local_namespace.get("head")
                if head is not None:
                    self.head = head
                    break
                print(
                    f"⚠️ Warning: 'head' not found (attempt {head_attempt + 1}/{self.max_retries})."
                )
            except Exception as e:
                print(
                    f"❌ Error executing head script (attempt {head_attempt + 1}/{self.max_retries}): {e}"
                )
            head_attempt += 1
            if head_attempt < self.max_retries:
                time.sleep(self.retry_interval)
        else:
            print(f"❌ Exceeded max retries ({self.max_retries}) while loading head. Quit.")
            return None, None

        # ## determine if the head is time series
        # if 'tiemstamp' in self.head.lower() or 'time' in self.head.lower():
        #     return 'time-series', 'time-series'

        if self.store_script and self.store_path and os.path.exists(self.store_path):
            generated_script = open(self.store_path).read()
        else:
            if self.head == 'graph':
                generated_script = self.generate_graph_script()
            else:
                generated_script = self.generate_script()


        # Create a controlled execution namespace
        local_namespace = {}

        exec_attempt = 0
        while exec_attempt < self.max_retries:
            local_namespace = {}
            try:
                # Execute the generated script safely
                exec(generated_script, local_namespace, local_namespace)
                
                # Retrieve X and y from the executed script
                X = local_namespace.get("X")
                y = local_namespace.get("y")


            # Print the extracted data
            # if X is not None:
            #     print("✅ Extracted X:\n", X)
            # else:
            #     print("⚠️ Warning: 'X' not found in the file.")
            
                if type(y) is str and y == 'graph':
                    return X, y
            
                if type(y) is str and y == 'Unsupervised':
                    if split_data:
                        return X, None, None, None
                    else:
                        return X, None
            
                if 'tiemstamp' in self.head.lower() or 'time' in self.head.lower():
                    return X, 'time-series'

            # if y is not None:
            #     print("✅ Extracted y:\n", y)
            # else:
            #     print("⚠️ Warning: 'y' not found in the file.")

            # Reshape y properly
                if y.shape[0] == 1 and y.shape[1] == X.shape[0]:
                    y = y.T

            # Convert y to 1D if required by train_test_split
                if len(y.shape) > 1 and y.shape[1] == 1:
                    y = y.ravel()


            # Ensure X and y now have matching samples
            
                if split_data:
                    if X.shape[0] != y.shape[0]:
                        print(
                            f"❌ Error: Mismatched samples. X has {X.shape[0]} rows, y has {y.shape[0]} rows."
                        )
                        return None, None, None, None
                    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
                        X, y, test_size=0.2, random_state=42
                    )
                    print("✅ Split data into training and testing sets.")
                    return X_train, X_test, y_train, y_test
                else:
                    return X, y
            except Exception as e:
                print(
                    f"❌ Error executing generated script (attempt {exec_attempt + 1}/{self.max_retries}): {e}"
                )
                exec_attempt += 1
                if exec_attempt < self.max_retries:
                    time.sleep(self.retry_interval)
        print(f"❌ Exceeded max retries ({self.max_retries}) while loading data. Quit.")
        return (None, None, None, None) if split_data else (None, None)




if __name__ == "__main__":
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from config.config import Config
    os.environ['OPENAI_API_KEY'] = Config.OPENAI_API_KEY
    # Example usage
    # file = 'data/yahoo_train.csv'
    # from orion.data import load_signal
    # print("Loading data from:", file)
    # data = load_signal(file)
    # print(data)
    # exit()

    if os.path.exists('head_generated_data_loader.py'):
        os.remove('head_generated_data_loader.py')
    if os.path.exists('generated_data_loader.py'):
        os.remove('generated_data_loader.py')

    data_loader = DataLoader("data/MSL", store_script=True)
    X_train, y_train = data_loader.load_data(split_data=False)

    print(X_train)
    print(y_train)

    print(len(X_train))
    #Run IForest on ./data/glass_train.mat and ./data/glass_test.mat with contamination=0.1
