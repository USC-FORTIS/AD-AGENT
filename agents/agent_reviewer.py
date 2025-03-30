import re
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
# Initialize OpenAI LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# promt template
template = PromptTemplate.from_template("""
You are an expert Python developer with deep experience in anomaly detection libraries.

You have the following original code (that caused an error) and the corresponding error message:

--- Original Code ---
{code}

--- Error Message ---
{error_message}

You also have the official documentation for the `{algorithm}` algorithm as follows:

--- BEGIN DOCUMENTATION ---
{algorithm_doc}
--- END DOCUMENTATION ---

Your task is to:
1. Analyze the error message to find the cause of the error.
2. Use the provided official documentation for `{algorithm}` to fix the code accordingly.
3. Identify the data path from the original code (if any). If no valid data path is found, print an appropriate message.
4. Write only **executable** Python code for anomaly detection and do not include any explanations or descriptions.
5. Base your new code strictly original code logic and the official documentation.

IMPORTANT:
- Do NOT input optional or incorrect parameters.
- **In your final answer, output only the Python code.**
""")

class AgentReviewer:
    def review_code(self,code_quality, vectorstore):
        
        revised_code = ""

        if code_quality.error_message != "" and code_quality.review_count < 2:
            print(f"\n=== [Reviewer] Error detected in {code_quality.algorithm} ===")
            algorithm_doc = self.query_docs(code_quality.algorithm, vectorstore)
            print(f"\n=== [Reviewer] Regenerate code for {code_quality.algorithm} ===\n")
            revised_code = llm.invoke(
                template.invoke({
                    "code": code_quality.code,
                    "error_message": code_quality.error_message,
                    "algorithm": code_quality.algorithm,
                    "algorithm_doc": algorithm_doc 
                })
            ).content
            revised_code = self.clean_generated_code(revised_code)
        else:
            if code_quality.error_message != "" and code_quality.review_count >= 2:
                print(f"\n=== [Reviewer] Reaching maximum review count ===")
            revised_code = code_quality.code + "\n# [Reviewer] Code has been reviewed and updated.\n"
            
        # with open("reviewed_script.py", "w", encoding="utf-8") as f:
        #     f.write(revised_code)

        return revised_code
    
    def query_docs(self, algorithm, vectorstore):
        """Searches for relevant documentation based on the query."""
        query = f"class pyod.models.{algorithm}.{algorithm}"
        doc_list = vectorstore.similarity_search(query, k=5)
        algorithm_doc = "\n\n".join([doc.page_content for doc in doc_list])
        return algorithm_doc
    def clean_generated_code(self, code):
        """Removes Markdown code block formatting from LLM output."""
        clean_code = re.sub(r"```(python)?", "", code)
        clean_code = re.sub(r"```", "", clean_code)
        return clean_code.strip()
