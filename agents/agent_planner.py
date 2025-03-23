from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
import os


class AgentPlanner:
    def __init__(self, user_input):
        self.tools = user_input['algorithm']
        self.parameters = user_input['parameters']
        self.data_path = user_input['dataset']
        self.documents = self.load_and_split_documents()
        self.vectorstore = self.build_vectorstore(self.documents)

    def load_and_split_documents(self,folder_path="./docs"):
      """
      load ./docs txt doc, divided into small blocks。
      """
      documents = []
      text_splitter = CharacterTextSplitter(separator="\n", chunk_size=700, chunk_overlap=150)

      for filename in os.listdir(folder_path):
         if filename.endswith(".txt"):
               file_path = os.path.join(folder_path, filename)
               with open(file_path, "r", encoding="utf-8") as file:
                  text = file.read()
                  chunks = text_splitter.split_text(text)
                  documents.extend(chunks)

      return documents
    def build_vectorstore(self,documents):
      """
      The segmented document blocks are converted into vectors and stored in the FAISS vector database.
      """
      embedding = OpenAIEmbeddings()
      vectorstore = FAISS.from_texts(documents, embedding)
      return vectorstore
    def generate_tools(self):
        """Generates the tools for the agent."""
        return ['ECOD', 'ABOD', 'FastABOD', 'COPOD', 'MAD', 'SOS', 'QMCD', 'KDE', 'Sampling', 'GMM', 'PCA', 'KPCA', 'MCD', 'CD', 'OCSVM', 'LMDD', 'LOF', 'COF', '(Incremental) COF', 'CBLOF', 'LOCI', 'HBOS', 'kNN', 'AvgKNN', 'MedKNN', 'SOD', 'ROD', 'IForest', 'INNE', 'DIF', 'FeatureBagging', 'LSCP', 'XGBOD', 'LODA', 'SUOD', 'AutoEncoder', 'VAE', 'Beta-VAE', 'SO_GAAL', 'MO_GAAL', 'DeepSVDD', 'AnoGAN', 'ALAD', 'AE1SVM', 'DevNet', 'R-Graph', 'LUNAR']
        # return ['DeepSVDD']

if __name__ == "__main__":
    agentPlanner = AgentPlanner()
    print(f"Tools: {agentPlanner.tools}")