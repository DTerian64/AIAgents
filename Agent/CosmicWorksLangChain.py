from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.schema import Document
from openai import AzureOpenAI
import os
import json
from azure.storage.blob import BlobServiceClient
from Helpers.MyCosmosDBHelper import CosmicWorksDb


class CosmicWorksLangChain:
    def __init__(self, chat_model: AzureChatOpenAI = None, embeddings: AzureOpenAIEmbeddings = None):
        
        self.chat_model = chat_model or AzureChatOpenAI(
            openai_api_key=os.getenv("AZURE_OPENAI_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),  # e.g. https://your-resource-name.openai.azure.com
            deployment_name=os.getenv("AZURE_CHAT_DEPLOYMENT"),  # Your deployed chat model name
            openai_api_version="2023-05-15",
            temperature=0.7
        )
        self.embeddings = embeddings or AzureOpenAIEmbeddings(
            openai_api_key=os.getenv("AZURE_OPENAI_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),  # e.g. https://your-resource-name.openai.azure.com
            deployment=os.getenv("AZURE_EMBEDDING_DEPLOYMENT"),  # Your deployed embedding model name
            openai_api_version="2023-05-15",
            chunk_size=1 
        )
    
  
    def create_local_faiss_index(self, container_name: str):
        """Create a local FAISS index from Cosmos DB."""
        
        try:
            cosmos_db = CosmicWorksDb(
                endpoint=os.getenv("COSMOS_ENDPOINT"),
                key=os.getenv("COSMOS_PRIMARY_KEY"),
                database_name=os.getenv("COSMICWORKS_DATABASE_NAME")
            )
            
            items = list(cosmos_db.get_container(container_name).read_all_items())
            documents = [
                Document(
                    page_content=f"{item['name']}. {item['description']} Price: ${item['price']}",
                    metadata={"source": "cosmicworks"}
                )
                for item in items
            ]
            
            vector_store = FAISS.from_documents(documents, self.embeddings)
            vector_store.save_local("faiss_cosmicworks")
            print("FAISS index created and saved localy.")
        except Exception as e:
            print(f"Error creating remote FAISS index: {e}")

    def upload_faiss_index_to_blob(self, local_path="faiss_cosmicworks"):
        conn_str = os.getenv("BLOB_CONNECTION_STRING")
        container = os.getenv("BLOB_CONTAINER_NAME")  # e.g. "vector-index"
        blob_dir = os.getenv("AZURE_FAISS_BLOB_DIR", "faiss_cosmicworks")

        blob_service = BlobServiceClient.from_connection_string(conn_str)
        container_client = blob_service.get_container_client(container)

        for fname in ["index.faiss", "index.pkl"]:
            file_path = f"{local_path}/{fname}"
            blob_name = f"{blob_dir}/{fname}"
            blob_client = container_client.get_blob_client(blob_name)

            with open(file_path, "rb") as f:
                blob_client.upload_blob(f, overwrite=True)

    def download_faiss_index_from_blob(self,local_path="faiss_cosmicworks"):
        print("⏬ Downloading FAISS index from Azure Blob Storage...")

        connection_string = os.getenv("BLOB_CONNECTION_STRING")
        container_name = os.getenv("BLOB_CONTAINER_NAME")
        blob_dir = os.getenv("AZURE_FAISS_BLOB_DIR", "faiss_cosmicworks")

        blob_service = BlobServiceClient.from_connection_string(connection_string)
        container_client = blob_service.get_container_client(container_name)

        os.makedirs(local_path, exist_ok=True)
    
        for filename in ["index.faiss", "index.pkl"]:
            blob_name = f"{blob_dir}/{filename}"
            blob_client = container_client.get_blob_client(blob_name)
            download_path = os.path.join(local_path, filename)

        with open(download_path, "wb") as f:
            download_stream = blob_client.download_blob()
            f.write(download_stream.readall())
    
    def faiss_index_exists(self, local_path="faiss_cosmicworks"):
        """Check if the FAISS index exists locally."""
        return (
            os.path.exists(os.path.join(local_path, "index.faiss")) and
            os.path.exists(os.path.join(local_path, "index.pkl"))
    )

    def get_langchain_retrievalqa_agent(self):
        """Create a LangChain RetrievalQA agent using the FAISS index."""
        print("Loading FAISS index and initializing QA chain...")

        local_faiss_path = "faiss_cosmicworks"

        if not self.faiss_index_exists(local_faiss_path):
            self.download_faiss_index_from_blob(local_faiss_path)

        vector_store = FAISS.load_local(local_faiss_path, self.embeddings, allow_dangerous_deserialization=True)                       
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.chat_model,
            chain_type="stuff",
            retriever=vector_store.as_retriever()
        )
        print("✅ RetrievalQA agent initialized successfully.")
        return qa_chain

    def get_process_langchain(self, query: str) -> str:
        """Answer a query using the RetrievalQA agent."""
        
        try:
            qa_chain = self.get_langchain_retrievalqa_agent()
        
            response = qa_chain.invoke(query)
            return response["result"]  
        except Exception as e:
            print(f"Error in get_process_langchain: {e}")
            return f"Error in get_process_langchain: {e}"
        