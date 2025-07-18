from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.schema import Document
from openai import AzureOpenAI
import os
import json


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
    
    def create_local_faiss_index(self, file_path: str):
        """Create a local FAISS index from a text file."""
        
        file_path = os.path.expanduser(file_path)
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Step 2: Convert to LangChain Documents
        documents = [
            Document(
                page_content=f"{item['name']}. {item['description']} Price: ${item['price']}",
                metadata={"source": "cosmicworks"}
            )
            for item in data
        ]   
        
       
        try:
            vector_store = FAISS.from_documents(documents, self.embeddings)
            vector_store.save_local("faiss_cosmicworks")
            print("FAISS index created and saved locally.")
        except Exception as e:
            print(f"Error creating FAISS index: {e}")

    def get_langchain_retrievalqa_agent(self):
        """Create a LangChain RetrievalQA agent using the FAISS index."""
        print("Loading FAISS index and initializing QA chain...")
        vector_store = FAISS.load_local("faiss_cosmicworks", self.embeddings, allow_dangerous_deserialization=True)                       
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.chat_model,
            chain_type="stuff",
            retriever=vector_store.as_retriever()
        )
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
        