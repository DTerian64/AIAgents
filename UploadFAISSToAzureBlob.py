from dotenv import load_dotenv
import os
from fastapi import FastAPI, Request, HTTPException, status, Depends

from openai import AzureOpenAI
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

import Helpers.MyCosmosDBHelper as MyCosmosDBHelper
from Helpers.Cosmicworks_ai_tool import CosmicworksAITool

import Agent.CosmicWorksLangChain as CosmicWorksLangChain

def main():
    load_dotenv()
    print("Starting FAISS index upload to Azure Blob Storage...")
    chat_model = AzureChatOpenAI(
            openai_api_key=os.getenv("AZURE_OPENAI_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),  # e.g. https://your-resource-name.openai.azure.com
            deployment_name=os.getenv("AZURE_CHAT_DEPLOYMENT"),  # Your deployed chat model name
            openai_api_version="2023-05-15",
            temperature=0.7
        )
    embeddings = AzureOpenAIEmbeddings(
            openai_api_key=os.getenv("AZURE_OPENAI_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),  # e.g. https://your-resource-name.openai.azure.com
            deployment=os.getenv("AZURE_EMBEDDING_DEPLOYMENT"),  # Your deployed embedding model name
            openai_api_version="2023-05-15",
            chunk_size=1 
        )

    cosmic_lang = CosmicWorksLangChain.CosmicWorksLangChain(chat_model, embeddings)               
    cosmic_lang.create_local_faiss_index(container_name="products")
    cosmic_lang.upload_faiss_index_to_blob(local_path="faiss_cosmicworks")
    #cosmic_lang.download_faiss_index_from_blob(local_path="faiss_cosmicworks")

if __name__ == "__main__":
    main()
    