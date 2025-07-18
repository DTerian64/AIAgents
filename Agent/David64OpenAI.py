from dotenv import load_dotenv
import os
from fastapi import FastAPI, Request, HTTPException, status, Depends

from openai import AzureOpenAI
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

import Helpers.MyCosmosDBHelper as MyCosmosDBHelper
from Helpers.Cosmicworks_ai_tool import CosmicworksAITool

import Agent.CosmicWorksRagChain_ChatGPT as CosmicWorksRAGChain_ChatGPT

class David64OpenAI:
    """A class to handle OpenAI API interactions."""
    
    def __init__(self):
        # Load environment variables
        load_dotenv()
        self._check_env()
        
        self.chat_model = AzureChatOpenAI(
            openai_api_key=os.getenv("AZURE_OPENAI_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),  # e.g. https://your-resource-name.openai.azure.com
            deployment_name=os.getenv("AZURE_CHAT_DEPLOYMENT"),  # Your deployed chat model name
            openai_api_version="2023-05-15",
            temperature=0.7
        )

    def _check_env(self):
        for var in ["AZURE_OPENAI_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_CHAT_DEPLOYMENT", "AZURE_EMBEDDING_DEPLOYMENT"]:
            if not os.getenv(var):
                raise ValueError(f"Missing required environment variable: {var}")


    def get_agent_response(self, user_input: str, knowledgeSource: str) -> str:
        "route AI call to General or Cosmic Works"    
        if knowledgeSource == "cosmic": 
            """Call the Cosmic Works RAG chain with user input."""
            try:        
                cosmic_rag = CosmicWorksRAGChain_ChatGPT.CosmicWorksRAGChain_ChatGPT(self.chat_model)
                cosmic_rag.create_local_faiss_index(file_path=r"C:\Users\David\source\repos\AIFoundry\Agents\App_data\sample_products.json")
                return cosmic_rag.get_process_langchain(user_input)
                                             
            except Exception as e:
                print(f"Error in get_agent_response: {e}")
            
        else:
            """Call the General MML model with user input."""
            return self.get_process_general_mml(user_input) 
             

    def get_process_general_mml(self, user_input: str) -> str:
            messages=[
                SystemMessage(content="You are a helpful assistant"),
                HumanMessage(content=user_input)
                ]

            # Create a conversation item
            response = self.chat_model.invoke(messages)
                
            return response.content
