from dotenv import load_dotenv
import os
from fastapi import FastAPI, Request, HTTPException, status, Depends

from openai import AzureOpenAI
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

import Helpers.MyCosmosDBHelper as MyCosmosDBHelper
from Helpers.Cosmicworks_ai_tool import CosmicworksAITool

import Agent.CosmicWorksLangChain as CosmicWorksLangChain

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
        self.embeddings = AzureOpenAIEmbeddings(
            openai_api_key=os.getenv("AZURE_OPENAI_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),  # e.g. https://your-resource-name.openai.azure.com
            deployment=os.getenv("AZURE_EMBEDDING_DEPLOYMENT"),  # Your deployed embedding model name
            openai_api_version="2023-05-15",
            chunk_size=1 
        )

    def _check_env(self):
        for var in ["AZURE_OPENAI_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_CHAT_DEPLOYMENT", "AZURE_EMBEDDING_DEPLOYMENT"]:
            if not os.getenv(var):
                raise ValueError(f"Missing required environment variable: {var}")


    def get_agent_response(self, user_input: str, knowledgeSource: str) -> str:
        "route AI call to General or Cosmic Works"    
        if knowledgeSource == "cosmic_lang": 
            """Call the Cosmic Works RAG chain with user input."""
            try:        
                cosmic_lang = CosmicWorksLangChain.CosmicWorksLangChain(self.chat_model, self.embeddings)               
                return cosmic_lang.get_process_langchain(user_input)
                                             
            except Exception as e:
                print(f"Error in get_agent_response: {e}")

        elif knowledgeSource == "cosmic_rag":
            raise NotImplementedError(f"{knowledgeSource} is not implemented yet.")

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
