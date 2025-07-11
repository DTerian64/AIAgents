from dotenv import load_dotenv
import os
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
import Helpers.MyCosmosDBHelper as MyCosmosDBHelper

from Helpers.Cosmicworks_ai_tool import CosmicworksAITool

load_dotenv()

endpoint = os.getenv("AZURE_ENDPOINT")
model_name = os.getenv("AI_MODEL_NAME")

client = ChatCompletionsClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(os.getenv("AZURE_KEY")),
    api_version="2024-05-01-preview"
)

def get_agent_response(user_input: str) -> str:
    
    (query_type, keyword, query) = get_parse_user_input(user_input)
        
    if query_type == "product":
            
            results = MyCosmosDBHelper.search_products_detailed(keyword)                         
            if not results:
                    context = f"No matching products found for '{keyword}'."
            else:
                     # Use the tool to aggregate
                    tool = CosmicworksAITool()
                    if tool.aggregate_products(results):                        
                        context = tool.generate_ai_context(max_products=10)   
                        print(f"Context: {context}")               
                    else:
                        context = "Error processing product data."     
        
    elif query_type == "employee":
            print("Searching for employees...get_agent_response")
            results = MyCosmosDBHelper.search_employees(keyword)
        
            if not results:
                context = f"No matching employees found for '{keyword}'."
            else:
                names = ", ".join([e["name"] for e in results])
                context = f"Here are the matching employees: {names}."
    else:
        context = ""

    if not context:
         messages=[
            SystemMessage(content="You are a helpful assistant"),
            UserMessage(content=user_input)
        ]
    else:
        messages=[
            SystemMessage(content=f" Use this data:\n{context}"),
            UserMessage(content=query or "Show the data.")
        ]


    # Create a conversation item
    response = client.complete(
        messages=messages,
        max_tokens=2048,
        temperature=0.8,
        top_p=0.1,
        model=model_name
    )
    return response.choices[0].message.content

def get_parse_user_input(user_input: str):
    """
    Parses the user input to extract query type, keyword, and query.
    Returns a tuple of (query_type, keyword, query).
    """
    parts = user_input.split(":", 2)
    
    if len(parts) < 2:
        return None, None, None
    
    query_type = parts[0].strip().lower()
    keyword = parts[1].strip()
    
    if len(parts) > 2:
        query = parts[2].strip()
    else:
        query = None
    
    return query_type, keyword, query