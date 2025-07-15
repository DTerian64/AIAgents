from dotenv import load_dotenv
import os
from fastapi import FastAPI, Request, HTTPException, status, Depends
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
#import spacy
from transformers import pipeline

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

#classifier = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")

#nlp = spacy.load("en_core_web_sm")

def get_agent_response(user_input: str, knowledgeSource: str) -> str:
    "route AI call to General or Cosmic Works"    
    if knowledgeSource == "cosmic": 
        return get_process_cosmic_langchain(user_input) 
    else:
        return get_process_general_mml(user_input) 
             
    

def get_process_cosmic_langchain(user_input):
    (keyword, query) = parse_natural_query(user_input)        
    if keyword:            
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

def get_process_general_mml(user_input: str):
    messages=[
        SystemMessage(content="You are a helpful assistant"),
        UserMessage(content=user_input)
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


#def parse_natural_query(user_input: str):
    doc = nlp(user_input)

    # Extract entities that might be products
    products = [ent.text for ent in doc.ents if ent.label_ in ["PRODUCT", "ORG", "PERSON"]]

    # Query keywords (lemmas)
    query_keywords = {"search", "find", "show", "get", "know", "have", "about", "info", "count", "many", "what", "who"}

    for i, token in enumerate(doc):
        if token.lemma_.lower() in query_keywords:
            # Try to extract the rest of the sentence after the keyword
            query_part = doc[i+1:].text if i + 1 < len(doc) else ""
            target = products[0] if products else "general"
            return f"{target}:{query_part.strip() or token.text}"

    # Fallback if no keyword matched but there's a product mention
    if products:
        return f"{products[0]}:info"

    return "general:" + user_input.strip()

   
def extract_with_transformers(user_input):
    # Define possible intents — tailor these to your app
    candidate_intents = [
        "inventory_check",
        "product_search",
        "general_question",
        "availability_query",        
        "pricing_request"
    ]

    result = classifier(user_input, candidate_labels=candidate_intents)
    top_intent = result['labels'][0]  # highest-confidence label

    # crude noun extraction
    import re
    nouns = re.findall(r'\b[a-zA-Z]+\b', user_input)
    object_guess = nouns[-1] if nouns else ""

    return {
        "intent": top_intent,
        "object": object_guess.lower(),
        "original_query": user_input
    }