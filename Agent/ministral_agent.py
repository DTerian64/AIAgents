from dotenv import load_dotenv
import os
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential

load_dotenv()

endpoint = os.getenv("AZURE_ENDPOINT")
model_name = os.getenv("AI_MODEL_NAME")

client = ChatCompletionsClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(os.getenv("AZURE_KEY")),
    api_version="2024-05-01-preview"
)

def get_agent_response(user_input: str) -> str:
    response = client.complete(
        messages=[
            SystemMessage(content="You are a helpful assistant."),
            UserMessage(content=user_input),
        ],
        max_tokens=2048,
        temperature=0.8,
        top_p=0.1,
        model=model_name
    )
    return response.choices[0].message.content