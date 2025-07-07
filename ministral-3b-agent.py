import os
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential

endpoint = "https://dterian64-6460-resource.services.ai.azure.com/models"
model_name = "Ministral-3B"

client = ChatCompletionsClient(
    endpoint=endpoint,
    credential=AzureKeyCredential("D5f65YQYa9Nl1UX6LtwBGAS48SILaTxLzz4gHU9Dz14lw2OtgbRdJQQJ99BGACHYHv6XJ3w3AAAAACOGkCy2"),
    api_version="2024-05-01-preview"
)

response = client.complete(
    messages=[
        SystemMessage(content="You are a helpful assistant."),
        UserMessage(content="I am going to Paris, what should I see?"),
    ],
    max_tokens=2048,
    temperature=0.8,
    top_p=0.1,
    model=model_name
)

print(response.choices[0].message.content)