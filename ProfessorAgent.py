"""Run this model in Python

> pip install azure-ai-inference
"""
import os
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import AssistantMessage, SystemMessage, UserMessage, ToolMessage
from azure.ai.inference.models import ImageContentItem, ImageUrl, TextContentItem
from azure.core.credentials import AzureKeyCredential

endpoint = "https://dterian64-6460-resource.services.ai.azure.com/models"
model_name = "Ministral-3B"

client = ChatCompletionsClient(
    endpoint=endpoint,
    credential=AzureKeyCredential("D5f65YQYa9Nl1UX6LtwBGAS48SILaTxLzz4gHU9Dz14lw2OtgbRdJQQJ99BGACHYHv6XJ3w3AAAAACOGkCy2"),
    api_version="2024-05-01-preview"
)
messages = [
    SystemMessage(content = "You are a professor in university teaching math"),
    UserMessage(content = [
        TextContentItem(text = "Explain to me the {{topic}} in simple terms"),
    ]),
]

tools = []

response_format = "text"

while True:
    response = client.complete(
        messages = messages,
        model = model_name,
        tools = tools,
        response_format = response_format,
        max_tokens=2048,
        temperature = 1,
        top_p = 1,
    )

    if response.choices[0].message.tool_calls:
        print(response.choices[0].message.tool_calls)
        messages.append(response.choices[0].message)
        for tool_call in response.choices[0].message.tool_calls:
            messages.append(ToolMessage(
                content=locals()[tool_call.function.name](),
                tool_call_id=tool_call.id,
            ))
    else:
        print(response.choices[0].message.content)
        break
