import os
import chainlit as cl
from openai import AzureOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize the Azure OpenAI client
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

# In Azure, you need to use the deployment name as the model name
model_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
opening_message = f"Hello! I'm a chatbot powered by {model_name} @ Azure OpenAI. How can I help you today?"

@cl.on_message
async def main(message: cl.Message):
    """
    This function is called every time a user sends a message in the chat.
    """
    # Call Azure OpenAI API
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": message.content}
        ]
    )
    
    # Send the response from Azure OpenAI back to the user
    await cl.Message(
        content=response.choices[0].message.content,
    ).send()

# Chainlit startup message
@cl.on_chat_start
async def start():
    await cl.Message(content=opening_message).send()