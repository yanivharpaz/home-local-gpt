import os
import chainlit as cl
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize the OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
model_name = "o1" # or "gpt-4o" or "gpt-3.5-turbo"
opening_message = f"Hello! I'm a chatbot powered by OpenAI's {model_name} model. How can I help you today?"

@cl.on_message
async def main(message: cl.Message):
    """
    This function is called every time a user sends a message in the chat.
    """
    # Call OpenAI API
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": message.content}
        ]
    )
    
    # Send the response from OpenAI back to the user
    await cl.Message(
        content=response.choices[0].message.content,
    ).send()

# Chainlit startup message
@cl.on_chat_start
async def start():
    await cl.Message(content=opening_message).send()