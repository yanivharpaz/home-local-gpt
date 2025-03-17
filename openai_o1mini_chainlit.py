import os
import chainlit as cl
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize the OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
model_name = "o1-mini"
opening_message = f"Hello! I'm a chatbot powered by OpenAI's {model_name} model. How can I help you today?"

@cl.on_message
async def main(message: cl.Message):
    """
    This function is called every time a user sends a message in the chat.
    """
    # Call OpenAI API
    # o1-mini doesn't support system messages, so we'll include instructions in user message
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "user", "content": "You are a helpful assistant. Please respond to the following: " + message.content}
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