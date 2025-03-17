import os
import chainlit as cl
from anthropic import Anthropic
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize the Anthropic client
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
model_name = "claude-3-haiku-20240307"  # Equivalent to o1-mini
opening_message = f"Hello! I'm a chatbot powered by Anthropic's {model_name} model. How can I help you today?"

@cl.on_message
async def main(message: cl.Message):
    """
    This function is called every time a user sends a message in the chat.
    """
    # Call Anthropic API
    response = client.messages.create(
        model=model_name,
        max_tokens=1000,
        system="You are a helpful assistant.",
        messages=[
            {"role": "user", "content": message.content}
        ]
    )
    
    # Send the response from Anthropic back to the user
    await cl.Message(
        content=response.content[0].text,
    ).send()

# Chainlit startup message
@cl.on_chat_start
async def start():
    await cl.Message(content=opening_message).send()