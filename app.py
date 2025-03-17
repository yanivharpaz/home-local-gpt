import chainlit as cl
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

# Initialize the OpenAI client with your API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@cl.on_message
async def on_message(message: cl.Message):
    """Handles incoming messages and generates a response using OpenAI."""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": message.content}]
    )
    
    reply = response.choices[0].message.content
    await cl.Message(content=reply).send()

if __name__ == "__main__":
    cl.run()