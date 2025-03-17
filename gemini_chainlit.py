import chainlit as cl
import google.generativeai as genai
from dotenv import load_dotenv
import os

# Configure the Gemini API
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=gemini_api_key)

# Select the Gemini model
model = genai.GenerativeModel('gemini-1.5-flash-latest')  # Or 'gemini-pro-vision' or another suitable model

@cl.on_message
async def handle_message(message: cl.Message):
    """Handles incoming messages from the user and sends them to the Gemini model for processing."""

    try:
        response = model.generate_content(message.content)

        await cl.Message(content=response.text).send()

    except Exception as e:
        await cl.Message(content=f"Error processing your request: {e}").send()
        print(f"Error during generation: {e}")  # Log the error for debugging

@cl.on_chat_start
async def main():
    """Initializes the chat and sends a welcome message."""
    await cl.Message(content="Hello! Welcome to the Gemini-powered chatbot.  How can I help you today?").send()


# No need for cl.run() if using chainlit run command in terminal
