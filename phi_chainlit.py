import chainlit as cl
import ollama

@cl.on_message
async def on_message(message: cl.Message):
    """Handles user messages and fetches AI responses from Ollama."""
    response = ollama.chat(model="phi", messages=[{"role": "user", "content": message.content}])
    reply = response["message"]["content"]
    
    await cl.Message(content=reply).send()

if __name__ == "__main__":
    cl.run()
