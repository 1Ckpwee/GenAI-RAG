"""
Demonstrates how to use the streaming output feature of the Deepseek model
"""

import os
from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.callbacks import StreamingStdOutCallbackHandler

# Load environment variables
load_dotenv()

def streaming_example():
    """
    Demonstrates how to use the streaming output feature of the Deepseek model
    """
    print("=" * 80)
    print("Deepseek Streaming Output Example")
    print("=" * 80)
    
    # Initialize Deepseek LLM with streaming enabled
    llm = ChatDeepSeek(
        model="deepseek-chat",
        temperature=0.7,
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()]
    )
    
    # Create messages
    messages = [
        SystemMessage(content="You are a helpful, creative, and intelligent AI assistant."),
        HumanMessage(content="Please write a short article about the applications of artificial intelligence in healthcare, approximately 200 words.")
    ]
    
    print("\nStarting streaming output...\n")
    
    # Call the model with streaming output
    response = llm.invoke(messages)
    
    print("\n\nStreaming output completed!")
    
    # You can also use the stream method for more granular control
    print("\nUsing the stream method for streaming output...\n")
    
    messages = [
        SystemMessage(content="You are a helpful, creative, and intelligent AI assistant."),
        HumanMessage(content="Please list 5 useful Python tips.")
    ]
    
    # Use the stream method
    for chunk in llm.stream(messages):
        # In a real application, you could process each chunk
        # Here we simply print it
        print(chunk.content, end="", flush=True)
    
    print("\n\nStreaming output completed!")

def async_streaming_example():
    """
    Demonstrates how to use the asynchronous streaming output feature of the Deepseek model
    Note: This function needs to be run in an asynchronous environment
    """
    import asyncio
    
    async def run_async_example():
        print("=" * 80)
        print("Deepseek Asynchronous Streaming Output Example")
        print("=" * 80)
        
        # Initialize Deepseek LLM
        llm = ChatDeepSeek(
            model="deepseek-chat",
            temperature=0.7
        )
        
        # Create messages
        messages = [
            SystemMessage(content="You are a helpful, creative, and intelligent AI assistant."),
            HumanMessage(content="Please briefly explain the basic principles of quantum computing.")
        ]
        
        print("\nStarting asynchronous streaming output...\n")
        
        # Use asynchronous streaming output
        async for chunk in llm.astream(messages):
            print(chunk.content, end="", flush=True)
        
        print("\n\nAsynchronous streaming output completed!")
    
    # Run the asynchronous function
    asyncio.run(run_async_example())

if __name__ == "__main__":
    streaming_example()
    # If you want to run the asynchronous example, uncomment the line below
    # async_streaming_example() 