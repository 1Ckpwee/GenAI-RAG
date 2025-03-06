"""
Deepseek example using LangChain agents and tools
"""

import os
from datetime import datetime
from utils import get_deepseek_llm, print_with_border
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import BaseTool, StructuredTool, tool
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_core.callbacks import StreamingStdOutCallbackHandler

def agent_with_tools_example():
    """
    Demonstrates how to interact with the Deepseek model using LangChain agents and tools
    """
    print_with_border("Agent and Tools Example")
    
    # Initialize Deepseek LLM
    llm = get_deepseek_llm(temperature=0.7)
    
    # Create tools
    
    # 1. Search tool (simulated)
    @tool
    def search(query: str) -> str:
        """Use this tool when you need to search the internet for information. The input should be a search query."""
        return f"Here are the search results for '{query}': [simulated search results]"
    
    # 2. Math calculation tool
    @tool
    def calculate(expression: str) -> str:
        """Use this tool when you need to perform mathematical calculations. The input should be a mathematical expression."""
        try:
            result = eval(expression)
            return f"Calculation result: {result}"
        except Exception as e:
            return f"Calculation error: {str(e)}"
    
    # 3. Date and time tool
    @tool
    def get_current_date() -> str:
        """Use this tool when you need to know the current date and time. No input parameters required."""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Create agent
    tools = [search, calculate, get_current_date]
    
    # Create agent prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant who can use tools to answer questions."),
        ("human", "{input}"),
        ("user", "Available tools:\n{tools}")
    ])
    
    # Create agent
    agent = create_openai_tools_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    # Use agent to answer questions
    questions = [
        "What date is it today?",
        "Calculate 23 multiplied by 45 plus 17?",
        "Can you tell me about the basic concepts of deep learning?"
    ]
    
    for question in questions:
        print(f"\nHuman: {question}")
        try:
            response = agent_executor.invoke({"input": question})
            print(f"AI Assistant: {response['output']}")
        except Exception as e:
            print(f"An error occurred: {str(e)}")

def streaming_agent_example():
    """
    Demonstrates how to use a LangChain agent with streaming output
    """
    print_with_border("Streaming Agent Example")
    
    # Initialize Deepseek LLM with streaming support
    llm = get_deepseek_llm(
        temperature=0.7,
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()]
    )
    
    # Create tools
    @tool
    def search(query: str) -> str:
        """Use this tool when you need to search the internet for information. The input should be a search query."""
        return f"Here are the search results for '{query}': [simulated search results]"
    
    @tool
    def calculate(expression: str) -> str:
        """Use this tool when you need to perform mathematical calculations. The input should be a mathematical expression."""
        try:
            result = eval(expression)
            return f"Calculation result: {result}"
        except Exception as e:
            return f"Calculation error: {str(e)}"
    
    # Create agent
    tools = [search, calculate]
    
    # Create agent prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant who can use tools to answer questions."),
        ("human", "{input}"),
        ("user", "Available tools:\n{tools}")
    ])
    
    # Create agent
    agent = create_openai_tools_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True,
        handle_parsing_errors=True
    )
    
    # Use streaming agent to answer a question
    question = "Please explain what machine learning is, and calculate what is 15 squared?"
    
    print(f"\nHuman: {question}")
    print("AI Assistant: ", end="", flush=True)
    
    try:
        # Use streaming output
        response = agent_executor.invoke(
            {"input": question},
            {"callbacks": [StreamingStdOutCallbackHandler()]}
        )
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")

if __name__ == "__main__":
    agent_with_tools_example()
    streaming_agent_example() 