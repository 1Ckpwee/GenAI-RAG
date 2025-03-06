"""
Simple LangChain with Deepseek chat example
"""

from utils import get_deepseek_llm, create_simple_chain, print_with_border
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

def simple_completion_example():
    """
    Demonstrates how to use the Deepseek model for simple text generation
    """
    print_with_border("Simple Text Generation Example")
    
    # Initialize Deepseek LLM
    llm = get_deepseek_llm(temperature=0.7)
    
    # Create a simple prompt template
    template = """
    Please write a short introduction for a company named "{company_name}" in the {industry} industry,
    highlighting its {key_feature} feature.
    """
    
    # Create chain
    chain = create_simple_chain(llm, template)
    
    # Run chain
    result = chain.invoke({
        "company_name": "Smart Cloud Technologies",
        "industry": "artificial intelligence",
        "key_feature": "low-code development platform"
    })
    
    print(result)
    print("\n")

def conversation_example():
    """
    Demonstrates how to use the Deepseek model for multi-turn conversations
    """
    print_with_border("Multi-turn Conversation Example")
    
    # Initialize Deepseek LLM
    llm = get_deepseek_llm(temperature=0.7)
    
    # Use message format for conversation
    messages = [
        SystemMessage(content="You are a helpful, creative, and intelligent AI assistant."),
        HumanMessage(content="Can you tell me about the basics of Python?")
    ]
    
    response = llm.invoke(messages)
    print("Human: Can you tell me about the basics of Python?")
    print(f"AI Assistant: {response.content}")
    
    # Continue conversation
    messages.append(response)
    messages.append(HumanMessage(content="What's the difference between lists and tuples in Python?"))
    
    response = llm.invoke(messages)
    print("\nHuman: What's the difference between lists and tuples in Python?")
    print(f"AI Assistant: {response.content}")
    
    # Continue conversation again
    messages.append(response)
    messages.append(HumanMessage(content="Can you give me a simple example of a Python function?"))
    
    response = llm.invoke(messages)
    print("\nHuman: Can you give me a simple example of a Python function?")
    print(f"AI Assistant: {response.content}")

if __name__ == "__main__":
    simple_completion_example()
    conversation_example() 