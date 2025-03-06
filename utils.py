"""
Helper functions module for LangChain with Deepseek example project
"""

import os
from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_weaviate import WeaviateVectorStore
import weaviate
from weaviate.auth import AuthApiKey
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

def get_deepseek_llm(temperature=0.7, max_tokens=1024, model_name="deepseek-chat", streaming=False, callbacks=None):
    """
    Initialize and return a Deepseek LLM instance
    
    Args:
        temperature (float): Randomness of generated text, higher values increase randomness
        max_tokens (int): Maximum number of tokens in generated text
        model_name (str): Name of the Deepseek model to use
        streaming (bool): Whether to enable streaming output
        callbacks (list): List of callback functions for handling streaming output
        
    Returns:
        ChatDeepSeek: Initialized Deepseek LLM instance
    """
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY environment variable not found, please set it in the .env file")
    
    return ChatDeepSeek(
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        streaming=streaming,
        callbacks=callbacks
    )

def get_deepseek_embeddings():
    """
    Initialize and return a HuggingFace embeddings model using multi-qa-MiniLM-L6-cos-v1
    
    Returns:
        HuggingFaceEmbeddings: Initialized HuggingFace embeddings model
    """
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

def create_simple_chain(llm, template):
    """
    Create a simple LangChain chain
    
    Args:
        llm: Language model instance
        template (str): Prompt template string
        
    Returns:
        Chain: A LangChain chain
    """
    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    return chain

def print_with_border(text, border_char="=", width=80):
    """
    Print text with a border
    
    Args:
        text (str): Text to print
        border_char (str): Border character
        width (int): Border width
    """
    border = border_char * width
    print(border)
    print(text)
    print(border)

def get_weaviate_client():
    """
    Initialize and return a Weaviate client
    
    Returns:
        weaviate.WeaviateClient: Initialized Weaviate client
    """
    weaviate_url = os.getenv("WEAVIATE_URL")
    weaviate_grpc_port = os.getenv("WEAVIATE_GRPC_PORT", "50051")  # Default gRPC port is 50051
    
    if not weaviate_url:
        raise ValueError("WEAVIATE_URL environment variable not found, please set it in the .env file")
    
    
    # Using the new WeaviateClient API (v4)
    return weaviate.WeaviateClient(
        connection_params=weaviate.ConnectionParams.from_url(
            url=weaviate_url,
            grpc_port=int(weaviate_grpc_port),  # Convert to integer as required
        )
    )

def create_vector_store(texts, embeddings, index_name="Documents"):
    """
    Create a Weaviate vector store from texts
    
    Args:
        texts (list): List of text documents
        embeddings: Embeddings model
        index_name (str): Name of the Weaviate class/index
        
    Returns:
        WeaviateVectorStore: Initialized Weaviate vector store
    """
    client = get_weaviate_client()
    
    # Check if class exists and delete if it does
    if client.collections.exists(index_name):
        client.collections.delete(index_name)
    
    # Create text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    
    # Split texts into chunks
    chunks = text_splitter.split_documents(texts)
    
    # Create vector store
    vector_store = WeaviateVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        client=client,
        index_name=index_name,
        text_key="text"
    )
    
    return vector_store

def get_retriever(vector_store, search_kwargs=None):
    """
    Get a retriever from a vector store
    
    Args:
        vector_store: Vector store instance
        search_kwargs (dict): Search parameters
        
    Returns:
        Retriever: A retriever instance
    """
    if search_kwargs is None:
        search_kwargs = {"k": 4}
    
    return vector_store.as_retriever(search_kwargs=search_kwargs) 