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
    Initialize and return a Weaviate client using v4 API
    
    Returns:
        weaviate.WeaviateClient: Initialized Weaviate client for v4 API
    """
    import weaviate
    import os
    from weaviate.classes.init import AdditionalConfig, Timeout
    
    weaviate_url = os.getenv("WEAVIATE_URL")
    
    if not weaviate_url:
        # Default to localhost if no URL is provided
        weaviate_url = "http://localhost:8080"
        print(f"WEAVIATE_URL not found, using default: {weaviate_url}")
    
    # Parse URL to extract host and port
    from urllib.parse import urlparse
    parsed_url = urlparse(weaviate_url)
    
    # Determine if connection is secure (https)
    is_secure = parsed_url.scheme == "https"
    
    # Extract host and port
    host = parsed_url.netloc.split(":")[0] or "localhost"
    port = parsed_url.port or (443 if is_secure else 8080)
    
    # Default gRPC port is typically 50051
    grpc_port = 50051
    
    # Configure timeouts
    additional_config = AdditionalConfig(
        timeout=Timeout(init=30, query=60, insert=120)  # Values in seconds
    )
    
    # Connect to custom Weaviate instance
    try:
        # For local instances, we don't need authentication
        client = weaviate.connect_to_custom(
            http_host=host,
            http_port=port,
            http_secure=is_secure,
            grpc_host=host,
            grpc_port=grpc_port,
            grpc_secure=is_secure,
            additional_config=additional_config
        )
        
        # Check if connection works - in v4 we can use the meta endpoint
        client.get_meta()
        print("Successfully connected to Weaviate")
        return client
    except Exception as e:
        raise ValueError(f"Failed to connect to Weaviate: {e}")

def create_weaviate_schema(client, index_name="Documents"):
    """
    Create and configure a Weaviate schema optimized for LangChain documents
    
    Args:
        client: Weaviate client instance
        index_name (str): Name of the collection to create
        
    Returns:
        The created collection object
    """
    import weaviate.classes as wvc
    
    # Check if collection exists and delete if it does
    try:
        # In v4, we need to check collection existence differently
        collection_exists = False
        try:
            # Try to get the collection - if it exists, this will succeed
            client.collections.get(index_name)
            collection_exists = True
        except Exception:
            # If we get here, the collection doesn't exist
            collection_exists = False
            
        if collection_exists:
            print(f"Collection {index_name} already exists, deleting it...")
            client.collections.delete(index_name)
            print(f"Deleted collection {index_name}")
    except Exception as e:
        print(f"Error checking/deleting collection: {e}")
    
    try:
        # Define properties for the collection
        properties = [
            wvc.config.Property(
                name="text",
                data_type=wvc.config.DataType.TEXT,
                description="The content of the document chunk"
            ),
            wvc.config.Property(
                name="source",
                data_type=wvc.config.DataType.TEXT,
                description="The source of the document"
            ),
            wvc.config.Property(
                name="metadata",
                data_type=wvc.config.DataType.OBJECT,
                description="Additional metadata about the document"
            )
        ]
        
        # Create the collection with vectorizer configuration
        collection = client.collections.create(
            name=index_name,
            description="Collection for storing document embeddings",
            properties=properties,
            # Use "none" vectorizer since we'll provide vectors from LangChain
            vectorizer_config=wvc.config.Configure.Vectorizer.none(),
            # Configure vector index
            vector_index_config=wvc.config.Configure.VectorIndex.hnsw(
                distance_metric=wvc.config.VectorDistances.COSINE
            )
        )
        
        print(f"Created Weaviate collection: {index_name}")
        return collection
    except Exception as e:
        print(f"Error creating collection: {e}")
        raise e

def create_vector_store(texts, embeddings, index_name="Documents"):
    """
    Create a Weaviate vector store from texts
    
    Args:
        texts (list): List of text documents
        embeddings: Embeddings model
        index_name (str): Name of the Weaviate collection
        
    Returns:
        WeaviateVectorStore: Initialized Weaviate vector store
    """
    from langchain_weaviate import WeaviateVectorStore
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    import time
    
    client = get_weaviate_client()
    
    try:
        # Create schema with proper configuration
        collection = create_weaviate_schema(client, index_name)
        
        # Give Weaviate a moment to set up the collection
        time.sleep(2)
        
        # Create text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        # Split texts into chunks
        chunks = text_splitter.split_documents(texts)
        print(f"Split documents into {len(chunks)} chunks")
        
        # Create vector store with explicit configuration
        try:
            # First try the newer approach
            vector_store = WeaviateVectorStore(
                client=client,
                index_name=index_name,
                text_key="text",
                embedding=embeddings,
                by_text=False  # Use embeddings from LangChain
            )
            
            # Add documents in batches to avoid overwhelming the server
            batch_size = 50
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i+batch_size]
                vector_store.add_documents(batch)
                print(f"Added batch of {len(batch)} documents ({i+len(batch)}/{len(chunks)})")
                
        except Exception as inner_e:
            print(f"First approach failed: {str(inner_e)}, trying alternative approach...")
            
            # Fall back to the from_documents approach
            vector_store = WeaviateVectorStore.from_documents(
                documents=chunks,
                embedding=embeddings,
                client=client,
                index_name=index_name,
                text_key="text"
            )
        
        print(f"Successfully created vector store with {len(chunks)} chunks")
        return vector_store
    except Exception as e:
        print(f"Error creating vector store: {str(e)}")
        client.close()  # Ensure client is closed on error
        raise e

def get_retriever(vector_store, search_kwargs=None):
    """
    Get a retriever from a vector store with configurable search parameters
    
    Args:
        vector_store: Vector store instance
        search_kwargs (dict): Search parameters, including:
            - k (int): Number of documents to retrieve (default: 4)
            - score_threshold (float): Minimum similarity score (0-1)
            - fetch_k (int): Number of documents to fetch before filtering
            - lambda_mult (float): Weight for hybrid search (0=sparse, 1=dense)
        
    Returns:
        Retriever: A retriever instance configured for the vector store
    """
    if search_kwargs is None:
        search_kwargs = {
            "k": 4,  # Number of documents to return
            "score_threshold": 0.7,  # Minimum similarity score (0-1)
            "fetch_k": 20,  # Fetch more documents than needed for better filtering
        }
    
    # Create retriever with specified search parameters
    retriever = vector_store.as_retriever(search_kwargs=search_kwargs)
    
    # Print retriever configuration
    print(f"Created retriever with search parameters: {search_kwargs}")
    
    return retriever

def close_weaviate_client(client):
    """
    Safely close a Weaviate client connection
    
    Args:
        client: Weaviate client instance to close
    """
    if client is not None:
        try:
            client.close()
            print("Weaviate client connection closed")
        except Exception as e:
            print(f"Warning: Error closing Weaviate client: {e}") 