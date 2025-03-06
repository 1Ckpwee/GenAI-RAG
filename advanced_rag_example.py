"""
Advanced RAG (Retrieval Augmented Generation) example using Weaviate and Deepseek
This example demonstrates loading documents from files and using them for RAG
"""

import os
import glob
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    CSVLoader,
    UnstructuredMarkdownLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils import (
    get_deepseek_llm,
    get_deepseek_embeddings,
    create_vector_store,
    get_retriever,
    print_with_border
)

# Load environment variables
load_dotenv()

def load_documents_from_directory(directory_path="./documents"):
    """
    Load documents from a directory
    
    Args:
        directory_path (str): Path to the directory containing documents
        
    Returns:
        list: List of Document objects
    """
    # Create documents directory if it doesn't exist
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Created directory: {directory_path}")
        print("Please add your documents to this directory and run the script again.")
        
        # Create a sample document
        with open(os.path.join(directory_path, "sample.txt"), "w") as f:
            f.write("""
            Artificial Intelligence (AI) is the simulation of human intelligence processes by machines, 
            especially computer systems. These processes include learning (the acquisition of information 
            and rules for using the information), reasoning (using rules to reach approximate or definite 
            conclusions) and self-correction. Particular applications of AI include expert systems, speech 
            recognition and machine vision.
            
            Machine Learning is a subset of artificial intelligence that provides systems the ability 
            to automatically learn and improve from experience without being explicitly programmed. 
            Machine learning focuses on the development of computer programs that can access data and 
            use it to learn for themselves.
            """)
        
        return []
    
    documents = []
    
    # Get all files in the directory
    file_paths = glob.glob(os.path.join(directory_path, "**"), recursive=True)
    
    for file_path in file_paths:
        if os.path.isfile(file_path):
            try:
                # Choose the appropriate loader based on file extension
                if file_path.endswith(".txt"):
                    loader = TextLoader(file_path)
                elif file_path.endswith(".pdf"):
                    loader = PyPDFLoader(file_path)
                elif file_path.endswith(".csv"):
                    loader = CSVLoader(file_path)
                elif file_path.endswith(".md"):
                    loader = UnstructuredMarkdownLoader(file_path)
                else:
                    # Skip unsupported file types
                    print(f"Skipping unsupported file: {file_path}")
                    continue
                
                # Load documents
                loaded_docs = loader.load()
                
                # Add source metadata
                for doc in loaded_docs:
                    doc.metadata["source"] = os.path.basename(file_path)
                
                documents.extend(loaded_docs)
                print(f"Loaded {len(loaded_docs)} documents from {file_path}")
            
            except Exception as e:
                print(f"Error loading {file_path}: {str(e)}")
    
    return documents

def setup_advanced_rag_pipeline(documents=None):
    """
    Set up an advanced RAG pipeline using Weaviate and Deepseek
    
    Args:
        documents (list, optional): List of Document objects. If None, documents will be loaded from the documents directory.
        
    Returns:
        tuple: (RAG chain, Retriever)
    """
    # Load documents if not provided
    if documents is None:
        documents = load_documents_from_directory()
    
    # If no documents were loaded, use sample documents
    if not documents:
        print("No documents loaded. Using sample documents.")
        from rag_example import create_sample_documents
        documents = create_sample_documents()
    
    # Create text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    
    # Split documents into chunks
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks")
    
    # Get Deepseek embeddings
    embeddings = get_deepseek_embeddings()
    
    # Create vector store
    vector_store = create_vector_store(chunks, embeddings, index_name="Documents")
    
    # Create retriever
    retriever = get_retriever(vector_store, search_kwargs={"k": 3})
    
    # Get Deepseek LLM
    llm = get_deepseek_llm(temperature=0.7)
    
    # Create prompt template
    template = """
    You are a knowledgeable AI assistant.
    Use the following retrieved information to answer the user's question.
    If you don't know the answer based on the retrieved information, just say that you don't know.
    
    Retrieved information:
    {context}
    
    User question: {question}
    
    Your answer:
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # Create RAG chain
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain, retriever

def advanced_rag_example():
    """
    Demonstrate advanced RAG using Weaviate and Deepseek
    """
    print_with_border("Advanced RAG Example with Weaviate and Deepseek")
    
    # Set up RAG pipeline
    rag_chain, retriever = setup_advanced_rag_pipeline()
    
    # Interactive Q&A
    print("\nEnter your questions (type 'exit' to quit):")
    
    while True:
        question = input("\nQuestion: ")
        
        if question.lower() in ["exit", "quit", "q"]:
            break
        
        # Get retrieved documents for explanation
        retrieved_docs = retriever.invoke(question)
        print(f"Retrieved {len(retrieved_docs)} documents:")
        for i, doc in enumerate(retrieved_docs):
            source = doc.metadata.get("source", "Unknown")
            print(f"  {i+1}. Source: {source}")
        
        # Generate answer
        answer = rag_chain.invoke(question)
        print(f"\nAnswer: {answer}")
        print("-" * 80)

if __name__ == "__main__":
    advanced_rag_example() 