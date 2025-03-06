"""
RAG (Retrieval Augmented Generation) example using Weaviate and Deepseek
"""

import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from utils import (
    get_deepseek_llm,
    get_deepseek_embeddings,
    create_vector_store,
    get_retriever,
    print_with_border
)

# Load environment variables
load_dotenv()

def create_sample_documents():
    """
    Create sample documents for the RAG example
    
    Returns:
        list: List of Document objects
    """
    # Sample documents about artificial intelligence
    documents = [
        Document(
            page_content="""
            Artificial Intelligence (AI) is the simulation of human intelligence processes by machines, 
            especially computer systems. These processes include learning (the acquisition of information 
            and rules for using the information), reasoning (using rules to reach approximate or definite 
            conclusions) and self-correction. Particular applications of AI include expert systems, speech 
            recognition and machine vision.
            """,
            metadata={"source": "AI Introduction", "topic": "artificial intelligence"}
        ),
        Document(
            page_content="""
            Machine Learning is a subset of artificial intelligence that provides systems the ability 
            to automatically learn and improve from experience without being explicitly programmed. 
            Machine learning focuses on the development of computer programs that can access data and 
            use it to learn for themselves. The process of learning begins with observations or data, 
            such as examples, direct experience, or instruction, in order to look for patterns in data 
            and make better decisions in the future based on the examples that we provide.
            """,
            metadata={"source": "Machine Learning Basics", "topic": "machine learning"}
        ),
        Document(
            page_content="""
            Deep Learning is a subset of machine learning that uses neural networks with many layers 
            (hence "deep") to analyze various factors of data. Deep learning is a key technology behind 
            driverless cars, enabling them to recognize a stop sign, or to distinguish a pedestrian from 
            a lamppost. It is also used in voice control in consumer devices like phones, tablets, TVs, 
            and hands-free speakers.
            """,
            metadata={"source": "Deep Learning Overview", "topic": "deep learning"}
        ),
        Document(
            page_content="""
            Natural Language Processing (NLP) is a field of artificial intelligence that gives computers 
            the ability to understand text and spoken words in much the same way human beings can. 
            NLP combines computational linguistics—rule-based modeling of human language—with statistical, 
            machine learning, and deep learning models. These technologies enable computers to process 
            human language in the form of text or voice data and to 'understand' its full meaning, 
            complete with the speaker or writer's intent and sentiment.
            """,
            metadata={"source": "NLP Introduction", "topic": "natural language processing"}
        ),
        Document(
            page_content="""
            Computer Vision is a field of artificial intelligence that trains computers to interpret and 
            understand the visual world. Using digital images from cameras and videos and deep learning 
            models, machines can accurately identify and classify objects — and then react to what they "see."
            """,
            metadata={"source": "Computer Vision Basics", "topic": "computer vision"}
        )
    ]
    
    return documents

def setup_rag_pipeline():
    """
    Set up the RAG pipeline using Weaviate and Deepseek
    
    Returns:
        tuple: (RAG chain, Retriever)
    """
    # Create sample documents
    documents = create_sample_documents()
    
    # Get Deepseek embeddings
    embeddings = get_deepseek_embeddings()
    
    # Create vector store
    vector_store = create_vector_store(documents, embeddings, index_name="AITopics")
    
    # Create retriever
    retriever = get_retriever(vector_store, search_kwargs={"k": 2})
    
    # Get Deepseek LLM
    llm = get_deepseek_llm(temperature=0.7)
    
    # Create prompt template
    template = """
    You are an AI assistant specialized in artificial intelligence topics.
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

def rag_example():
    """
    Demonstrate RAG using Weaviate and Deepseek
    """
    print_with_border("RAG Example with Weaviate and Deepseek")
    
    # Set up RAG pipeline
    rag_chain, retriever = setup_rag_pipeline()
    
    # Example questions
    questions = [
        "What is artificial intelligence?",
        "How is machine learning related to AI?",
        "What are some applications of deep learning?",
        "What is NLP used for?",
        "How does computer vision work?"
    ]
    
    # Answer questions using RAG
    for question in questions:
        print(f"\nQuestion: {question}")
        
        # Get retrieved documents for explanation
        retrieved_docs = retriever.invoke(question)
        print(f"Retrieved {len(retrieved_docs)} documents:")
        for i, doc in enumerate(retrieved_docs):
            source = doc.metadata.get("source", "Unknown")
            topic = doc.metadata.get("topic", "Unknown")
            print(f"  {i+1}. Source: {source}, Topic: {topic}")
        
        # Generate answer
        answer = rag_chain.invoke(question)
        print(f"\nAnswer: {answer}\n")
        print("-" * 80)

if __name__ == "__main__":
    rag_example() 