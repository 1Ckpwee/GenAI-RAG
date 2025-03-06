"""
Web server for RAG implementation using Flask, Weaviate, and Deepseek
"""

import os
import json
import uuid
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    CSVLoader,
    UnstructuredMarkdownLoader
)
from utils import (
    get_deepseek_llm,
    get_deepseek_embeddings,
    create_vector_store,
    get_retriever,
    print_with_border
)

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__, static_folder='static')
CORS(app)  # Enable CORS for all routes

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'csv', 'md'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max upload size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Create upload folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Global variables
vector_store = None
retriever = None
llm = None
embeddings = None

def allowed_file(filename):
    """
    Check if the file extension is allowed
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_document(file_path):
    """
    Load a document based on its file extension
    """
    try:
        if file_path.endswith('.txt'):
            loader = TextLoader(file_path)
        elif file_path.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith('.csv'):
            loader = CSVLoader(file_path)
        elif file_path.endswith('.md'):
            loader = UnstructuredMarkdownLoader(file_path)
        else:
            return None
        
        return loader.load()
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
        return None

def initialize_rag_components():
    """
    Initialize the RAG components (embeddings, LLM)
    """
    global embeddings, llm
    
    # Initialize embeddings if not already done
    if embeddings is None:
        embeddings = get_deepseek_embeddings()
    
    # Initialize LLM if not already done
    if llm is None:
        llm = get_deepseek_llm(temperature=0.7)
    
    return embeddings, llm

def update_vector_store(documents):
    """
    Update the vector store with new documents
    """
    global vector_store, retriever, embeddings
    
    # Initialize components if needed
    embeddings, _ = initialize_rag_components()
    
    # Create text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    
    # Split documents into chunks
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks")
    
    # Create or update vector store
    if vector_store is None:
        # Create new vector store
        vector_store = create_vector_store(chunks, embeddings, index_name="WebDocuments")
    else:
        # Add documents to existing vector store
        vector_store.add_documents(chunks)
    
    # Update retriever
    retriever = get_retriever(vector_store, search_kwargs={"k": 3})
    
    return len(chunks)

@app.route('/')
def index():
    """
    Serve the main page
    """
    return render_template('index.html')

@app.route('/static/<path:path>')
def serve_static(path):
    """
    Serve static files
    """
    return send_from_directory('static', path)

@app.route('/upload', methods=['POST'])
def upload_file():
    """
    Handle file upload
    """
    # Check if a file was uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    # Check if the file is empty
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Check if the file type is allowed
    if not allowed_file(file.filename):
        return jsonify({'error': f'File type not allowed. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}'}), 400
    
    try:
        # Generate a unique filename
        original_filename = secure_filename(file.filename)
        file_extension = original_filename.rsplit('.', 1)[1].lower()
        unique_filename = f"{uuid.uuid4().hex}.{file_extension}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        # Save the file
        file.save(file_path)
        
        # Load the document
        documents = load_document(file_path)
        
        if documents is None or len(documents) == 0:
            return jsonify({'error': 'Failed to load document'}), 400
        
        # Add metadata to documents
        for doc in documents:
            doc.metadata['source'] = original_filename
            doc.metadata['path'] = file_path
        
        # Update vector store
        chunks_count = update_vector_store(documents)
        
        return jsonify({
            'success': True,
            'filename': original_filename,
            'document_count': len(documents),
            'chunks_count': chunks_count
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/query', methods=['POST'])
def query():
    """
    Handle RAG query
    """
    global retriever, llm
    
    # Check if vector store is initialized
    if vector_store is None or retriever is None:
        return jsonify({'error': 'No documents have been uploaded yet'}), 400
    
    # Get query from request
    data = request.json
    if not data or 'query' not in data:
        return jsonify({'error': 'No query provided'}), 400
    
    query_text = data['query']
    
    try:
        # Initialize components if needed
        _, llm = initialize_rag_components()
        
        # Retrieve relevant documents
        retrieved_docs = retriever.invoke(query_text)
        
        # Prepare context for LLM
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        
        # Create prompt
        prompt = f"""
        You are a knowledgeable AI assistant.
        Use the following retrieved information to answer the user's question.
        If you don't know the answer based on the retrieved information, just say that you don't know.
        
        Retrieved information:
        {context}
        
        User question: {query_text}
        
        Your answer:
        """
        
        # Generate answer
        answer = llm.invoke(prompt)
        
        # Prepare sources information
        sources = []
        for doc in retrieved_docs:
            sources.append({
                'source': doc.metadata.get('source', 'Unknown'),
                'content': doc.page_content[:200] + '...' if len(doc.page_content) > 200 else doc.page_content
            })
        
        return jsonify({
            'answer': answer.content,
            'sources': sources
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/documents', methods=['GET'])
def list_documents():
    """
    List all uploaded documents
    """
    if not os.path.exists(UPLOAD_FOLDER):
        return jsonify({'documents': []})
    
    documents = []
    for filename in os.listdir(UPLOAD_FOLDER):
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        if os.path.isfile(file_path):
            documents.append({
                'filename': filename,
                'size': os.path.getsize(file_path),
                'uploaded_at': os.path.getctime(file_path)
            })
    
    return jsonify({'documents': documents})

if __name__ == '__main__':
    # Initialize RAG components
    initialize_rag_components()
    print_with_border("RAG Web Server Started")
    print("Access the web interface at http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000) 