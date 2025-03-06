#!/bin/bash

# Check if Python virtual environment exists
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
else
    echo "Virtual environment not found. Creating one..."
    python -m venv .venv
    source .venv/bin/activate
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

# Check if Weaviate is running
if curl -s -o /dev/null -w "%{http_code}" http://localhost:8080/v1/.well-known/ready | grep -q "200"; then
    echo "Weaviate is running."
else
    echo "Warning: Weaviate does not seem to be running at http://localhost:8080."
    echo "Make sure Weaviate is running before using the RAG functionality."
    echo "You can start Weaviate using Docker with:"
    echo "docker run -d -p 8080:8080 --name weaviate-server semitechnologies/weaviate:1.19.6"
fi

# Run the web server
echo "Starting web server..."
python web_server.py 