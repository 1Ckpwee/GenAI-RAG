#!/bin/bash

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Start Weaviate using Docker Compose
echo "Starting Weaviate..."
docker-compose up -d

# Wait for Weaviate to be ready
echo "Waiting for Weaviate to be ready..."
while ! curl -s -o /dev/null -w "%{http_code}" http://localhost:8080/v1/.well-known/ready | grep -q "200"; do
    echo "Weaviate is not ready yet. Waiting..."
    sleep 5
done

echo "Weaviate is ready at http://localhost:8080"
echo "You can now run the web server with: ./run_web_server.sh" 