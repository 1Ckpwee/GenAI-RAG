# LangChain with Deepseek Example

This project demonstrates how to interact with the Deepseek large language model using the LangChain framework, including RAG implementation with Weaviate.

## Features

- Text generation using the Deepseek model
- Implementation of simple conversation chains
- Demonstration of LangChain's tools and agent capabilities
- Showcase of LangChain's memory components
- RAG (Retrieval Augmented Generation) implementation using Weaviate and Deepseek
- Web server for document upload and RAG search

## Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file and add your Deepseek API key and Weaviate configuration:
   ```
   DEEPSEEK_API_KEY=your_api_key_here
   WEAVIATE_URL=http://localhost:8080
   WEAVIATE_API_KEY=your_weaviate_api_key_here
   ```

## Weaviate Setup

For the RAG functionality, you need to have Weaviate running. You can use Docker Compose to start Weaviate:

```
./start_weaviate.sh
```

Or manually with Docker Compose:

```
docker-compose up -d
```

## Usage

### Command-line Examples

Run the basic example:
```
python simple_chat.py
```

Run the agent with tools example:
```
python agent_with_tools.py
```

Run the RAG example:
```
python rag_example.py
```

Run the advanced RAG example:
```
python advanced_rag_example.py
```

### Web Server

Run the web server:
```
./run_web_server.sh
```

Or manually:
```
python web_server.py
```

Then open your browser and navigate to:
```
http://localhost:5000
```

The web interface allows you to:
- Upload documents (.txt, .pdf, .csv, .md)
- View uploaded documents
- Search documents using RAG

## Project Structure

- `simple_chat.py`: Basic Deepseek chat example
- `agent_with_tools.py`: Deepseek agent example with tools
- `rag_example.py`: RAG implementation using Weaviate and Deepseek
- `advanced_rag_example.py`: Advanced RAG implementation with document loading
- `web_server.py`: Web server for document upload and RAG search
- `utils.py`: Helper functions
- `.env`: Environment variables configuration file (needs to be created manually)
- `templates/`: HTML templates for the web interface
- `static/`: Static files (CSS, JavaScript) for the web interface
- `uploads/`: Directory for uploaded documents
- `docker-compose.yml`: Docker Compose configuration for Weaviate
- `start_weaviate.sh`: Script to start Weaviate
- `run_web_server.sh`: Script to run the web server 