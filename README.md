# Agentic_RAG
An Agentic RAG System that uses AI agents to intelligently retrieve and answer questions from documents.

## Setup Guide

### Prerequisites

1. **Ollama**
	- Install Ollama from [https://ollama.com/download](https://ollama.com/download).
	- Download required models (e.g., llama3, etc.) using:
	  ```
	  ollama pull llama3
	  ollama pull llama3.1:8b
	  ```
	- Start Ollama server:
	  ```
	  ollama serve
	  ```

2. **Milvus (for vector DB)**
	- Follow the official Milvus Docker setup guide: [Milvus Prerequisite Docker](https://milvus.io/docs/prerequisite-docker.md)
	- Start Milvus using Docker:
	- If you do **not** want to use Milvus, you can use ChromaDB instead (see below).

### Backend Selection

You can choose between Milvus and ChromaDB as your vector database backend:

- **ChromaDB** (simpler, no Docker required) [default]:
  - In `database/database_config.py`, change:
	 ```python
	 DATABASE_BACKEND = "chroma"
	 ```
- **Milvus** :
  - In `database/database_config.py`, ensure:
	 ```python
	 DATABASE_BACKEND = "milvus"
	 ```

### Python Environment Setup

1. Create and activate a virtual environment:
	```
	python -m venv .venv
	# On Windows:
	.venv\Scripts\activate
	# On Linux/Mac:
	source .venv/bin/activate
	```

2. Install required packages:
	```
	pip install -r requirements.txt
	```

### Model Serving

- Make sure Ollama is running (`ollama serve`) and the required models are downloaded.
- If using Milvus, ensure the Docker container is running.

### Running the Application

1. Start Ollama server (if not already running):
	```
	ollama serve
	```
2. Start Milvus (if using Milvus backend):
	```
	docker start milvus | docker compose up (as per your OS)
	```
3. Run the application (example):
	```
	python app.py
	```

---


---


## Project File Structure (Detailed)

```
Agentic_RAG/
├── app.py
│     Main entry point for backend logic and app orchestration.
├── requirements.txt
│     Python dependencies for the project.
├── README.md
│     Project documentation and setup instructions.
├── LICENSE
│     License file.

├── Agent/
│   ├── agent.py
│   │     Main agent workflow and orchestration logic.
│   └── nodes/
│         ├── final_ans_generator.py      # Generates final answers from retrieved docs
│         ├── query_generator.py          # Generates search queries from user input
│         ├── question_rewriter.py        # Rewrites/clarifies user questions
│         ├── retrieved_doc_grader.py     # Grades and filters retrieved documents
│         └── retriever.py                # Handles retrieval logic (hybrid, rerank, etc.)

├── backend/
│   ├── agent_logger.py           # Logging for agent actions
│   ├── api.py                    # (Optional) FastAPI endpoints for backend
│   ├── doc_loader.py             # Document loading utilities
│   ├── exceptions.py             # Custom exceptions and error handling
│   ├── ingestion_logger.py       # Logging for ingestion pipeline
│   ├── ingestion_pipeline.py     # Document ingestion and indexing logic
│   ├── splitter.py               # Text splitting utilities
│   ├── storing.py                # Storage helpers for vector DBs
│   └── data/                     # Example or default data files
│   └── logs/                     # Log files

├── database/
│   ├── database_config.py        # Switch between Milvus/Chroma backends
│   ├── chroma_db_setup.py        # ChromaDB setup/configuration
│   ├── milvus_db_setup.py        # Milvus setup/configuration
│   ├── reset_db.py               # Utility to reset/clear the database
│   └── data/                     # Example data files for ingestion

├── models/
│   ├── gemini_emb_model.py       # Gemini embedding model wrapper
│   ├── gemini_LLM.py             # Gemini LLM wrapper
│   ├── ollama_emb.py             # Ollama embedding model wrapper
│   ├── ollama_LLM.py             # Ollama LLM wrapper
│   └── __init__.py

├── tests/
│   ├── database_test.py          # Tests for database logic
│   ├── ingestion_test.py         # Tests for ingestion pipeline
│   └── old_loader.py             # (Legacy) loader tests

├── UI/
│   ├── streamlit_app.py          # Streamlit-based user interface
│   └── __init__.py

├── chroma_langchain_db/
│     ChromaDB local storage files (auto-generated if Chroma is used)

├── database/
│   └── data/                     # Example or default data files for ingestion

```

## About the Project

Agentic_RAG is a Retrieval-Augmented Generation (RAG) system that leverages AI agents to:
- Ingest and index documents using dense (Ollama) and sparse (BM25) embeddings
- Retrieve relevant information using hybrid search (Milvus or ChromaDB)
- Use LLMs to answer user queries based on retrieved context
- Provide a modular, extensible backend and a Streamlit UI for interaction

## Usage Rules & Recommendations

- Ensure Ollama is running and required models are downloaded before starting the app.
- If using Milvus, make sure the Docker container is running and accessible.
- For simple/local use, switch to ChromaDB backend in `database/database_config.py`.
- Place your documents for ingestion in the appropriate data folder or use the UI to upload.
- Use the Streamlit UI (`UI/streamlit_app.py`) for interactive querying and document management.
- For development, run tests in the `tests/` folder to validate changes.

---

For more details, see code comments and individual module docstrings.
