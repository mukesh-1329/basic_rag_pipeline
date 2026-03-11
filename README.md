📚 Local PDF RAG Pipeline
A robust, Privacy-First Retrieval-Augmented Generation (RAG) system built with Python. This project allows you to ingest PDF documents and "chat" with their content using entirely local Large Language Models (LLMs) and Vector Databases. No data ever leaves your machine, and no expensive API keys are required.

🚀 Overview
This pipeline transforms static PDF documents into a searchable "knowledge base." It uses semantic search to find the most relevant parts of your document to answer specific user queries accurately.

The Pipeline in Two Steps:
Ingestion: Reads Google.pdf, breaks it into 1000-character chunks, converts them into mathematical vectors (embeddings), and stores them in a local ChromaDB.

Retrieval: Takes a user's question, finds the top 3 most relevant chunks from the database, and passes them to Llama 3 to generate a natural language response.

🛠️ Technology Stack
Orchestration: LangChain (using modern LCEL syntax)

LLM & Embeddings: Ollama (running llama3 and nomic-embed-text)

Vector Database: ChromaDB

PDF Processing: pypdf

Language: Python 3.10+
