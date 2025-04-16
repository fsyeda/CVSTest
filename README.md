```markdown
# RAG Chatbot for Aetna Handbook

## Overview
A RAG-based chatbot that ingests the Aetna Illinois Member Handbook and answers user queries.

## Tech Stack
- Python, LangChain
- OpenAI text-embedding-3-small & GPT-4o
- FAISS vector store with section metadata filtering
- FastAPI backend
- Chainlit chat UI
- Loguru + Sentry for logging
- Docker + Render for deployment

## Setup
1. `git clone https://github.com/fsyeda/CVSTest.git`
2. Copy `.env.example` to `.env` and fill in keys
3. `docker build -t rag-chatbot .`
4. `docker run -p 8000:8000 rag-chatbot`

## Usage
- Chat via Chainlit UI at `http://localhost:8000`
- REST endpoint: POST `/chat` with `{"question": "...", "section": "..."}`
