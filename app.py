from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag_pipeline.retriever import ingest_and_index, retrieve
from rag_pipeline.generator import generate_answer

app = FastAPI()

class QueryRequest(BaseModel):
    question: str
    section: str | None = None  # Optional section filter

# On startup, ingest PDF once and build index
@app.on_event("startup")
async def startup_event():
    await ingest_and_index("/mnt/data/ABHIL_Member_Handbook.pdf")

@app.post("/chat")
async def chat(req: QueryRequest):
    try:
        # Step 1: retrieve relevant chunks
        chunks = await retrieve(req.question, k=5, section=req.section)
        # Step 2: generate answer from LLM
        answer = generate_answer(chunks, req.question)
        return {"answer": answer}
    except Exception as e:
        # Log error and return generic 500
        from loguru import logger
        logger.error(f"Error in /chat: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")