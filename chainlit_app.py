import sys, os
# Ensure project root is on Python path for module imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))  # ensure current dir is on PYTHONPATH, '..')))
import chainlit as cl
from rag_pipeline.retriever import ingest_and_index, retrieve
from rag_pipeline.generator import generate_answer

@cl.on_chat_start
async def start():
    """
    Ingest and index the PDF once at the start of the chat session.
    """
    await ingest_and_index("data/ABHIL_Member_Handbook.pdf")
    # Log how many chunks were indexed (import store from retriever)
    from rag_pipeline.retriever import store as faiss_store
    from loguru import logger
    logger.info(f"Indexed {len(faiss_store.docs)} chunks into FAISS")
    cl.user_session.set("history", [])

@cl.on_message
async def main(message: cl.Message):
    """
    On each message, retrieve top-5 chunks and generate a grounded response.
    """
    from loguru import logger
    question = message.content
    logger.info(f"Received question: {question}")
    chunks = await retrieve(question, k=5)
    logger.info(f"Retrieved {len(chunks)} chunks")
    if not chunks:
        await cl.Message(content="Sorry, I couldn’t find any relevant information.").send()
        return

    answer = generate_answer(chunks, question)
    logger.info(f"Sending answer: {answer[:150]}…")
    await cl.Message(content=answer).send()

# end of chainlit_app.py
async def main(message: cl.Message):
    """
    Handle each incoming chat message by retrieving relevant chunks
    and generating a grounded response via the RAG pipeline.
    """
    question = message.content
    # Retrieve top 5 chunks from FAISS (no section filter here)
    chunks = await retrieve(question, k=5)
    # Generate a deterministic answer using GPT-4o
    answer = generate_answer(chunks, question)
    await cl.Message(content=answer).send()
