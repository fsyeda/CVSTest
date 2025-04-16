from .embeddings import get_embedding
from .store import FaissStore

# Embedding dimension for text-embedding-3-small (1536)
VECTOR_DIM = 1536
# Initialize FAISS store
store = FaissStore(dim=VECTOR_DIM)

async def ingest_and_index(pdf_path: str):
    """
    Ingest PDF, extract text chunks, generate embeddings, and index in FAISS.
    """
    from pdf_ingestion.parse_pdf import extract_pages
    from pdf_ingestion.chunker import chunk_pages

    # Step 1: Parse PDF into pages
    pages = extract_pages(pdf_path)
    # Step 2: Chunk pages into manageable pieces
    chunks = chunk_pages(pages)
    # Step 3: Embed each chunk
    embs = [get_embedding(doc.page_content) for doc in chunks]
    # Step 4: Add embeddings and docs to FAISS store
    store.add(embs, chunks)

async def retrieve(query: str, k: int = 5, section: str = None):
    """
    Retrieve top-k relevant chunks for a user query, optionally filtering by section.
    """
    qv = get_embedding(query)
    return store.search(qv, k=k, section=section)