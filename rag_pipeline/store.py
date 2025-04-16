import faiss
import numpy as np
from langchain.schema import Document

class FaissStore:
    """
    Wrapper around FAISS index for storing embeddings and retrieving Documents.
    """
    def __init__(self, dim: int):
        self.dim = dim
        # Use Inner Product index for cosine similarity after normalization
        self.index = faiss.IndexFlatIP(dim)
        self.docs: list[Document] = []  # Store corresponding Documents

    def add(self, embeddings: list[list[float]], docs: list[Document]):
        """
        Add embeddings and their Documents to the FAISS index.

        Args:
            embeddings: List of embedding vectors.
            docs: List of LangChain Document objects.
        """
        # Convert embeddings to NumPy float32 array
        vectors = np.array(embeddings, dtype="float32")
        self.index.add(vectors)
        self.docs.extend(docs)

    def search(self, query_vec: list[float], k: int = 5, section: str = None) -> list[Document]:
        """
        Search top-k similar documents for the query vector, optionally filtering by section.

        Args:
            query_vec: Embedding vector for the query.
            k: Number of top results to return.
            section: If provided, only return chunks from this section.
        Returns:
            List of Document objects matching the criteria.
        """
        xq = np.array([query_vec], dtype="float32")
        D, I = self.index.search(xq, k)
        results = []
        for idx in I[0]:
            doc = self.docs[idx]
            if section is None or doc.metadata.get("section") == section:
                results.append(doc)
        return results