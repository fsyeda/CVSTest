from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

def chunk_pages(pages: list[dict], chunk_size: int = 500, overlap: int = 50) -> list[Document]:
    """
    Take extracted page dicts and convert to LangChain Documents, preserving metadata.
    Then split into smaller chunks for embedding and retrieval.

    Args:
        pages: List of dicts with 'page_number', 'section', and 'text'.
        chunk_size: Maximum tokens per chunk.
        overlap: Overlap between chunks to preserve context.

    Returns:
        List of Document objects, each with page_content and metadata.
    """
    docs = []
    for p in pages:
        # Create a LangChain Document per page, storing section and page metadata
        docs.append(Document(
            page_content=p["text"],
            metadata={
                "page": p["page_number"],
                "section": p["section"]
            }
        ))
    # Use LangChain's recursive splitter to chunk the documents
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_documents(docs)