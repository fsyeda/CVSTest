import fitz  # PyMuPDF library to read PDFs

def extract_pages(file_path: str) -> list[dict]:
    """
    Open the PDF at file_path and extract each page's full text and a detected section heading.

    Returns:
        List of dicts with keys: page_number, section (first ALL-CAPS line), text (full page text).
    """
    doc = fitz.open(file_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text()
        # Detect headings: lines in ALL CAPS and under a reasonable length
        headings = [line.strip() for line in text.splitlines() if line.isupper() and len(line) < 100]
        section = headings[0] if headings else None  # Use first detected heading, or None
        pages.append({
            "page_number": i + 1,
            "section": section,
            "text": text
        })
    return pages