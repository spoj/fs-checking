"""PDF utilities for document processing.

Provides:
- pdf_to_images: Convert PDF pages to JPEG images
- Vision API calls for table extraction (optional)
"""

import fitz  # PyMuPDF


def pdf_to_images(pdf_bytes: bytes, dpi: int = 150, quality: int = 80) -> list[bytes]:
    """Convert PDF to list of JPEG images (one per page).

    Args:
        pdf_bytes: Raw PDF file content
        dpi: Resolution for rendering (150 is good balance of quality/size)
        quality: JPEG quality 1-100 (80 is good balance for text documents)

    Returns:
        List of JPEG image bytes, one per page
    """
    images = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    try:
        for page_num in range(len(doc)):
            page = doc[page_num]
            pix = page.get_pixmap(matrix=matrix, colorspace=fitz.csRGB)
            images.append(pix.tobytes("jpeg", quality))
    finally:
        doc.close()

    return images


def get_page_count(pdf_bytes: bytes) -> int:
    """Get number of pages in a PDF."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        return len(doc)
    finally:
        doc.close()


def extract_text(pdf_bytes: bytes) -> list[str]:
    """Extract text from each page of a PDF.

    Returns:
        List of text strings, one per page
    """
    texts = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    try:
        for page_num in range(len(doc)):
            page = doc[page_num]
            texts.append(page.get_text())
    finally:
        doc.close()

    return texts


def shuffle_pdf_pages(pdf_bytes: bytes, seed: int) -> bytes:
    """Shuffle PDF pages in a new order (lossless).

    Creates a new PDF with pages reordered randomly based on seed.
    The original page numbers in headers/footers are preserved,
    allowing the model to use document page numbers for error reporting.

    Args:
        pdf_bytes: Original PDF content
        seed: Random seed for reproducible shuffling

    Returns:
        New PDF bytes with shuffled page order
    """
    import random

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    num_pages = len(doc)

    # Generate shuffled order
    shuffled_order = list(range(num_pages))
    random.seed(seed)
    random.shuffle(shuffled_order)

    # Create new PDF with shuffled pages
    new_doc = fitz.open()
    for page_num in shuffled_order:
        new_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)

    shuffled_bytes = new_doc.tobytes()

    doc.close()
    new_doc.close()

    return shuffled_bytes
