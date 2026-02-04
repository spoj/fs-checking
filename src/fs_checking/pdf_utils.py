"""PDF utilities for document processing.

Provides:
- pdf_to_images: Convert PDF pages to PNG images
- Vision API calls for table extraction (optional)
"""

import fitz  # PyMuPDF


def pdf_to_images(pdf_bytes: bytes, dpi: int = 150) -> list[bytes]:
    """Convert PDF to list of PNG images (one per page).

    Args:
        pdf_bytes: Raw PDF file content
        dpi: Resolution for rendering (150 is good balance of quality/size)

    Returns:
        List of PNG image bytes, one per page
    """
    images = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    try:
        for page_num in range(len(doc)):
            page = doc[page_num]
            pix = page.get_pixmap(matrix=matrix, colorspace=fitz.csRGB)
            images.append(pix.tobytes("png"))
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
