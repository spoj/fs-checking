"""PDF utilities for document processing.

Provides:
- pdf_to_images: Convert PDF pages to WebP/JPEG images
- pdf_to_image_content: Convert PDF to OpenRouter image content blocks
- shuffle_pdf_pages: Lossless page reorder for ensemble diversity
"""

import base64

import fitz  # PyMuPDF


def pdf_to_images(
    pdf_bytes: bytes,
    dpi: int = 150,
    quality: int = 70,
    fmt: str = "webp",
) -> list[bytes]:
    """Convert PDF to list of images (one per page).

    Args:
        pdf_bytes: Raw PDF file content
        dpi: Resolution for rendering (default 150 — good for FS tables)
        quality: Image quality 1-100 (default 70)
        fmt: Image format — "webp" (default, ~50% smaller than JPEG) or "jpeg"

    Returns:
        List of image bytes, one per page
    """
    if fmt == "webp":
        import io

        from PIL import Image

    images = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    try:
        for page_num in range(len(doc)):
            page = doc[page_num]
            pix = page.get_pixmap(matrix=matrix, colorspace=fitz.csRGB)
            if fmt == "webp":
                img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                buf = io.BytesIO()
                img.save(buf, format="WEBP", quality=quality)
                images.append(buf.getvalue())
            else:
                images.append(pix.tobytes("jpeg", quality))
    finally:
        doc.close()

    return images


def pdf_to_image_content(
    pdf_bytes: bytes,
    dpi: int = 150,
    quality: int = 70,
    fmt: str = "webp",
    shuffle_seed: int | None = None,
) -> list[dict]:
    """Convert PDF pages to OpenRouter-compatible image content blocks.

    Each page gets a text label with its original document page number,
    followed by the rendered image. Pages can optionally be shuffled
    for ensemble diversity while preserving page number labels.

    Args:
        pdf_bytes: Raw PDF file content
        dpi: Render resolution (default 150)
        quality: Image quality (default 70)
        fmt: Image format — "webp" (default, ~50% smaller) or "jpeg"
        shuffle_seed: If set, shuffle the page order using this seed.
            Page number labels are preserved so the model can still
            report errors by document page number.

    Returns:
        List of content block dicts ready for OpenRouter messages API.
    """
    import random as _random

    mime = f"image/{fmt}"
    images = pdf_to_images(pdf_bytes, dpi=dpi, quality=quality, fmt=fmt)
    num_pages = len(images)

    # Build (page_number_1based, image_bytes) pairs
    pages = list(enumerate(images, 1))

    if shuffle_seed is not None:
        _random.seed(shuffle_seed)
        _random.shuffle(pages)

    content: list[dict] = []
    for page_num, img_bytes in pages:
        content.append({"type": "text", "text": f"[Page {page_num} of {num_pages}]"})
        b64 = base64.b64encode(img_bytes).decode()
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}"},
            }
        )
    return content


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
