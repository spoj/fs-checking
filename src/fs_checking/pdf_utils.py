"""PDF utilities for document processing.

Provides:
- pdf_to_images: Convert PDF pages to WebP/JPEG images
- rasterize_pdf: Strip text layer, produce image-only PDF
- shuffle_pdf_pages: Lossless page reorder for ensemble diversity
"""

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


def rasterize_pdf(
    pdf_bytes: bytes,
    dpi: int = 150,
    quality: int = 70,
) -> bytes:
    """Rasterize a PDF: render each page as an image and recombine into a new PDF.

    The output PDF contains only images — no embedded text, no selectable text,
    no hidden text layer. Forces the model to rely purely on visual recognition.

    Uses JPEG internally (PyMuPDF's insert_image doesn't support WebP).

    Args:
        pdf_bytes: Original PDF file content
        dpi: Render resolution (default 150)
        quality: JPEG quality (default 70)

    Returns:
        New PDF bytes with image-only pages (same page count, same page sizes).
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    out = fitz.open()
    try:
        for page_num in range(len(doc)):
            page = doc[page_num]
            pix = page.get_pixmap(matrix=matrix, colorspace=fitz.csRGB)
            jpg_bytes = pix.tobytes("jpeg", quality)

            # Create a new page matching original dimensions
            orig_rect = page.rect
            new_page = out.new_page(width=orig_rect.width, height=orig_rect.height)
            new_page.insert_image(orig_rect, stream=jpg_bytes)
    finally:
        doc.close()

    result = out.tobytes(deflate=True)
    out.close()
    return result


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


def ring_offset_pages(pdf_bytes: bytes, seed: int) -> bytes:
    """Rotate PDF pages by a random offset (lossless, preserves adjacency).

    Each seed produces a different random starting page. Pages are kept in
    their original order but shifted circularly — e.g. for a 10-page doc
    with offset 7: [7, 8, 9, 0, 1, 2, 3, 4, 5, 6].

    This preserves page adjacency (table → note references stay close)
    while varying what the model sees first across runs, combating
    attention fading on long documents.

    Args:
        pdf_bytes: Original PDF content
        seed: Random seed — determines the starting offset

    Returns:
        New PDF bytes with circularly rotated page order
    """
    import random

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    num_pages = len(doc)

    if num_pages <= 1:
        result = doc.tobytes()
        doc.close()
        return result

    # Pick a random offset from seed
    random.seed(seed)
    offset = random.randint(0, num_pages - 1)

    # Build ring order: [offset, offset+1, ..., n-1, 0, 1, ..., offset-1]
    ring_order = [(offset + i) % num_pages for i in range(num_pages)]

    new_doc = fitz.open()
    for page_num in ring_order:
        new_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)

    result = new_doc.tobytes()

    doc.close()
    new_doc.close()

    return result
