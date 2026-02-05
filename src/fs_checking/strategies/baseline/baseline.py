"""Baseline single-pass verification - one model call, JSON output.

Supports two modes:
- PDF direct: Pass PDF as base64 (smaller, for large documents)
- Images: Convert to JPEG images (needed for page shuffling in ensemble)

Usage:
    from fs_checking.strategies.baseline import run_baseline
    result = await run_baseline(Path("document.pdf"))
"""

import asyncio
import base64
import json
import re
import sys
import time
from pathlib import Path

from ...api import OpenRouterClient
from ...pdf_utils import pdf_to_images, get_page_count

DEFAULT_MODEL = "google/gemini-3-flash-preview"

DETECT_PROMPT = """\
You are a financial statement auditor. Analyze these financial statements for errors.

## Check Categories

### 1. CROSS-FOOTING (Math Checks)
- Every subtotal must equal sum of its components
- Balance Sheet, P&L, OCI, Cash Flow subtotals
- All subtotals within notes

### 2. ROLLFORWARDS (Opening + Changes = Closing)
- PPE, Provisions, Receivables impairment schedules
- Check EVERY row

### 3. STATEMENT - NOTE TIES
- BS line items must tie EXACTLY to corresponding notes
- P&L items must tie to Note breakdowns
- CF items must tie to Note 31 reconciliations

### 4. PRESENTATION
- Title dates match column headers
- Labels match values (positive/negative)
- Note references are valid and sequential

## Instructions

1. Work through EVERY page systematically
2. Report ALL errors found
3. Be thorough - missing errors is worse than false positives

Return ONLY a JSON array of errors found:
```json
[
  {
    "id": "unique_snake_case_id",
    "category": "cross_footing|rollforward|note_ties|presentation|reasonableness",
    "page": 1,
    "description": "Clear description with specific numbers",
    "expected": 12345,
    "actual": 12346
  }
]
```

Return `[]` if no errors found. Return ONLY the JSON array, no other text.
"""


async def run_baseline(
    pdf_path: Path,
    output_path: Path | None = None,
    model: str = DEFAULT_MODEL,
    use_images: bool = False,
) -> dict:
    """Run single-pass baseline detection with JSON output.

    Args:
        pdf_path: Path to PDF file
        output_path: Output JSON path (default: pdf_path.with_suffix('.baseline.json'))
        model: Model to use
        use_images: If True, convert to JPEG images; if False, pass PDF directly

    Returns:
        Result dict with checks and metadata
    """
    output_path = output_path or pdf_path.with_suffix(".baseline.json")

    print(f"Loading {pdf_path}...", file=sys.stderr)
    print(f"Model: {model}", file=sys.stderr)

    pdf_bytes = pdf_path.read_bytes()
    num_pages = get_page_count(pdf_bytes)
    print(f"Pages: {num_pages}", file=sys.stderr)

    # Build message content
    user_content = []

    if use_images:
        # Convert to JPEG images (sequential order)
        print("Mode: JPEG images", file=sys.stderr)
        page_images = pdf_to_images(pdf_bytes, dpi=150)
        for i, img_bytes in enumerate(page_images):
            user_content.append({"type": "text", "text": f"\n=== Page {i + 1} ==="})
            img_b64 = base64.b64encode(img_bytes).decode()
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                }
            )
    else:
        # Pass PDF directly as base64
        print("Mode: PDF direct", file=sys.stderr)
        pdf_b64 = base64.b64encode(pdf_bytes).decode()
        user_content.append(
            {
                "type": "file",
                "file": {
                    "filename": pdf_path.name,
                    "file_data": f"data:application/pdf;base64,{pdf_b64}",
                },
            }
        )

    user_content.append({"type": "text", "text": f"\n\n{DETECT_PROMPT}"})

    messages = [{"role": "user", "content": user_content}]
    client = OpenRouterClient(reasoning_effort="high", timeout=1800.0)

    start = time.time()
    print("Running detection...", file=sys.stderr)

    # Single pass - no tools
    response = await client.chat(model=model, messages=messages)
    content = response.get("message", {}).get("content", "")
    usage = response.get("usage", {})

    # Parse JSON array
    checks = []
    try:
        json_match = re.search(r"\[[\s\S]*\]", content)
        if json_match:
            parsed = json.loads(json_match.group())
            if isinstance(parsed, list):
                checks = parsed
    except json.JSONDecodeError as e:
        print(f"JSON parse error: {e}", file=sys.stderr)

    elapsed = time.time() - start

    result = {
        "metadata": {
            "strategy": "baseline",
            "model": model,
            "mode": "images" if use_images else "pdf",
            "num_pages": num_pages,
            "elapsed_seconds": round(elapsed, 1),
            "usage": usage,
        },
        "checks": checks,
        "summary": {
            "total": len(checks),
        },
    }

    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    print(f"\n{'=' * 60}", file=sys.stderr)
    print(f"Completed in {elapsed:.1f}s", file=sys.stderr)
    print(f"Found {len(checks)} errors", file=sys.stderr)
    print(f"Output: {output_path}", file=sys.stderr)

    return result


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Baseline single-pass detection")
    parser.add_argument("pdf", type=Path, help="PDF file to analyze")
    parser.add_argument("-o", "--output", type=Path, help="Output JSON path")
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL, help="Model to use")
    parser.add_argument(
        "--images", action="store_true", help="Use JPEG images instead of PDF direct"
    )

    args = parser.parse_args()

    await run_baseline(args.pdf, args.output, args.model, args.images)


if __name__ == "__main__":
    asyncio.run(main())
