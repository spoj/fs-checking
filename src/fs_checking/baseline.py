"""Baseline single-pass verification - no swarm, just one model call.

A simple baseline for comparison with the swarm approach.
Sends all pages to a single model call with detailed instructions.

Usage:
    from fs_checking.baseline import run_baseline
    result = await run_baseline(Path("document.pdf"))
"""

import asyncio
import base64
import json
import sys
import time
from pathlib import Path
from typing import Any

from .api import OpenRouterClient
from .pdf_utils import pdf_to_images

DEFAULT_MODEL = "openai/gpt-5.2"


BASELINE_PROMPT = """\
You are a financial statement auditor. Analyze these IFRS financial statements for internal consistency and mathematical accuracy.

## Your Task

Examine EVERY page carefully and perform ALL of the following checks:

### Balance Sheet (Statement of Financial Position)
- Cross-foot all subtotals (non-current assets, current assets, etc.)
- Verify Total Assets = Total Liabilities + Total Equity (both years)
- Check net current assets/liabilities calculation
- Verify prior year comparatives

### Profit & Loss (Income Statement)
- Gross profit = Revenue - Cost of sales
- Operating profit = Gross profit - Operating expenses
- Cross-foot all subtotals
- Verify Note 4 (Operating profit details) ties to P&L line items

### Statement of Comprehensive Income (OCI)
- Items that will/won't be reclassified subtotals
- Total comprehensive income = Net profit + OCI items

### Cash Flow Statement
- Operating/Investing/Financing subtotals cross-foot
- Net change in cash = Operating + Investing + Financing
- Closing cash = Opening cash + Net change
- Closing cash must tie to Balance Sheet cash balance

### Notes Verification
- **Rollforwards**: Opening + Additions - Disposals = Closing (for PPE, provisions, receivables, etc.)
- **Note ties**: Note totals must match corresponding statement line items
- **Internal math**: All subtotals within notes must cross-foot
- **Look for round number discrepancies** ($100k, $50k) - these often indicate missing entries

### Cross-Statement Consistency
- Net profit: P&L = OCI = Cash Flow reconciliation
- Closing cash: Cash Flow = Balance Sheet
- Retained earnings movement ties to net profit

## Output Format

Return a JSON object with this structure:
```json
{
  "metadata": {
    "company": "Company Name",
    "period": "Year ended 31 December 2023",
    "currency": "US$'000"
  },
  "checks": [
    {
      "id": "unique_snake_case_id",
      "category": "cross_footing|internal_consistency|note_ties",
      "status": "pass|fail|warn",
      "expected": 1234,
      "actual": 1234,
      "difference": 0,
      "description": "What was checked",
      "page": 3
    }
  ],
  "summary": {
    "total_checks": 50,
    "passed": 45,
    "failed": 5,
    "key_findings": ["List of significant issues found"]
  }
}
```

## Important Instructions

1. Check EVERY subtotal on EVERY page
2. Record ALL checks, including passes (creates audit trail)
3. Be thorough - don't skip any calculations
4. For failures, show expected vs actual with difference
5. Pay special attention to note rollforwards and ties

Return ONLY the JSON object, no other text.
"""


async def run_baseline(
    pdf_path: Path,
    output_path: Path | None = None,
    model: str | None = None,
) -> dict:
    """Run single-pass baseline verification.

    Args:
        pdf_path: Path to PDF file
        output_path: Output JSON path (default: pdf_path.with_suffix('.baseline.json'))
        model: Model to use (default: openai/gpt-5.2)

    Returns:
        The parsed result dict
    """
    model = model or DEFAULT_MODEL
    output_path = output_path or pdf_path.with_suffix(".baseline.json")

    print(f"Loading {pdf_path}...", file=sys.stderr)
    print(f"Model: {model}", file=sys.stderr)
    print("Strategy: single-pass baseline", file=sys.stderr)

    # Convert PDF to images
    page_images = pdf_to_images(pdf_path.read_bytes(), dpi=150)
    print(f"Pages: {len(page_images)}", file=sys.stderr)

    # Build message with all pages
    user_content = []
    for i, img_bytes in enumerate(page_images):
        user_content.append({"type": "text", "text": f"\n=== Page {i + 1} ==="})
        img_b64 = base64.b64encode(img_bytes).decode()
        user_content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_b64}"},
            }
        )

    user_content.append({"type": "text", "text": f"\n\n{BASELINE_PROMPT}"})

    messages = [
        {"role": "user", "content": user_content},
    ]

    # Make single API call
    client = OpenRouterClient(reasoning_effort="high", timeout=900.0)

    start = time.time()
    print("Sending to model...", file=sys.stderr)

    response = await client.chat(
        model=model,
        messages=messages,
        tools=None,
    )

    elapsed = time.time() - start

    # Parse response
    message = response.get("message", {})
    content = message.get("content", "")
    usage = response.get("usage", {})

    # Try to extract JSON from response
    result = None
    try:
        # Try direct parse
        result = json.loads(content)
    except json.JSONDecodeError:
        # Try to find JSON in markdown code block
        import re

        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content)
        if json_match:
            try:
                result = json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

    if result is None:
        print("WARNING: Could not parse JSON from response", file=sys.stderr)
        result = {"raw_response": content, "parse_error": True}

    # Write output
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    # Summary
    checks = result.get("checks", [])
    if isinstance(checks, list):
        pass_count = sum(1 for c in checks if c.get("status") == "pass")
        fail_count = sum(1 for c in checks if c.get("status") == "fail")
        warn_count = sum(1 for c in checks if c.get("status") == "warn")
    else:
        pass_count = fail_count = warn_count = 0

    print(f"\n{'=' * 60}", file=sys.stderr)
    print(f"Completed in {elapsed:.1f}s", file=sys.stderr)
    print(
        f"Tokens: {usage.get('prompt_tokens', 0):,} in, {usage.get('completion_tokens', 0):,} out",
        file=sys.stderr,
    )
    print(
        f"Checks: {len(checks)} total ({pass_count} pass, {fail_count} fail, {warn_count} warn)",
        file=sys.stderr,
    )
    print(f"Output: {output_path}", file=sys.stderr)

    return result


# CLI entry point
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Baseline single-pass verification")
    parser.add_argument("pdf", type=Path, help="PDF file to verify")
    parser.add_argument("-o", "--output", type=Path, help="Output JSON path")
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL, help="Model to use")

    args = parser.parse_args()

    asyncio.run(run_baseline(args.pdf, args.output, args.model))
