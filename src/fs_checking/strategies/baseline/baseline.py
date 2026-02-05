"""Baseline single-pass verification - one model call.

Usage:
    from fs_checking.strategies.baseline import run_baseline
    result = await run_baseline(Path("document.pdf"))
"""

import asyncio
import base64
import json
import sys
import time
from pathlib import Path

from ...api import OpenRouterClient
from ...pdf_utils import pdf_to_images

DEFAULT_MODEL = "google/gemini-3-flash-preview"

DETECT_PROMPT = """\
You are a financial statement auditor. Analyze these financial statements for errors.

## Your Task

Examine EVERY page. Call `save_check` for each ERROR found.

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
2. Call `save_check()` for each error found
3. When done, call `finish()`

Be thorough. Missing errors is worse than false positives.
"""

TOOL_SAVE_CHECK = {
    "type": "function",
    "function": {
        "name": "save_check",
        "description": "Record an error found",
        "parameters": {
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "Unique snake_case id"},
                "category": {
                    "type": "string",
                    "enum": [
                        "cross_footing",
                        "rollforward",
                        "note_ties",
                        "presentation",
                        "reasonableness",
                    ],
                },
                "page": {"type": "integer", "description": "Primary page number"},
                "description": {
                    "type": "string",
                    "description": "Clear description with specific numbers",
                },
                "expected": {
                    "type": "number",
                    "description": "Calculated/expected value",
                },
                "actual": {"type": "number", "description": "Stated value in document"},
            },
            "required": ["id", "category", "page", "description"],
        },
    },
}

TOOL_FINISH = {
    "type": "function",
    "function": {
        "name": "finish",
        "description": "Complete the analysis",
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "Brief summary of findings",
                },
            },
            "required": ["summary"],
        },
    },
}


async def run_baseline(
    pdf_path: Path,
    output_path: Path | None = None,
    model: str = DEFAULT_MODEL,
    shuffle_pages: bool = False,
    seed: int | None = None,
) -> dict:
    """Run single-pass baseline detection.

    Args:
        pdf_path: Path to PDF file
        output_path: Output JSON path (default: pdf_path.with_suffix('.baseline.json'))
        model: Model to use
        shuffle_pages: Whether to shuffle page order
        seed: Random seed for shuffle

    Returns:
        Result dict with checks and metadata
    """
    import random

    output_path = output_path or pdf_path.with_suffix(".baseline.json")

    print(f"Loading {pdf_path}...", file=sys.stderr)
    print(f"Model: {model}", file=sys.stderr)

    page_images = pdf_to_images(pdf_path.read_bytes(), dpi=150)
    print(f"Pages: {len(page_images)}", file=sys.stderr)

    # Page ordering
    page_indices = list(range(len(page_images)))
    if shuffle_pages:
        if seed is not None:
            random.seed(seed)
        random.shuffle(page_indices)
        print(f"Shuffled order (seed={seed})", file=sys.stderr)

    # Build message
    user_content = []
    for idx in page_indices:
        user_content.append({"type": "text", "text": f"\n=== Page {idx + 1} ==="})
        img_b64 = base64.b64encode(page_images[idx]).decode()
        user_content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_b64}"},
            }
        )
    user_content.append({"type": "text", "text": f"\n\n{DETECT_PROMPT}"})

    messages = [{"role": "user", "content": user_content}]
    tools = [TOOL_SAVE_CHECK, TOOL_FINISH]

    client = OpenRouterClient(reasoning_effort="high", timeout=900.0)

    checks = []
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0}

    start = time.time()
    print("Running detection...", file=sys.stderr)

    for iteration in range(50):
        response = await client.chat(model=model, messages=messages, tools=tools)

        usage = response.get("usage", {})
        total_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
        total_usage["completion_tokens"] += usage.get("completion_tokens", 0)

        message = response.get("message", {})
        tool_calls = message.get("tool_calls", [])

        if not tool_calls:
            break

        messages.append(message)
        tool_results = []
        finished = False

        for tc in tool_calls:
            name = tc["function"]["name"]
            try:
                args = json.loads(tc["function"]["arguments"])
            except json.JSONDecodeError:
                args = {}

            if name == "save_check":
                checks.append(args)
                tool_results.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": f"Saved: {args.get('id')}",
                    }
                )
                print(f"  [{iteration + 1}] {args.get('id')}", file=sys.stderr)

            elif name == "finish":
                tool_results.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": "Done",
                    }
                )
                finished = True

        messages.extend(tool_results)
        if finished:
            break

    elapsed = time.time() - start

    result = {
        "metadata": {
            "strategy": "baseline",
            "model": model,
            "elapsed_seconds": round(elapsed, 1),
            "usage": total_usage,
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
    parser.add_argument("--shuffle", action="store_true", help="Shuffle page order")
    parser.add_argument("--seed", type=int, help="Random seed for shuffle")

    args = parser.parse_args()

    await run_baseline(args.pdf, args.output, args.model, args.shuffle, args.seed)


if __name__ == "__main__":
    asyncio.run(main())
