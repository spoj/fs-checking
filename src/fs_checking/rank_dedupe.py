"""Rank and deduplicate findings - no validation, just organize.

Pass full document + all candidates to Gemini Pro.
Output: high/medium/low priority buckets, duplicates merged.

Usage:
    uv run python -m fs_checking.rank_dedupe "samples/Written test_Case.pdf" findings.json
"""

import asyncio
import base64
import json
import sys
import time
from pathlib import Path

from .api import OpenRouterClient
from .pdf_utils import pdf_to_images


RANK_DEDUPE_PROMPT = """\
You are organizing error findings from a financial statement review.

## Candidates ({num_candidates} findings)

{candidates_json}

## Task

1. DEDUPLICATE: Merge findings that describe the same underlying error
2. RANK by priority:
   - HIGH: Material misstatements, math errors affecting totals, broken ties between statements
   - MEDIUM: Minor calculation errors, presentation issues with numbers
   - LOW: Formatting, labeling, cosmetic issues

Do NOT validate whether errors are real - assume they are. Just organize them.

Return JSON only:
```json
{{
  "high": [
    {{"id": "best_id", "page": 1, "description": "clear description", "merged_from": ["id1", "id2"]}}
  ],
  "medium": [
    {{"id": "...", "page": 1, "description": "...", "merged_from": []}}
  ],
  "low": [
    {{"id": "...", "page": 1, "description": "...", "merged_from": []}}
  ]
}}
```
"""


async def rank_and_dedupe(
    pdf_path: Path,
    candidates: list[dict],
    output_path: Path | None = None,
    model: str = "google/gemini-3-pro-preview",
) -> dict:
    """Rank and deduplicate findings using full document context."""
    import re

    output_path = output_path or pdf_path.with_suffix(".ranked.json")

    print(f"Loading {pdf_path}...", file=sys.stderr)
    page_images = pdf_to_images(pdf_path.read_bytes(), dpi=150)
    print(f"Pages: {len(page_images)}, Candidates: {len(candidates)}", file=sys.stderr)

    # Build message with all pages in sequence
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

    # Format candidates
    candidates_for_prompt = [
        {
            "id": c.get("id"),
            "page": c.get("page"),
            "category": c.get("category"),
            "description": c.get("description"),
        }
        for c in candidates
    ]

    prompt = RANK_DEDUPE_PROMPT.format(
        num_candidates=len(candidates),
        candidates_json=json.dumps(candidates_for_prompt, indent=2),
    )
    user_content.append({"type": "text", "text": prompt})

    client = OpenRouterClient(reasoning_effort="high", timeout=900.0)

    print(f"Sending to {model}...", file=sys.stderr)
    start = time.time()

    response = await client.chat(
        model=model,
        messages=[{"role": "user", "content": user_content}],
    )

    elapsed = time.time() - start
    content = response.get("message", {}).get("content", "")

    # Parse response
    try:
        json_match = re.search(r"\{[\s\S]*\}", content)
        if json_match:
            result = json.loads(json_match.group())
        else:
            result = {"error": "no_json", "raw": content[:500]}
    except json.JSONDecodeError as e:
        result = {"error": str(e), "raw": content[:500]}

    # Add metadata
    result["metadata"] = {
        "model": model,
        "elapsed_seconds": round(elapsed, 1),
        "input_candidates": len(candidates),
    }

    # Count outputs
    high = result.get("high", [])
    medium = result.get("medium", [])
    low = result.get("low", [])

    result["summary"] = {
        "high": len(high),
        "medium": len(medium),
        "low": len(low),
        "total": len(high) + len(medium) + len(low),
    }

    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    print(f"\n{'=' * 60}", file=sys.stderr)
    print(f"Completed in {elapsed:.1f}s", file=sys.stderr)
    print(f"Input: {len(candidates)} candidates", file=sys.stderr)
    print(
        f"Output: {len(high)} high, {len(medium)} medium, {len(low)} low",
        file=sys.stderr,
    )
    print(f"Saved: {output_path}", file=sys.stderr)

    return result


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Rank and dedupe findings")
    parser.add_argument("pdf", type=Path, help="PDF file")
    parser.add_argument(
        "findings",
        type=Path,
        nargs="?",
        help="Findings JSON (or use --from-benchmarks)",
    )
    parser.add_argument("-o", "--output", type=Path, help="Output path")
    parser.add_argument(
        "--from-benchmarks", action="store_true", help="Load from benchmark flash runs"
    )

    args = parser.parse_args()

    if args.from_benchmarks:
        # Load union of all flash benchmark runs
        candidates = []
        benchmark_dir = Path("benchmarks")
        for f in sorted(benchmark_dir.glob("gemini-3-flash-preview_seed*.json")):
            data = json.loads(f.read_text())
            candidates.extend(data.get("findings", []))
        print(f"Loaded {len(candidates)} from benchmark runs", file=sys.stderr)
    elif args.findings:
        data = json.loads(args.findings.read_text())
        candidates = data.get("findings", data.get("checks", []))
    else:
        parser.error("Provide findings JSON or use --from-benchmarks")

    await rank_and_dedupe(args.pdf, candidates, args.output)


if __name__ == "__main__":
    asyncio.run(main())
