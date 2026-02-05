"""Ensemble detection: 10x Flash + Rank/Dedupe with Pro.

Based on extensive benchmarking:
- 10x flash runs achieve 88.9% recall on 27-error test set
- Gemini Pro rank+dedupe reduces 166 candidates to 21 unique findings
- Final: 90.9% F1, 86.2% recall, 96.2% precision
- Total cost ~$0.80, time ~3-4 minutes

Usage:
    from fs_checking.strategies.ensemble import run_ensemble
    result = await run_ensemble(Path("document.pdf"))
"""

import asyncio
import base64
import json
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from ...api import OpenRouterClient
from ...pdf_utils import pdf_to_images

DEFAULT_DETECT_MODEL = "google/gemini-3-flash-preview"
DEFAULT_RANK_MODEL = "google/gemini-3-pro-preview"
DEFAULT_NUM_RUNS = 10


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


@dataclass
class RunConfig:
    """Configuration for a detection run."""

    run_id: str
    model: str
    seed: int


async def _run_detection_pass(
    config: RunConfig,
    page_images: list[bytes],
    client: OpenRouterClient,
) -> list[dict]:
    """Run a single detection pass with shuffled page order (single-pass JSON output)."""
    # Shuffle pages
    page_indices = list(range(len(page_images)))
    random.seed(config.seed)
    random.shuffle(page_indices)

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

    # Single pass - no tools, just JSON output
    response = await client.chat(model=config.model, messages=messages)
    content = response.get("message", {}).get("content", "")

    # Parse JSON array from response
    findings = []
    try:
        # Find JSON array in response
        json_match = re.search(r"\[[\s\S]*\]", content)
        if json_match:
            parsed = json.loads(json_match.group())
            if isinstance(parsed, list):
                for item in parsed:
                    item["_run"] = config.run_id
                    findings.append(item)
    except json.JSONDecodeError as e:
        print(f"[{config.run_id}] JSON parse error: {e}", file=sys.stderr)

    print(f"[{config.run_id}] Found {len(findings)} errors", file=sys.stderr)
    return findings


async def _rank_and_dedupe(
    candidates: list[dict],
    page_images: list[bytes],
    client: OpenRouterClient,
    model: str,
) -> dict:
    """Rank and deduplicate findings using full document context."""
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

    response = await client.chat(
        model=model,
        messages=[{"role": "user", "content": user_content}],
    )

    content = response.get("message", {}).get("content", "")

    # Parse response
    try:
        json_match = re.search(r"\{[\s\S]*\}", content)
        if json_match:
            return json.loads(json_match.group())
    except json.JSONDecodeError as e:
        print(f"Parse error in rank_dedupe: {e}", file=sys.stderr)

    return {"high": [], "medium": [], "low": [], "error": "parse_failed"}


async def run_ensemble(
    pdf_path: Path,
    output_path: Path | None = None,
    detect_model: str = DEFAULT_DETECT_MODEL,
    rank_model: str = DEFAULT_RANK_MODEL,
    num_runs: int = DEFAULT_NUM_RUNS,
) -> dict:
    """Run ensemble detection with rank/dedupe.

    Pipeline:
    1. Detection: 10x flash parallel runs with shuffled page orders
    2. Rank/Dedupe: Gemini Pro organizes and deduplicates findings

    Args:
        pdf_path: Path to PDF file
        output_path: Output JSON path
        detect_model: Model for detection phase (default: flash)
        rank_model: Model for rank/dedupe phase (default: pro)
        num_runs: Number of parallel detection runs (default: 10)

    Returns:
        Result dict with prioritized checks
    """
    output_path = output_path or pdf_path.with_suffix(".ensemble.json")

    print(f"Loading {pdf_path}...", file=sys.stderr)
    page_images = pdf_to_images(pdf_path.read_bytes(), dpi=150)
    print(f"Pages: {len(page_images)}", file=sys.stderr)
    print(
        f"Strategy: {num_runs}x {detect_model.split('/')[-1]} + {rank_model.split('/')[-1]} rank/dedupe",
        file=sys.stderr,
    )

    client = OpenRouterClient(reasoning_effort="high", timeout=900.0)
    total_start = time.time()

    # Phase 1: Detection (parallel)
    print(f"\n{'=' * 60}", file=sys.stderr)
    print(
        f"PHASE 1: Detection ({num_runs}x {detect_model.split('/')[-1]})",
        file=sys.stderr,
    )
    print(f"{'=' * 60}", file=sys.stderr)

    configs = [
        RunConfig(run_id=f"run_{i + 1}", model=detect_model, seed=i + 1)
        for i in range(num_runs)
    ]

    detection_tasks = [
        _run_detection_pass(config, page_images, client) for config in configs
    ]
    detection_results = await asyncio.gather(*detection_tasks)

    all_findings = []
    for findings in detection_results:
        all_findings.extend(findings)

    print(f"\nPhase 1 complete: {len(all_findings)} raw findings", file=sys.stderr)

    # Phase 2: Rank and Dedupe
    print(f"\n{'=' * 60}", file=sys.stderr)
    print(f"PHASE 2: Rank/Dedupe ({rank_model.split('/')[-1]})", file=sys.stderr)
    print(f"{'=' * 60}", file=sys.stderr)

    ranked = await _rank_and_dedupe(all_findings, page_images, client, rank_model)

    high = ranked.get("high", [])
    medium = ranked.get("medium", [])
    low = ranked.get("low", [])
    total_ranked = len(high) + len(medium) + len(low)

    print(
        f"Phase 2 complete: {total_ranked} unique findings (H:{len(high)} M:{len(medium)} L:{len(low)})",
        file=sys.stderr,
    )

    elapsed = time.time() - total_start

    # Build result
    result = {
        "metadata": {
            "strategy": "ensemble-10x-flash-pro-rankdedupe",
            "detect_model": detect_model,
            "rank_model": rank_model,
            "num_runs": num_runs,
            "elapsed_seconds": round(elapsed, 1),
        },
        "high": high,
        "medium": medium,
        "low": low,
        "raw_findings": all_findings,
        "summary": {
            "raw_findings": len(all_findings),
            "high": len(high),
            "medium": len(medium),
            "low": len(low),
            "total_unique": total_ranked,
        },
    }

    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    print(f"\n{'=' * 60}", file=sys.stderr)
    print(f"Completed in {elapsed:.1f}s", file=sys.stderr)
    print(
        f"Raw: {len(all_findings)} -> Unique: {total_ranked} (H:{len(high)} M:{len(medium)} L:{len(low)})",
        file=sys.stderr,
    )
    print(f"Output: {output_path}", file=sys.stderr)

    return result


async def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Ensemble detection (10x flash + pro rank/dedupe)"
    )
    parser.add_argument("pdf", type=Path, help="PDF file to analyze")
    parser.add_argument("-o", "--output", type=Path, help="Output JSON path")
    parser.add_argument(
        "--detect-model", default=DEFAULT_DETECT_MODEL, help="Detection model"
    )
    parser.add_argument(
        "--rank-model", default=DEFAULT_RANK_MODEL, help="Rank/dedupe model"
    )
    parser.add_argument(
        "-n",
        "--runs",
        type=int,
        default=DEFAULT_NUM_RUNS,
        help="Number of detection runs",
    )

    args = parser.parse_args()

    await run_ensemble(
        args.pdf,
        args.output,
        args.detect_model,
        args.rank_model,
        args.runs,
    )


if __name__ == "__main__":
    asyncio.run(main())
