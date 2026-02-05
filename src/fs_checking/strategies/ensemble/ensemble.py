"""Ensemble detection: 10x Flash + Rank/Dedupe with Pro.

Based on extensive benchmarking:
- 10x flash runs achieve 88.9% recall on 27-error test set
- Gemini Pro rank+dedupe reduces 166 candidates to 21 unique findings
- Final: 90.9% F1, 86.2% recall, 96.2% precision
- Total cost ~$0.15, time ~3-4 minutes

Uses native PDF page shuffling for diversity (lossless, no image conversion).
Each detection run sees pages in a different random order, but page numbers
in document headers are preserved so the model reports correct page references.

Usage:
    from fs_checking.strategies.ensemble import run_ensemble
    result = await run_ensemble(Path("document.pdf"))
"""

import asyncio
import base64
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from ...api import OpenRouterClient
from ...pdf_utils import get_page_count, shuffle_pdf_pages

DEFAULT_DETECT_MODEL = "google/gemini-3-flash-preview"
DEFAULT_RANK_MODEL = "google/gemini-3-pro-preview"
DEFAULT_NUM_RUNS = 10

# Pricing per 1M tokens (from OpenRouter, as of Jan 2025)
MODEL_PRICING = {
    "google/gemini-3-flash-preview": {"input": 0.50, "output": 3.00},
    "google/gemini-3-pro-preview": {"input": 2.00, "output": 12.00},
    "google/gemini-2.5-flash-preview": {"input": 0.15, "output": 0.60},
    "google/gemini-2.5-pro-preview": {"input": 1.25, "output": 10.00},
}


DETECT_PROMPT = """\
You are a financial statement auditor. Analyze these financial statements for errors.

IMPORTANT: Pages may appear in shuffled order. Use the page numbers shown in the 
document headers/footers (e.g., "Page 8" or "8" at top of page), NOT the PDF position.

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
4. Use DOCUMENT page numbers (from headers), not PDF position

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


def _calculate_cost(usage: dict, model: str) -> float:
    """Calculate cost in USD from usage dict."""
    pricing = MODEL_PRICING.get(model, {"input": 0, "output": 0})
    input_tokens = usage.get("prompt_tokens", 0)
    output_tokens = usage.get("completion_tokens", 0)
    return (
        input_tokens * pricing["input"] + output_tokens * pricing["output"]
    ) / 1_000_000


async def _run_detection_pass(
    config: RunConfig,
    pdf_bytes: bytes,
    pdf_name: str,
    client: OpenRouterClient,
    shuffle: bool = True,
) -> tuple[list[dict], dict]:
    """Run a single detection pass on PDF.

    Args:
        config: Run configuration with seed for shuffling
        pdf_bytes: Original PDF bytes
        pdf_name: Filename for the PDF
        client: API client
        shuffle: If True, shuffle pages using config.seed

    Returns:
        Tuple of (findings list, usage dict)
    """
    # Optionally shuffle pages
    if shuffle:
        pdf_to_send = shuffle_pdf_pages(pdf_bytes, config.seed)
    else:
        pdf_to_send = pdf_bytes

    # Build message with PDF
    pdf_b64 = base64.b64encode(pdf_to_send).decode()
    user_content = [
        {
            "type": "file",
            "file": {
                "filename": pdf_name,
                "file_data": f"data:application/pdf;base64,{pdf_b64}",
            },
        },
        {"type": "text", "text": DETECT_PROMPT},
    ]

    messages = [{"role": "user", "content": user_content}]

    # Single pass - no tools, just JSON output
    response = await client.chat(model=config.model, messages=messages)
    content = response.get("message", {}).get("content", "")
    usage = response.get("usage", {})

    # Parse JSON array from response
    findings = []
    try:
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
    return findings, usage


async def _rank_and_dedupe(
    candidates: list[dict],
    pdf_bytes: bytes,
    pdf_name: str,
    client: OpenRouterClient,
    model: str,
) -> tuple[dict, dict]:
    """Rank and deduplicate findings using PDF context.

    Returns:
        Tuple of (ranked results dict, usage dict)
    """
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

    pdf_b64 = base64.b64encode(pdf_bytes).decode()
    user_content = [
        {
            "type": "file",
            "file": {
                "filename": pdf_name,
                "file_data": f"data:application/pdf;base64,{pdf_b64}",
            },
        },
        {"type": "text", "text": prompt},
    ]

    response = await client.chat(
        model=model,
        messages=[{"role": "user", "content": user_content}],
    )

    content = response.get("message", {}).get("content", "")
    usage = response.get("usage", {})

    # Parse response
    try:
        json_match = re.search(r"\{[\s\S]*\}", content)
        if json_match:
            return json.loads(json_match.group()), usage
    except json.JSONDecodeError as e:
        print(f"Parse error in rank_dedupe: {e}", file=sys.stderr)

    return {"high": [], "medium": [], "low": [], "error": "parse_failed"}, usage


async def run_ensemble(
    pdf_path: Path,
    output_path: Path | None = None,
    detect_model: str = DEFAULT_DETECT_MODEL,
    rank_model: str = DEFAULT_RANK_MODEL,
    num_runs: int = DEFAULT_NUM_RUNS,
    shuffle: bool = True,
) -> dict:
    """Run ensemble detection with rank/dedupe.

    Pipeline:
    1. Detection: N parallel runs with shuffled page order
    2. Rank/Dedupe: Gemini Pro organizes and deduplicates findings

    Args:
        pdf_path: Path to PDF file
        output_path: Output JSON path (default: <pdf>.ensemble.json)
        detect_model: Model for detection phase (default: gemini-3-flash)
        rank_model: Model for rank/dedupe phase (default: gemini-3-pro)
        num_runs: Number of parallel detection runs (default: 10)
        shuffle: If True (default), shuffle PDF pages for each run for diversity

    Returns:
        Result dict with prioritized findings and metadata
    """
    output_path = output_path or pdf_path.with_suffix(".ensemble.json")

    print(f"Loading {pdf_path}...", file=sys.stderr)
    pdf_bytes = pdf_path.read_bytes()
    num_pages = get_page_count(pdf_bytes)
    print(f"Pages: {num_pages}", file=sys.stderr)

    mode = "PDF shuffled" if shuffle else "PDF sequential"
    print(f"Mode: {mode}", file=sys.stderr)
    print(
        f"Strategy: {num_runs}x {detect_model.split('/')[-1]} + {rank_model.split('/')[-1]} rank/dedupe",
        file=sys.stderr,
    )

    client = OpenRouterClient(reasoning_effort="high", timeout=1800.0)
    total_start = time.time()
    total_cost = 0.0

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
        _run_detection_pass(config, pdf_bytes, pdf_path.name, client, shuffle=shuffle)
        for config in configs
    ]

    detection_results = await asyncio.gather(*detection_tasks)

    all_findings = []
    for findings, usage in detection_results:
        all_findings.extend(findings)
        total_cost += _calculate_cost(usage, detect_model)

    print(f"\nPhase 1 complete: {len(all_findings)} raw findings", file=sys.stderr)

    # Phase 2: Rank and Dedupe
    print(f"\n{'=' * 60}", file=sys.stderr)
    print(f"PHASE 2: Rank/Dedupe ({rank_model.split('/')[-1]})", file=sys.stderr)
    print(f"{'=' * 60}", file=sys.stderr)

    ranked, rank_usage = await _rank_and_dedupe(
        all_findings, pdf_bytes, pdf_path.name, client, rank_model
    )
    total_cost += _calculate_cost(rank_usage, rank_model)

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
            "strategy": "ensemble-flash-pro-rankdedupe",
            "detect_model": detect_model,
            "rank_model": rank_model,
            "num_runs": num_runs,
            "mode": "pdf_shuffled" if shuffle else "pdf_sequential",
            "num_pages": num_pages,
            "elapsed_seconds": round(elapsed, 1),
            "estimated_cost_usd": round(total_cost, 4),
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
    print(f"Completed in {elapsed:.1f}s (est. ${total_cost:.4f})", file=sys.stderr)
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
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Disable page shuffling (all runs see same page order)",
    )

    args = parser.parse_args()

    await run_ensemble(
        args.pdf,
        args.output,
        args.detect_model,
        args.rank_model,
        args.runs,
        shuffle=not args.no_shuffle,
    )


if __name__ == "__main__":
    asyncio.run(main())
