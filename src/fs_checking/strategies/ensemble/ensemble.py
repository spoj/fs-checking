"""Ensemble detection: Nx Flash + Pro rank/dedupe.

Pipeline:
  1. Launch N parallel Gemini Flash runs, each seeing shuffled page order
  2. Race pattern: launch more than needed, keep first K to finish
  3. Gemini Pro deduplicates and ranks all findings

Best result on mixed doping (29 errors, ar2019):
  25x Flash race 30/25 — 89.7% recall, 85.2% F1, ~$4

Uses native PDF page shuffling for diversity (lossless PyMuPDF page reorder).
Page numbers in document headers are preserved so the model reports correct
page references despite seeing pages in random order.

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


LOG_ISSUE_TOOL = {
    "type": "function",
    "function": {
        "name": "log_issue",
        "description": "Log a financial statement error you have found. Call this each time you identify an issue.",
        "parameters": {
            "type": "object",
            "properties": {
                "id": {
                    "type": "string",
                    "description": "Unique snake_case identifier for this error (e.g. 'ppe_rollforward_total_mismatch')",
                },
                "category": {
                    "type": "string",
                    "enum": [
                        "cross_footing",
                        "rollforward",
                        "note_ties",
                        "presentation",
                        "reasonableness",
                    ],
                    "description": "Error category",
                },
                "page": {
                    "type": "integer",
                    "description": "Document page number (from headers/footers, NOT PDF position)",
                },
                "description": {
                    "type": "string",
                    "description": "Clear description with specific numbers showing the discrepancy",
                },
                "expected": {
                    "type": "number",
                    "description": "The correct/expected value",
                },
                "actual": {
                    "type": "number",
                    "description": "The incorrect value found in the document",
                },
            },
            "required": ["id", "category", "page", "description"],
        },
    },
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
- CF items must tie to Note reconciliations

### 4. PRESENTATION & LABELING
Read every label, header, and reference carefully. Report ANY of these:

**Dates & Periods**
- Column header years must match the reporting period (e.g. "2019" column under \
"Year ended 31 December 2019")
- Title dates, subtitle dates, and column headers must all be consistent
- "(Restated)" or "(Re-presented)" labels must appear on ALL comparative columns \
that were restated — flag if missing from any column that should have it

**Labels & Classifications**
- Section headings must match their content (e.g. "Non-current assets" section \
should not contain current items)
- "Continuing operations" vs "Discontinued operations" labels must be used correctly
- Direction words must be consistent: "Due from" (asset) vs "Due to" (liability), \
"inflow" vs "outflow" in cash flow statements
- Sign words must match values: "profit" with positive, "loss" with negative; \
"receivable" in receivable notes, "payable" in payable notes

**References**
- Every "(Note X)" cross-reference must point to the correct note number — verify \
the referenced note actually discusses the line item
- Accounting standard references must be correct (e.g. IFRS 16 is Leases, not \
IFRS 17 which is Insurance Contracts; IAS 19 is Employee Benefits, not IAS 20)
- Note numbering should be sequential with no gaps

**Currency & Units**
- Currency labels (e.g. US$, HK$, EUR) must be consistent within each statement
- Unit labels (e.g. "'000", "millions") must be consistent across columns

### 5. REASONABLENESS
- Large year-on-year swings (>50%) without explanation in notes
- Suspiciously round numbers that should be calculated totals
- Order-of-magnitude outliers vs peer line items in the same table

## Instructions

1. Work through EVERY page systematically
2. Each time you find an error, call the `log_issue` tool immediately
3. Be thorough - missing errors is worse than false positives
4. Use DOCUMENT page numbers (from headers), not PDF position
5. Pay special attention to PRESENTATION errors — they are easy to overlook but \
just as important as math errors. Read every label and reference, don't just check numbers.
6. After checking all pages, write a brief summary of your review

Log each error as you find it using the log_issue tool.
"""

RANK_DEDUPE_PROMPT = """\
Deduplicate and rank the following {num_candidates} error findings from a financial \
statement review. Use the check categories below to judge severity — anything that \
falls under these categories is at least MEDIUM, not LOW.

{detect_prompt}

## Candidates

{candidates_json}

## Instructions

1. DEDUPLICATE: Merge findings about the exact same error. Keep DIFFERENT errors on \
the same page as separate entries (e.g. a math error and a label error on the same \
line are two distinct issues).
2. RANK as HIGH / MEDIUM / LOW based on the check categories above.
3. Do NOT validate whether errors are real — assume they are. Just organize them.

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
    """Calculate cost in USD from usage dict.

    Prefers OpenRouter's authoritative `cost` field when available.
    Falls back to naive token×rate calculation (which overestimates
    because it doesn't account for cached token discounts).
    """
    # OpenRouter returns cost directly (accounts for caching discounts)
    if "cost" in usage and usage["cost"] is not None:
        return float(usage["cost"])

    # Fallback: naive calculation (no cache discount)
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
    stagger_max: float = 0.0,
    partial_results: dict | None = None,
) -> tuple[list[dict], dict]:
    """Run a single detection pass on PDF using log_issue tool calls.

    The model calls log_issue() incrementally as it finds errors.
    We loop, feeding back tool results, until the model stops calling tools.

    Args:
        config: Run configuration with seed for shuffling
        pdf_bytes: Original PDF bytes (already rasterized if force_visual)
        pdf_name: Filename for the PDF
        client: API client
        shuffle: If True, shuffle pages using config.seed
        stagger_max: Max random delay in seconds before starting (0 = no delay)
        partial_results: Shared dict keyed by run_id. Findings are written here
            as they arrive so they survive task cancellation in race mode.

    Returns:
        Tuple of (findings list, aggregated usage dict)
    """
    # Stagger start to avoid thundering herd
    if stagger_max > 0:
        delay = random.uniform(0, stagger_max)
        await asyncio.sleep(delay)

    # Shuffle pages if requested
    if shuffle:
        pdf_to_send = shuffle_pdf_pages(pdf_bytes, config.seed)
    else:
        pdf_to_send = pdf_bytes

    # Build message with PDF
    pdf_b64 = base64.b64encode(pdf_to_send).decode()
    user_content: list[dict] = [
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
    tools = [LOG_ISSUE_TOOL]

    findings: list[dict] = []
    # Register in shared partial_results so findings survive cancellation
    if partial_results is not None:
        partial_results[config.run_id] = findings

    total_usage: dict = {"prompt_tokens": 0, "completion_tokens": 0, "cost": 0.0}
    max_turns = 30  # safety limit
    turn = 0

    for turn in range(max_turns):
        response = await client.chat(model=config.model, messages=messages, tools=tools)
        msg = response.get("message", {})
        usage = response.get("usage", {})
        total_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
        total_usage["completion_tokens"] += usage.get("completion_tokens", 0)
        if "cost" in usage and usage["cost"] is not None:
            total_usage["cost"] += float(usage["cost"])

        tool_calls = msg.get("tool_calls", [])

        # Append assistant message to conversation
        messages.append(msg)

        if not tool_calls:
            break

        # Process each tool call
        for tc in tool_calls:
            fn = tc.get("function", {})
            fn_name = fn.get("name", "")
            call_id = tc.get("id", "")

            if fn_name == "log_issue":
                try:
                    args = json.loads(fn.get("arguments", "{}"))
                    args["_run"] = config.run_id
                    findings.append(args)
                    ack = json.dumps(
                        {"status": "logged", "issue_number": len(findings)}
                    )
                except (json.JSONDecodeError, TypeError) as e:
                    ack = json.dumps({"status": "error", "message": str(e)})
            else:
                ack = json.dumps(
                    {"status": "error", "message": f"unknown tool: {fn_name}"}
                )

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": ack,
                }
            )

    print(
        f"[{config.run_id}] Found {len(findings)} errors ({turn + 1} turns)",
        file=sys.stderr,
    )
    return findings, total_usage


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
        detect_prompt=DETECT_PROMPT,
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
    num_launch: int | None = None,
    stagger_max: float = 0.0,
) -> dict:
    """Run ensemble detection with rank/dedupe.

    Pipeline:
    1. Detection: launch N parallel runs, keep first K to complete
    2. Rank/Dedupe: Gemini Pro organizes and deduplicates findings

    Accepts any PDF — native or pre-rasterized (via ``rasterize_pdf``).
    For visual-only mode, rasterize offline first and pass the result.

    Args:
        pdf_path: Path to PDF file (native or pre-rasterized)
        output_path: Output JSON path (default: <pdf>.ensemble.json)
        detect_model: Model for detection phase (default: gemini-3-flash)
        rank_model: Model for rank/dedupe phase (default: gemini-3-pro)
        num_runs: Number of detection results to keep (default: 10)
        shuffle: If True (default), shuffle PDF pages for each run for diversity
        num_launch: Total runs to launch (default: same as num_runs, no race)
        stagger_max: Max random delay in seconds before each run starts

    Returns:
        Result dict with prioritized findings and metadata
    """
    if num_launch is None:
        num_launch = num_runs
    num_launch = max(num_launch, num_runs)

    output_path = output_path or pdf_path.with_suffix(".ensemble.json")

    print(f"Loading {pdf_path}...", file=sys.stderr)
    pdf_bytes = pdf_path.read_bytes()
    num_pages = get_page_count(pdf_bytes)
    print(f"Pages: {num_pages}", file=sys.stderr)

    mode = "shuffled" if shuffle else "sequential"
    print(f"Mode: {mode}", file=sys.stderr)
    race_str = (
        f" (launch {num_launch}, keep {num_runs})" if num_launch > num_runs else ""
    )
    stagger_str = f", stagger {stagger_max:.0f}s" if stagger_max > 0 else ""
    print(
        f"Strategy: {num_runs}x {detect_model.split('/')[-1]}"
        f"{race_str}{stagger_str}"
        f" + {rank_model.split('/')[-1]} rank/dedupe",
        file=sys.stderr,
    )

    client = OpenRouterClient(reasoning_effort="high", timeout=1800.0)
    total_start = time.time()
    total_cost = 0.0

    # Phase 1: Detection
    print(f"\n{'=' * 60}", file=sys.stderr)
    print(
        f"PHASE 1: Detection ({num_launch}x {detect_model.split('/')[-1]}"
        f", keep first {num_runs})",
        file=sys.stderr,
    )
    print(f"{'=' * 60}", file=sys.stderr)

    configs = [
        RunConfig(run_id=f"run_{i + 1}", model=detect_model, seed=i + 1)
        for i in range(num_launch)
    ]

    # Shared dict for partial findings — survives task cancellation
    partial_results: dict[str, list[dict]] = {}

    # Create tasks with stagger
    tasks = [
        asyncio.create_task(
            _run_detection_pass(
                config,
                pdf_bytes,
                pdf_path.name,
                client,
                shuffle=shuffle,
                stagger_max=stagger_max,
                partial_results=partial_results,
            ),
            name=config.run_id,
        )
        for config in configs
    ]

    # Race: collect first num_runs successful completions, discard the rest
    all_findings = []
    completed_runs: set[str] = set()
    pending = set(tasks)
    completed_count = 0
    failed_count = 0

    while pending and completed_count < num_runs:
        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            try:
                findings, usage = task.result()
                completed_count += 1
                completed_runs.add(task.get_name())
                all_findings.extend(findings)
                total_cost += _calculate_cost(usage, detect_model)
                elapsed = time.time() - total_start
                print(
                    f"  [{task.get_name()}] done: {len(findings)} findings "
                    f"({completed_count}/{num_runs} kept, {elapsed:.0f}s)",
                    file=sys.stderr,
                )
            except BaseException as e:
                failed_count += 1
                print(
                    f"  [{task.get_name()}] FAILED: {e}",
                    file=sys.stderr,
                )

    # Cancel remaining slow runners but harvest their partial findings
    cancelled = 0
    partial_harvested = 0
    for task in pending:
        task.cancel()
        cancelled += 1
    if cancelled:
        # Wait for cancellation to complete
        await asyncio.gather(*pending, return_exceptions=True)
        # Harvest partial findings from cancelled runs
        for run_id, findings in partial_results.items():
            if run_id not in completed_runs and findings:
                all_findings.extend(findings)
                partial_harvested += len(findings)
        msg = f"  Cancelled {cancelled} slow runners"
        if partial_harvested:
            msg += f", harvested {partial_harvested} partial findings"
        print(msg, file=sys.stderr)

    print(
        f"\nPhase 1 complete: {len(all_findings)} raw findings "
        f"({completed_count} runs kept, {failed_count} failed, {cancelled} cancelled)",
        file=sys.stderr,
    )

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
            "num_launched": num_launch,
            "stagger_max": stagger_max,
            "input_mode": "pdf",
            "shuffled": shuffle,
            "num_pages": num_pages,
            "elapsed_seconds": round(elapsed, 1),
            "cost_usd": round(total_cost, 4),
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
    parser.add_argument(
        "--launch",
        type=int,
        default=None,
        help="Total runs to launch (keep first N fastest). Default: same as --runs",
    )
    parser.add_argument(
        "--stagger",
        type=float,
        default=0.0,
        help="Max random delay in seconds before each run starts (default: 0)",
    )
    args = parser.parse_args()

    await run_ensemble(
        args.pdf,
        args.output,
        args.detect_model,
        args.rank_model,
        args.runs,
        shuffle=not args.no_shuffle,
        num_launch=args.launch,
        stagger_max=args.stagger,
    )


if __name__ == "__main__":
    asyncio.run(main())
