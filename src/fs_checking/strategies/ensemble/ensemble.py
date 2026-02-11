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
from ...pdf_utils import get_page_count, ring_offset_pages, shuffle_pdf_pages

DEFAULT_DETECT_MODEL = "google/gemini-3-flash-preview"
DEFAULT_RANK_MODEL = "openai/gpt-5.2"
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

DETECT_PROMPT_STRUCTURED = """\
You are a financial statement auditor performing a systematic error detection review.
You MUST complete ALL THREE passes below. Do NOT stop after one pass. Continue \
calling log_issue until every page has been checked through all passes.

IMPORTANT: Pages may appear in shuffled order. Use the page numbers shown in the \
document headers/footers (e.g., "Page 8" or "8" at top of page), NOT the PDF position.

═══════════════════════════════════════════════════════════════
PASS 1 — FACE STATEMENTS (Income Statement, Balance Sheet, SOCIE, Cash Flow)
═══════════════════════════════════════════════════════════════

For EACH face statement, perform these checks IN ORDER:

### 1A. Income Statement / Profit & Loss
Locate the P&L. Read every line. Verify:
- Revenue − Cost of sales = Gross profit (check label: should say "profit" \
if positive, "loss" if negative)
- Gross profit − Operating expenses = Operating profit
- Operating profit + Finance income − Finance costs = Profit before tax
- Profit before tax − Tax = Profit for the year
- Check column headers: do the years match the reporting period?
- Check currency label (e.g., HK$'000)
- Check section heading: "Continuing operations" vs "Discontinued operations" — \
is the label correct for the content?
- For EVERY line with a note reference "(Note X)", verify Note X actually \
discusses that line item

### 1B. Balance Sheet / Statement of Financial Position
Locate the BS. Read every line. Verify:
- Non-current assets subtotal = sum of all non-current asset lines
- Current assets subtotal = sum of all current asset lines
- Total assets = Non-current + Current assets
- Similarly for liabilities: Non-current + Current = Total liabilities
- Total assets = Total liabilities + Total equity
- Check column headers: do the years match? Is "(Restated)" present on \
comparative columns that were restated?
- Check labels: "Due from" (asset) vs "Due to" (liability) — is the \
direction word correct?
- Check currency label consistency
- For EVERY line with a note reference, verify the note number is correct

### 1C. Statement of Changes in Equity (SOCIE)
Locate the SOCIE. Verify:
- Opening balance + changes during year = Closing balance (for EACH column)
- Total comprehensive income ties to the OCI statement
- Dividends, share buybacks, other movements foot correctly
- Column headers and dates are consistent

### 1D. Cash Flow Statement
Locate the Cash Flow Statement. Verify:
- Operating + Investing + Financing = Net change in cash
- Net change + Opening cash = Closing cash
- Check direction words: "inflow" vs "outflow" must match the sign
- Operating cash flow reconciliation (if shown) ties to profit before tax
- Individual line items foot to subtotals

After completing Pass 1, call log_issue for every error found, then proceed \
IMMEDIATELY to Pass 2. DO NOT STOP HERE.

═══════════════════════════════════════════════════════════════
PASS 2 — NOTES TO THE FINANCIAL STATEMENTS
═══════════════════════════════════════════════════════════════

For EACH note that contains a numerical table:
- Verify all subtotals and totals foot correctly (components sum to total)
- For rollforward schedules (PPE, provisions, receivables impairment): \
Opening + Additions − Disposals ± Transfers = Closing
- Verify the note total ties back to the corresponding face statement line
- Check currency labels match the face statements
- Check years in column headers

Pay special attention to:
- Segment notes: do segment totals reconcile to group totals?
- Related party notes: do amounts tie to face statement disclosures?
- Financial summary pages: do key figures match the face statements?

After completing Pass 2, call log_issue for every error found, then proceed \
IMMEDIATELY to Pass 3. DO NOT STOP HERE.

═══════════════════════════════════════════════════════════════
PASS 3 — PRESENTATION SWEEP (page by page)
═══════════════════════════════════════════════════════════════

Go through EVERY page and check this checklist:

□ Column header years — correct and consistent?
□ Currency labels (US$, HK$, EUR) — consistent within each statement?
□ Unit labels ('000, millions) — consistent across columns?
□ "(Restated)" / "(Re-presented)" — present on all comparative columns that \
should have it?
□ Section headings — match content? (e.g., "Non-current" section has no \
current items)
□ "Continuing" vs "Discontinued" labels — correct?
□ Direction words: "Due from" vs "Due to", "inflow" vs "outflow" — correct?
□ Sign words: "profit" vs "loss", "receivable" vs "payable" — match the values?
□ Note references "(Note X)" — does Note X actually discuss that line item?
□ Accounting standard references (IFRS/IAS/HKFRS numbers) — correct standard \
for the topic?
□ Note numbering — sequential with no gaps?
□ Any other presentation anomalies

═══════════════════════════════════════════════════════════════

## Output Instructions

- Each time you find an error, call the `log_issue` tool IMMEDIATELY
- Be thorough — missing errors is worse than false positives
- Use DOCUMENT page numbers (from headers), not PDF position
- You MUST complete all three passes before stopping
- After all three passes, write a brief summary confirming you reviewed all pages

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

1. DEDUPLICATE: Merge findings ONLY when they describe the exact same error — same \
page, same numbers, same issue. Err on the side of keeping entries separate.
   - Two errors on the same page are usually DIFFERENT errors (e.g. a math error and \
a label error on the same line). Keep them separate.
   - Two errors mentioning the same line item but on different pages are usually \
DIFFERENT errors (e.g. a tie error on the face statement vs. a footing error in the \
note). Keep them separate.
   - Only merge when the descriptions are clearly restating the identical finding \
(same page, same category, same numbers).
   - When in doubt, keep as separate entries. Missing real errors is worse than \
having duplicates.
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


VISION_TOOL = {
    "type": "function",
    "function": {
        "name": "vision",
        "description": (
            "Ask a vision model to look at the ENTIRE PDF document and answer "
            "your query. The model reads all pages and reports exactly what it "
            "sees. Use this to verify numbers, labels, cross-references, or "
            "any claim about the document."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "What to look for. Be specific: "
                        "'What is the gross profit for 2019 in the P&L? "
                        "List the exact numbers for Turnover and COGS.'"
                    ),
                },
            },
            "required": ["query"],
        },
    },
}

VALIDATE_PROMPT = """\
You are a financial statement auditor VALIDATING detection findings.

You have {num_candidates} candidate errors from an automated detection scan. \
Your job is to deduplicate, verify each finding by looking at the actual PDF, \
and produce a final clean list of confirmed errors.

## Candidates

{candidates_json}

## Tools

- **vision(query)**: Ask a vision model to read the ENTIRE PDF and answer your query. \
The model sees all pages. Use this to verify numbers, labels, cross-references, or \
any claim. You can batch multiple vision calls in a single turn — they execute in parallel.

## Workflow

1. Group candidates by section/topic for efficient checking
2. Call `vision(query=...)` to verify each group — ask specific questions about the \
numbers the candidates claim are wrong. Batch multiple vision calls per turn.
3. Based on the vision evidence, decide which candidates are real errors and which \
are false positives or duplicates.
4. When done verifying, output your final answer as a JSON array.

## Final Output

After all verification is complete, output a JSON array of confirmed, deduplicated errors. \
Each entry has only two fields:

```json
[
  {{"location": "page and section/table", "description": "what is wrong, with specific numbers"}},
  ...
]
```

- **location**: Where in the document (e.g. "Page 12, Note 21 Trade and Other Receivables, currency breakdown table")
- **description**: Clear description with the specific numbers showing the discrepancy \
(e.g. "Sum of currency components (815,732 + 26,329 + ...) = 1,157,953 but total shows 1,157,971")

## Rules

- Use vision liberally — it's cheap. Batch multiple calls per turn.
- Merge duplicates: if multiple candidates describe the same underlying error, output it once.
- Reject false positives: if vision shows the numbers are actually correct, drop it.
- Err on the side of keeping — missing real errors is worse than keeping a false positive.
- Do NOT include IDs, priorities, or categories. Just location and description.
"""


async def _vision_call(
    query: str,
    pdf_bytes: bytes,
    num_pages: int,
    client: OpenRouterClient,
    vision_model: str,
) -> str:
    """Send the full PDF to Flash with a query. Returns Flash's answer.

    Uses low reasoning effort to prevent the model from "thinking" numbers
    into consistency — we want it to read and report, not rationalize.
    """
    pdf_b64 = base64.b64encode(pdf_bytes).decode()
    messages = [
        {
            "role": "system",
            "content": (
                "You are a precise document reader for a financial statement audit. "
                "This document MAY CONTAIN ERRORS — that is the whole point of the audit. "
                "Your job is to report EXACTLY what you see on the page, not what you think "
                "should be there. NEVER adjust, round, or correct numbers to make them consistent. "
                "If a total does not match the sum of its components, report both the individual "
                "numbers AND the printed total exactly as shown. Do NOT assume the total is correct. "
                "Do NOT recompute numbers to fill gaps — only report digits you can read on the page."
            ),
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "file",
                    "file": {
                        "filename": "document.pdf",
                        "file_data": f"data:application/pdf;base64,{pdf_b64}",
                    },
                },
                {
                    "type": "text",
                    "text": (
                        f"This is a {num_pages}-page financial statement document.\n\n"
                        f"Question: {query}\n\n"
                        "IMPORTANT: Report numbers EXACTLY as printed. Do not correct or "
                        "adjust any figures. If you see a table, quote every row verbatim."
                    ),
                },
            ],
        },
    ]

    resp = await client.chat(
        model=vision_model, messages=messages, reasoning_effort="low"
    )
    answer = resp.get("message", {}).get("content", "")
    return answer or "(no response)"


async def _validate_findings(
    candidates: list[dict],
    pdf_bytes: bytes,
    client: OpenRouterClient,
    validator_model: str,
    vision_model: str,
    detect_prompt: str | None = None,
    max_turns: int = 120,
) -> tuple[list[dict], dict]:
    """Validate and deduplicate findings using a validator model with vision tool.

    The validator uses vision(query) to check findings against the PDF,
    then outputs a final JSON array of confirmed, deduplicated issues.

    Args:
        candidates: Raw detection findings to validate
        pdf_bytes: Original PDF bytes
        client: API client
        validator_model: Model for the validator (e.g., GPT-5.2)
        vision_model: Model for vision tool calls (e.g., Gemini Flash)
        detect_prompt: Detection prompt for context (unused, kept for API compat)
        max_turns: Max conversation turns

    Returns:
        Tuple of (list of validated issues, usage dict)
    """
    from ...agent_loop import run_agent_loop
    from ...pdf_utils import get_page_count

    num_pages = get_page_count(pdf_bytes)

    candidates_for_prompt = [
        {
            "id": c.get("id"),
            "page": c.get("page"),
            "category": c.get("category"),
            "description": c.get("description"),
        }
        for c in candidates
    ]

    prompt = VALIDATE_PROMPT.format(
        num_candidates=len(candidates),
        candidates_json=json.dumps(candidates_for_prompt, indent=2),
    )

    messages: list[dict] = [{"role": "user", "content": prompt}]
    tools = [VISION_TOOL]

    vision_calls = 0

    async def call_model(msgs: list[dict]) -> tuple[dict, dict]:
        resp = await client.chat(model=validator_model, messages=msgs, tools=tools)
        return resp.get("message", {}), resp.get("usage", {})

    async def execute_tool(name: str, args: dict) -> str:
        nonlocal vision_calls

        if name == "vision":
            query = args.get("query", "")
            vision_calls += 1
            print(
                f"      vision('{query[:80]}...')",
                file=sys.stderr,
            )
            return await _vision_call(
                query,
                pdf_bytes,
                num_pages,
                client,
                vision_model,
            )
        else:
            return json.dumps({"error": f"unknown tool: {name}"})

    result = await run_agent_loop(
        call_model=call_model,
        tool_executor=execute_tool,
        initial_messages=messages,
        max_iterations=max_turns,
    )

    print(
        f"    Validation complete: "
        f"{vision_calls} vision calls, {result.iterations} turns",
        file=sys.stderr,
    )

    # Parse the final message as JSON array
    issues = []
    try:
        # Look for JSON array in the final message
        json_match = re.search(r"\[[\s\S]*\]", result.final_message)
        if json_match:
            issues = json.loads(json_match.group())
    except json.JSONDecodeError as e:
        print(f"    WARNING: Failed to parse validator output: {e}", file=sys.stderr)

    if not issues:
        print(
            "    WARNING: No issues parsed from validator output, "
            "keeping all candidates as fallback",
            file=sys.stderr,
        )
        issues = [
            {
                "location": f"Page {c.get('page', '?')}",
                "description": c.get("description", ""),
            }
            for c in candidates
        ]

    return issues, result.usage


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


def _get_detect_prompt(prompt_style: str) -> str:
    """Return the detection prompt for the given style."""
    if prompt_style == "structured":
        return DETECT_PROMPT_STRUCTURED
    return DETECT_PROMPT


async def _run_detection_pass(
    config: RunConfig,
    pdf_bytes: bytes,
    pdf_name: str,
    client: OpenRouterClient,
    shuffle_mode: str = "random",
    stagger_max: float = 0.0,
    partial_results: dict | None = None,
    prompt_style: str = "default",
) -> tuple[list[dict], dict]:
    """Run a single detection pass on PDF using log_issue tool calls.

    Uses the generic agent loop. The model calls log_issue() incrementally
    as it finds errors.

    Args:
        config: Run configuration with seed for shuffling
        pdf_bytes: Original PDF bytes (already rasterized if force_visual)
        pdf_name: Filename for the PDF
        client: API client
        shuffle_mode: Page reorder mode — "random" (full shuffle), "ring"
            (circular offset preserving adjacency), or "none"
        stagger_max: Max random delay in seconds before starting (0 = no delay)
        partial_results: Shared dict keyed by run_id. Findings are written here
            as they arrive so they survive task cancellation in race mode.
        prompt_style: Detection prompt variant — "default" or "structured"

    Returns:
        Tuple of (findings list, aggregated usage dict)
    """
    from ...agent_loop import run_agent_loop

    # Stagger start to avoid thundering herd
    if stagger_max > 0:
        delay = random.uniform(0, stagger_max)
        await asyncio.sleep(delay)

    # Reorder pages based on shuffle mode
    if shuffle_mode == "random":
        pdf_to_send = shuffle_pdf_pages(pdf_bytes, config.seed)
    elif shuffle_mode == "ring":
        pdf_to_send = ring_offset_pages(pdf_bytes, config.seed)
    else:
        pdf_to_send = pdf_bytes

    # Build message with PDF
    pdf_b64 = base64.b64encode(pdf_to_send).decode()
    detect_prompt = _get_detect_prompt(prompt_style)
    user_content: list[dict] = [
        {
            "type": "file",
            "file": {
                "filename": pdf_name,
                "file_data": f"data:application/pdf;base64,{pdf_b64}",
            },
        },
        {"type": "text", "text": detect_prompt},
    ]

    initial_messages = [{"role": "user", "content": user_content}]
    tools = [LOG_ISSUE_TOOL]

    findings: list[dict] = []
    # Register in shared partial_results so findings survive cancellation
    if partial_results is not None:
        partial_results[config.run_id] = findings

    async def call_model(msgs: list[dict]) -> tuple[dict, dict]:
        resp = await client.chat(model=config.model, messages=msgs, tools=tools)
        return resp.get("message", {}), resp.get("usage", {})

    async def execute_tool(name: str, args: dict) -> str:
        if name == "log_issue":
            args["_run"] = config.run_id
            findings.append(args)
            return json.dumps({"status": "logged", "issue_number": len(findings)})
        return json.dumps({"status": "error", "message": f"unknown tool: {name}"})

    result = await run_agent_loop(
        call_model=call_model,
        tool_executor=execute_tool,
        initial_messages=initial_messages,
        max_iterations=30,
    )

    print(
        f"[{config.run_id}] Found {len(findings)} errors ({result.iterations} turns)",
        file=sys.stderr,
    )
    return findings, result.usage


async def _rank_and_dedupe(
    candidates: list[dict],
    client: OpenRouterClient,
    model: str,
    detect_prompt: str | None = None,
) -> tuple[dict, dict]:
    """Rank and deduplicate findings without PDF context.

    The PDF is intentionally excluded — giving the ranker the document
    lets it second-guess the detectors and silently drop findings it
    thinks are wrong.  Without the PDF it can only compare descriptions
    to each other, which is all dedup needs.

    Returns:
        Tuple of (ranked results dict, usage dict)
    """
    if detect_prompt is None:
        detect_prompt = DETECT_PROMPT

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
        detect_prompt=detect_prompt,
        candidates_json=json.dumps(candidates_for_prompt, indent=2),
    )

    response = await client.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
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
    shuffle_mode: str = "random",
    num_launch: int | None = None,
    stagger_max: float = 0.0,
    prompt_style: str = "default",
    timeout: float = 1800.0,
    validate: bool = False,
    vision_model: str = DEFAULT_DETECT_MODEL,
) -> dict:
    """Run ensemble detection with rank/dedupe or validation.

    Pipeline:
    1. Detection: launch N parallel runs, keep first K to complete
    2a. Rank/Dedupe (default): text-only dedup and ranking
    2b. Validate (--validate): GPT-5.2 validates each finding using a
        vision(pages, query) tool backed by Gemini Flash

    Accepts any PDF — native or pre-rasterized (via ``rasterize_pdf``).
    For visual-only mode, rasterize offline first and pass the result.

    Args:
        pdf_path: Path to PDF file (native or pre-rasterized)
        output_path: Output JSON path (default: <pdf>.ensemble.json)
        detect_model: Model for detection phase (default: gemini-3-flash)
        rank_model: Model for rank/dedupe phase (default: gpt-5.2)
        num_runs: Number of detection results to keep (default: 10)
        shuffle_mode: Page reorder mode — "random" (full shuffle, default),
            "ring" (circular offset preserving adjacency), or "none"
        num_launch: Total runs to launch (default: same as num_runs, no race)
        stagger_max: Max random delay in seconds before each run starts
        prompt_style: Detection prompt variant — "default" or "structured"
            (multi-pass procedure, better for models that stop early)
        timeout: HTTP client timeout in seconds (default: 1800)
        validate: Use vision-powered validation instead of text-only rank/dedupe
        vision_model: Model for vision tool calls during validation

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

    mode = shuffle_mode if shuffle_mode != "none" else "sequential"
    print(f"Mode: {mode}, Prompt: {prompt_style}", file=sys.stderr)
    race_str = (
        f" (launch {num_launch}, keep {num_runs})" if num_launch > num_runs else ""
    )
    stagger_str = f", stagger {stagger_max:.0f}s" if stagger_max > 0 else ""
    phase2_label = (
        f"{rank_model.split('/')[-1]} validate (vision: {vision_model.split('/')[-1]})"
        if validate
        else f"{rank_model.split('/')[-1]} rank/dedupe"
    )
    print(
        f"Strategy: {num_runs}x {detect_model.split('/')[-1]}"
        f"{race_str}{stagger_str}"
        f" + {phase2_label}",
        file=sys.stderr,
    )

    client = OpenRouterClient(reasoning_effort="high", timeout=timeout)
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
                shuffle_mode=shuffle_mode,
                stagger_max=stagger_max,
                partial_results=partial_results,
                prompt_style=prompt_style,
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

    # Phase 2: Rank/Dedupe or Validate
    print(f"\n{'=' * 60}", file=sys.stderr)
    if validate:
        print(
            f"PHASE 2: Validate ({rank_model.split('/')[-1]} + {vision_model.split('/')[-1]} vision)",
            file=sys.stderr,
        )
        print(f"{'=' * 60}", file=sys.stderr)

        validated_issues, validate_usage = await _validate_findings(
            all_findings,
            pdf_bytes,
            client,
            validator_model=rank_model,
            vision_model=vision_model,
            detect_prompt=_get_detect_prompt(prompt_style),
        )
        total_cost += _calculate_cost(validate_usage, rank_model)

        print(
            f"Phase 2 complete: {len(validated_issues)} validated issues",
            file=sys.stderr,
        )

        elapsed = time.time() - total_start

        result = {
            "metadata": {
                "strategy": "ensemble-validate",
                "detect_model": detect_model,
                "validator_model": rank_model,
                "vision_model": vision_model,
                "num_runs": num_runs,
                "num_launched": num_launch,
                "stagger_max": stagger_max,
                "input_mode": "pdf",
                "shuffled": shuffle_mode != "none",
                "shuffle_mode": shuffle_mode,
                "prompt_style": prompt_style,
                "num_pages": num_pages,
                "elapsed_seconds": round(elapsed, 1),
                "cost_usd": round(total_cost, 4),
            },
            "issues": validated_issues,
            "raw_findings": all_findings,
            "summary": {
                "raw_findings": len(all_findings),
                "validated_issues": len(validated_issues),
            },
        }

        output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))

        print(f"\n{'=' * 60}", file=sys.stderr)
        print(f"Completed in {elapsed:.1f}s (est. ${total_cost:.4f})", file=sys.stderr)
        print(
            f"Raw: {len(all_findings)} -> Validated: {len(validated_issues)}",
            file=sys.stderr,
        )
        print(f"Output: {output_path}", file=sys.stderr)

        return result
    else:
        print(f"PHASE 2: Rank/Dedupe ({rank_model.split('/')[-1]})", file=sys.stderr)
        print(f"{'=' * 60}", file=sys.stderr)

        ranked, rank_usage = await _rank_and_dedupe(
            all_findings,
            client,
            rank_model,
            detect_prompt=_get_detect_prompt(prompt_style),
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
            "shuffled": shuffle_mode != "none",
            "shuffle_mode": shuffle_mode,
            "prompt_style": prompt_style,
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
        "--shuffle-mode",
        choices=["random", "ring", "none"],
        default="random",
        help="Page reorder mode: random (full shuffle), ring (circular offset), none",
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
    parser.add_argument(
        "--prompt-style",
        choices=["default", "structured"],
        default="default",
        help="Detection prompt variant: default (standard), structured (multi-pass, better for GPT-5.2)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=1800.0,
        help="HTTP client timeout in seconds (default: 1800 = 30 min)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Use vision-powered validation instead of text-only rank/dedupe. "
        "GPT-5.2 validates each finding by calling vision(pages, query) "
        "backed by Gemini Flash.",
    )
    parser.add_argument(
        "--vision-model",
        default=DEFAULT_DETECT_MODEL,
        help="Model for vision tool calls during validation (default: gemini-3-flash)",
    )
    args = parser.parse_args()

    await run_ensemble(
        args.pdf,
        args.output,
        args.detect_model,
        args.rank_model,
        args.runs,
        shuffle_mode=args.shuffle_mode,
        num_launch=args.launch,
        stagger_max=args.stagger,
        prompt_style=args.prompt_style,
        timeout=args.timeout,
        validate=args.validate,
        vision_model=args.vision_model,
    )


if __name__ == "__main__":
    asyncio.run(main())
