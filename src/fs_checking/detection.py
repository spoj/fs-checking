"""Core detection module - single-pass error detection.

Provides:
- _run_single_detection_pass(): Tool-based detection with page shuffling
- RunConfig: Configuration for detection runs
- run_baseline(): Simple single-run detection

Used by:
- benchmark.py: For variance analysis
- strategies/ensemble: For parallel detection runs

Usage:
    from fs_checking.detection import run_baseline
    result = await run_baseline(Path("document.pdf"))
"""

import asyncio
import base64
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .api import OpenRouterClient
from .pdf_utils import pdf_to_images

DEFAULT_MODEL = "google/gemini-3-flash-preview"


# === Tool-based prompt (for incremental output) ===

TOOL_PROMPT = """\
You are a financial statement auditor. Analyze these financial statements for errors.

## Your Task

Examine EVERY page. Call `save_check` for each ERROR found.

### 1. CROSS-FOOTING (Math Checks)
- Every subtotal must equal sum of its components
- Balance Sheet: NCA total, CA total, CL total, Net current assets, Total equity
- P&L: Gross profit, Operating profit, PBT, Profit for year
- OCI: Items that will/won't be reclassified subtotals, Total comprehensive income
- Cash Flow: Operating/Investing/Financing subtotals, Net change in cash
- Notes: All subtotals within tables

### 2. ROLLFORWARDS (Opening + Changes = Closing)
Check EVERY rollforward table row by row:
- PPE: Opening NBV + Additions - Depreciation - Disposals +/- Revaluation +/- FX = Closing NBV
- Provisions: Opening + Charge - Utilization - Reversal +/- FX = Closing
- Receivables impairment: Opening + Increase - Write-off - Reversal +/- FX = Closing
- Any other movement schedule

### 3. STATEMENT ↔ NOTE TIES (Critical!)
Primary statements must tie EXACTLY to notes:
- BS PPE → Note PPE closing NBV (MUST match exactly)
- BS Cash → CF closing cash
- P&L Cost of sales → Note 4 cost of inventories
- P&L Depreciation → Note 4 depreciation → PPE Note depreciation
- P&L Amortization → Note 4 amortization
- P&L Interest → Note 4 interest → CF reconciliation interest
- CF items → Note 31/Cash flow reconciliation note
- Any line item referencing a note → that note's total

### 4. PRESENTATION & CONSISTENCY
- Title/header dates match column headers (e.g., "Year ended 2023" with 2023/2022 columns)
- Labels match values (e.g., "Net current liabilities" should be negative, not positive)
- Note references valid (e.g., "Note 1" actually exists and is relevant)
- Note numbering sequential (no gaps like Note 9, Note 11 skipping Note 10)
- "CONSOLIDATED" label consistent across all statements
- Negative values in aging tables (shouldn't happen)

### 5. REASONABLENESS
- Unusual large swings YoY (>50% change in major items)
- Round number differences ($100k, $50k) often indicate errors

## Instructions

1. Work through EVERY page systematically
2. Check EVERY subtotal, EVERY rollforward row, EVERY note tie
3. Call `save_check()` for each error found
4. When done, call `finish()`

Be extremely thorough. Missing errors is worse than false positives.
"""

TOOL_SAVE_CHECK = {
    "type": "function",
    "function": {
        "name": "save_check",
        "description": "Record a verification check result",
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
                "status": {"type": "string", "enum": ["fail", "warn"]},
                "expected": {
                    "type": "number",
                    "description": "Calculated/expected value",
                },
                "actual": {"type": "number", "description": "Stated value in document"},
                "difference": {"type": "number", "description": "expected - actual"},
                "description": {"type": "string", "description": "What was checked"},
                "page": {"type": "integer", "description": "Page number"},
            },
            "required": ["id", "category", "status", "description", "page"],
        },
    },
}

TOOL_FINISH = {
    "type": "function",
    "function": {
        "name": "finish",
        "description": "Complete the verification with a summary",
        "parameters": {
            "type": "object",
            "properties": {
                "key_findings": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of significant issues found",
                },
            },
            "required": ["key_findings"],
        },
    },
}


# === Single-shot prompt (original) ===

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


async def run_baseline_with_tools(
    pdf_path: Path,
    output_path: Path | None = None,
    model: str | None = None,
    max_iterations: int = 50,
    shuffle_pages: bool = False,
    seed: int | None = None,
) -> dict:
    """Run tool-based baseline - model calls save_check() incrementally.

    This approach can produce unlimited checks since each is saved via tool call.
    """
    import random

    model = model or DEFAULT_MODEL
    output_path = output_path or pdf_path.with_suffix(".baseline.json")

    print(f"Loading {pdf_path}...", file=sys.stderr)
    print(f"Model: {model}", file=sys.stderr)
    print("Strategy: tool-based baseline (incremental)", file=sys.stderr)

    # Convert PDF to images
    page_images = pdf_to_images(pdf_path.read_bytes(), dpi=150)
    print(f"Pages: {len(page_images)}", file=sys.stderr)

    # Create page indices (0-based internally, 1-based for display)
    page_indices = list(range(len(page_images)))
    if shuffle_pages:
        if seed is not None:
            random.seed(seed)
        random.shuffle(page_indices)
        print(
            f"Shuffled order (seed={seed}): {[i + 1 for i in page_indices]}",
            file=sys.stderr,
        )

    # Build message with all pages
    user_content = []
    for idx in page_indices:
        img_bytes = page_images[idx]
        # Label with original page number so model knows actual page refs
        user_content.append({"type": "text", "text": f"\n=== Page {idx + 1} ==="})
        img_b64 = base64.b64encode(img_bytes).decode()
        user_content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
            }
        )

    user_content.append({"type": "text", "text": f"\n\n{TOOL_PROMPT}"})

    messages = [{"role": "user", "content": user_content}]
    tools = [TOOL_SAVE_CHECK, TOOL_FINISH]

    client = OpenRouterClient(reasoning_effort="high", timeout=900.0)

    # Collect results
    checks = []
    key_findings = []
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0}

    start = time.time()
    print("Running verification...", file=sys.stderr)

    iteration = 0
    for iteration in range(max_iterations):
        response = await client.chat(
            model=model,
            messages=messages,
            tools=tools,
        )

        usage = response.get("usage", {})
        total_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
        total_usage["completion_tokens"] += usage.get("completion_tokens", 0)

        message = response.get("message", {})
        tool_calls = message.get("tool_calls", [])

        if not tool_calls:
            # Model finished without calling finish() - that's ok
            print(f"[iter {iteration + 1}] No tool calls, done", file=sys.stderr)
            break

        messages.append(message)

        # Process tool calls
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
                        "content": f"Saved check {args.get('id')} ({args.get('status')})",
                    }
                )
                print(
                    f"[iter {iteration + 1}] save_check: {args.get('id')} -> {args.get('status')}",
                    file=sys.stderr,
                )

            elif name == "finish":
                key_findings = args.get("key_findings", [])
                tool_results.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": "Verification complete.",
                    }
                )
                finished = True
                print(f"[iter {iteration + 1}] finish() called", file=sys.stderr)

        messages.extend(tool_results)

        if finished:
            break

    elapsed = time.time() - start

    # Build result
    pass_count = sum(1 for c in checks if c.get("status") == "pass")
    fail_count = sum(1 for c in checks if c.get("status") == "fail")
    warn_count = sum(1 for c in checks if c.get("status") == "warn")

    result = {
        "metadata": {},
        "checks": checks,
        "summary": {
            "total_checks": len(checks),
            "passed": pass_count,
            "failed": fail_count,
            "warned": warn_count,
            "key_findings": key_findings,
        },
    }

    # Write output
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    print(f"\n{'=' * 60}", file=sys.stderr)
    print(f"Completed in {elapsed:.1f}s ({iteration + 1} iterations)", file=sys.stderr)
    print(
        f"Tokens: {total_usage['prompt_tokens']:,} in, {total_usage['completion_tokens']:,} out",
        file=sys.stderr,
    )
    print(
        f"Checks: {len(checks)} total ({pass_count} pass, {fail_count} fail, {warn_count} warn)",
        file=sys.stderr,
    )
    print(f"Output: {output_path}", file=sys.stderr)

    return result


async def run_baseline(
    pdf_path: Path,
    output_path: Path | None = None,
    model: str | None = None,
    use_tools: bool = False,
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
                "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
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


VERIFY_PROMPT_SINGLE = """\
You are verifying a potential error found in financial statements.

## Claimed Error
{error_description}

- ID: {error_id}
- Page: {page}
- Expected: {expected}
- Actual: {actual}
- Difference: {difference}

## Task
Look at the relevant page(s) and verify if this is a REAL error or a FALSE POSITIVE.

1. Find the exact numbers mentioned
2. Perform the calculation yourself
3. Determine if there's actually an error

Respond with a JSON object:
```json
{{
  "verified": true/false,
  "reasoning": "Brief explanation of your verification",
  "calculated": <your calculated value if applicable>
}}
```

Return ONLY the JSON, no other text.
"""

VERIFY_BATCH_PROMPT = """\
You are verifying potential errors found in financial statements.

## Candidates for Page {page}

The following {count} potential errors were flagged by multiple detection passes.
Some may be duplicates describing the same underlying issue.

{findings_json}

## Task

1. Look at the page image carefully
2. For each candidate, verify if it's a REAL error by checking the actual numbers
3. Identify duplicates - findings that describe the same underlying error
4. Return only UNIQUE, VERIFIED errors

Respond with a JSON object:
```json
{{
  "verified_errors": [
    {{
      "id": "keep the original id of the best description",
      "description": "clear description of the verified error",
      "expected": <calculated value>,
      "actual": <stated value>,
      "difference": <expected - actual>,
      "reasoning": "brief explanation of verification",
      "duplicate_ids": ["ids", "of", "duplicates", "merged", "into", "this"]
    }}
  ],
  "rejected": [
    {{
      "id": "original id",
      "reason": "why this is not a real error or is a duplicate"
    }}
  ]
}}
```

Return ONLY the JSON, no other text.
"""


async def verify_finding(
    finding: dict,
    page_images: list[bytes],
    client: OpenRouterClient,
    model: str,
) -> dict:
    """Verify a single finding against the PDF pages."""
    # Get relevant pages
    pages = finding.get("page", 1)
    if isinstance(pages, str):
        # Handle "3,8" format
        page_nums = [int(p.strip()) for p in pages.split(",")]
    elif isinstance(pages, list):
        page_nums = pages
    else:
        page_nums = [int(pages)]

    # Build message with relevant pages only
    user_content = []
    for pn in page_nums:
        if 1 <= pn <= len(page_images):
            img_bytes = page_images[pn - 1]
            user_content.append({"type": "text", "text": f"\n=== Page {pn} ==="})
            img_b64 = base64.b64encode(img_bytes).decode()
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                }
            )

    prompt = VERIFY_PROMPT_SINGLE.format(
        error_description=finding.get("description", ""),
        error_id=finding.get("id", ""),
        page=finding.get("page", ""),
        expected=finding.get("expected", "N/A"),
        actual=finding.get("actual", "N/A"),
        difference=finding.get("difference", "N/A"),
    )
    user_content.append({"type": "text", "text": prompt})

    response = await client.chat(
        model=model,
        messages=[{"role": "user", "content": user_content}],
    )

    content = response.get("message", {}).get("content", "")

    # Parse response
    try:
        # Try to extract JSON
        import re

        json_match = re.search(r"\{[\s\S]*\}", content)
        if json_match:
            result = json.loads(json_match.group())
            return {
                **finding,
                "verified": result.get("verified", False),
                "verification_reasoning": result.get("reasoning", ""),
            }
    except (json.JSONDecodeError, AttributeError):
        pass

    # Default to not verified if can't parse
    return {**finding, "verified": False, "verification_reasoning": "Parse error"}


async def verify_page_batch(
    page_num: int,
    findings: list[dict],
    page_images: list[bytes],
    client: OpenRouterClient,
    model: str,
) -> tuple[list[dict], list[dict]]:
    """Verify all findings for a single page, deduplicating in the process.

    Returns (verified_findings, rejected_findings)
    """
    import re

    if not findings:
        return [], []

    # Build message with the page image
    user_content = []
    if 1 <= page_num <= len(page_images):
        img_bytes = page_images[page_num - 1]
        user_content.append({"type": "text", "text": f"\n=== Page {page_num} ==="})
        img_b64 = base64.b64encode(img_bytes).decode()
        user_content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
            }
        )

    # Format findings for the prompt
    findings_for_prompt = [
        {
            "id": f.get("id"),
            "category": f.get("category"),
            "description": f.get("description"),
            "expected": f.get("expected"),
            "actual": f.get("actual"),
            "difference": f.get("difference"),
        }
        for f in findings
    ]

    prompt = VERIFY_BATCH_PROMPT.format(
        page=page_num,
        count=len(findings),
        findings_json=json.dumps(findings_for_prompt, indent=2),
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
            result = json.loads(json_match.group())

            verified = []
            for v in result.get("verified_errors", []):
                verified.append(
                    {
                        "id": v.get("id"),
                        "page": page_num,
                        "category": "verified",
                        "description": v.get("description"),
                        "expected": v.get("expected"),
                        "actual": v.get("actual"),
                        "difference": v.get("difference"),
                        "verified": True,
                        "verification_reasoning": v.get("reasoning", ""),
                        "duplicate_ids": v.get("duplicate_ids", []),
                    }
                )

            rejected = []
            for r in result.get("rejected", []):
                rejected.append(
                    {
                        "id": r.get("id"),
                        "page": page_num,
                        "verified": False,
                        "rejection_reason": r.get("reason", ""),
                    }
                )

            return verified, rejected
    except (json.JSONDecodeError, AttributeError) as e:
        print(f"  [Page {page_num}] Parse error: {e}", file=sys.stderr)

    # On parse failure, return nothing verified
    return [], [
        {
            "id": f.get("id"),
            "page": page_num,
            "verified": False,
            "rejection_reason": "Parse error",
        }
        for f in findings
    ]


@dataclass
class RunConfig:
    """Configuration for a single detection run."""

    model: str
    shuffle: bool = True
    seed: int | None = None
    label: str = ""

    def __post_init__(self):
        if not self.label:
            model_short = self.model.split("/")[-1]
            order = "shuffle" if self.shuffle else "seq"
            self.label = f"{model_short}({order})"


# Default ensemble configuration: 3x flash with different page orders
DEFAULT_ENSEMBLE_RUNS = [
    RunConfig("google/gemini-3-flash-preview", shuffle=False, label="flash(seq)"),
    RunConfig(
        "google/gemini-3-flash-preview", shuffle=True, seed=1, label="flash(shuf1)"
    ),
    RunConfig(
        "google/gemini-3-flash-preview", shuffle=True, seed=2, label="flash(shuf2)"
    ),
]

DEFAULT_VALIDATOR_MODEL = "google/gemini-3-flash-preview"


async def _run_single_detection_pass(
    run_config: RunConfig,
    run_idx: int,
    page_images: list[bytes],
    client: OpenRouterClient,
) -> tuple[list[dict], dict]:
    """Run a single detection pass. Returns (findings, usage)."""
    import random

    page_indices = list(range(len(page_images)))

    if run_config.shuffle:
        seed = run_config.seed if run_config.seed is not None else (run_idx + 1)
        random.seed(seed)
        random.shuffle(page_indices)
        order_str = f"shuffle(seed={seed})"
    else:
        order_str = "sequential"

    print(
        f"[{run_config.label}] {run_config.model}, {order_str}",
        file=sys.stderr,
    )

    # Build message with pages in specified order
    user_content = []
    for idx in page_indices:
        img_bytes = page_images[idx]
        user_content.append({"type": "text", "text": f"\n=== Page {idx + 1} ==="})
        img_b64 = base64.b64encode(img_bytes).decode()
        user_content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
            }
        )
    user_content.append({"type": "text", "text": f"\n\n{TOOL_PROMPT}"})

    messages = [{"role": "user", "content": user_content}]
    tools = [TOOL_SAVE_CHECK, TOOL_FINISH]

    findings = []
    usage = {"prompt_tokens": 0, "completion_tokens": 0}

    # Run detection loop
    for iteration in range(50):
        response = await client.chat(
            model=run_config.model, messages=messages, tools=tools
        )

        resp_usage = response.get("usage", {})
        usage["prompt_tokens"] += resp_usage.get("prompt_tokens", 0)
        usage["completion_tokens"] += resp_usage.get("completion_tokens", 0)

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
                args["_run"] = run_config.label
                args["_model"] = run_config.model
                findings.append(args)
                tool_results.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": f"Saved: {args.get('id')}",
                    }
                )
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

    print(f"[{run_config.label}] Found {len(findings)} issues", file=sys.stderr)
    return findings, usage


async def run_ensemble_with_verification(
    pdf_path: Path,
    output_path: Path | None = None,
    runs: list[RunConfig] | None = None,
    validator_model: str | None = None,
) -> dict:
    """Run configurable ensemble, then batch verify + dedupe.

    Strategy:
    1. Run detection passes with different models/orderings (in parallel)
    2. Group all findings by page
    3. Batch verify + dedupe each page using validator model (in parallel)
    4. Return verified findings

    Args:
        pdf_path: Path to PDF file
        output_path: Output JSON path
        runs: List of RunConfig for each detection pass (default: DEFAULT_ENSEMBLE_RUNS)
        validator_model: Model for verification (default: gemini-3-pro-preview)
    """
    runs = runs or DEFAULT_ENSEMBLE_RUNS
    validator_model = validator_model or DEFAULT_VALIDATOR_MODEL
    output_path = output_path or pdf_path.with_suffix(".ensemble.json")

    print(f"Loading {pdf_path}...", file=sys.stderr)
    print(f"Ensemble: {len(runs)} runs", file=sys.stderr)
    for r in runs:
        print(f"  - {r.label}: {r.model}", file=sys.stderr)
    print(f"Validator: {validator_model}", file=sys.stderr)

    # Convert PDF to images once
    page_images = pdf_to_images(pdf_path.read_bytes(), dpi=150)
    print(f"Pages: {len(page_images)}", file=sys.stderr)

    client = OpenRouterClient(reasoning_effort="high", timeout=900.0)
    start = time.time()

    # Phase 1: Run all detection passes IN PARALLEL
    print(
        f"\n=== Phase 1: {len(runs)} detection passes (parallel) ===", file=sys.stderr
    )

    tasks = [
        _run_single_detection_pass(run_config, run_idx, page_images, client)
        for run_idx, run_config in enumerate(runs)
    ]
    results = await asyncio.gather(*tasks)

    all_findings = []
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0}
    for findings, usage in results:
        all_findings.extend(findings)
        total_usage["prompt_tokens"] += usage["prompt_tokens"]
        total_usage["completion_tokens"] += usage["completion_tokens"]

    # Phase 2: Group findings by page
    print(f"\n=== Phase 2: Group by page ===", file=sys.stderr)
    print(f"Total raw findings: {len(all_findings)}", file=sys.stderr)

    from collections import defaultdict, Counter

    # Show per-run counts
    run_counts = Counter(f.get("_run") for f in all_findings)
    for run_label, count in sorted(run_counts.items(), key=lambda x: str(x[0])):
        print(f"  {run_label}: {count} findings", file=sys.stderr)

    page_groups = defaultdict(list)
    for f in all_findings:
        page = f.get("page", 1)
        # Normalize page to int
        if isinstance(page, str):
            # Take first page if comma-separated
            page = int(page.split(",")[0].strip())
        page_groups[int(page)].append(f)

    print(f"Grouped into {len(page_groups)} pages", file=sys.stderr)

    # Phase 3: Verify + dedupe each page batch IN PARALLEL using validator model
    print(
        f"\n=== Phase 3: Verify + dedupe ({len(page_groups)} pages) with {validator_model.split('/')[-1]} ===",
        file=sys.stderr,
    )

    verify_tasks = [
        verify_page_batch(page_num, findings, page_images, client, validator_model)
        for page_num, findings in page_groups.items()
    ]
    batch_results = await asyncio.gather(*verify_tasks)

    accepted = []
    rejected = []
    for verified, rej in batch_results:
        for v in verified:
            accepted.append(v)
            dupes = v.get("duplicate_ids", [])
            dupe_str = f" (merged {len(dupes)} dupes)" if dupes else ""
            print(f"  VERIFIED: {v.get('id')}{dupe_str}", file=sys.stderr)
        for r in rej:
            rejected.append(r)
            print(
                f"  rejected: {r.get('id')} ({r.get('rejection_reason', '')[:40]})",
                file=sys.stderr,
            )

    elapsed = time.time() - start

    # Build result
    run_labels = [r.label for r in runs]
    result = {
        "metadata": {
            "strategy": "configurable ensemble + validation",
            "ensemble_runs": run_labels,
            "validator_model": validator_model,
        },
        "checks": accepted,
        "rejected": rejected,
        "summary": {
            "total_raw": len(all_findings),
            "pages_verified": len(page_groups),
            "verified": len(accepted),
            "rejected": len(rejected),
        },
    }

    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    print(f"\n{'=' * 60}", file=sys.stderr)
    print(f"Completed in {elapsed:.1f}s", file=sys.stderr)
    print(
        f"Tokens: {total_usage['prompt_tokens']:,} in, {total_usage['completion_tokens']:,} out",
        file=sys.stderr,
    )
    print(
        f"Results: {len(all_findings)} raw → {len(page_groups)} pages → {len(accepted)} verified",
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
    parser.add_argument(
        "--tools",
        action="store_true",
        help="Use tool-based mode (incremental save_check calls)",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle page order (only with --tools)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for shuffle",
    )
    parser.add_argument(
        "--ensemble",
        action="store_true",
        help="Run configurable ensemble with verification",
    )
    parser.add_argument(
        "--validator",
        default=DEFAULT_VALIDATOR_MODEL,
        help="Model for validation phase",
    )

    args = parser.parse_args()

    if args.ensemble:
        # Use default ensemble config, but allow overriding validator
        asyncio.run(
            run_ensemble_with_verification(
                args.pdf,
                args.output,
                runs=DEFAULT_ENSEMBLE_RUNS,
                validator_model=args.validator,
            )
        )
    elif args.tools:
        asyncio.run(
            run_baseline_with_tools(
                args.pdf,
                args.output,
                args.model,
                shuffle_pages=args.shuffle,
                seed=args.seed,
            )
        )
    else:
        asyncio.run(run_baseline(args.pdf, args.output, args.model))
