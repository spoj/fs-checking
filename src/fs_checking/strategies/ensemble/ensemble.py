"""Ensemble detection: Nx parallel single-shot detectors + validate.

Pipeline:
  1. Run N parallel single-shot detection passes, each seeing a different page order.
  2. Phase 2 (default): Validate candidates with full-PDF context.
     Optional: output raw findings only.

Detectors are intentionally single-shot (no tool-call loop) to keep each run
independent and simple. Tool-call detection lives in
`fs_checking.strategies.single_agent`.
"""

from __future__ import annotations

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
from ...prompts import DETECT_PROMPT_JSON_OUTPUT

DEFAULT_DETECT_MODEL = "google/gemini-3-flash-preview"
DEFAULT_PHASE2_MODEL = "openai/gpt-5.2"
DEFAULT_NUM_RUNS = 10

# Pricing per 1M tokens (from OpenRouter, as of Jan 2025)
MODEL_PRICING = {
    "google/gemini-3-flash-preview": {"input": 0.50, "output": 3.00},
    "google/gemini-3-pro-preview": {"input": 2.00, "output": 12.00},
    "google/gemini-2.5-flash-preview": {"input": 0.15, "output": 0.60},
    "google/gemini-2.5-pro-preview": {"input": 1.25, "output": 10.00},
}


VALIDATE_PROMPT = """\
You are a financial statement auditor. The attached PDF is a set of financial statements \
that MAY CONTAIN ERRORS - that is the whole point of this audit.

An automated scan produced {num_candidates} candidate errors (listed below). \
Your job: look at the actual PDF, verify each candidate, deduplicate, and return \
only the confirmed errors.

## Candidates

{candidates_json}

## Instructions

1. For each candidate, check the relevant page/table in the PDF.
2. Verify the numbers: read every component, compute the sum yourself, compare \
   to the printed total.
3. Merge duplicates - keep one entry per distinct error.
4. Drop false positives - if the PDF shows the numbers are actually correct.
5. Err on the side of keeping. Missing real errors is worse than a false positive.

## Output

Return ONLY a JSON array of confirmed errors.

```json
[
  {{"location": "page and section/table", "description": "what is wrong, with specific numbers"}}
]
```
"""


@dataclass
class RunConfig:
    run_id: str
    model: str
    seed: int


def _calculate_cost(usage: dict, model: str) -> float:
    """Calculate cost in USD from usage dict.

    Prefers OpenRouter's authoritative `cost` field when available.
    """
    if "cost" in usage and usage["cost"] is not None:
        return float(usage["cost"])

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
    shuffle_mode: str = "random",
    stagger_max: float = 0.0,
) -> tuple[list[dict], dict]:
    """Single-shot detector: one call, parse JSON array from response."""

    if stagger_max > 0:
        await asyncio.sleep(random.uniform(0, stagger_max))

    if shuffle_mode == "random":
        pdf_to_send = shuffle_pdf_pages(pdf_bytes, config.seed)
    elif shuffle_mode == "ring":
        pdf_to_send = ring_offset_pages(pdf_bytes, config.seed)
    else:
        pdf_to_send = pdf_bytes

    pdf_b64 = base64.b64encode(pdf_to_send).decode()
    user_content: list[dict] = [
        {
            "type": "file",
            "file": {
                "filename": pdf_name,
                "file_data": f"data:application/pdf;base64,{pdf_b64}",
            },
        },
        {"type": "text", "text": DETECT_PROMPT_JSON_OUTPUT},
    ]
    messages = [{"role": "user", "content": user_content}]

    resp = await client.chat(model=config.model, messages=messages)
    content = resp.get("message", {}).get("content", "")
    usage = resp.get("usage", {})

    findings: list[dict] = []
    try:
        json_match = re.search(r"\[[\s\S]*\]", content)
        if json_match:
            parsed = json.loads(json_match.group())
            if isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, dict):
                        item["_run"] = config.run_id
                        findings.append(item)
    except json.JSONDecodeError as e:
        print(f"[{config.run_id}] JSON parse error: {e}", file=sys.stderr)

    print(
        f"[{config.run_id}] Found {len(findings)} errors (single-shot)", file=sys.stderr
    )
    return findings, usage


async def _validate_findings(
    candidates: list[dict],
    pdf_bytes: bytes,
    client: OpenRouterClient,
    model: str,
) -> tuple[list[dict], dict]:
    """PDF-aware validation: one call with PDF + candidates."""

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

    pdf_b64 = base64.b64encode(pdf_bytes).decode()
    messages: list[dict] = [
        {
            "role": "user",
            "content": [
                {
                    "type": "file",
                    "file": {
                        "filename": "financial_statements.pdf",
                        "file_data": f"data:application/pdf;base64,{pdf_b64}",
                    },
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    num_pages = get_page_count(pdf_bytes)
    print(
        f"    Sending {num_pages}-page PDF + {len(candidates)} candidates to {model}...",
        file=sys.stderr,
    )

    resp = await client.chat(model=model, messages=messages)
    usage = resp.get("usage", {})
    content = resp.get("message", {}).get("content", "")

    issues: list[dict] = []
    try:
        json_match = re.search(r"\[[\s\S]*\]", content)
        if json_match:
            parsed = json.loads(json_match.group())
            if isinstance(parsed, list):
                issues = parsed
    except json.JSONDecodeError as e:
        print(f"    WARNING: Failed to parse validator output: {e}", file=sys.stderr)

    if not issues:
        print(
            "    WARNING: No issues parsed from validator output; keeping all candidates",
            file=sys.stderr,
        )
        issues = [
            {
                "location": f"Page {c.get('page', '?')}",
                "description": c.get("description", ""),
            }
            for c in candidates
        ]

    print(
        f"    Validation complete: {len(candidates)} candidates -> {len(issues)} confirmed",
        file=sys.stderr,
    )
    return issues, usage


async def run_ensemble(
    pdf_path: Path,
    output_path: Path | None = None,
    detect_model: str = DEFAULT_DETECT_MODEL,
    phase2_model: str = DEFAULT_PHASE2_MODEL,
    num_runs: int = DEFAULT_NUM_RUNS,
    shuffle_mode: str = "random",
    stagger_max: float = 0.0,
    timeout: float = 1800.0,
    detect_only: bool = False,
) -> dict:
    """Run ensemble detection.

    If detect_only=True, outputs raw findings and skips phase 2.
    Otherwise, phase 2 validates with the PDF and deduplicates findings.
    """

    output_path = output_path or pdf_path.with_suffix(".ensemble.json")

    print(f"Loading {pdf_path}...", file=sys.stderr)
    pdf_bytes = pdf_path.read_bytes()
    num_pages = get_page_count(pdf_bytes)
    print(f"Pages: {num_pages}", file=sys.stderr)

    mode = shuffle_mode if shuffle_mode != "none" else "sequential"
    stagger_str = f", stagger {stagger_max:.0f}s" if stagger_max > 0 else ""
    phase2_label = "raw" if detect_only else "validate"
    print(f"Mode: {mode}, Detect: single-shot", file=sys.stderr)
    print(
        f"Strategy: {num_runs}x {detect_model.split('/')[-1]}{stagger_str} + {phase2_model.split('/')[-1]} {phase2_label}",
        file=sys.stderr,
    )

    client = OpenRouterClient(reasoning_effort="high", timeout=timeout)
    total_start = time.time()
    total_cost = 0.0

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
    tasks = [
        asyncio.create_task(
            _run_detection_pass(
                cfg,
                pdf_bytes,
                pdf_path.name,
                client,
                shuffle_mode=shuffle_mode,
                stagger_max=stagger_max,
            ),
            name=cfg.run_id,
        )
        for cfg in configs
    ]

    all_findings: list[dict] = []
    run_usages: list[dict] = []
    failed = 0
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for cfg, res in zip(configs, results, strict=True):
        if isinstance(res, BaseException):
            failed += 1
            print(f"  [{cfg.run_id}] FAILED: {res}", file=sys.stderr)
            continue
        findings, usage = res
        all_findings.extend(findings)
        run_cost = _calculate_cost(usage, detect_model)
        total_cost += run_cost
        run_usages.append(
            {
                "run_id": cfg.run_id,
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "reasoning_tokens": usage.get("reasoning_tokens", 0),
                "cached_tokens": usage.get("cached_tokens", 0),
                "cost_usd": round(run_cost, 4),
                "findings": len(findings),
            }
        )

    print(
        f"\nPhase 1 complete: {len(all_findings)} raw findings ({len(run_usages)} succeeded, {failed} failed)",
        file=sys.stderr,
    )

    elapsed = time.time() - total_start
    detect_usage_agg = {
        "prompt_tokens": sum(r["prompt_tokens"] for r in run_usages),
        "completion_tokens": sum(r["completion_tokens"] for r in run_usages),
        "reasoning_tokens": sum(r["reasoning_tokens"] for r in run_usages),
        "cached_tokens": sum(r["cached_tokens"] for r in run_usages),
        "cost_usd": round(sum(r["cost_usd"] for r in run_usages), 4),
    }

    if detect_only:
        result = {
            "metadata": {
                "strategy": "ensemble-detect-only",
                "detect_model": detect_model,
                "num_runs": num_runs,
                "stagger_max": stagger_max,
                "shuffle_mode": shuffle_mode,
                "num_pages": num_pages,
                "elapsed_seconds": round(elapsed, 1),
                "cost_usd": round(total_cost, 4),
                "detection_usage": detect_usage_agg,
                "run_details": run_usages,
            },
            "raw_findings": all_findings,
            "summary": {"raw_findings": len(all_findings)},
        }
        output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"Output: {output_path}", file=sys.stderr)
        return result

    print(f"\n{'=' * 60}", file=sys.stderr)
    print(f"PHASE 2: Validate ({phase2_model.split('/')[-1]})", file=sys.stderr)
    print(f"{'=' * 60}", file=sys.stderr)

    issues, usage = await _validate_findings(
        all_findings, pdf_bytes, client, model=phase2_model
    )
    phase2_cost = _calculate_cost(usage, phase2_model)
    total_cost += phase2_cost
    result = {
        "metadata": {
            "strategy": "ensemble-validate",
            "detect_model": detect_model,
            "validator_model": phase2_model,
            "num_runs": num_runs,
            "stagger_max": stagger_max,
            "shuffle_mode": shuffle_mode,
            "num_pages": num_pages,
            "elapsed_seconds": round(time.time() - total_start, 1),
            "cost_usd": round(total_cost, 4),
            "detection_usage": detect_usage_agg,
            "validation_usage": usage,
            "run_details": run_usages,
        },
        "issues": issues,
        "raw_findings": all_findings,
        "summary": {
            "raw_findings": len(all_findings),
            "issues": len(issues),
        },
    }
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Output: {output_path}", file=sys.stderr)
    return result


async def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Ensemble detection")
    parser.add_argument("pdf", type=Path, help="PDF file to analyze")
    parser.add_argument("-o", "--output", type=Path, help="Output JSON path")
    parser.add_argument(
        "--detect-model", default=DEFAULT_DETECT_MODEL, help="Detection model"
    )
    parser.add_argument(
        "--phase2-model", default=DEFAULT_PHASE2_MODEL, help="Validation model"
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
        "--stagger",
        type=float,
        default=0.0,
        help="Max random delay in seconds before each run starts (default: 0)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=1800.0,
        help="HTTP client timeout in seconds (default: 1800 = 30 min)",
    )
    parser.add_argument(
        "--raw", action="store_true", help="Output raw detector findings only"
    )
    args = parser.parse_args()

    await run_ensemble(
        pdf_path=args.pdf,
        output_path=args.output,
        detect_model=args.detect_model,
        phase2_model=args.phase2_model,
        num_runs=args.runs,
        shuffle_mode=args.shuffle_mode,
        stagger_max=args.stagger,
        timeout=args.timeout,
        detect_only=args.raw,
    )


if __name__ == "__main__":
    asyncio.run(main())
