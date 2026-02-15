"""Single-agent detection: one tool-call loop, incremental `log_issue()`.

This strategy is intentionally simple and self-contained:
- One model run
- One tool (`log_issue`) for incremental logging
- Optional PDF page reordering for diversity
"""

import asyncio
import base64
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from ...api import OpenRouterClient
from ...pdf_utils import get_page_count, ring_offset_pages, shuffle_pdf_pages
from ...prompts import DETECT_PROMPT_TOOL_OUTPUT

DEFAULT_MODEL = "google/gemini-3-flash-preview"


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
                    "description": "Unique snake_case identifier",
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
                },
                "page": {
                    "type": "integer",
                    "description": "Document page number from headers/footers, not PDF position",
                },
                "description": {
                    "type": "string",
                    "description": "Clear description with specific numbers",
                },
                "expected": {"type": "number"},
                "actual": {"type": "number"},
            },
            "required": ["id", "category", "page", "description"],
        },
    },
}


@dataclass
class RunConfig:
    run_id: str
    model: str
    seed: int


async def _run_tool_call_loop(
    config: RunConfig,
    pdf_bytes: bytes,
    pdf_name: str,
    client: OpenRouterClient,
    shuffle_mode: str = "none",
    stagger_max: float = 0.0,
) -> tuple[list[dict], dict]:
    from ...agent_loop import run_agent_loop

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
        {"type": "text", "text": DETECT_PROMPT_TOOL_OUTPUT},
    ]

    initial_messages = [{"role": "user", "content": user_content}]
    tools = [LOG_ISSUE_TOOL]
    findings: list[dict] = []

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


async def run_single_agent(
    pdf_path: Path,
    output_path: Path | None = None,
    model: str = DEFAULT_MODEL,
    shuffle_mode: str = "none",
    stagger_max: float = 0.0,
    timeout: float = 1800.0,
) -> dict:
    """Run a single tool-call agent loop and output raw findings."""
    output_path = output_path or pdf_path.with_suffix(".single_agent.json")

    print(f"Loading {pdf_path}...", file=sys.stderr)
    pdf_bytes = pdf_path.read_bytes()
    num_pages = get_page_count(pdf_bytes)
    print(f"Pages: {num_pages}", file=sys.stderr)
    print(
        f"Strategy: single-agent {model.split('/')[-1]} (shuffle={shuffle_mode})",
        file=sys.stderr,
    )

    client = OpenRouterClient(reasoning_effort="high", timeout=timeout)
    start = time.time()

    findings, usage = await _run_tool_call_loop(
        RunConfig(run_id="single", model=model, seed=1),
        pdf_bytes,
        pdf_path.name,
        client,
        shuffle_mode=shuffle_mode,
        stagger_max=stagger_max,
    )

    elapsed = time.time() - start
    result = {
        "metadata": {
            "strategy": "single-agent-tool-call",
            "model": model,
            "shuffle_mode": shuffle_mode,
            "stagger_max": stagger_max,
            "num_pages": num_pages,
            "elapsed_seconds": round(elapsed, 1),
            "usage": usage,
        },
        "raw_findings": findings,
        "summary": {"raw_findings": len(findings)},
    }

    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Output: {output_path}", file=sys.stderr)
    return result


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Single-agent tool-call detection")
    parser.add_argument("pdf", type=Path, help="PDF file to analyze")
    parser.add_argument("-o", "--output", type=Path, help="Output JSON path")
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL, help="Model to use")
    parser.add_argument(
        "--shuffle-mode",
        choices=["random", "ring", "none"],
        default="none",
        help="Page reorder mode",
    )
    parser.add_argument(
        "--stagger",
        type=float,
        default=0.0,
        help="Max random delay in seconds before starting (default: 0)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=1800.0,
        help="HTTP client timeout in seconds (default: 1800 = 30 min)",
    )
    args = parser.parse_args()

    await run_single_agent(
        args.pdf,
        output_path=args.output,
        model=args.model,
        shuffle_mode=args.shuffle_mode,
        stagger_max=args.stagger,
        timeout=args.timeout,
    )


if __name__ == "__main__":
    asyncio.run(main())
