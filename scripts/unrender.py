#!/usr/bin/env python3
"""Unrender: PDF -> GFM Markdown via iterative LLM refinement.

Part of the eval pipeline:
  1. unrender.py  — vetted FS PDF -> faithful GFM markdown reproduction
  2. (future) dope the markdown with known errors
  3. (future) render doped markdown -> PDF
  4. (future) run detection model, eval against known errors

Architecture — three nested loops:

  OUTER (window advance):
    Slides a window of `window_size` pages across the original PDF with
    `overlap` pages of overlap. When a window is done, records the current
    rendered page count as the anchor for the next window.

  MIDDLE (multi-pass convergence):
    For the current window, run multiple editing passes until markdown
    stabilises. Each pass provides the model with:
      - original_window.pdf  — the current window slice from the source PDF
      - rendered_tail.pdf    — the tail of the rendered PDF from
                               (anchor - overlap) onward, so the model sees
                               continuity with what it already built
    The model is NOT force-injected with the markdown buffer — it uses tools
    to read/edit it.

  INNER (agentic tool-call loop):
    The model calls read_range / replace_range / insert_lines / delete_range
    to build and refine markdown in a single shared buffer. Tool responses are
    minimal ("OK (N lines)"). No rendering during edits.

Resumability:
  Intermediate markdown is saved to <pdf>.unrender/body.md after each window.
  Re-running with the same PDF picks up where it left off.

Usage:
    uv run python scripts/unrender.py samples/doc.pdf
    uv run python scripts/unrender.py samples/doc.pdf --window 10 --overlap 2
"""

import asyncio
import base64
import json
import re
import sys
import tempfile
import time
from pathlib import Path

import fitz
import markdown
from playwright.async_api import async_playwright

# Allow importing from src/
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from fs_checking.api import OpenRouterClient

DEFAULT_MODEL = "google/gemini-3-flash-preview"
DEFAULT_WINDOW = 10
DEFAULT_OVERLAP = 2

# ═══════════════════════════════════════════════════════════════
# Markdown → HTML rendering
# ═══════════════════════════════════════════════════════════════

DEFAULT_CSS = """\
@page { size: A4; margin: 12mm; }

html, body {
  padding: 0;
  margin: 0;
  font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial;
  font-size: 10.5pt;
  color: #111;
}

h1 { font-size: 18pt; font-weight: 700; margin: 18pt 0 8pt 0; }
h2 { font-size: 13pt; font-weight: 700; margin: 14pt 0 6pt 0; }
h3 { font-size: 11pt; font-weight: 700; margin: 10pt 0 4pt 0; }
h4 { font-size: 10.5pt; font-weight: 700; margin: 8pt 0 4pt 0; }

table { border-collapse: collapse; width: 100%; margin: 6pt 0; }
th, td { border: 1px solid #222; padding: 4px 6px; vertical-align: top; }
th { background: #f3f4f6; font-weight: 700; }
thead { display: table-header-group; }

blockquote { margin: 4pt 0 4pt 12pt; padding-left: 8pt; border-left: 2px solid #999; color: #333; font-size: 9pt; }
"""

_md = markdown.Markdown(extensions=["tables"])


def md_to_html(md_body: str) -> str:
    """Convert GFM markdown to a full HTML document with CSS for PDF rendering."""
    _md.reset()
    body_html = _md.convert(md_body)
    return (
        "<!doctype html>\n<html>\n<head>\n"
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<style>\n{DEFAULT_CSS}</style>\n"
        "</head>\n<body>\n"
        f"{body_html}\n"
        "</body>\n</html>\n"
    )


# ═══════════════════════════════════════════════════════════════
# PDF utilities
# ═══════════════════════════════════════════════════════════════


async def render_html_to_pdf(
    html: str, pdf_path: Path, timeout_ms: int = 30000
) -> tuple[bool, str]:
    """Render HTML string to PDF. Returns (ok, error_message)."""
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            await page.set_content(html, wait_until="load", timeout=timeout_ms)
            await page.emulate_media(media="print")
            await page.pdf(
                path=str(pdf_path),
                print_background=True,
                format="A4",
                margin={
                    "top": "12mm",
                    "bottom": "12mm",
                    "left": "12mm",
                    "right": "12mm",
                },
            )
            await browser.close()
        return True, ""
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def _pdf_page_count(pdf_bytes: bytes) -> int:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    n = len(doc)
    doc.close()
    return n


def _extract_pdf_pages(pdf_bytes: bytes, from_page: int, to_page: int) -> bytes:
    """Extract pages [from_page, to_page] (0-indexed, inclusive) into a new PDF."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    out = fitz.open()
    for i in range(from_page, min(to_page + 1, len(doc))):
        out.insert_pdf(doc, from_page=i, to_page=i)
    result = out.tobytes()
    out.close()
    doc.close()
    return result


def _extract_pdf_tail(pdf_bytes: bytes, from_page: int) -> bytes:
    """Extract pages from from_page (0-indexed) to end."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    n = len(doc)
    if from_page >= n:
        from_page = max(0, n - 1)
    out = fitz.open()
    for i in range(from_page, n):
        out.insert_pdf(doc, from_page=i, to_page=i)
    result = out.tobytes()
    out.close()
    doc.close()
    return result


# ═══════════════════════════════════════════════════════════════
# Line buffer
# ═══════════════════════════════════════════════════════════════


class LineBuffer:
    """Line-based text buffer (1-indexed). Format-agnostic."""

    def __init__(self, content: str = ""):
        self._lines: list[str] = content.split("\n") if content else []

    @property
    def content(self) -> str:
        return "\n".join(self._lines)

    @property
    def line_count(self) -> int:
        return len(self._lines)

    def read_range(self, from_line: int, to_line: int) -> str:
        f = max(1, from_line)
        t = min(self.line_count, to_line)
        if f > t:
            return "(empty)"
        numbered = [
            f"{i:4d}| {l}" for i, l in enumerate(self._lines[f - 1 : t], start=f)
        ]
        return "\n".join(numbered)

    def replace_range(self, from_line: int, to_line: int, content: str) -> str:
        f = max(1, from_line)
        t = min(self.line_count, to_line)
        if f > t and self.line_count > 0:
            return f"Error: invalid range {from_line}-{to_line}"
        new = content.split("\n")
        self._lines[f - 1 : t] = new
        return f"OK ({self.line_count} lines)"

    def insert_lines(self, after_line: int, content: str) -> str:
        pos = max(0, min(self.line_count, after_line))
        new = content.split("\n")
        self._lines[pos:pos] = new
        return f"OK ({self.line_count} lines)"

    def delete_range(self, from_line: int, to_line: int) -> str:
        f = max(1, from_line)
        t = min(self.line_count, to_line)
        if f > t:
            return f"Error: invalid range {from_line}-{to_line}"
        del self._lines[f - 1 : t]
        return f"OK ({self.line_count} lines)"


# ═══════════════════════════════════════════════════════════════
# Tool definitions
# ═══════════════════════════════════════════════════════════════

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_range",
            "description": "Read lines from markdown source (numbered). Omit params for full doc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "from_line": {
                        "type": "integer",
                        "description": "Start line (1-based)",
                    },
                    "to_line": {
                        "type": "integer",
                        "description": "End line (1-based, inclusive)",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "replace_range",
            "description": "Replace lines from_line..to_line with new content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "from_line": {"type": "integer"},
                    "to_line": {"type": "integer"},
                    "content": {
                        "type": "string",
                        "description": "New markdown content",
                    },
                },
                "required": ["from_line", "to_line", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "insert_lines",
            "description": "Insert content after a line (0 = prepend).",
            "parameters": {
                "type": "object",
                "properties": {
                    "after_line": {"type": "integer"},
                    "content": {"type": "string"},
                },
                "required": ["after_line", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_range",
            "description": "Delete lines from_line..to_line (inclusive).",
            "parameters": {
                "type": "object",
                "properties": {
                    "from_line": {"type": "integer"},
                    "to_line": {"type": "integer"},
                },
                "required": ["from_line", "to_line"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_info",
            "description": "Get current line count.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


# ═══════════════════════════════════════════════════════════════
# Tool executor
# ═══════════════════════════════════════════════════════════════


def execute_tool(name: str, args: dict, buf: LineBuffer) -> str:
    """Execute tool, return minimal status string."""
    if name == "read_range":
        return buf.read_range(
            args.get("from_line", 1),
            args.get("to_line", buf.line_count),
        )
    elif name == "replace_range":
        return buf.replace_range(args["from_line"], args["to_line"], args["content"])
    elif name == "insert_lines":
        return buf.insert_lines(args["after_line"], args["content"])
    elif name == "delete_range":
        return buf.delete_range(args["from_line"], args["to_line"])
    elif name == "get_info":
        return f"{buf.line_count} lines"
    return f"Unknown tool: {name}"


# ═══════════════════════════════════════════════════════════════
# Prompt
# ═══════════════════════════════════════════════════════════════

SYSTEM_PROMPT = r"""You are converting a PDF document into GFM (GitHub-Flavored Markdown), working through it section by section.

## Format

You edit a plain-text markdown buffer. The harness renders it to HTML+PDF.
Use standard GFM syntax:

- `# Title`, `## Section`, `### Subsection`, `#### Sub-subsection`
- `**bold**`, `*italic*`
- GFM pipe tables with alignment:
  ```
  | Label | Note | 2019 US$'000 | 2018 US$'000 |
  |-------|-----:|-------------:|-------------:|
  | Revenue | 3 | 11,413,312 | 12,700,744 |
  ```
- `> blockquote` for notes, footnotes, fine print
- Bullet lists (`- item`) and numbered lists (`1. item`)
- `---` for horizontal rules / section separators

## Table conventions

- Use `---:` for right-aligned columns (numbers, amounts)
- For section sub-headers within a table (e.g. "Continuing Operations"),
  use a bold row: `| **Continuing Operations** | | | |`
- For indented sub-items, prefix with spaces: `|   Sub-item | ... |`
- For footnote references, use `^1`, `^2` etc.
- Keep tables compact: one row per line, no extra whitespace
- IMPORTANT: every table MUST have a header row and a separator row

## Tools

- read_range(from_line, to_line) — read numbered lines (omit for full doc)
- replace_range(from_line, to_line, content) — replace range
- insert_lines(after_line, content) — insert after line (0 = prepend)
- delete_range(from_line, to_line) — delete range
- get_info() — line count

Mutations return just "OK (N lines)" or error. No rendering during edits.

## Attached files

- **original_window.pdf** — the section of the source PDF to reproduce NOW.
  This is a window of pages from a larger document.
- **rendered_tail.pdf** — the tail of what you've built so far, rendered to
  PDF (for continuity context). May be absent on the first window.

## Workflow

1. Study original_window.pdf — note structure, pages, tables, headers.
2. Use read_range to see the end of the existing buffer (if non-empty)
   so you know where to continue from.
3. APPEND new content for this window using insert_lines at the end.
   Do NOT rewrite earlier content unless fixing an error at the boundary.
4. Content completeness is the priority: ALL text, ALL tables, ALL numbers.
5. Layout should be readable, NOT pixel-perfect.
6. When done editing, respond with a short summary (no tool call).

## Rules

- Pure GFM markdown only — no HTML tags, no images, no external assets.
- Completeness > visual fidelity.
- You are building a CONTINUOUS document — each window extends the buffer.
- SKIP per-page headers and footers (e.g. repeated company name, "Annual Report 2019",
  page numbers like "143", "144"). These are PDF artifacts, not document content.
  Only include headers/footers once if they contain unique content (e.g. the document
  title on the first page).
"""


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════


def _file_part(filename: str, pdf_bytes: bytes) -> dict:
    b64 = base64.b64encode(pdf_bytes).decode()
    return {
        "type": "file",
        "file": {
            "filename": filename,
            "file_data": f"data:application/pdf;base64,{b64}",
        },
    }


def _log(step: int, name: str, args: dict, result: str):
    if name == "read_range":
        desc = f"read({args.get('from_line', 1)}..{args.get('to_line', 'end')})"
    elif name == "replace_range":
        desc = f"replace({args.get('from_line')}..{args.get('to_line')}, {len(args.get('content', ''))} chars)"
    elif name == "insert_lines":
        desc = f"insert(after={args.get('after_line')}, {len(args.get('content', ''))} chars)"
    elif name == "delete_range":
        desc = f"delete({args.get('from_line')}..{args.get('to_line')})"
    else:
        desc = name
    first = result.split("\n")[0][:100]
    print(f"      [{step}] {desc} -> {first}", file=sys.stderr)


# ═══════════════════════════════════════════════════════════════
# INNER LOOP: agentic tool-call editing (one pass, no rendering)
# ═══════════════════════════════════════════════════════════════


async def run_inner_loop(
    window_pdf: bytes,
    tail_pdf: bytes | None,
    buf: LineBuffer,
    client: OpenRouterClient,
    model: str,
    window_label: str,
    max_iter: int = 30,
) -> dict:
    """One editing pass. Modifies buf in-place. Returns usage dict."""

    parts: list[dict] = [_file_part("original_window.pdf", window_pdf)]
    if tail_pdf:
        parts.append(_file_part("rendered_tail.pdf", tail_pdf))
    parts.append(
        {
            "type": "text",
            "text": (
                f"Current markdown buffer: {buf.line_count} lines. "
                f"Processing window: {window_label}."
                + (
                    " Buffer is empty — start building."
                    if buf.line_count == 0
                    else " Use read_range to see the end of the buffer, then append new content."
                )
            ),
        }
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": parts},
    ]

    usage = {"prompt_tokens": 0, "completion_tokens": 0, "cost": 0.0}

    for i in range(max_iter):
        resp = await client.chat(model=model, messages=messages, tools=TOOLS)
        msg = resp.get("message", {})
        u = resp.get("usage", {})
        usage["prompt_tokens"] += u.get("prompt_tokens", 0)
        usage["completion_tokens"] += u.get("completion_tokens", 0)
        if u.get("cost") is not None:
            usage["cost"] += float(u["cost"])

        tool_calls = msg.get("tool_calls", [])
        messages.append(msg)

        if not tool_calls:
            text = (msg.get("content", "") or "").strip().split("\n")[0][:80]
            print(f"      [{i + 1}] Done: {text}", file=sys.stderr)
            break

        for tc in tool_calls:
            fn = tc.get("function", {})
            fn_name = fn.get("name", "")
            call_id = tc.get("id", "")
            try:
                args = json.loads(fn.get("arguments", "{}"))
            except json.JSONDecodeError:
                args = {}

            result = execute_tool(fn_name, args, buf)
            messages.append(
                {"role": "tool", "tool_call_id": call_id, "content": result}
            )
            _log(i + 1, fn_name, args, result)

    return usage


# ═══════════════════════════════════════════════════════════════
# MIDDLE LOOP: multi-pass convergence for one window
# ═══════════════════════════════════════════════════════════════


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("\r\n", "\n")).strip()


async def run_window(
    window_pdf: bytes,
    anchor_page: int,
    overlap: int,
    buf: LineBuffer,
    client: OpenRouterClient,
    model: str,
    window_label: str,
    save_path: Path | None = None,
    max_passes: int = 3,
    stable_count: int = 1,
) -> tuple[dict, int]:
    """Multi-pass refinement for one window.

    Args:
        window_pdf: slice of original PDF for this window
        anchor_page: rendered page count when this window started (0-indexed).
            Used to extract the tail of the rendered PDF for continuity.
        overlap: number of overlap pages
        buf: shared markdown buffer (modified in-place)
        client, model: API
        window_label: for logging
        save_path: if set, persist buffer to this file after each pass
        max_passes: max passes for this window
        stable_count: consecutive stable passes to declare done

    Returns:
        (usage_dict, new_rendered_page_count)
    """
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "cost": 0.0}
    rendered_pdf_bytes: bytes | None = None
    rendered_pages = 0
    stable = 0

    for pass_num in range(1, max_passes + 1):
        prev = buf.content
        print(f"    Pass {pass_num}...", file=sys.stderr)

        # Build the rendered tail PDF for continuity
        tail_pdf: bytes | None = None
        if rendered_pdf_bytes and anchor_page > 0:
            tail_start = max(0, anchor_page - overlap)
            tail_pdf = _extract_pdf_tail(rendered_pdf_bytes, tail_start)

        pass_usage = await run_inner_loop(
            window_pdf, tail_pdf, buf, client, model, window_label
        )

        # Persist after each pass so progress survives interruptions
        if save_path:
            save_path.write_text(buf.content, encoding="utf-8")
        for k in total_usage:
            total_usage[k] += pass_usage.get(k, 0.0)

        # Render full buffer once after the pass
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td) / "render.pdf"
            ok, err = await render_html_to_pdf(md_to_html(buf.content), tmp)
            if ok:
                rendered_pdf_bytes = tmp.read_bytes()
                rendered_pages = _pdf_page_count(rendered_pdf_bytes)
            else:
                rendered_pdf_bytes = None
                rendered_pages = 0
                stable = 0
                print(f"    Pass {pass_num} render failed: {err[:80]}", file=sys.stderr)
                continue

        # Convergence
        curr = buf.content
        delta = len(curr) - len(prev)
        if _normalize(prev) == _normalize(curr):
            stable += 1
            print(
                f"    Pass {pass_num} stable ({delta:+d} chars, {stable}/{stable_count}, {rendered_pages}p)",
                file=sys.stderr,
            )
            if stable >= stable_count:
                break
        else:
            stable = 0
            print(
                f"    Pass {pass_num} changed ({delta:+d} chars, {buf.line_count} lines, {rendered_pages}p)",
                file=sys.stderr,
            )

    return total_usage, rendered_pages


# ═══════════════════════════════════════════════════════════════
# OUTER LOOP: sliding window over original PDF
# ═══════════════════════════════════════════════════════════════


async def run_unrender(
    pdf_path: Path,
    output_path: Path | None = None,
    model: str = DEFAULT_MODEL,
    window_size: int = DEFAULT_WINDOW,
    overlap: int = DEFAULT_OVERLAP,
    max_passes: int = 3,
    stable_count: int = 1,
    timeout: float = 600.0,
) -> dict:
    """PDF -> Markdown via sliding-window iterative LLM refinement.

    Work directory: <pdf>.unrender/ — contains body.md.
    Re-running loads existing markdown and re-runs all windows; the model
    will see the buffer is already populated and converge quickly (stable
    on pass 1) for already-completed sections.
    """
    output_path = output_path or pdf_path.with_suffix(".md")
    original_pdf = pdf_path.read_bytes()
    total_pages = _pdf_page_count(original_pdf)

    # Work directory
    work_dir = pdf_path.with_suffix(".unrender")
    work_dir.mkdir(exist_ok=True)
    body_md_path = work_dir / "body.md"

    # Plan windows
    step = max(1, window_size - overlap)
    windows: list[tuple[int, int]] = []
    start = 0
    while start < total_pages:
        end = min(start + window_size - 1, total_pages - 1)
        windows.append((start, end))
        start += step
        if end == total_pages - 1:
            break

    print(f"Loading {pdf_path} ({total_pages} pages)", file=sys.stderr)
    print(f"Model: {model}", file=sys.stderr)
    print(
        f"Windows: {len(windows)} ({window_size}p, {overlap}p overlap)", file=sys.stderr
    )
    print(f"Work dir: {work_dir}", file=sys.stderr)

    # Resume: load existing markdown if present
    if body_md_path.exists():
        buf = LineBuffer(body_md_path.read_text(encoding="utf-8"))
        print(f"Loaded existing buffer: {buf.line_count} lines", file=sys.stderr)
    else:
        buf = LineBuffer()
    anchor_page = 0

    client = OpenRouterClient(reasoning_effort="high", timeout=timeout)
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "cost": 0.0}
    start_time = time.time()

    for win_idx in range(len(windows)):
        from_pg, to_pg = windows[win_idx]
        label = f"pages {from_pg + 1}-{to_pg + 1}"

        print(f"\n{'─' * 50}", file=sys.stderr)
        print(
            f"Window {win_idx + 1}/{len(windows)}: {label} ({to_pg - from_pg + 1}p)",
            file=sys.stderr,
        )

        window_pdf = _extract_pdf_pages(original_pdf, from_pg, to_pg)
        win_usage, rendered_pages = await run_window(
            window_pdf,
            anchor_page,
            overlap,
            buf,
            client,
            model,
            label,
            save_path=body_md_path,
            max_passes=max_passes,
            stable_count=stable_count,
        )
        for k in total_usage:
            total_usage[k] += win_usage.get(k, 0.0)

        # Update anchor for next window
        anchor_page = rendered_pages

        cost_so_far = total_usage["cost"]
        print(
            f"  -> {buf.line_count} lines, {rendered_pages}p rendered, ${cost_so_far:.4f} so far",
            file=sys.stderr,
        )

    # Final output — save markdown and rendered HTML
    output_path.write_text(buf.content, encoding="utf-8")
    full_html = md_to_html(buf.content)

    # Render final PDF
    final_pdf_path = pdf_path.with_suffix(".unrendered.pdf")
    rendered_pages = 0
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / "final.pdf"
        ok, err = await render_html_to_pdf(full_html, tmp)
        if ok:
            final_pdf_path.write_bytes(tmp.read_bytes())
            rendered_pages = _pdf_page_count(tmp.read_bytes())
            print(
                f"\nRendered PDF: {final_pdf_path} ({rendered_pages} pages)",
                file=sys.stderr,
            )
        else:
            print(f"\nFinal render failed: {err}", file=sys.stderr)

    elapsed = time.time() - start_time

    print(f"\n{'=' * 60}", file=sys.stderr)
    print(f"Done in {elapsed:.1f}s ({len(windows)} windows)", file=sys.stderr)
    print(
        f"Tokens: {total_usage['prompt_tokens']:,} in, {total_usage['completion_tokens']:,} out",
        file=sys.stderr,
    )
    if total_usage["cost"] > 0:
        print(f"Cost: ${total_usage['cost']:.4f}", file=sys.stderr)
    print(f"Markdown: {output_path} ({buf.line_count} lines)", file=sys.stderr)

    return {
        "md_path": str(output_path),
        "pdf_path": str(final_pdf_path),
        "work_dir": str(work_dir),
        "metadata": {
            "model": model,
            "original_pages": total_pages,
            "rendered_pages": rendered_pages,
            "md_lines": buf.line_count,
            "windows": len(windows),
            "window_size": window_size,
            "overlap": overlap,
            "elapsed_seconds": round(elapsed, 1),
            "cost_usd": round(total_usage["cost"], 4),
        },
        "usage": total_usage,
    }


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════


async def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="PDF -> GFM Markdown via iterative LLM refinement"
    )
    parser.add_argument("pdf", type=Path, help="Input PDF")
    parser.add_argument("-o", "--output", type=Path, help="Output markdown path")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--window",
        type=int,
        default=DEFAULT_WINDOW,
        help=f"Pages per window (default: {DEFAULT_WINDOW})",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=DEFAULT_OVERLAP,
        help=f"Overlap pages between windows (default: {DEFAULT_OVERLAP})",
    )
    parser.add_argument(
        "--max-passes", type=int, default=3, help="Max passes per window"
    )
    parser.add_argument(
        "--stable", type=int, default=1, help="Stable passes to converge per window"
    )
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args()

    if not args.pdf.exists():
        print(f"Error: {args.pdf} not found", file=sys.stderr)
        raise SystemExit(1)

    await run_unrender(
        args.pdf,
        args.output,
        model=args.model,
        window_size=args.window,
        overlap=args.overlap,
        max_passes=args.max_passes,
        stable_count=args.stable,
        timeout=args.timeout,
    )


if __name__ == "__main__":
    asyncio.run(main())
