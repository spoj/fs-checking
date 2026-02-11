#!/usr/bin/env python3
"""Annotate a PDF with detection findings, color-coded by GT/TP/FP.

Sends the full PDF + all findings to Gemini Pro in one call, gets bboxes back,
then draws three layers of colored annotations on the PDF:

  - Blue (dashed):  Known injected errors (ground truth locations)
  - Green (solid):  Model detected, eval matched (true positive)
  - Red (solid):    Model detected, eval unmatched (false positive)

Usage:
    # Experiment (with eval + GT):
    uv run python scripts/annotate_findings.py \
      samples/ar2019/unrender/ar2019_fs.injected.pdf \
      samples/ar2019/unrender/ar2019_fs.injected.result.json \
      --eval samples/ar2019/unrender/ar2019_fs.injected.eval.json \
      --gt samples/ar2019/unrender/ar2019_fs.injected.ground_truth.json \
      -o samples/ar2019/unrender/ar2019_fs.injected.annotated.pdf

    # Control (no eval, all findings colored blue):
    uv run python scripts/annotate_findings.py \
      samples/ar2019/unrender/ar2019_fs.unrendered.pdf \
      samples/ar2019/unrender/ar2019_fs.control.json \
      -o samples/ar2019/unrender/ar2019_fs.control.annotated.pdf
"""

import asyncio
import base64
import json
import sys
import time
from pathlib import Path

import fitz

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from fs_checking.api import OpenRouterClient

BBOX_MODEL = "google/gemini-3-pro-preview"

# Colors (R, G, B) in 0-1 range for PyMuPDF
COLOR_GT = (0.15, 0.3, 0.9)  # Blue — ground truth (known injection)
COLOR_TP = (0.1, 0.7, 0.2)  # Green — true positive (model found + eval confirmed)
COLOR_FP = (0.9, 0.15, 0.15)  # Red — false positive (model found, not in GT)
COLOR_DEFAULT = (0.5, 0.5, 0.5)  # Grey — unclassified


async def get_all_bboxes(
    client: OpenRouterClient,
    pdf_bytes: bytes,
    findings: list[dict],
    label: str = "",
) -> list[dict]:
    """Send full PDF + all findings to Gemini Pro, get bboxes back."""

    issue_list = []
    for f in findings:
        issue_list.append(
            {
                "issue_id": f["id"],
                "reported_page": f.get("page", 0),
                "description": f["description"][:400],
            }
        )

    issues_json = json.dumps(issue_list, indent=2)

    prompt = f"""You are given a PDF document and a list of reported issues found in it.

For each issue, locate the most relevant cell, number, or text region on the PDF page and return its bounding box.

## Issues

```json
{issues_json}
```

## Task

For each issue, return a JSON object:
```
{{"issue_id": "...", "page": <1-based page number>, "bbox": [ymin, xmin, ymax, xmax]}}
```

Where bbox coordinates are integers in range 0-1000, representing normalized positions:
- 0 = top/left edge of the page
- 1000 = bottom/right edge of the page

Draw the box tightly around the specific number or table row that is wrong.
If the reported_page seems incorrect, use the actual page where you find the relevant content.

Return a JSON array of all results. Return ONLY the JSON array, no other text.
"""

    pdf_b64 = base64.b64encode(pdf_bytes).decode()
    messages = [
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
                {"type": "text", "text": prompt},
            ],
        }
    ]

    tag = f" [{label}]" if label else ""
    print(
        f"Sending {len(findings)} findings + PDF to {BBOX_MODEL}{tag}...",
        file=sys.stderr,
    )
    t0 = time.time()
    resp = await client.chat(model=BBOX_MODEL, messages=messages)
    elapsed = time.time() - t0

    usage = resp.get("usage", {})
    cost = usage.get("cost", 0)
    print(f"Response in {elapsed:.0f}s, cost: ${cost:.4f}", file=sys.stderr)

    text = resp.get("message", {}).get("content", "")
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

    try:
        results = json.loads(text)
        if isinstance(results, list):
            print(f"Got {len(results)} bbox results", file=sys.stderr)
            return results
    except json.JSONDecodeError as e:
        print(f"JSON parse error: {e}", file=sys.stderr)
        print(f"Response text (first 500 chars): {text[:500]}", file=sys.stderr)

    return []


def draw_bbox(
    doc: fitz.Document,
    page_num: int,
    bbox: list,
    color: tuple,
    fid: str,
    desc: str,
    dashes: str | None = None,
):
    """Draw a colored rectangle annotation on a page."""
    num_pages = len(doc)
    if not (1 <= page_num <= num_pages) or len(bbox) != 4:
        return False

    ymin, xmin, ymax, xmax = bbox
    ymin = max(0, min(1000, ymin))
    xmin = max(0, min(1000, xmin))
    ymax = max(0, min(1000, ymax))
    xmax = max(0, min(1000, xmax))

    if ymin >= ymax or xmin >= xmax:
        return False

    page = doc[page_num - 1]
    rect = page.rect
    x1 = rect.x0 + (xmin / 1000) * rect.width
    y1 = rect.y0 + (ymin / 1000) * rect.height
    x2 = rect.x0 + (xmax / 1000) * rect.width
    y2 = rect.y0 + (ymax / 1000) * rect.height
    annot_rect = fitz.Rect(x1, y1, x2, y2)

    annot = page.add_rect_annot(annot_rect)
    annot.set_colors(stroke=color)
    border = {"width": 2}
    if dashes:
        border["dashes"] = [3, 3]
    annot.set_border(**border)
    annot.set_opacity(0.8)
    annot.set_info(title=fid, content=desc[:500])
    annot.update()
    return True


async def annotate_pdf(
    pdf_path: Path,
    results_path: Path,
    eval_path: Path | None,
    gt_path: Path | None,
    output_path: Path,
):
    """Annotate PDF with three layers: GT (blue), TP (green), FP (red)."""

    pdf_bytes = pdf_path.read_bytes()
    doc = fitz.open(str(pdf_path))
    num_pages = len(doc)
    print(f"PDF: {pdf_path} ({num_pages} pages)", file=sys.stderr)

    # Load detection results
    with open(results_path) as f:
        results = json.load(f)
    all_findings = (
        results.get("high", []) + results.get("medium", []) + results.get("low", [])
    )
    print(f"Detection findings: {len(all_findings)}", file=sys.stderr)

    # Load eval (if available)
    tp_ids: set[str] = set()
    fp_ids: set[str] = set()
    if eval_path and eval_path.exists():
        with open(eval_path) as f:
            ev = json.load(f)
        tp_ids = {m["model_id"] for m in ev.get("matches", [])}
        fp_ids = {u["model_id"] for u in ev.get("unmatched_model", [])}
        print(f"Eval: {len(tp_ids)} TP, {len(fp_ids)} FP", file=sys.stderr)

    # Load ground truth (if available)
    gt_issues: list[dict] = []
    if gt_path and gt_path.exists():
        with open(gt_path) as f:
            gt = json.load(f)
        gt_issues = gt.get("issues", [])
        # Normalize GT into same format as findings for bbox lookup
        for issue in gt_issues:
            if "id" not in issue:
                issue["id"] = issue.get("_id", f"gt_{gt_issues.index(issue)}")
            if "description" not in issue:
                issue["description"] = str(issue)
        print(f"Ground truth: {len(gt_issues)} known errors", file=sys.stderr)

    client = OpenRouterClient(timeout=600.0)

    # Collect all items that need bboxes — combine into one call
    all_items: list[dict] = []

    # Add GT issues (prefixed to avoid ID collision)
    for issue in gt_issues:
        all_items.append(
            {
                "id": f"GT_{issue['id']}",
                "page": issue.get("page", 0),
                "description": issue["description"][:400],
                "_layer": "gt",
                "_orig_id": issue["id"],
            }
        )

    # Add detection findings
    for f in all_findings:
        layer = "default"
        if eval_path and eval_path.exists():
            if f["id"] in tp_ids:
                layer = "tp"
            elif f["id"] in fp_ids:
                layer = "fp"
        all_items.append(
            {
                "id": f["id"],
                "page": f.get("page", 0),
                "description": f["description"][:400],
                "_layer": layer,
            }
        )

    print(f"\nTotal items to locate: {len(all_items)}", file=sys.stderr)

    # Get all bboxes in one call
    bbox_results = await get_all_bboxes(client, pdf_bytes, all_items, label="all")

    # Build lookup
    bbox_by_id: dict[str, dict] = {}
    for item in bbox_results:
        bbox_by_id[item.get("issue_id", "")] = item

    # Draw annotations in order: GT first (bottom), then TP, then FP (top)
    stats = {"gt": 0, "tp": 0, "fp": 0, "default": 0, "skipped": 0}

    # Layer 1: GT (blue, dashed)
    for item in all_items:
        if item["_layer"] != "gt":
            continue
        bbox_item = bbox_by_id.get(item["id"])
        if not bbox_item:
            stats["skipped"] += 1
            continue
        ok = draw_bbox(
            doc,
            bbox_item["page"],
            bbox_item["bbox"],
            COLOR_GT,
            item.get("_orig_id", item["id"]),
            f"[INJECTED] {item['description']}",
            dashes="yes",
        )
        if ok:
            stats["gt"] += 1
        else:
            stats["skipped"] += 1

    # Layer 2: TP (green, solid)
    for item in all_items:
        if item["_layer"] != "tp":
            continue
        bbox_item = bbox_by_id.get(item["id"])
        if not bbox_item:
            stats["skipped"] += 1
            continue
        ok = draw_bbox(
            doc,
            bbox_item["page"],
            bbox_item["bbox"],
            COLOR_TP,
            item["id"],
            f"[TP] {item['description']}",
        )
        if ok:
            stats["tp"] += 1
        else:
            stats["skipped"] += 1

    # Layer 3: FP (red, solid)
    for item in all_items:
        if item["_layer"] != "fp":
            continue
        bbox_item = bbox_by_id.get(item["id"])
        if not bbox_item:
            stats["skipped"] += 1
            continue
        ok = draw_bbox(
            doc,
            bbox_item["page"],
            bbox_item["bbox"],
            COLOR_FP,
            item["id"],
            f"[FP] {item['description']}",
        )
        if ok:
            stats["fp"] += 1
        else:
            stats["skipped"] += 1

    # Layer 4: Default/unclassified (grey, for control runs)
    for item in all_items:
        if item["_layer"] != "default":
            continue
        bbox_item = bbox_by_id.get(item["id"])
        if not bbox_item:
            stats["skipped"] += 1
            continue
        ok = draw_bbox(
            doc,
            bbox_item["page"],
            bbox_item["bbox"],
            COLOR_DEFAULT,
            item["id"],
            item["description"],
        )
        if ok:
            stats["default"] += 1
        else:
            stats["skipped"] += 1

    print(f"\nAnnotations drawn:", file=sys.stderr)
    print(f"  Blue (GT):     {stats['gt']}", file=sys.stderr)
    print(f"  Green (TP):    {stats['tp']}", file=sys.stderr)
    print(f"  Red (FP):      {stats['fp']}", file=sys.stderr)
    print(f"  Grey (default):{stats['default']}", file=sys.stderr)
    print(f"  Skipped:       {stats['skipped']}", file=sys.stderr)

    doc.save(str(output_path))
    doc.close()
    print(f"\nOutput: {output_path}", file=sys.stderr)


async def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Annotate PDF with detection findings (bbox via Gemini Pro)"
    )
    parser.add_argument("pdf", type=Path, help="Input PDF to annotate")
    parser.add_argument("results", type=Path, help="Detection results JSON")
    parser.add_argument(
        "--eval",
        type=Path,
        default=None,
        help="Eval JSON for TP/FP classification",
    )
    parser.add_argument(
        "--gt",
        type=Path,
        default=None,
        help="Ground truth JSON for injected error locations (blue boxes)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output annotated PDF path",
    )
    args = parser.parse_args()

    await annotate_pdf(args.pdf, args.results, args.eval, args.gt, args.output)


if __name__ == "__main__":
    asyncio.run(main())
