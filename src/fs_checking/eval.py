"""LLM-based evaluation of model findings against ground truth."""

import asyncio
import json
import sys
from pathlib import Path

from .api import OpenRouterClient

MATCH_PROMPT = """\
You are evaluating a financial statement error detection system.

## Ground Truth Issues (GT)
These are the known errors in the document:

{gt_json}

## Model Findings
These are the errors detected by the model:

{model_json}

## Task

Match each model finding to the GT issue it corresponds to (if any).
Two items match if they describe the SAME underlying error, even if:
- Wording differs
- Page numbers differ slightly (e.g., cross-page issues)
- One is more detailed than the other
- IDs are completely different

Return a JSON object:
```json
{{
  "matches": [
    {{
      "model_id": "model finding id",
      "gt_id": "matching ground truth id",
      "confidence": "high|medium|low",
      "reasoning": "brief explanation"
    }}
  ],
  "unmatched_model": [
    {{
      "model_id": "id",
      "reasoning": "why this doesn't match any GT issue - false positive or missing from GT?"
    }}
  ],
  "unmatched_gt": [
    {{
      "gt_id": "id", 
      "reasoning": "why model missed this - hard to detect? presentation issue?"
    }}
  ]
}}
```

Be thorough - every model finding should appear in either matches or unmatched_model.
Every GT issue should appear in either matches or unmatched_gt.

Return ONLY the JSON, no other text.
"""


async def evaluate_with_llm(
    gt_path: Path,
    model_results_path: Path,
    model: str = "google/gemini-3-flash-preview",
) -> dict:
    """Use LLM to match model findings to ground truth."""

    # Load files
    with open(gt_path) as f:
        gt = json.load(f)
    with open(model_results_path) as f:
        model_results = json.load(f)

    gt_issues = gt["issues"]

    # Support both old "checks" format and new ensemble "high/medium/low" format
    if "checks" in model_results:
        model_findings = model_results["checks"]
    else:
        # Ensemble format: combine high, medium, low
        model_findings = (
            model_results.get("high", [])
            + model_results.get("medium", [])
            + model_results.get("low", [])
        )

    # Prepare simplified versions for the prompt
    gt_simplified = [
        {
            "id": i["id"],
            "page": i["page"],
            "category": i.get("category"),
            "description": i["description"],
            "severity": i.get("severity"),
        }
        for i in gt_issues
    ]

    model_simplified = [
        {
            "id": i["id"],
            "page": i["page"],
            "description": i["description"],
        }
        for i in model_findings
    ]

    prompt = MATCH_PROMPT.format(
        gt_json=json.dumps(gt_simplified, indent=2),
        model_json=json.dumps(model_simplified, indent=2),
    )

    client = OpenRouterClient(timeout=120.0)

    print(
        f"Evaluating {len(model_findings)} model findings against {len(gt_issues)} GT issues...",
        file=sys.stderr,
    )

    response = await client.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )

    content = response.get("message", {}).get("content", "")

    # Parse response
    import re

    try:
        json_match = re.search(r"\{[\s\S]*\}", content)
        if json_match:
            result = json.loads(json_match.group())
        else:
            print("ERROR: Could not parse JSON from response", file=sys.stderr)
            return {"error": "parse_failed", "raw": content}
    except json.JSONDecodeError as e:
        print(f"ERROR: JSON parse error: {e}", file=sys.stderr)
        return {"error": "parse_failed", "raw": content}

    # Calculate scores
    matches = result.get("matches", [])
    unmatched_model = result.get("unmatched_model", [])
    unmatched_gt = result.get("unmatched_gt", [])

    # Count unique GT issues matched (not match pairs — multiple model
    # findings can match the same GT issue, inflating TP if counted naively)
    unique_gt_matched = len(set(m["gt_id"] for m in matches))
    unique_model_matched = len(set(m["model_id"] for m in matches))
    total_gt = unique_gt_matched + len(unmatched_gt)
    total_model = unique_model_matched + len(unmatched_model)

    precision = unique_model_matched / total_model if total_model > 0 else 0
    recall = unique_gt_matched / total_gt if total_gt > 0 else 0
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )

    # Add scores to result
    result["scores"] = {
        "gt_matched": unique_gt_matched,
        "gt_total": total_gt,
        "model_matched": unique_model_matched,
        "model_total": total_model,
        "false_positives": len(unmatched_model),
        "false_negatives": len(unmatched_gt),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
    }

    return result


def print_eval_report(result: dict):
    """Print a formatted evaluation report."""

    if "error" in result:
        print(f"Evaluation failed: {result['error']}")
        return

    scores = result["scores"]
    matches = result.get("matches", [])
    unmatched_model = result.get("unmatched_model", [])
    unmatched_gt = result.get("unmatched_gt", [])

    print("\n" + "=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)

    print(f"\n📊 SCORES")
    print(
        f"\n   Recall:    {scores['recall']:.1%}  ({scores['gt_matched']}/{scores['gt_total']} GT errors detected)"
    )
    print(
        f"   Precision: {scores['precision']:.1%}  ({scores['model_matched']}/{scores['model_total']} model findings correct)"
    )
    print(f"   F1 Score:  {scores['f1']:.1%}")
    print(f"\n   FP: {scores['false_positives']} | FN: {scores['false_negatives']}")

    print(f"\n✅ MATCHED ({len(matches)})")
    for m in matches:
        conf = m.get("confidence", "?")
        print(f"   [{conf[0].upper()}] {m['model_id']} → {m['gt_id']}")

    if unmatched_model:
        print(f"\n❌ FALSE POSITIVES ({len(unmatched_model)})")
        for u in unmatched_model:
            reason = u.get("reasoning", "")[:50]
            print(f"   {u['model_id']}: {reason}...")

    if unmatched_gt:
        print(f"\n⚠️  MISSED ({len(unmatched_gt)})")
        for u in unmatched_gt:
            reason = u.get("reasoning", "")[:50]
            print(f"   {u['gt_id']}: {reason}...")

    print("\n" + "=" * 60)


async def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate model findings against ground truth"
    )
    parser.add_argument("gt", type=Path, help="Ground truth JSON file")
    parser.add_argument("results", type=Path, help="Model results JSON file")
    parser.add_argument(
        "-m",
        "--model",
        default="google/gemini-3-flash-preview",
        help="Model for evaluation",
    )
    parser.add_argument("-o", "--output", type=Path, help="Output JSON path")

    args = parser.parse_args()

    result = await evaluate_with_llm(args.gt, args.results, args.model)

    print_eval_report(result)

    if args.output:
        args.output.write_text(json.dumps(result, indent=2))
        print(f"\nFull results saved to: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
