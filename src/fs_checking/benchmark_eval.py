"""Evaluate benchmark runs against ground truth using LLM matching.

Runs eval.py against each benchmark result to get recall/precision per run,
then aggregates to understand model variance.

Usage:
    uv run python -m fs_checking.benchmark_eval benchmarks/ samples/Written_test_Case.ground_truth.json
"""

import asyncio
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .api import OpenRouterClient
from .eval import evaluate_with_llm


@dataclass
class EvalResult:
    """Evaluation result for a single benchmark run."""

    run_id: str
    model_short: str
    seed: int
    num_findings: int
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    matched_gt_ids: list[str]
    missed_gt_ids: list[str]


async def evaluate_single_run(
    benchmark_file: Path,
    gt_path: Path,
    eval_model: str = "google/gemini-3-flash-preview",
) -> EvalResult | None:
    """Evaluate a single benchmark run against ground truth."""
    try:
        data = json.loads(benchmark_file.read_text())
    except (json.JSONDecodeError, IOError) as e:
        print(f"Skipping {benchmark_file}: {e}", file=sys.stderr)
        return None

    # Skip non-benchmark files
    if "num_findings" not in data:
        return None

    run_id = data.get("run_id", benchmark_file.stem)
    model_short = data.get("model_short", "unknown")
    seed = data.get("seed", 0)
    num_findings = data.get("num_findings", 0)

    # Create temp file with findings in expected format
    temp_results = {
        "checks": data.get("findings", []),
        "metadata": {"model": model_short},
    }
    temp_file = benchmark_file.with_suffix(".temp_eval.json")
    temp_file.write_text(json.dumps(temp_results, indent=2))

    try:
        result = await evaluate_with_llm(gt_path, temp_file, eval_model)
    finally:
        temp_file.unlink(missing_ok=True)

    if "error" in result:
        print(f"Eval error for {run_id}: {result['error']}", file=sys.stderr)
        return None

    scores = result.get("scores", {})
    matches = result.get("matches", [])
    unmatched_gt = result.get("unmatched_gt", [])

    return EvalResult(
        run_id=run_id,
        model_short=model_short,
        seed=seed,
        num_findings=num_findings,
        tp=scores.get("true_positives", 0),
        fp=scores.get("false_positives", 0),
        fn=scores.get("false_negatives", 0),
        precision=scores.get("precision", 0),
        recall=scores.get("recall", 0),
        f1=scores.get("f1", 0),
        matched_gt_ids=[m.get("gt_id", "") for m in matches],
        missed_gt_ids=[u.get("gt_id", "") for u in unmatched_gt],
    )


async def evaluate_all_benchmarks(
    benchmark_dir: Path,
    gt_path: Path,
    eval_model: str = "google/gemini-3-flash-preview",
    max_concurrent: int = 10,
) -> list[EvalResult]:
    """Evaluate all benchmark runs against ground truth."""
    # Find all benchmark files
    benchmark_files = [
        f
        for f in benchmark_dir.glob("*.json")
        if f.name not in ("summary.json", "analysis.json")
        and not f.name.endswith("_eval.json")
    ]

    print(f"Evaluating {len(benchmark_files)} benchmark runs...", file=sys.stderr)

    # Run evaluations with concurrency limit
    semaphore = asyncio.Semaphore(max_concurrent)

    async def limited_eval(f: Path) -> EvalResult | None:
        async with semaphore:
            result = await evaluate_single_run(f, gt_path, eval_model)
            if result:
                print(
                    f"[{result.run_id}] P={result.precision:.1%} R={result.recall:.1%} F1={result.f1:.1%} (TP={result.tp} FP={result.fp} FN={result.fn})",
                    file=sys.stderr,
                )
            return result

    tasks = [limited_eval(f) for f in benchmark_files]
    results = await asyncio.gather(*tasks)

    return [r for r in results if r is not None]


def analyze_eval_results(results: list[EvalResult], gt_path: Path) -> dict:
    """Analyze evaluation results for variance and coverage."""
    # Load GT for reference
    gt = json.loads(gt_path.read_text())
    gt_ids = [i["id"] for i in gt["issues"]]
    total_gt = len(gt_ids)

    # Group by model
    by_model: dict[str, list[EvalResult]] = defaultdict(list)
    for r in results:
        by_model[r.model_short].append(r)

    analysis = {
        "total_gt_errors": total_gt,
        "models": {},
        "gt_error_detection_rates": {},
    }

    # Per-model analysis
    for model, runs in by_model.items():
        precisions = [r.precision for r in runs]
        recalls = [r.recall for r in runs]
        f1s = [r.f1 for r in runs]

        # Which GT errors were found in at least one run?
        all_matched = set()
        for r in runs:
            all_matched.update(r.matched_gt_ids)

        # Per-GT-error detection rate
        gt_detection_counts = defaultdict(int)
        for r in runs:
            for gt_id in r.matched_gt_ids:
                gt_detection_counts[gt_id] += 1

        analysis["models"][model] = {
            "num_runs": len(runs),
            "precision": {
                "mean": round(sum(precisions) / len(precisions), 3),
                "min": round(min(precisions), 3),
                "max": round(max(precisions), 3),
            },
            "recall": {
                "mean": round(sum(recalls) / len(recalls), 3),
                "min": round(min(recalls), 3),
                "max": round(max(recalls), 3),
            },
            "f1": {
                "mean": round(sum(f1s) / len(f1s), 3),
                "min": round(min(f1s), 3),
                "max": round(max(f1s), 3),
            },
            "unique_gt_found": len(all_matched),
            "coverage_pct": round(len(all_matched) / total_gt * 100, 1),
            "gt_detection_counts": dict(gt_detection_counts),
        }

    # Overall GT error detection rates (across all models)
    all_gt_detection = defaultdict(
        lambda: {"found_by_models": set(), "total_detections": 0}
    )
    for r in results:
        for gt_id in r.matched_gt_ids:
            all_gt_detection[gt_id]["found_by_models"].add(r.model_short)
            all_gt_detection[gt_id]["total_detections"] += 1

    for gt_id in gt_ids:
        info = all_gt_detection.get(
            gt_id, {"found_by_models": set(), "total_detections": 0}
        )
        analysis["gt_error_detection_rates"][gt_id] = {
            "found_by_models": list(info["found_by_models"]),
            "num_models": len(info["found_by_models"]),
            "total_detections": info["total_detections"],
        }

    # Sort by detection rate
    never_found = [
        gt_id
        for gt_id, info in analysis["gt_error_detection_rates"].items()
        if info["total_detections"] == 0
    ]
    analysis["never_found"] = never_found

    return analysis


def print_eval_analysis(analysis: dict, gt_path: Path):
    """Print formatted evaluation analysis."""
    gt = json.loads(gt_path.read_text())
    gt_by_id = {i["id"]: i for i in gt["issues"]}

    print("\n" + "=" * 70)
    print("BENCHMARK EVALUATION ANALYSIS")
    print("=" * 70)
    print(f"Ground truth: {analysis['total_gt_errors']} errors")

    for model, stats in analysis["models"].items():
        print(f"\n## {model} ({stats['num_runs']} runs)")
        print(
            f"   Precision: {stats['precision']['mean']:.1%} ({stats['precision']['min']:.1%}-{stats['precision']['max']:.1%})"
        )
        print(
            f"   Recall:    {stats['recall']['mean']:.1%} ({stats['recall']['min']:.1%}-{stats['recall']['max']:.1%})"
        )
        print(
            f"   F1:        {stats['f1']['mean']:.1%} ({stats['f1']['min']:.1%}-{stats['f1']['max']:.1%})"
        )
        print(
            f"   Coverage:  {stats['unique_gt_found']}/{analysis['total_gt_errors']} = {stats['coverage_pct']:.0f}% of GT errors found in at least one run"
        )

        # Most reliably found errors
        print(f"\n   Most reliably detected (by this model):")
        sorted_gt = sorted(stats["gt_detection_counts"].items(), key=lambda x: -x[1])
        for gt_id, count in sorted_gt[:5]:
            pct = count / stats["num_runs"] * 100
            desc = gt_by_id.get(gt_id, {}).get("description", "")[:40]
            print(
                f"      [{count}/{stats['num_runs']} = {pct:.0f}%] {gt_id}: {desc}..."
            )

    # Never found errors
    never_found = analysis.get("never_found", [])
    if never_found:
        print(f"\n## NEVER DETECTED ({len(never_found)} errors)")
        for gt_id in never_found:
            info = gt_by_id.get(gt_id, {})
            print(f"   - {gt_id}: {info.get('description', '')[:60]}...")

    # Cross-model detection
    print(f"\n## CROSS-MODEL DETECTION RATES")
    rates = analysis["gt_error_detection_rates"]
    by_num_models = defaultdict(list)
    for gt_id, info in rates.items():
        by_num_models[info["num_models"]].append((gt_id, info))

    for n in sorted(by_num_models.keys(), reverse=True):
        items = by_num_models[n]
        print(f"\n   Found by {n} model(s): {len(items)} errors")
        for gt_id, info in items[:5]:
            models = ", ".join(info["found_by_models"])
            desc = gt_by_id.get(gt_id, {}).get("description", "")[:30]
            print(f"      {gt_id}: {models} ({info['total_detections']} detections)")

    print("\n" + "=" * 70)


async def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate benchmark runs against ground truth"
    )
    parser.add_argument(
        "benchmark_dir", type=Path, help="Directory with benchmark JSON files"
    )
    parser.add_argument("gt", type=Path, help="Ground truth JSON file")
    parser.add_argument(
        "-m",
        "--model",
        default="google/gemini-3-flash-preview",
        help="Model for LLM matching",
    )
    parser.add_argument(
        "-c", "--concurrent", type=int, default=10, help="Max concurrent evaluations"
    )
    parser.add_argument("-o", "--output", type=Path, help="Output JSON path")

    args = parser.parse_args()

    results = await evaluate_all_benchmarks(
        args.benchmark_dir, args.gt, args.model, args.concurrent
    )

    analysis = analyze_eval_results(results, args.gt)
    print_eval_analysis(analysis, args.gt)

    if args.output:
        args.output.write_text(json.dumps(analysis, indent=2, default=str))
        print(f"\nAnalysis saved to: {args.output}")
    else:
        # Default output
        out_file = args.benchmark_dir / "eval_analysis.json"
        out_file.write_text(json.dumps(analysis, indent=2, default=str))
        print(f"\nAnalysis saved to: {out_file}")


if __name__ == "__main__":
    asyncio.run(main())
