"""Scaling synthesis: subsample from pooled Flash shuffled runs.

Pools all single-shot Flash shuffled raw findings (~100 runs), draws
random subsets at different sizes, validates each with GPT-5.2,
evaluates against ground truth. All 9 validate+eval tasks run in parallel.

Usage:
    uv run python scripts/scaling_synthesis.py
"""

import asyncio
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from fs_checking.api import OpenRouterClient
from fs_checking.strategies.ensemble.ensemble import _validate_findings, _calculate_cost
from fs_checking.eval import evaluate_with_llm

# Single-shot Flash shuffled result files (all independent runs)
RESULT_FILES = [
    "samples/ar2019/unrender/singleshot_flash_shuf_10.json",
    "samples/ar2019/unrender/singleshot_flash_shuf_40_50.json",
    "samples/ar2019/unrender/singleshot_flash_shuf_50_raw.json",
]

GT_PATH = Path("samples/ar2019/unrender/ar2019_fs.injected.ground_truth.json")
PDF_PATH = Path("samples/ar2019/unrender/ar2019_fs.injected.pdf")
OUTPUT_DIR = Path("samples/ar2019/unrender/scaling")

SAMPLING_PLAN = [
    (10, 3),
    (20, 3),
    (40, 3),
]

SEED = 77
VALIDATOR_MODEL = "openai/gpt-5.2"


def load_pooled_runs() -> dict[str, list[dict]]:
    """Load all raw findings, keyed by globally unique run ID."""
    all_runs: dict[str, list[dict]] = {}
    file_idx = 0

    for filepath in RESULT_FILES:
        path = Path(filepath)
        if not path.exists():
            print(f"  WARNING: {filepath} not found, skipping", file=sys.stderr)
            continue

        data = json.loads(path.read_text())
        raw = data.get("raw_findings", [])
        if not raw:
            print(f"  WARNING: {filepath} has no raw_findings", file=sys.stderr)
            continue

        # Group by run ID
        runs_in_file: dict[str, list[dict]] = {}
        for finding in raw:
            run_id = finding.get("_run", "unknown")
            if run_id not in runs_in_file:
                runs_in_file[run_id] = []
            runs_in_file[run_id].append(finding)

        for orig_run_id, findings in runs_in_file.items():
            global_id = f"f{file_idx}_{orig_run_id}"
            all_runs[global_id] = findings

        print(
            f"  {path.name}: {len(runs_in_file)} runs, {len(raw)} findings",
            file=sys.stderr,
        )
        file_idx += 1

    return all_runs


async def validate_and_eval(
    label: str,
    sampled_findings: list[dict],
    sampled_ids: list[str],
    n_runs: int,
    sample_idx: int,
    pdf_bytes: bytes,
    client: OpenRouterClient,
) -> dict:
    """Validate a sample then evaluate. Returns summary dict."""
    t0 = time.time()

    # Validate
    validated_issues, val_usage = await _validate_findings(
        sampled_findings, pdf_bytes, client, VALIDATOR_MODEL
    )
    val_cost = _calculate_cost(val_usage, VALIDATOR_MODEL)

    print(
        f"  [{label}] validated: {len(sampled_findings)} raw -> "
        f"{len(validated_issues)} confirmed (${val_cost:.2f})",
        file=sys.stderr,
    )

    # Write result file
    result_path = OUTPUT_DIR / f"scaling_{label}.json"
    result_data = {
        "metadata": {
            "strategy": "scaling-synthesis",
            "n_runs": n_runs,
            "sample_idx": sample_idx,
            "sampled_run_ids": sampled_ids,
            "raw_findings_count": len(sampled_findings),
            "validator_model": VALIDATOR_MODEL,
            "validation_cost": round(val_cost, 4),
            "validation_usage": {
                "prompt_tokens": val_usage.get("prompt_tokens", 0),
                "completion_tokens": val_usage.get("completion_tokens", 0),
            },
        },
        "issues": validated_issues,
        "raw_findings": sampled_findings,
    }
    result_path.write_text(json.dumps(result_data, indent=2, ensure_ascii=False))

    # Evaluate
    eval_result = await evaluate_with_llm(GT_PATH, result_path)
    scores = eval_result.get("scores", {})

    eval_path = OUTPUT_DIR / f"scaling_{label}.eval.json"
    eval_path.write_text(json.dumps(eval_result, indent=2, ensure_ascii=False))

    elapsed = time.time() - t0
    recall = scores.get("recall", 0)
    precision = scores.get("precision", 0)
    f1 = scores.get("f1", 0)

    print(
        f"  [{label}] recall={recall:.1%} precision={precision:.1%} "
        f"F1={f1:.1%} (${val_cost:.2f}, {elapsed:.0f}s)",
        file=sys.stderr,
    )

    return {
        "label": label,
        "n_runs": n_runs,
        "sample_idx": sample_idx,
        "raw_findings": len(sampled_findings),
        "validated_issues": len(validated_issues),
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "gt_matched": scores.get("gt_matched", 0),
        "gt_total": scores.get("gt_total", 0),
        "fp": scores.get("false_positives", 0),
        "fn": scores.get("false_negatives", 0),
        "val_cost": round(val_cost, 4),
    }


async def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60, file=sys.stderr)
    print("SCALING SYNTHESIS", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    # Load pool
    print("\nLoading runs...", file=sys.stderr)
    all_runs = load_pooled_runs()
    run_ids = list(all_runs.keys())
    total_findings = sum(len(f) for f in all_runs.values())
    print(
        f"\nPool: {len(run_ids)} runs, {total_findings} findings\n",
        file=sys.stderr,
    )

    pdf_bytes = PDF_PATH.read_bytes()
    client = OpenRouterClient(reasoning_effort="high", timeout=1800.0)

    # Prepare all samples upfront
    rng = random.Random(SEED)
    tasks = []

    for n_runs, num_samples in SAMPLING_PLAN:
        if n_runs > len(run_ids):
            print(
                f"  Skip N={n_runs}: only {len(run_ids)} runs available",
                file=sys.stderr,
            )
            continue
        for s in range(num_samples):
            label = f"n{n_runs}_s{s + 1}"
            sampled_ids = rng.sample(run_ids, n_runs)
            sampled_findings = []
            for rid in sampled_ids:
                sampled_findings.extend(all_runs[rid])

            print(
                f"  {label}: {n_runs} runs, {len(sampled_findings)} raw findings",
                file=sys.stderr,
            )

            tasks.append(
                validate_and_eval(
                    label,
                    sampled_findings,
                    sampled_ids,
                    n_runs,
                    s + 1,
                    pdf_bytes,
                    client,
                )
            )

    # Run all validate+eval in parallel
    print(
        f"\nLaunching {len(tasks)} validate+eval tasks in parallel...\n",
        file=sys.stderr,
    )
    results = await asyncio.gather(*tasks)

    # Summary
    print(f"\n{'=' * 60}", file=sys.stderr)
    print("RESULTS", file=sys.stderr)
    print(f"{'=' * 60}", file=sys.stderr)
    print(
        f"{'Label':<12} {'N':>3} {'Raw':>5} {'Val':>4} "
        f"{'Recall':>8} {'Prec':>8} {'F1':>8} {'FP':>3} {'FN':>3} {'$Val':>6}",
        file=sys.stderr,
    )
    print("-" * 72, file=sys.stderr)
    for r in results:
        print(
            f"{r['label']:<12} {r['n_runs']:>3} {r['raw_findings']:>5} "
            f"{r['validated_issues']:>4} "
            f"{r['recall']:>7.1%} {r['precision']:>7.1%} {r['f1']:>7.1%} "
            f"{r['fp']:>3} {r['fn']:>3} ${r['val_cost']:>5.2f}",
            file=sys.stderr,
        )

    # Averages
    print(f"\n{'=' * 60}", file=sys.stderr)
    print("AVERAGES", file=sys.stderr)
    print(f"{'=' * 60}", file=sys.stderr)
    for n_runs, _ in SAMPLING_PLAN:
        samples = [r for r in results if r["n_runs"] == n_runs]
        if not samples:
            continue
        avg_recall = sum(r["recall"] for r in samples) / len(samples)
        avg_prec = sum(r["precision"] for r in samples) / len(samples)
        avg_f1 = sum(r["f1"] for r in samples) / len(samples)
        avg_gt = sum(r["gt_matched"] for r in samples) / len(samples)
        min_r = min(r["recall"] for r in samples)
        max_r = max(r["recall"] for r in samples)
        avg_val = sum(r["val_cost"] for r in samples) / len(samples)
        print(
            f"  N={n_runs:>2}: recall={avg_recall:.1%} [{min_r:.1%}..{max_r:.1%}], "
            f"prec={avg_prec:.1%}, F1={avg_f1:.1%}, "
            f"GT={avg_gt:.1f}/31, val=${avg_val:.2f}",
            file=sys.stderr,
        )

    # Save
    summary_path = OUTPUT_DIR / "scaling_summary.json"
    summary_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {summary_path}", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
