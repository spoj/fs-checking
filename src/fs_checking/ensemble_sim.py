"""Simulate ensemble combinations to find optimal cost/performance tradeoffs.

Analyzes:
1. How recall scales with N flash runs
2. Benefit of mixing model families vs same model
3. Optimal ensemble composition for different budgets

Usage:
    uv run python -m fs_checking.ensemble_sim benchmarks/ samples/Written_test_Case.ground_truth.json
"""

import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from random import sample, seed as set_seed

# Approximate costs per run (USD)
COSTS = {
    "gemini-3-flash-preview": 0.05,
    "gemini-3-pro-preview": 0.30,
    "gpt-5.2": 5.00,
}

# Approximate time per run (seconds)
TIMES = {
    "gemini-3-flash-preview": 100,
    "gemini-3-pro-preview": 185,
    "gpt-5.2": 580,
}


@dataclass
class RunData:
    """Data from a single benchmark run."""

    run_id: str
    model_short: str
    seed: int
    matched_gt_ids: set[str]
    num_findings: int


def load_eval_data(benchmark_dir: Path) -> dict[str, list[RunData]]:
    """Load evaluation results and group by model."""
    eval_file = benchmark_dir / "eval_analysis.json"
    if not eval_file.exists():
        raise FileNotFoundError(f"Run benchmark_eval first: {eval_file}")

    analysis = json.loads(eval_file.read_text())

    # Load individual benchmark files to get matched_gt_ids per run
    runs_by_model: dict[str, list[RunData]] = defaultdict(list)

    for f in sorted(benchmark_dir.glob("*.json")):
        if f.name in ("summary.json", "analysis.json", "eval_analysis.json"):
            continue
        if f.name.endswith("_eval.json"):
            continue

        try:
            data = json.loads(f.read_text())
            if "num_findings" not in data:
                continue

            model_short = data.get("model_short", "unknown")
            run_id = data.get("run_id", f.stem)
            seed = data.get("seed", 0)

            # We need to re-extract matched GT IDs from the eval
            # For now, use the detection counts from analysis
            runs_by_model[model_short].append(
                RunData(
                    run_id=run_id,
                    model_short=model_short,
                    seed=seed,
                    matched_gt_ids=set(),  # Will populate below
                    num_findings=data.get("num_findings", 0),
                )
            )
        except (json.JSONDecodeError, IOError):
            continue

    return dict(runs_by_model)


def load_run_matches(benchmark_dir: Path, gt_path: Path) -> dict[str, set[str]]:
    """Load which GT errors each run matched.

    This requires the temp eval files or re-running eval.
    For now, we'll use a simpler approach based on the model detection patterns.
    """
    # Load the detailed eval results if available
    eval_details_file = benchmark_dir / "eval_details.json"
    if eval_details_file.exists():
        details = json.loads(eval_details_file.read_text())
        return {run_id: set(matches) for run_id, matches in details.items()}

    # Otherwise, we need to reconstruct from individual evals
    # Check if we saved per-run eval results
    run_matches = {}
    for f in sorted(benchmark_dir.glob("*.json")):
        if f.name in ("summary.json", "analysis.json", "eval_analysis.json"):
            continue
        if "_eval.json" in f.name:
            continue

        # Try to find corresponding eval result
        eval_result_file = f.with_suffix(".eval_result.json")
        if eval_result_file.exists():
            result = json.loads(eval_result_file.read_text())
            matches = result.get("matches", [])
            run_matches[f.stem] = set(m.get("gt_id", "") for m in matches)

    return run_matches


async def rebuild_run_matches(
    benchmark_dir: Path, gt_path: Path, max_concurrent: int = 30
) -> dict[str, set[str]]:
    """Re-run evaluations to get per-run GT matches - ALL CONCURRENT."""
    import asyncio
    from .benchmark_eval import evaluate_single_run

    # Collect all benchmark files
    benchmark_files = []
    for f in sorted(benchmark_dir.glob("*.json")):
        if f.name in ("summary.json", "analysis.json", "eval_analysis.json"):
            continue
        if f.name.endswith("_eval.json"):
            continue

        try:
            data = json.loads(f.read_text())
            if "num_findings" in data:
                benchmark_files.append(f)
        except (json.JSONDecodeError, IOError):
            continue

    print(f"Evaluating {len(benchmark_files)} runs CONCURRENTLY...", file=sys.stderr)

    # Run all evaluations concurrently
    async def eval_one(f: Path):
        result = await evaluate_single_run(f, gt_path)
        if result:
            print(
                f"[{result.run_id}] matched {len(result.matched_gt_ids)} GT errors",
                file=sys.stderr,
            )
        return result

    results = await asyncio.gather(*[eval_one(f) for f in benchmark_files])

    # Build run_matches dict
    run_matches = {}
    for result in results:
        if result:
            run_matches[result.run_id] = set(result.matched_gt_ids)

    # Save for future use
    save_data = {k: list(v) for k, v in run_matches.items()}
    (benchmark_dir / "eval_details.json").write_text(json.dumps(save_data, indent=2))

    return run_matches


def simulate_ensemble(
    run_matches: dict[str, set[str]],
    run_ids: list[str],
    total_gt: int,
) -> dict:
    """Simulate an ensemble of specific runs."""
    combined_matches = set()
    for run_id in run_ids:
        if run_id in run_matches:
            combined_matches.update(run_matches[run_id])

    tp = len(combined_matches)
    fn = total_gt - tp
    recall = tp / total_gt if total_gt > 0 else 0

    return {
        "runs": run_ids,
        "num_runs": len(run_ids),
        "tp": tp,
        "fn": fn,
        "recall": recall,
        "coverage": tp,
    }


def analyze_scaling(
    run_matches: dict[str, set[str]],
    model_short: str,
    total_gt: int,
    num_trials: int = 100,
) -> dict:
    """Analyze how recall scales with number of runs for a single model."""
    # Get all runs for this model
    model_runs = [rid for rid in run_matches.keys() if model_short in rid]

    if not model_runs:
        return {}

    max_runs = len(model_runs)
    results = {}

    for n in range(1, max_runs + 1):
        recalls = []
        coverages = []

        for trial in range(num_trials):
            set_seed(trial)
            selected = sample(model_runs, n)
            sim = simulate_ensemble(run_matches, selected, total_gt)
            recalls.append(sim["recall"])
            coverages.append(sim["coverage"])

        results[n] = {
            "mean_recall": sum(recalls) / len(recalls),
            "min_recall": min(recalls),
            "max_recall": max(recalls),
            "mean_coverage": sum(coverages) / len(coverages),
        }

    return results


def analyze_cross_model_benefit(
    run_matches: dict[str, set[str]],
    total_gt: int,
    num_trials: int = 100,
) -> dict:
    """Compare same-model vs cross-model ensembles at same cost."""
    flash_runs = [rid for rid in run_matches.keys() if "flash" in rid]
    pro_runs = [rid for rid in run_matches.keys() if "pro" in rid]
    gpt_runs = [rid for rid in run_matches.keys() if "gpt" in rid]

    results = {}

    # Cost of 1 GPT run ≈ 100 flash runs ≈ 17 pro runs
    # Cost of 1 pro run ≈ 6 flash runs

    # Scenario 1: Budget = 1 GPT run ($5)
    # Option A: 1 GPT
    # Option B: 100 flash (but we only have 10, so use all 10)
    # Option C: ~17 pro (but we only have 10)
    # Option D: Mixed - e.g., 5 flash + 2 pro + 0 gpt ≈ $0.85 (too cheap)
    #           Better: 1 pro + 9 flash ≈ $0.75
    #           Or: 10 flash + 3 pro ≈ $1.40

    results["budget_5_dollar"] = {
        "description": "Budget ≈ $5 (1 GPT run)",
        "options": {},
    }

    # 1 GPT
    if gpt_runs:
        gpt_recalls = []
        for trial in range(num_trials):
            set_seed(trial)
            sim = simulate_ensemble(run_matches, sample(gpt_runs, 1), total_gt)
            gpt_recalls.append(sim["recall"])
        results["budget_5_dollar"]["options"]["1_gpt"] = {
            "cost": 5.00,
            "mean_recall": sum(gpt_recalls) / len(gpt_recalls),
            "min_recall": min(gpt_recalls),
            "max_recall": max(gpt_recalls),
        }

    # All 10 flash ($0.50)
    if flash_runs:
        sim = simulate_ensemble(run_matches, flash_runs, total_gt)
        results["budget_5_dollar"]["options"]["10_flash"] = {
            "cost": 0.50,
            "mean_recall": sim["recall"],
            "min_recall": sim["recall"],
            "max_recall": sim["recall"],
        }

    # All 10 pro ($3.00)
    if pro_runs:
        sim = simulate_ensemble(run_matches, pro_runs, total_gt)
        results["budget_5_dollar"]["options"]["10_pro"] = {
            "cost": 3.00,
            "mean_recall": sim["recall"],
            "min_recall": sim["recall"],
            "max_recall": sim["recall"],
        }

    # Mixed: 10 flash + 10 pro ($3.50)
    if flash_runs and pro_runs:
        sim = simulate_ensemble(run_matches, flash_runs + pro_runs, total_gt)
        results["budget_5_dollar"]["options"]["10_flash_10_pro"] = {
            "cost": 3.50,
            "mean_recall": sim["recall"],
            "min_recall": sim["recall"],
            "max_recall": sim["recall"],
        }

    # All 30 runs
    all_runs = flash_runs + pro_runs + gpt_runs
    if all_runs:
        sim = simulate_ensemble(run_matches, all_runs, total_gt)
        results["budget_5_dollar"]["options"]["all_30"] = {
            "cost": 0.50 + 3.00 + 50.00,  # 10 flash + 10 pro + 10 gpt
            "mean_recall": sim["recall"],
            "min_recall": sim["recall"],
            "max_recall": sim["recall"],
        }

    # Scenario 2: Fixed number of runs (e.g., 3 runs)
    results["fixed_3_runs"] = {"description": "Exactly 3 runs", "options": {}}

    # 3 flash
    if len(flash_runs) >= 3:
        recalls = []
        for trial in range(num_trials):
            set_seed(trial)
            sim = simulate_ensemble(run_matches, sample(flash_runs, 3), total_gt)
            recalls.append(sim["recall"])
        results["fixed_3_runs"]["options"]["3_flash"] = {
            "cost": 0.15,
            "mean_recall": sum(recalls) / len(recalls),
        }

    # 3 pro
    if len(pro_runs) >= 3:
        recalls = []
        for trial in range(num_trials):
            set_seed(trial)
            sim = simulate_ensemble(run_matches, sample(pro_runs, 3), total_gt)
            recalls.append(sim["recall"])
        results["fixed_3_runs"]["options"]["3_pro"] = {
            "cost": 0.90,
            "mean_recall": sum(recalls) / len(recalls),
        }

    # 3 gpt
    if len(gpt_runs) >= 3:
        recalls = []
        for trial in range(num_trials):
            set_seed(trial)
            sim = simulate_ensemble(run_matches, sample(gpt_runs, 3), total_gt)
            recalls.append(sim["recall"])
        results["fixed_3_runs"]["options"]["3_gpt"] = {
            "cost": 15.00,
            "mean_recall": sum(recalls) / len(recalls),
        }

    # 1 of each
    if flash_runs and pro_runs and gpt_runs:
        recalls = []
        for trial in range(num_trials):
            set_seed(trial)
            selected = sample(flash_runs, 1) + sample(pro_runs, 1) + sample(gpt_runs, 1)
            sim = simulate_ensemble(run_matches, selected, total_gt)
            recalls.append(sim["recall"])
        results["fixed_3_runs"]["options"]["1_each"] = {
            "cost": 5.35,
            "mean_recall": sum(recalls) / len(recalls),
        }

    # 2 flash + 1 gpt
    if len(flash_runs) >= 2 and gpt_runs:
        recalls = []
        for trial in range(num_trials):
            set_seed(trial)
            selected = sample(flash_runs, 2) + sample(gpt_runs, 1)
            sim = simulate_ensemble(run_matches, selected, total_gt)
            recalls.append(sim["recall"])
        results["fixed_3_runs"]["options"]["2_flash_1_gpt"] = {
            "cost": 5.10,
            "mean_recall": sum(recalls) / len(recalls),
        }

    return results


def print_analysis(
    scaling: dict[str, dict],
    cross_model: dict,
    total_gt: int,
):
    """Print formatted analysis."""
    print("\n" + "=" * 70)
    print("ENSEMBLE SIMULATION ANALYSIS")
    print("=" * 70)
    print(f"Ground truth: {total_gt} errors")

    # Scaling analysis
    print("\n## RECALL SCALING (same model, varying N)")
    print("-" * 70)

    for model, results in scaling.items():
        print(f"\n### {model}")
        print(
            f"{'N':>3} | {'Mean Recall':>12} | {'Range':>15} | {'Coverage':>10} | {'Cost':>8}"
        )
        print("-" * 60)

        cost_per_run = COSTS.get(model, 0.10)
        for n, stats in sorted(results.items()):
            cost = n * cost_per_run
            print(
                f"{n:3d} | {stats['mean_recall']:>11.1%} | "
                f"{stats['min_recall']:.1%}-{stats['max_recall']:.1%} | "
                f"{stats['mean_coverage']:>9.1f} | ${cost:>7.2f}"
            )

    # Cross-model analysis
    print("\n## CROSS-MODEL COMPARISON")
    print("-" * 70)

    for scenario_name, scenario in cross_model.items():
        print(f"\n### {scenario['description']}")
        print(f"{'Option':>20} | {'Cost':>8} | {'Mean Recall':>12}")
        print("-" * 50)

        sorted_options = sorted(
            scenario["options"].items(), key=lambda x: x[1]["mean_recall"], reverse=True
        )
        for opt_name, opt_stats in sorted_options:
            print(
                f"{opt_name:>20} | ${opt_stats['cost']:>7.2f} | "
                f"{opt_stats['mean_recall']:>11.1%}"
            )

    print("\n" + "=" * 70)


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Simulate ensemble combinations")
    parser.add_argument(
        "benchmark_dir", type=Path, help="Directory with benchmark results"
    )
    parser.add_argument("gt", type=Path, help="Ground truth JSON file")
    parser.add_argument(
        "--rebuild", action="store_true", help="Rebuild per-run match data"
    )

    args = parser.parse_args()

    # Load ground truth
    gt = json.loads(args.gt.read_text())
    total_gt = len(gt["issues"])

    # Load or rebuild run matches
    eval_details_file = args.benchmark_dir / "eval_details.json"

    if args.rebuild or not eval_details_file.exists():
        print(
            "Building per-run match data (this runs 30 LLM evals)...", file=sys.stderr
        )
        run_matches = await rebuild_run_matches(args.benchmark_dir, args.gt)
    else:
        print(f"Loading cached match data from {eval_details_file}", file=sys.stderr)
        details = json.loads(eval_details_file.read_text())
        run_matches = {k: set(v) for k, v in details.items()}

    print(f"Loaded {len(run_matches)} runs", file=sys.stderr)

    # Analyze scaling for each model
    scaling = {}
    for model in ["gemini-3-flash-preview", "gemini-3-pro-preview", "gpt-5.2"]:
        scaling[model] = analyze_scaling(run_matches, model, total_gt)

    # Cross-model analysis
    cross_model = analyze_cross_model_benefit(run_matches, total_gt)

    # Print results
    print_analysis(scaling, cross_model, total_gt)

    # Save results
    results = {
        "total_gt": total_gt,
        "scaling": scaling,
        "cross_model": cross_model,
    }
    out_file = args.benchmark_dir / "ensemble_sim.json"
    out_file.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to: {out_file}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
