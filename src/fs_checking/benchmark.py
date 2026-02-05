"""Benchmark script for systematic model variance testing.

Runs multiple single detection passes to understand:
1. Per-model variance (what does the same model find/miss across runs?)
2. Cross-model coverage (which errors are found by which models?)
3. Optimal ensemble composition

Usage:
    # Run 10x flash
    uv run python -m fs_checking.benchmark samples/test.pdf --model flash --runs 10

    # Run benchmark suite (10x each model)
    uv run python -m fs_checking.benchmark samples/test.pdf --suite

    # Analyze existing benchmark results
    uv run python -m fs_checking.benchmark --analyze benchmarks/
"""

import asyncio
import json
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

from .api import OpenRouterClient
from .detection import _run_single_detection_pass, RunConfig
from .pdf_utils import pdf_to_images

# Model shortcuts
MODELS = {
    "flash": "google/gemini-3-flash-preview",
    "pro": "google/gemini-3-pro-preview",
    "gpt": "openai/gpt-5.2",
}


@dataclass
class BenchmarkRun:
    """Result of a single benchmark run."""

    run_id: str
    model: str
    model_short: str
    seed: int
    shuffle: bool
    timestamp: str
    elapsed_seconds: float
    prompt_tokens: int
    completion_tokens: int
    findings: list[dict]
    num_findings: int
    finding_ids: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.finding_ids:
            self.finding_ids = [f.get("id", "") for f in self.findings]


async def run_single_benchmark(
    pdf_path: Path,
    model: str,
    seed: int,
    shuffle: bool = True,
) -> BenchmarkRun:
    """Run a single detection pass and return structured result."""
    model_short = model.split("/")[-1]
    run_id = f"{model_short}_seed{seed}_{datetime.now().strftime('%H%M%S')}"

    run_config = RunConfig(
        model=model,
        shuffle=shuffle,
        seed=seed,
        label=run_id,
    )

    # Load PDF and convert to images
    page_images = pdf_to_images(pdf_path.read_bytes(), dpi=150)
    client = OpenRouterClient(reasoning_effort="high", timeout=900.0)

    start = time.time()
    findings, usage = await _run_single_detection_pass(
        run_config, 0, page_images, client
    )
    elapsed = time.time() - start

    return BenchmarkRun(
        run_id=run_id,
        model=model,
        model_short=model_short,
        seed=seed,
        shuffle=shuffle,
        timestamp=datetime.now().isoformat(),
        elapsed_seconds=round(elapsed, 1),
        prompt_tokens=usage.get("prompt_tokens", 0),
        completion_tokens=usage.get("completion_tokens", 0),
        findings=findings,
        num_findings=len(findings),
    )


async def _run_single_benchmark_task(
    run_idx: int,
    seed: int,
    model: str,
    model_short: str,
    page_images: list[bytes],
    client: OpenRouterClient,
    output_dir: Path,
) -> BenchmarkRun:
    """Run a single benchmark task (for parallel execution)."""
    run_id = f"{model_short}_seed{seed}"

    run_config = RunConfig(
        model=model,
        shuffle=True,
        seed=seed,
        label=run_id,
    )

    start = time.time()
    findings, usage = await _run_single_detection_pass(
        run_config, run_idx, page_images, client
    )
    elapsed = time.time() - start

    run_result = BenchmarkRun(
        run_id=run_id,
        model=model,
        model_short=model_short,
        seed=seed,
        shuffle=True,
        timestamp=datetime.now().isoformat(),
        elapsed_seconds=round(elapsed, 1),
        prompt_tokens=usage.get("prompt_tokens", 0),
        completion_tokens=usage.get("completion_tokens", 0),
        findings=findings,
        num_findings=len(findings),
    )

    # Save individual result immediately
    out_file = output_dir / f"{run_id}.json"
    out_file.write_text(json.dumps(asdict(run_result), indent=2))
    print(
        f"[{run_id}] Found {len(findings)} issues in {elapsed:.1f}s",
        file=sys.stderr,
    )

    return run_result


async def run_benchmark_batch(
    pdf_path: Path,
    model: str,
    num_runs: int,
    output_dir: Path,
    start_seed: int = 1,
) -> list[BenchmarkRun]:
    """Run multiple detection passes for one model CONCURRENTLY."""
    model_short = model.split("/")[-1]
    print(f"\n{'=' * 60}", file=sys.stderr)
    print(f"Benchmarking {model_short}: {num_runs} runs (PARALLEL)", file=sys.stderr)
    print(f"{'=' * 60}", file=sys.stderr)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load PDF once
    page_images = pdf_to_images(pdf_path.read_bytes(), dpi=150)
    client = OpenRouterClient(reasoning_effort="high", timeout=900.0)

    total_start = time.time()

    # Create all tasks
    tasks = [
        _run_single_benchmark_task(
            run_idx=i,
            seed=start_seed + i,
            model=model,
            model_short=model_short,
            page_images=page_images,
            client=client,
            output_dir=output_dir,
        )
        for i in range(num_runs)
    ]

    # Run all concurrently
    results = await asyncio.gather(*tasks)

    total_elapsed = time.time() - total_start
    print(
        f"\nCompleted {num_runs} runs in {total_elapsed:.1f}s (parallel)",
        file=sys.stderr,
    )

    return list(results)


def analyze_benchmark_results(benchmark_dir: Path, gt_path: Path | None = None) -> dict:
    """Analyze benchmark results for variance and coverage."""
    # Load all benchmark files (as dicts, not BenchmarkRun objects)
    runs_by_model: dict[str, list[dict]] = {}

    for f in sorted(benchmark_dir.glob("*.json")):
        if f.name.startswith("analysis") or f.name == "summary.json":
            continue
        try:
            data = json.loads(f.read_text())
            # Skip non-benchmark files (must have num_findings)
            if "num_findings" not in data:
                print(f"Skipping {f.name}: not a benchmark result", file=sys.stderr)
                continue
            model_short = data.get("model_short", "unknown")
            if model_short not in runs_by_model:
                runs_by_model[model_short] = []
            runs_by_model[model_short].append(data)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Skipping {f}: {e}", file=sys.stderr)

    analysis = {"models": {}, "all_findings": {}}

    # Per-model analysis
    for model_short, runs in runs_by_model.items():
        findings_counts = [r["num_findings"] for r in runs]
        all_finding_ids = []
        finding_freq: dict[str, int] = {}

        for r in runs:
            for f in r.get("findings", []):
                fid = f.get("id", "")
                if fid:
                    all_finding_ids.append(fid)
                    # Normalize ID for grouping (remove run-specific prefixes)
                    base_id = normalize_finding_id(fid)
                    finding_freq[base_id] = finding_freq.get(base_id, 0) + 1

        unique_findings = set(all_finding_ids)

        analysis["models"][model_short] = {
            "num_runs": len(runs),
            "findings_per_run": {
                "min": min(findings_counts) if findings_counts else 0,
                "max": max(findings_counts) if findings_counts else 0,
                "mean": round(sum(findings_counts) / len(findings_counts), 1)
                if findings_counts
                else 0,
            },
            "unique_finding_ids": len(unique_findings),
            "avg_time_seconds": round(
                sum(r["elapsed_seconds"] for r in runs) / len(runs), 1
            )
            if runs
            else 0,
            "avg_tokens": {
                "prompt": round(sum(r["prompt_tokens"] for r in runs) / len(runs))
                if runs
                else 0,
                "completion": round(
                    sum(r["completion_tokens"] for r in runs) / len(runs)
                )
                if runs
                else 0,
            },
            "finding_frequency": finding_freq,
        }

    # Cross-model analysis: which findings appear in how many models?
    all_findings_by_base_id: dict[str, dict] = {}
    for model_short, runs in runs_by_model.items():
        for r in runs:
            for f in r.get("findings", []):
                base_id = normalize_finding_id(f.get("id", ""))
                if base_id not in all_findings_by_base_id:
                    all_findings_by_base_id[base_id] = {
                        "models_found_by": set(),
                        "total_occurrences": 0,
                        "sample_description": f.get("description", "")[:100],
                        "page": f.get("page"),
                    }
                all_findings_by_base_id[base_id]["models_found_by"].add(model_short)
                all_findings_by_base_id[base_id]["total_occurrences"] += 1

    # Convert sets to lists for JSON
    for base_id, info in all_findings_by_base_id.items():
        info["models_found_by"] = list(info["models_found_by"])

    analysis["all_findings"] = all_findings_by_base_id

    return analysis


def normalize_finding_id(fid: str) -> str:
    """Normalize finding ID to group similar findings across runs.

    Many findings will have similar IDs like:
    - bs_total_assets_2023
    - balance_sheet_total_assets_2023
    - total_assets_crossfoot_2023

    This tries to extract the semantic core.
    """
    # For now, just return as-is - we'll refine based on actual data
    return fid.lower().strip()


def print_analysis(analysis: dict):
    """Print formatted analysis report."""
    print("\n" + "=" * 70)
    print("BENCHMARK ANALYSIS")
    print("=" * 70)

    for model, stats in analysis["models"].items():
        print(f"\n## {model}")
        print(f"   Runs: {stats['num_runs']}")
        fpr = stats["findings_per_run"]
        print(
            f"   Findings/run: {fpr['mean']} avg (min: {fpr['min']}, max: {fpr['max']})"
        )
        print(f"   Unique finding IDs: {stats['unique_finding_ids']}")
        print(f"   Avg time: {stats['avg_time_seconds']}s")
        print(
            f"   Avg tokens: {stats['avg_tokens']['prompt']:,} in, {stats['avg_tokens']['completion']:,} out"
        )

        # Most frequent findings
        freq = stats.get("finding_frequency", {})
        if freq:
            print(f"\n   Most frequent findings:")
            for fid, count in sorted(freq.items(), key=lambda x: -x[1])[:10]:
                pct = count / stats["num_runs"] * 100
                print(f"      [{count}/{stats['num_runs']} = {pct:.0f}%] {fid[:50]}")

    # Cross-model findings
    all_findings = analysis.get("all_findings", {})
    if all_findings:
        print(f"\n## Cross-Model Coverage ({len(all_findings)} unique findings)")

        # Categorize by number of models that found each
        by_model_count: dict[int, list] = {}
        for fid, info in all_findings.items():
            n = len(info["models_found_by"])
            if n not in by_model_count:
                by_model_count[n] = []
            by_model_count[n].append((fid, info))

        for n in sorted(by_model_count.keys(), reverse=True):
            findings = by_model_count[n]
            print(f"\n   Found by {n} model(s): {len(findings)} findings")
            for fid, info in findings[:5]:
                models = ", ".join(info["models_found_by"])
                print(f"      {fid[:40]}: {models}")

    print("\n" + "=" * 70)


async def run_full_benchmark_suite(
    pdf_path: Path,
    output_dir: Path,
    runs_per_model: int = 10,
):
    """Run the full benchmark suite: 10x flash, 10x pro, 10x gpt - ALL CONCURRENT."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 70}", file=sys.stderr)
    print(
        f"FULL BENCHMARK SUITE: {len(MODELS)} models x {runs_per_model} runs = {len(MODELS) * runs_per_model} total",
        file=sys.stderr,
    )
    print(
        f"Running ALL {len(MODELS) * runs_per_model} jobs CONCURRENTLY", file=sys.stderr
    )
    print(f"{'=' * 70}", file=sys.stderr)

    # Load PDF once
    page_images = pdf_to_images(pdf_path.read_bytes(), dpi=150)
    client = OpenRouterClient(reasoning_effort="high", timeout=900.0)

    total_start = time.time()

    # Create ALL tasks across ALL models
    all_tasks = []
    for short_name, model in MODELS.items():
        model_short = model.split("/")[-1]
        for i in range(runs_per_model):
            seed = i + 1
            all_tasks.append(
                _run_single_benchmark_task(
                    run_idx=i,
                    seed=seed,
                    model=model,
                    model_short=model_short,
                    page_images=page_images,
                    client=client,
                    output_dir=output_dir,
                )
            )

    # Run ALL concurrently
    results = await asyncio.gather(*all_tasks, return_exceptions=True)

    # Count successes/failures
    successes = [r for r in results if not isinstance(r, Exception)]
    failures = [r for r in results if isinstance(r, Exception)]

    total_elapsed = time.time() - total_start
    print(f"\n{'=' * 70}", file=sys.stderr)
    print(
        f"Completed {len(successes)} runs in {total_elapsed:.1f}s ({len(failures)} failures)",
        file=sys.stderr,
    )

    if failures:
        for f in failures[:5]:
            print(f"  Error: {f}", file=sys.stderr)

    # Group results by model
    all_results = {}
    for r in successes:
        if isinstance(r, BenchmarkRun):
            if r.model_short not in all_results:
                all_results[r.model_short] = []
            all_results[r.model_short].append(r)

    # Save summary
    summary = {
        "pdf": str(pdf_path),
        "timestamp": datetime.now().isoformat(),
        "runs_per_model": runs_per_model,
        "models": list(MODELS.keys()),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    return all_results


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark model variance")
    parser.add_argument("pdf", type=Path, nargs="?", help="PDF file to test")
    parser.add_argument(
        "-m",
        "--model",
        choices=list(MODELS.keys()),
        help="Model to benchmark (flash, pro, gpt)",
    )
    parser.add_argument(
        "-n", "--runs", type=int, default=10, help="Number of runs (default: 10)"
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=Path("benchmarks"), help="Output directory"
    )
    parser.add_argument(
        "--suite", action="store_true", help="Run full benchmark suite (all models)"
    )
    parser.add_argument(
        "--analyze", type=Path, help="Analyze existing benchmark results"
    )
    parser.add_argument("--gt", type=Path, help="Ground truth file for analysis")

    args = parser.parse_args()

    if args.analyze:
        # Analysis mode
        analysis = analyze_benchmark_results(args.analyze, args.gt)
        print_analysis(analysis)

        # Save analysis
        out_file = args.analyze / "analysis.json"
        # Convert sets to lists for JSON serialization
        out_file.write_text(json.dumps(analysis, indent=2, default=str))
        print(f"\nAnalysis saved to: {out_file}")

    elif args.suite:
        # Full suite mode
        if not args.pdf:
            parser.error("PDF path required for benchmark suite")
        await run_full_benchmark_suite(args.pdf, args.output, args.runs)

    elif args.model:
        # Single model mode
        if not args.pdf:
            parser.error("PDF path required for benchmark")
        model = MODELS[args.model]
        args.output.mkdir(parents=True, exist_ok=True)
        await run_benchmark_batch(args.pdf, model, args.runs, args.output)

    else:
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())
