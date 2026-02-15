"""CLI entry point for financial statement checking."""

import asyncio
import click
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root
load_dotenv(Path(__file__).parent.parent / ".env")


@click.command()
@click.argument("pdf_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--strategy",
    type=click.Choice(["ensemble", "baseline", "single-agent", "swarm"]),
    default="ensemble",
    show_default=True,
    help="Which strategy to run",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Output JSON file (default depends on strategy)",
)
@click.option(
    "--detect-model",
    "-d",
    default="google/gemini-3-flash-preview",
    help="Model for detection phase (default: gemini-3-flash-preview)",
)
@click.option(
    "--phase2-model",
    "-r",
    "rank_model",
    default="openai/gpt-5.2",
    help="Model for validation phase (default: gpt-5.2)",
)
@click.option(
    "--rank-model",
    "rank_model",
    hidden=True,
)
@click.option(
    "--raw",
    is_flag=True,
    help="(Ensemble) Skip validation and output raw detector findings",
)
@click.option(
    "--runs",
    "-n",
    type=int,
    default=10,
    help="Number of parallel detection runs (default: 10)",
)
@click.option(
    "--shuffle-mode",
    type=click.Choice(["random", "ring", "none"]),
    default="random",
    help="Page reorder mode: random (full shuffle), ring (circular offset), none",
)
@click.option(
    "--images",
    is_flag=True,
    help="(Baseline) Convert PDF to JPEG images before sending",
)
def main(
    pdf_file: Path,
    strategy: str,
    output: Path | None,
    detect_model: str,
    rank_model: str,
    raw: bool,
    runs: int,
    shuffle_mode: str,
    images: bool,
):
    """Check IFRS financial statements in a PDF for errors.

    Ensemble defaults to PDF-aware validation (use --raw to skip phase 2).
    """
    click.echo(f"Checking: {pdf_file}")
    click.echo(f"Strategy: {strategy}")

    if strategy == "ensemble":
        from fs_checking.strategies.ensemble import run_ensemble

        click.echo(f"Mode: {shuffle_mode}")
        phase2 = "raw" if raw else "validate"
        click.echo(
            f"Plan: {runs}x {detect_model.split('/')[-1]} + {rank_model.split('/')[-1]} {phase2}"
        )
        result = asyncio.run(
            run_ensemble(
                pdf_path=pdf_file,
                output_path=output,
                detect_model=detect_model,
                phase2_model=rank_model,
                num_runs=runs,
                shuffle_mode=shuffle_mode,
                detect_only=raw,
            )
        )
    elif strategy == "baseline":
        from fs_checking.strategies.baseline import run_baseline

        click.echo(f"Model: {detect_model}")
        click.echo(f"Mode: {'images' if images else 'pdf'}")
        result = asyncio.run(
            run_baseline(
                pdf_path=pdf_file,
                output_path=output,
                model=detect_model,
                use_images=images,
            )
        )
    elif strategy == "single-agent":
        from fs_checking.strategies.single_agent import run_single_agent

        click.echo(f"Model: {detect_model}")
        click.echo(f"Mode: {shuffle_mode}")
        result = asyncio.run(
            run_single_agent(
                pdf_path=pdf_file,
                output_path=output,
                model=detect_model,
                shuffle_mode=shuffle_mode,
            )
        )
    else:
        from fs_checking.strategies.swarm import run_swarm

        result = asyncio.run(run_swarm(pdf_file, output))

    # Print summary
    summary = result.get("summary", {})
    metadata = result.get("metadata", {})

    if "high" in result or "medium" in result or "low" in result:
        high = summary.get("high", 0)
        medium = summary.get("medium", 0)
        low = summary.get("low", 0)
        total = summary.get("total_unique", 0)
        raw = summary.get("raw_findings", 0)
        cost = metadata.get("cost_usd", metadata.get("estimated_cost_usd", 0))

        click.echo(f"\nFindings: {raw} raw -> {total} unique (${cost:.4f})")
        click.echo(f"  HIGH:   {high}")
        click.echo(f"  MEDIUM: {medium}")
        click.echo(f"  LOW:    {low}")
    else:
        total = summary.get("total", summary.get("raw_findings", 0))
        click.echo(f"\nFindings: {total}")


if __name__ == "__main__":
    main()
