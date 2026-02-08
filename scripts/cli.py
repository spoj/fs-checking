"""CLI entry point for financial statement checking.

Uses ensemble strategy: 10x Flash detection + Pro rank/dedupe.
Achieves 90.9% F1, 86.2% recall, 96.2% precision on test set.
"""

import asyncio
import click
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root
load_dotenv(Path(__file__).parent.parent / ".env")


@click.command()
@click.argument("pdf_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Output JSON file (default: <pdf_file>.ensemble.json)",
)
@click.option(
    "--detect-model",
    "-d",
    default="google/gemini-3-flash-preview",
    help="Model for detection phase (default: gemini-3-flash-preview)",
)
@click.option(
    "--rank-model",
    "-r",
    default="openai/gpt-5.2",
    help="Model for rank/dedupe phase (default: gpt-5.2)",
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
def main(
    pdf_file: Path,
    output: Path | None,
    detect_model: str,
    rank_model: str,
    runs: int,
    shuffle_mode: str,
):
    """
    Check IFRS financial statements in a PDF for errors.

    Uses ensemble strategy: multiple detection passes with rank/dedupe.

    PDF_FILE: Path to the financial statements PDF

    Example:
        fs-check annual_report.pdf
        fs-check financial_statements.pdf -o results.json
        fs-check document.pdf --runs 5  # Faster, less thorough
    """
    # Import here to avoid import errors when just showing help
    from fs_checking.strategies.ensemble import run_ensemble

    click.echo(f"Checking: {pdf_file}")
    click.echo(f"Mode: {shuffle_mode}")
    click.echo(
        f"Strategy: {runs}x {detect_model.split('/')[-1]} + {rank_model.split('/')[-1]} rank/dedupe"
    )

    result = asyncio.run(
        run_ensemble(
            pdf_path=pdf_file,
            output_path=output,
            detect_model=detect_model,
            rank_model=rank_model,
            num_runs=runs,
            shuffle_mode=shuffle_mode,
        )
    )

    # Print summary
    summary = result.get("summary", {})
    metadata = result.get("metadata", {})
    high = summary.get("high", 0)
    medium = summary.get("medium", 0)
    low = summary.get("low", 0)
    total = summary.get("total_unique", 0)
    raw = summary.get("raw_findings", 0)
    cost = metadata.get("estimated_cost_usd", 0)

    click.echo(f"\nFindings: {raw} raw -> {total} unique (${cost:.4f})")
    click.echo(f"  HIGH:   {high}")
    click.echo(f"  MEDIUM: {medium}")
    click.echo(f"  LOW:    {low}")

    # Show high priority items
    if result.get("high"):
        click.echo("\nHigh priority findings:")
        for item in result["high"][:5]:
            desc = item.get("description", "")[:70]
            click.echo(f"  p{item.get('page', '?')}: {desc}")


if __name__ == "__main__":
    main()
