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
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Output JSON file (default: <pdf_file>.checks.json)",
)
@click.option(
    "--model",
    "-m",
    default=None,
    help="Model to use (default: FS_CHECK_MODEL env or google/gemini-3-flash-preview)",
)
@click.option(
    "--max-depth",
    type=int,
    default=2,
    help="Maximum delegation depth for swarm (default: 2)",
)
@click.option(
    "--prompt",
    "-p",
    type=str,
    default=None,
    help="Custom prompt for the swarm (default: standard IFRS checking)",
)
def main(
    pdf_file: Path,
    output: Path | None,
    model: str | None,
    max_depth: int,
    prompt: str | None,
):
    """
    Check IFRS financial statements in a PDF for consistency.

    PDF_FILE: Path to the financial statements PDF

    Example:
        fs-check annual_report.pdf -o results.json
        fs-check financial_statements.pdf --model anthropic/claude-opus-4.5
    """
    # Import here to avoid import errors when just showing help
    from fs_checking.swarm import run_swarm

    click.echo(f"Checking: {pdf_file}")

    result = asyncio.run(
        run_swarm(
            pdf_path=pdf_file,
            output_path=output,
            model=model,
            max_depth=max_depth,
            prompt=prompt,
        )
    )

    # Print summary
    checks = result.get("checks", [])
    pass_count = sum(1 for c in checks if c.get("status") == "pass")
    fail_count = sum(1 for c in checks if c.get("status") == "fail")
    warn_count = sum(1 for c in checks if c.get("status") == "warn")

    click.echo(f"\nSummary: {pass_count} pass, {fail_count} fail, {warn_count} warn")

    if fail_count > 0:
        click.echo("\nFailed checks:")
        for c in checks:
            if c.get("status") == "fail":
                click.echo(f"  - {c.get('id', '?')}: {c.get('reason', '')[:60]}")


if __name__ == "__main__":
    main()
