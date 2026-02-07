"""CLI for PDF error injection.

Inject known errors into financial statement PDFs for systematic
evaluation of the detection system.

Commands:
    fs-inject list   PDF    — Show all numeric spans (exploration)
    fs-inject random PDF    — Inject N random errors
    fs-inject batch  PDF    — Generate M variants with N errors each
"""

import json

import click
from pathlib import Path


@click.group()
def main():
    """Inject errors into financial statement PDFs for evaluation."""
    pass


@main.command()
@click.argument("pdf_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--min-value", type=int, default=100, help="Minimum absolute value to include"
)
def list(pdf_file: Path, min_value: int):
    """List all numeric spans in a PDF.

    Shows every extractable number with its page, position, and value.
    Useful for exploring the document before injecting errors.
    """
    from fs_checking.error_inject import extract_numeric_spans, list_spans

    pdf_bytes = pdf_file.read_bytes()
    output = list_spans(pdf_bytes)
    click.echo(output)


@main.command()
@click.argument("pdf_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "-n", "--n-errors", type=int, default=5, help="Number of errors to inject"
)
@click.option("-s", "--seed", type=int, default=42, help="Random seed")
@click.option(
    "-o",
    "--output",
    type=click.Path(path_type=Path),
    help="Output PDF path (default: <input>.injected.pdf)",
)
@click.option(
    "--types",
    "-t",
    multiple=True,
    type=click.Choice(
        ["magnitude", "offset", "transposition", "sign_flip", "tie_break"],
        case_sensitive=False,
    ),
    help="Error types to use (default: all)",
)
@click.option(
    "--min-value", type=int, default=1000, help="Minimum value of numbers to target"
)
def random(
    pdf_file: Path,
    n_errors: int,
    seed: int,
    output: Path | None,
    types: tuple[str, ...],
    min_value: int,
):
    """Inject N random errors into a PDF.

    Produces a mutated PDF and a ground truth JSON manifest.

    Example:
        fs-inject random report.pdf -n 10 -s 42
        fs-inject random report.pdf -n 3 -t tie_break -t offset
    """
    from fs_checking.error_inject import random_inject

    pdf_bytes = pdf_file.read_bytes()
    error_types = list(types) if types else None

    result = random_inject(
        pdf_bytes,
        n_errors=n_errors,
        seed=seed,
        error_types=error_types,
        source_document=pdf_file.name,
        min_value=min_value,
    )

    # Output paths
    if output is None:
        output = pdf_file.with_suffix(f".injected_s{seed}.pdf")
    gt_path = output.with_suffix(".ground_truth.json")

    output.write_bytes(result.pdf_bytes)
    gt_path.write_text(json.dumps(result.to_ground_truth_json(), indent=2))

    click.echo(f"Injected {len(result.ground_truth)} errors (seed={seed})")
    click.echo(f"  PDF:          {output}")
    click.echo(f"  Ground truth: {gt_path}")
    click.echo()
    for item in result.ground_truth:
        click.echo(f"  [{item.id}] p{item.page}: {item.description}")


@main.command()
@click.argument("pdf_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "-m", "--variants", type=int, default=10, help="Number of PDF variants to generate"
)
@click.option("-n", "--n-errors", type=int, default=5, help="Errors per variant")
@click.option("-s", "--base-seed", type=int, default=0, help="Starting seed")
@click.option(
    "-o",
    "--output-dir",
    type=click.Path(path_type=Path),
    help="Output directory (default: <input>_injected/)",
)
@click.option(
    "--min-value", type=int, default=1000, help="Minimum value of numbers to target"
)
def batch(
    pdf_file: Path,
    variants: int,
    n_errors: int,
    base_seed: int,
    output_dir: Path | None,
    min_value: int,
):
    """Generate M variant PDFs, each with N injected errors.

    Creates a directory of mutated PDFs + ground truth files,
    ready for batch evaluation with the detection pipeline.

    Example:
        fs-inject batch report.pdf -m 20 -n 5
        fs-inject batch report.pdf -m 10 -n 3 -o test_variants/
    """
    from fs_checking.error_inject import random_inject

    pdf_bytes = pdf_file.read_bytes()

    if output_dir is None:
        output_dir = pdf_file.parent / f"{pdf_file.stem}_injected"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_manifests: list[dict] = []

    for i in range(variants):
        seed = base_seed + i
        result = random_inject(
            pdf_bytes,
            n_errors=n_errors,
            seed=seed,
            source_document=pdf_file.name,
            min_value=min_value,
        )

        variant_name = f"variant_{seed:04d}"
        pdf_path = output_dir / f"{variant_name}.pdf"
        gt_path = output_dir / f"{variant_name}.ground_truth.json"

        pdf_path.write_bytes(result.pdf_bytes)
        manifest = result.to_ground_truth_json()
        gt_path.write_text(json.dumps(manifest, indent=2))
        all_manifests.append(manifest)

        n_actual = len(result.ground_truth)
        click.echo(
            f"  [{i + 1:3d}/{variants}] seed={seed:4d}  errors={n_actual}  → {pdf_path.name}"
        )

    # Write combined manifest
    combined_path = output_dir / "batch_manifest.json"
    combined = {
        "source_document": pdf_file.name,
        "variants": variants,
        "errors_per_variant": n_errors,
        "base_seed": base_seed,
        "manifests": all_manifests,
    }
    combined_path.write_text(json.dumps(combined, indent=2))

    click.echo(f"\nGenerated {variants} variants in {output_dir}/")
    click.echo(f"Combined manifest: {combined_path}")


if __name__ == "__main__":
    main()
