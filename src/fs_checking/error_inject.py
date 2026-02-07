"""PDF error injection for systematic FS checking evaluation.

Extracts numeric spans from financial statement PDFs, applies mutations
(magnitude shifts, offsets, sign flips, transpositions, tie-out breaks),
and produces a mutated PDF + ground truth manifest.

Usage:
    # Extract all numbers from a PDF
    spans = extract_numeric_spans(pdf_bytes)

    # Random injection: N errors with reproducible seed
    mutated_pdf, ground_truth = random_inject(pdf_bytes, n_errors=5, seed=42)

    # Targeted injection: specific mutations
    mutations = [
        Mutation(page=0, span_index=3, kind="magnitude", factor=10),
        Mutation(page=2, span_index=7, kind="offset", delta=10_000),
    ]
    mutated_pdf, ground_truth = inject_errors(pdf_bytes, mutations)
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field, asdict
from decimal import Decimal
from enum import Enum
from typing import Literal

import fitz  # PyMuPDF


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class NumericSpan:
    """A numeric value extracted from a PDF with its position and font info."""

    page: int  # 0-based page index
    text: str  # original text e.g. "(10,221,721)" or "383,802"
    value: int  # parsed integer value (negative if parenthesized)
    bbox: tuple[float, float, float, float]  # (x0, y0, x1, y1)
    font: str  # e.g. "ArialMT"
    fontsize: float  # e.g. 9.96
    color: int  # e.g. 0 (black)
    is_negative: bool  # True if parenthesized
    span_index: int  # index within the page's numeric spans (for targeting)


MutationKind = Literal[
    "magnitude",  # multiply by factor (e.g. 10x)
    "offset",  # add/subtract a fixed amount
    "transposition",  # swap two adjacent digits
    "sign_flip",  # positive ↔ negative
    "replace",  # replace with explicit value
    "tie_break",  # change one occurrence of a cross-page number
]


@dataclass
class Mutation:
    """Specification for a single error injection."""

    page: int  # 0-based page index
    span_index: int  # index within page's numeric spans
    kind: MutationKind
    # Parameters (used depending on kind):
    factor: float = 10.0  # for "magnitude"
    delta: int = 0  # for "offset"
    new_value: int | None = None  # for "replace"


@dataclass
class GroundTruthItem:
    """Record of a single injected error."""

    id: str
    page: int  # 1-based (for consistency with existing ground truth format)
    category: str
    severity: str
    description: str
    original_text: str
    mutated_text: str
    original_value: int
    mutated_value: int
    mutation_kind: str
    bbox: tuple[float, float, float, float]


@dataclass
class InjectionResult:
    """Result of error injection."""

    pdf_bytes: bytes
    ground_truth: list[GroundTruthItem]
    seed: int | None = None
    source_document: str = ""

    def to_ground_truth_json(self) -> dict:
        """Export in format compatible with existing eval.py."""
        return {
            "document": self.source_document,
            "injection_seed": self.seed,
            "description": f"{len(self.ground_truth)} injected errors",
            "issues": [
                {
                    "id": item.id,
                    "page": item.page,
                    "category": item.category,
                    "severity": item.severity,
                    "description": item.description,
                }
                for item in self.ground_truth
            ],
            "summary": {
                "total": len(self.ground_truth),
                "by_category": _count_by(self.ground_truth, "category"),
                "by_severity": _count_by(self.ground_truth, "severity"),
            },
            "_injection_details": [asdict(item) for item in self.ground_truth],
        }


def _count_by(items: list[GroundTruthItem], attr: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        key = getattr(item, attr)
        counts[key] = counts.get(key, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# Number parsing
# ---------------------------------------------------------------------------

# Matches: 1,234,567  or  (1,234,567)  or  1234567  (at least 3+ digits total)
_NUM_RE = re.compile(
    r"""
    (?P<neg>\()?           # optional opening paren (negative)
    (?P<digits>
        \d{1,3}(?:,\d{3})+ # with commas: 1,234 or 1,234,567
        | \d{4,}            # or plain 4+ digits: 1234567
    )
    (?(neg)\))             # closing paren required if opening was present
    """,
    re.VERBOSE,
)


def _parse_number(text: str) -> tuple[int, bool] | None:
    """Parse a financial number string, return (value, is_negative) or None."""
    text = text.strip()
    m = _NUM_RE.search(text)
    if not m:
        return None
    digits = m.group("digits").replace(",", "")
    value = int(digits)
    is_neg = m.group("neg") is not None
    if is_neg:
        value = -value
    return value, is_neg


# Spans that are primarily a number (possibly with parens/commas)
# Rejects sentences like "At 31 December 2023, the Group..."
# and headers like "FOR THE YEAR ENDED 31 DECEMBER 2022"
_NUMERIC_SPAN_RE = re.compile(
    r"""^\s*
    (?:\()?                        # optional opening paren
    \d{1,3}(?:,\d{3})*            # digits with commas
    (?:\.\d+)?                     # optional decimal
    (?:\))?                        # optional closing paren
    \s*$""",
    re.VERBOSE,
)


def _is_numeric_span(text: str) -> bool:
    """Check if a span is primarily a numeric value (not prose with embedded numbers)."""
    return bool(_NUMERIC_SPAN_RE.match(text.strip()))


def _format_number(value: int, is_negative: bool | None = None) -> str:
    """Format integer as financial number string.

    Uses accounting convention: negatives in parentheses, positives plain.
    """
    if is_negative is None:
        is_negative = value < 0
    abs_val = abs(value)
    formatted = f"{abs_val:,}"
    if is_negative:
        return f"({formatted})"
    return formatted


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def extract_numeric_spans(pdf_bytes: bytes) -> dict[int, list[NumericSpan]]:
    """Extract all numeric spans from a PDF, grouped by page.

    Returns:
        Dict mapping page index (0-based) to list of NumericSpan objects,
        sorted by vertical position (top to bottom) then horizontal (left to right).
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    result: dict[int, list[NumericSpan]] = {}

    try:
        for page_idx in range(len(doc)):
            page = doc[page_idx]
            page_data = page.get_text("dict")
            spans: list[NumericSpan] = []

            for block in page_data["blocks"]:
                if block["type"] != 0:  # text block only
                    continue
                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span["text"].strip()
                        if not text:
                            continue
                        # Only match spans that are primarily numeric
                        # (skip sentences with embedded numbers like
                        # "At 31 December 2023, the Group...")
                        if not _is_numeric_span(text):
                            continue
                        parsed = _parse_number(text)
                        if parsed is None:
                            continue
                        value, is_neg = parsed
                        # Skip trivially small numbers (note refs, etc.)
                        # Keep anything ≥100 to catch meaningful financial amounts
                        if abs(value) < 100:
                            continue
                        spans.append(
                            NumericSpan(
                                page=page_idx,
                                text=text,
                                value=value,
                                bbox=tuple(span["bbox"]),
                                font=span["font"],
                                fontsize=span["size"],
                                color=span["color"],
                                is_negative=is_neg,
                                span_index=0,  # set below
                            )
                        )

            # Sort by y then x
            spans.sort(key=lambda s: (s.bbox[1], s.bbox[0]))
            for i, s in enumerate(spans):
                s.span_index = i

            result[page_idx] = spans
    finally:
        doc.close()

    return result


def find_cross_page_numbers(
    spans_by_page: dict[int, list[NumericSpan]],
    min_value: int = 3000,
) -> list[tuple[NumericSpan, NumericSpan]]:
    """Find numbers that appear on multiple pages (potential tie-outs).

    Returns pairs of spans with matching absolute values on different pages.
    These are prime candidates for tie_break mutations.

    Args:
        min_value: Minimum absolute value to consider (default 3000,
                   which excludes years like 2022/2023).
    """
    # Build value → list of spans index
    value_map: dict[int, list[NumericSpan]] = {}
    for page_spans in spans_by_page.values():
        for span in page_spans:
            abs_val = abs(span.value)
            if abs_val < min_value:
                continue
            if abs_val not in value_map:
                value_map[abs_val] = []
            value_map[abs_val].append(span)

    pairs: list[tuple[NumericSpan, NumericSpan]] = []
    for abs_val, span_list in value_map.items():
        # Only interested in cross-page matches
        pages = {s.page for s in span_list}
        if len(pages) < 2:
            continue
        # Create pairs across different pages
        for i in range(len(span_list)):
            for j in range(i + 1, len(span_list)):
                if span_list[i].page != span_list[j].page:
                    pairs.append((span_list[i], span_list[j]))

    return pairs


# ---------------------------------------------------------------------------
# Mutation logic
# ---------------------------------------------------------------------------


def _apply_mutation(span: NumericSpan, mutation: Mutation) -> tuple[int, str]:
    """Apply a mutation to a span, return (new_value, description)."""
    original = span.value

    if mutation.kind == "magnitude":
        new_val = int(original * mutation.factor)
        desc = f"Magnitude ×{mutation.factor}: {_format_number(original)} → {_format_number(new_val)} on page {span.page + 1}"

    elif mutation.kind == "offset":
        new_val = original + mutation.delta
        sign = "+" if mutation.delta > 0 else ""
        desc = f"Offset {sign}{mutation.delta:,}: {_format_number(original)} → {_format_number(new_val)} on page {span.page + 1}"

    elif mutation.kind == "transposition":
        digits = list(str(abs(original)))
        if len(digits) >= 2:
            # Find a swap position that changes the number
            # Prefer swapping non-adjacent digits for bigger visible change
            pos = min(len(digits) - 2, max(0, len(digits) // 2))
            digits[pos], digits[pos + 1] = digits[pos + 1], digits[pos]
        new_abs = int("".join(digits))
        new_val = -new_abs if original < 0 else new_abs
        if new_val == original:
            # Fallback: offset by a small amount
            new_val = original + (1000 if abs(original) > 10000 else 100)
        desc = f"Transposition: {_format_number(original)} → {_format_number(new_val)} on page {span.page + 1}"

    elif mutation.kind == "sign_flip":
        new_val = -original
        desc = f"Sign flip: {_format_number(original)} → {_format_number(new_val)} on page {span.page + 1}"

    elif mutation.kind == "replace":
        new_val = mutation.new_value if mutation.new_value is not None else original
        desc = f"Replace: {_format_number(original)} → {_format_number(new_val)} on page {span.page + 1}"

    elif mutation.kind == "tie_break":
        # Offset by ~1-5% to create a visible but plausible difference
        pct = random.uniform(0.01, 0.05)
        delta = max(1, int(abs(original) * pct))
        new_val = original + delta
        desc = f"Tie break ({delta:+,}): {_format_number(original)} → {_format_number(new_val)} on page {span.page + 1}"

    else:
        raise ValueError(f"Unknown mutation kind: {mutation.kind}")

    return new_val, desc


# ---------------------------------------------------------------------------
# PDF mutation (redact + stamp)
# ---------------------------------------------------------------------------

# Built-in font closest to Arial
_FALLBACK_FONT = "helv"


def _mutate_pdf_span(
    page: fitz.Page,
    span: NumericSpan,
    new_value: int,
) -> str:
    """Redact a numeric span in the PDF and stamp the new value.

    Returns the formatted new text that was stamped.
    """
    new_is_neg = new_value < 0
    new_text = _format_number(new_value, is_negative=new_is_neg)

    # Redact original text with white fill
    rect = fitz.Rect(span.bbox)
    # Expand rect slightly to ensure full coverage
    rect.x0 -= 1
    rect.x1 += 1
    page.add_redact_annot(rect, fill=(1, 1, 1))
    page.apply_redactions()

    # Calculate right-aligned position for new text
    tw = fitz.get_text_length(new_text, fontname=_FALLBACK_FONT, fontsize=span.fontsize)
    right_edge = span.bbox[2]
    baseline_y = span.bbox[3] - 1.5  # approx baseline offset from bottom

    x = right_edge - tw

    page.insert_text(
        fitz.Point(x, baseline_y),
        new_text,
        fontsize=span.fontsize,
        fontname=_FALLBACK_FONT,
        color=(0, 0, 0),
    )

    return new_text


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def inject_errors(
    pdf_bytes: bytes,
    mutations: list[Mutation],
    source_document: str = "",
) -> InjectionResult:
    """Apply specific mutations to a PDF.

    Args:
        pdf_bytes: Original PDF content
        mutations: List of mutation specifications
        source_document: Name of the source document (for metadata)

    Returns:
        InjectionResult with mutated PDF bytes and ground truth manifest
    """
    # Extract spans first to validate mutations
    spans_by_page = extract_numeric_spans(pdf_bytes)
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    ground_truth: list[GroundTruthItem] = []

    try:
        for i, mutation in enumerate(mutations):
            page_spans = spans_by_page.get(mutation.page, [])
            if mutation.span_index >= len(page_spans):
                raise IndexError(
                    f"Mutation {i}: span_index {mutation.span_index} out of range "
                    f"(page {mutation.page} has {len(page_spans)} numeric spans)"
                )

            span = page_spans[mutation.span_index]
            new_value, description = _apply_mutation(span, mutation)

            # Skip no-ops
            if new_value == span.value:
                continue

            page = doc[mutation.page]
            new_text = _mutate_pdf_span(page, span, new_value)

            ground_truth.append(
                GroundTruthItem(
                    id=f"inject_{i:03d}",
                    page=mutation.page + 1,  # 1-based
                    category=_category_for_kind(mutation.kind),
                    severity="material"
                    if abs(new_value - span.value) > 1000
                    else "minor",
                    description=description,
                    original_text=span.text,
                    mutated_text=new_text,
                    original_value=span.value,
                    mutated_value=new_value,
                    mutation_kind=mutation.kind,
                    bbox=span.bbox,
                )
            )

        result_bytes = doc.tobytes()
    finally:
        doc.close()

    return InjectionResult(
        pdf_bytes=result_bytes,
        ground_truth=ground_truth,
        source_document=source_document,
    )


def random_inject(
    pdf_bytes: bytes,
    n_errors: int = 5,
    seed: int = 42,
    error_types: list[MutationKind] | None = None,
    source_document: str = "",
    min_value: int = 1000,
) -> InjectionResult:
    """Randomly inject N errors into a PDF.

    Selects random numeric spans and applies random mutations.
    Prefers cross-page tie-break mutations (most useful for testing).

    Args:
        pdf_bytes: Original PDF content
        n_errors: Number of errors to inject
        seed: Random seed for reproducibility
        error_types: Mutation kinds to choose from (default: all)
        source_document: Name of the source document
        min_value: Minimum absolute value of numbers to target

    Returns:
        InjectionResult with mutated PDF and ground truth
    """
    rng = random.Random(seed)

    if error_types is None:
        error_types = ["magnitude", "offset", "transposition", "sign_flip", "tie_break"]

    spans_by_page = extract_numeric_spans(pdf_bytes)

    # Collect all eligible spans
    all_spans: list[NumericSpan] = []
    for page_spans in spans_by_page.values():
        for span in page_spans:
            if abs(span.value) >= min_value:
                all_spans.append(span)

    if not all_spans:
        raise ValueError("No numeric spans found with value >= min_value")

    # Find cross-page pairs for tie_break mutations
    cross_page_pairs = find_cross_page_numbers(spans_by_page)

    # Build mutation list
    mutations: list[Mutation] = []
    used_spans: set[tuple[int, int]] = set()  # (page, span_index)

    # Allocate some tie_break mutations if available
    n_tie_breaks = 0
    if "tie_break" in error_types and cross_page_pairs:
        n_tie_breaks = min(len(cross_page_pairs), n_errors // 2 + 1)
        rng.shuffle(cross_page_pairs)
        for pair in cross_page_pairs[:n_tie_breaks]:
            # Mutate one side of the pair (randomly choose which)
            target = rng.choice(pair)
            key = (target.page, target.span_index)
            if key in used_spans:
                continue
            used_spans.add(key)
            mutations.append(
                Mutation(
                    page=target.page,
                    span_index=target.span_index,
                    kind="tie_break",
                )
            )

    # Fill remaining slots with random mutations
    remaining = n_errors - len(mutations)
    non_tie_types = [t for t in error_types if t != "tie_break"]
    if not non_tie_types:
        non_tie_types = ["offset"]

    eligible = [s for s in all_spans if (s.page, s.span_index) not in used_spans]
    rng.shuffle(eligible)

    for span in eligible[:remaining]:
        kind = rng.choice(non_tie_types)
        m = Mutation(page=span.page, span_index=span.span_index, kind=kind)

        # Set kind-specific params
        if kind == "magnitude":
            m.factor = rng.choice([0.1, 10])
        elif kind == "offset":
            pct = rng.uniform(0.05, 0.3)
            m.delta = int(abs(span.value) * pct) * rng.choice([1, -1])
            if m.delta == 0:
                m.delta = rng.choice([1000, -1000])

        used_spans.add((span.page, span.span_index))
        mutations.append(m)

    # Apply mutations
    result = inject_errors(pdf_bytes, mutations, source_document=source_document)
    result.seed = seed
    return result


def _category_for_kind(kind: MutationKind) -> str:
    """Map mutation kind to error category for ground truth."""
    return {
        "magnitude": "cross_footing",
        "offset": "cross_footing",
        "transposition": "cross_footing",
        "sign_flip": "cross_footing",
        "replace": "cross_footing",
        "tie_break": "note_ties",
    }.get(kind, "numeric_mutation")


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------


def list_spans(pdf_bytes: bytes) -> str:
    """Pretty-print all numeric spans in a PDF (for exploration)."""
    spans_by_page = extract_numeric_spans(pdf_bytes)
    lines: list[str] = []
    for page_idx in sorted(spans_by_page):
        lines.append(
            f"\n=== Page {page_idx + 1} ({len(spans_by_page[page_idx])} numbers) ==="
        )
        for span in spans_by_page[page_idx]:
            lines.append(
                f"  [{span.span_index:3d}] {span.text:>20s}  "
                f"(val={span.value:>15,})  "
                f"bbox=({span.bbox[0]:.0f},{span.bbox[1]:.0f},{span.bbox[2]:.0f},{span.bbox[3]:.0f})"
            )

    # Cross-page numbers
    pairs = find_cross_page_numbers(spans_by_page)
    if pairs:
        lines.append(f"\n=== Cross-page tie-outs ({len(pairs)} pairs) ===")
        seen: set[int] = set()
        for s1, s2 in pairs:
            val = abs(s1.value)
            if val in seen:
                continue
            seen.add(val)
            lines.append(
                f"  {_format_number(s1.value):>15s}: "
                f"page {s1.page + 1}[{s1.span_index}] ↔ page {s2.page + 1}[{s2.span_index}]"
            )

    total = sum(len(v) for v in spans_by_page.values())
    lines.insert(0, f"Total: {total} numeric spans across {len(spans_by_page)} pages")
    return "\n".join(lines)
