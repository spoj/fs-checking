"""PDF error injection for systematic FS checking evaluation.

Extracts numeric and text spans from financial statement PDFs, applies mutations
(magnitude shifts, offsets, sign flips, transpositions, tie-out breaks,
label swaps, note reference errors, year swaps, currency swaps, standard
reference errors, and restated-label mutations), and produces a mutated PDF +
ground truth manifest.

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

    # Text mutation types (new):
    #   note_ref_wrong, year_swap, currency_swap, standard_ref_wrong,
    #   label_swap_direction, label_swap_classification, label_swap_sign_word,
    #   restated_label
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


@dataclass
class TextSpan:
    """A text span extracted from a PDF for text-level mutation."""

    page: int  # 0-based page index
    text: str  # original text content
    bbox: tuple[float, float, float, float]  # (x0, y0, x1, y1)
    font: str  # e.g. "Roboto-Light"
    fontsize: float  # e.g. 8.0
    color: int  # e.g. 0 (black)
    span_index: int  # index within the page's text spans of this type
    mutation_kind: str  # which text mutation type this span is eligible for
    replacement: str  # what to replace the text with


# ---------------------------------------------------------------------------
# Text mutation swap pairs and patterns
# ---------------------------------------------------------------------------

# Note references: "(Note 25)" or "Note 25" or "Note 1.22"
_NOTE_REF_PAREN_RE = re.compile(r"\(Note\s+(\d+(?:\.\d+)*)\)")
_NOTE_REF_INLINE_RE = re.compile(r"(?<!\()Note\s+(\d+(?:\.\d+)*)")

# Year references: "2019", "2018", etc.
_YEAR_RE = re.compile(r"\b(20[12]\d)\b")

# Currency: "US$" or "HK$"
_CURRENCY_USD_RE = re.compile(r"US\s*\$")
_CURRENCY_HKD_RE = re.compile(r"HK\s*\$")

# Accounting standard references: "HKFRS 16", "HKAS 19", "IFRS 9", "IAS 17"
_STANDARD_REF_RE = re.compile(r"((?:HKFRS|HKAS|IFRS|IAS)\s*)(\d+)")

# Restated label
_RESTATED_RE = re.compile(r"\(Restated\)")

# Label swap pairs — each is (pattern, replacement_text) bidirectional
_DIRECTION_SWAPS: list[tuple[str, str]] = [
    ("increase", "decrease"),
    ("Increase", "Decrease"),
    ("Due from", "Due to"),
    ("Due From", "Due To"),
    ("due from", "due to"),
    ("inflow", "outflow"),
    ("Inflow", "Outflow"),
]

_CLASSIFICATION_SWAPS: list[tuple[str, str]] = [
    ("Non-current assets", "Non-current liabilities"),
    ("Non-current Assets", "Non-current Liabilities"),
    ("Current assets", "Current liabilities"),
    ("Current Assets", "Current Liabilities"),
    ("Non-current liabilities", "Non-current assets"),
    ("Non-current Liabilities", "Non-current Assets"),
    ("Current liabilities", "Current assets"),
    ("Current Liabilities", "Current Assets"),
    ("Continuing operations", "Discontinued operations"),
    ("Continuing Operations", "Discontinued Operations"),
    ("continuing operations", "discontinued operations"),
]

_SIGN_WORD_SWAPS: list[tuple[str, str]] = [
    ("profit", "loss"),
    ("Profit", "Loss"),
    ("gains", "losses"),
    ("Gains", "Losses"),
    ("gain", "loss"),
    ("Gain", "Loss"),
    ("receivable", "payable"),
    ("Receivable", "Payable"),
    ("receivables", "payables"),
    ("Receivables", "Payables"),
    ("credit", "debit"),
    ("Credit", "Debit"),
]


# Build regex for each label swap group (match any of the left-hand terms)
def _build_swap_re(swaps: list[tuple[str, str]]) -> re.Pattern:
    """Build a regex that matches any left-hand term in the swap pairs."""
    # Escape and sort by length (longest first) to avoid partial matches
    terms = sorted({s[0] for s in swaps}, key=len, reverse=True)
    pattern = "|".join(re.escape(t) for t in terms)
    return re.compile(rf"\b({pattern})\b")


_DIRECTION_RE = _build_swap_re(_DIRECTION_SWAPS)
_CLASSIFICATION_RE = _build_swap_re(_CLASSIFICATION_SWAPS)
_SIGN_WORD_RE = _build_swap_re(_SIGN_WORD_SWAPS)

# Lookup dicts for fast swap
_DIRECTION_MAP = {a: b for a, b in _DIRECTION_SWAPS}
_CLASSIFICATION_MAP = {a: b for a, b in _CLASSIFICATION_SWAPS}
_SIGN_WORD_MAP = {a: b for a, b in _SIGN_WORD_SWAPS}


# All mutation kinds (numeric + text)
MutationKind = Literal[
    # Numeric mutations
    "magnitude",  # multiply by factor (e.g. 10x)
    "offset",  # add/subtract a fixed amount
    "transposition",  # swap two adjacent digits
    "sign_flip",  # positive ↔ negative
    "replace",  # replace with explicit value
    "tie_break",  # change one occurrence of a cross-page number
    # Text mutations
    "note_ref_wrong",  # change note reference number
    "year_swap",  # change year reference (e.g. 2019 → 2018)
    "currency_swap",  # change currency (US$ ↔ HK$)
    "standard_ref_wrong",  # change accounting standard number
    "label_swap_direction",  # increase ↔ decrease, due from ↔ due to
    "label_swap_classification",  # current ↔ non-current, assets ↔ liabilities
    "label_swap_sign_word",  # profit ↔ loss, gain ↔ loss, receivable ↔ payable
    "restated_label",  # remove or add "(Restated)"
]

NUMERIC_MUTATION_KINDS = {
    "magnitude",
    "offset",
    "transposition",
    "sign_flip",
    "replace",
    "tie_break",
}
TEXT_MUTATION_KINDS = {
    "note_ref_wrong",
    "year_swap",
    "currency_swap",
    "standard_ref_wrong",
    "label_swap_direction",
    "label_swap_classification",
    "label_swap_sign_word",
    "restated_label",
}


@dataclass
class Mutation:
    """Specification for a single error injection (numeric)."""

    page: int  # 0-based page index
    span_index: int  # index within page's numeric spans
    kind: MutationKind
    # Parameters (used depending on kind):
    factor: float = 10.0  # for "magnitude"
    delta: int = 0  # for "offset"
    new_value: int | None = None  # for "replace"


@dataclass
class TextMutation:
    """Specification for a single text-level error injection."""

    page: int  # 0-based page index
    span_index: int  # index within page's text spans for this kind
    kind: MutationKind
    replacement: str = ""  # the new text to stamp


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
    original_value: int  # 0 for text mutations
    mutated_value: int  # 0 for text mutations
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
# Text span extraction
# ---------------------------------------------------------------------------


def _iter_pdf_spans(
    pdf_bytes: bytes,
) -> list[tuple[int, dict]]:
    """Iterate over all text spans in a PDF, yielding (page_idx, span_dict).

    Each span_dict has keys: text, bbox, font, size, color.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    results = []
    try:
        for page_idx in range(len(doc)):
            page = doc[page_idx]
            page_data = page.get_text("dict")
            for block in page_data["blocks"]:
                if block["type"] != 0:
                    continue
                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span["text"]
                        if not text.strip():
                            continue
                        results.append((page_idx, span))
    finally:
        doc.close()
    return results


def extract_text_spans(
    pdf_bytes: bytes,
    kinds: set[str] | None = None,
) -> dict[str, list[TextSpan]]:
    """Extract text spans eligible for text mutations, grouped by mutation kind.

    Args:
        pdf_bytes: PDF file content.
        kinds: Which mutation kinds to extract for (default: all text kinds).

    Returns:
        Dict mapping mutation_kind → list of TextSpan, each with its replacement.
    """
    if kinds is None:
        kinds = TEXT_MUTATION_KINDS

    raw_spans = _iter_pdf_spans(pdf_bytes)
    result: dict[str, list[TextSpan]] = {k: [] for k in kinds}

    for page_idx, span in raw_spans:
        text = span["text"]
        bbox = tuple(span["bbox"])
        font = span["font"]
        fontsize = span["size"]
        color = span["color"]

        def _make_ts(kind: str, orig: str, repl: str, idx: int = 0) -> TextSpan:
            return TextSpan(
                page=page_idx,
                text=orig,
                bbox=bbox,
                font=font,
                fontsize=fontsize,
                color=color,
                span_index=idx,  # re-indexed below
                mutation_kind=kind,
                replacement=repl,
            )

        # --- Note references ---
        if "note_ref_wrong" in kinds:
            for m in _NOTE_REF_PAREN_RE.finditer(text):
                note_num = int(m.group(1).split(".")[0])
                # Offset by ±1 or ±2 (but stay positive)
                delta = 1 if note_num > 1 else 2
                new_num = note_num + delta
                orig = m.group(0)
                repl = orig.replace(str(note_num), str(new_num), 1)
                result["note_ref_wrong"].append(_make_ts("note_ref_wrong", orig, repl))
            for m in _NOTE_REF_INLINE_RE.finditer(text):
                note_num = int(m.group(1).split(".")[0])
                delta = 1 if note_num > 1 else 2
                new_num = note_num + delta
                orig = m.group(0)
                repl = orig.replace(str(note_num), str(new_num), 1)
                result["note_ref_wrong"].append(_make_ts("note_ref_wrong", orig, repl))

        # --- Year swap ---
        if "year_swap" in kinds:
            for m in _YEAR_RE.finditer(text):
                year = int(m.group(1))
                # Swap 2019↔2018, 2020→2019, etc.
                new_year = year - 1 if year % 2 == 1 else year + 1
                # Only target years in the document's likely reporting range
                if 2015 <= year <= 2025:
                    orig_text = text
                    repl_text = text[: m.start(1)] + str(new_year) + text[m.end(1) :]
                    result["year_swap"].append(
                        _make_ts("year_swap", orig_text, repl_text)
                    )

        # --- Currency swap ---
        if "currency_swap" in kinds:
            if _CURRENCY_USD_RE.search(text):
                repl_text = _CURRENCY_USD_RE.sub("HK$", text)
                result["currency_swap"].append(
                    _make_ts("currency_swap", text, repl_text)
                )
            elif _CURRENCY_HKD_RE.search(text):
                repl_text = _CURRENCY_HKD_RE.sub("US$", text)
                result["currency_swap"].append(
                    _make_ts("currency_swap", text, repl_text)
                )

        # --- Standard reference ---
        if "standard_ref_wrong" in kinds:
            for m in _STANDARD_REF_RE.finditer(text):
                std_num = int(m.group(2))
                # Offset by ±1 to point to a different standard
                new_num = std_num + 1
                orig_text = text
                repl_text = text[: m.start(2)] + str(new_num) + text[m.end(2) :]
                result["standard_ref_wrong"].append(
                    _make_ts("standard_ref_wrong", orig_text, repl_text)
                )

        # --- Label swap: direction ---
        if "label_swap_direction" in kinds:
            m = _DIRECTION_RE.search(text)
            if m:
                matched = m.group(0)
                if matched in _DIRECTION_MAP:
                    repl_text = (
                        text[: m.start()] + _DIRECTION_MAP[matched] + text[m.end() :]
                    )
                    result["label_swap_direction"].append(
                        _make_ts("label_swap_direction", text, repl_text)
                    )

        # --- Label swap: classification ---
        if "label_swap_classification" in kinds:
            m = _CLASSIFICATION_RE.search(text)
            if m:
                matched = m.group(0)
                if matched in _CLASSIFICATION_MAP:
                    repl_text = (
                        text[: m.start()]
                        + _CLASSIFICATION_MAP[matched]
                        + text[m.end() :]
                    )
                    result["label_swap_classification"].append(
                        _make_ts("label_swap_classification", text, repl_text)
                    )

        # --- Label swap: sign word ---
        if "label_swap_sign_word" in kinds:
            m = _SIGN_WORD_RE.search(text)
            if m:
                matched = m.group(0)
                if matched in _SIGN_WORD_MAP:
                    repl_text = (
                        text[: m.start()] + _SIGN_WORD_MAP[matched] + text[m.end() :]
                    )
                    result["label_swap_sign_word"].append(
                        _make_ts("label_swap_sign_word", text, repl_text)
                    )

        # --- Restated label ---
        if "restated_label" in kinds:
            m = _RESTATED_RE.search(text)
            if m:
                # Remove "(Restated)" — replace with empty string
                repl_text = text[: m.start()] + text[m.end() :]
                result["restated_label"].append(
                    _make_ts("restated_label", text, repl_text.strip())
                )

    # Re-index span_index per kind
    for kind, spans in result.items():
        for i, s in enumerate(spans):
            s.span_index = i

    return result


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

# Built-in font fallback (only used when embedded font not found)
_FALLBACK_FONT = "helv"


class _FontCache:
    """Cache of font reference names per page in a PDF document.

    Maps span font names (e.g. "Roboto-Light") to the page-level font
    reference names (e.g. "TT2") used by ``page.insert_text(fontname=...)``.

    Using the page's own font ref name reuses the embedded font exactly,
    producing pixel-perfect matches. No temp files or font extraction needed.

    Usage::

        with _FontCache(doc) as fc:
            fontname = fc.resolve(page, span_font_name)
            page.insert_text(..., fontname=fontname, ...)
    """

    def __init__(self, doc: fitz.Document) -> None:
        # page_idx -> {clean_font_name -> ref_name}
        self._page_fonts: dict[int, dict[str, str]] = {}
        self._build(doc)

    def _build(self, doc: fitz.Document) -> None:
        """Scan all pages and build font ref name mappings."""
        for page_idx in range(len(doc)):
            font_map: dict[str, str] = {}
            for f in doc[page_idx].get_fonts():
                base_name = f[3]  # e.g. "LECDCP+Roboto-Light"
                ref_name = f[4]  # e.g. "TT2"
                clean = base_name.split("+", 1)[-1] if "+" in base_name else base_name
                # Keep first mapping for each clean name (in case of duplicates)
                if clean not in font_map:
                    font_map[clean] = ref_name
            self._page_fonts[page_idx] = font_map

    def resolve(self, page_idx: int, span_font: str) -> str:
        """Resolve a span font name to the page's font reference name.

        Args:
            page_idx: 0-based page index
            span_font: Font name from get_text("dict") span, e.g. "Roboto-Light"

        Returns:
            Font reference name for use with insert_text(fontname=...).
            Falls back to Helvetica if no embedded font match found.
        """
        font_map = self._page_fonts.get(page_idx, {})

        # Direct match
        if span_font in font_map:
            return font_map[span_font]

        # Try stripping subset prefix
        if "+" in span_font:
            clean = span_font.split("+", 1)[1]
            if clean in font_map:
                return font_map[clean]

        # Fallback to Helvetica
        return _FALLBACK_FONT

    def close(self) -> None:
        """No-op (no temp files to clean up)."""
        pass

    def __enter__(self) -> _FontCache:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def _color_int_to_rgb(color: int) -> tuple[float, float, float]:
    """Convert an integer color (e.g. 0x231f20) to an (r, g, b) float tuple."""
    r = ((color >> 16) & 0xFF) / 255.0
    g = ((color >> 8) & 0xFF) / 255.0
    b = (color & 0xFF) / 255.0
    return (r, g, b)


def _mutate_pdf_span(
    page: fitz.Page,
    span: NumericSpan,
    new_value: int,
    font_cache: _FontCache | None = None,
) -> str:
    """Redact a numeric span in the PDF and stamp the new value.

    Uses the page's embedded font matching the span's original font,
    falling back to Helvetica only if the embedded font is not found.

    Returns the formatted new text that was stamped.
    """
    new_is_neg = new_value < 0
    new_text = _format_number(new_value, is_negative=new_is_neg)

    # Redact original text with white fill
    rect = fitz.Rect(span.bbox)
    rect.x0 -= 1
    rect.x1 += 1
    page.add_redact_annot(rect, fill=(1, 1, 1))
    page.apply_redactions()

    # Resolve font via page's font ref name
    if font_cache:
        fontname = font_cache.resolve(span.page, span.font)
    else:
        fontname = _FALLBACK_FONT

    # Calculate right-aligned position.
    # Note: fitz.get_text_length only works with built-in font names, not
    # page font refs like "TT2". Use Helvetica metrics for width estimation —
    # character widths are close enough for alignment purposes.
    tw = fitz.get_text_length(new_text, fontname=_FALLBACK_FONT, fontsize=span.fontsize)
    right_edge = span.bbox[2]
    baseline_y = span.bbox[3] - 1.5

    page.insert_text(
        fitz.Point(right_edge - tw, baseline_y),
        new_text,
        fontsize=span.fontsize,
        fontname=fontname,
        color=_color_int_to_rgb(span.color),
    )

    return new_text


def _mutate_pdf_text_span(
    page: fitz.Page,
    span: TextSpan,
    font_cache: _FontCache | None = None,
) -> str:
    """Redact a text span in the PDF and stamp the replacement text.

    Uses the page's embedded font matching the span's original font,
    preserving the original color. Text is left-aligned (unlike numeric
    spans which are right-aligned).

    Returns the replacement text that was stamped.
    """
    new_text = span.replacement

    # For restated_label removal, if replacement is empty, just redact
    if not new_text.strip():
        rect = fitz.Rect(span.bbox)
        rect.x0 -= 1
        rect.x1 += 1
        page.add_redact_annot(rect, fill=(1, 1, 1))
        page.apply_redactions()
        return ""

    # Redact original text with white fill
    rect = fitz.Rect(span.bbox)
    rect.x0 -= 1
    rect.x1 += 1
    page.add_redact_annot(rect, fill=(1, 1, 1))
    page.apply_redactions()

    # Resolve font via page's font ref name
    if font_cache:
        fontname = font_cache.resolve(span.page, span.font)
    else:
        fontname = _FALLBACK_FONT

    # Left-aligned: stamp new text starting at original x0
    page.insert_text(
        fitz.Point(span.bbox[0], span.bbox[3] - 1.5),
        new_text,
        fontsize=span.fontsize,
        fontname=fontname,
        color=_color_int_to_rgb(span.color),
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

    with _FontCache(doc) as fc:
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
                new_text = _mutate_pdf_span(page, span, new_value, font_cache=fc)

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


def inject_text_errors(
    pdf_bytes: bytes,
    text_mutations: list[TextMutation],
    source_document: str = "",
    id_offset: int = 0,
) -> InjectionResult:
    """Apply text-level mutations to a PDF.

    Args:
        pdf_bytes: Original PDF content
        text_mutations: List of text mutation specifications
        source_document: Name of the source document
        id_offset: Starting offset for ground truth IDs (for mixing with numeric)

    Returns:
        InjectionResult with mutated PDF bytes and ground truth manifest
    """
    # Extract all text spans to find the targets
    kinds_needed = {m.kind for m in text_mutations}
    text_spans = extract_text_spans(pdf_bytes, kinds=kinds_needed)

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    ground_truth: list[GroundTruthItem] = []

    with _FontCache(doc) as fc:
        for i, mutation in enumerate(text_mutations):
            kind_spans = text_spans.get(mutation.kind, [])
            if mutation.span_index >= len(kind_spans):
                raise IndexError(
                    f"TextMutation {i}: span_index {mutation.span_index} out of range "
                    f"(kind '{mutation.kind}' has {len(kind_spans)} spans)"
                )

            span = kind_spans[mutation.span_index]
            page = doc[mutation.page]

            # Use explicit replacement if provided, else use span's pre-computed one
            replacement = (
                mutation.replacement if mutation.replacement else span.replacement
            )

            # Skip no-ops
            if replacement == span.text:
                continue

            # Create a temporary TextSpan with the replacement for stamping
            stamp_span = TextSpan(
                page=span.page,
                text=span.text,
                bbox=span.bbox,
                font=span.font,
                fontsize=span.fontsize,
                color=span.color,
                span_index=span.span_index,
                mutation_kind=span.mutation_kind,
                replacement=replacement,
            )
            new_text = _mutate_pdf_text_span(page, stamp_span, font_cache=fc)

            desc = _describe_text_mutation(
                mutation.kind, span.text, replacement, span.page
            )

            ground_truth.append(
                GroundTruthItem(
                    id=f"inject_{id_offset + i:03d}",
                    page=mutation.page + 1,  # 1-based
                    category=_category_for_kind(mutation.kind),
                    severity=_severity_for_text_kind(mutation.kind),
                    description=desc,
                    original_text=span.text,
                    mutated_text=new_text,
                    original_value=0,
                    mutated_value=0,
                    mutation_kind=mutation.kind,
                    bbox=span.bbox,
                )
            )

    try:
        result_bytes = doc.tobytes()
    finally:
        doc.close()

    return InjectionResult(
        pdf_bytes=result_bytes,
        ground_truth=ground_truth,
        source_document=source_document,
    )


def _describe_text_mutation(
    kind: str, original: str, replacement: str, page: int
) -> str:
    """Generate a human-readable description of a text mutation."""
    page_1 = page + 1  # 1-based

    labels = {
        "note_ref_wrong": "Wrong note reference",
        "year_swap": "Year swap",
        "currency_swap": "Currency swap",
        "standard_ref_wrong": "Wrong accounting standard reference",
        "label_swap_direction": "Direction label swap",
        "label_swap_classification": "Classification label swap",
        "label_swap_sign_word": "Sign-word label swap",
        "restated_label": "Restated label removed",
    }
    label = labels.get(kind, kind)

    if not replacement.strip():
        trunc_orig = original[:60] + "..." if len(original) > 60 else original
        return f"{label}: removed '{trunc_orig}' on page {page_1}"

    # For long spans, find and show just the changed part with context
    if len(original) > 60:
        # Find first difference
        for i, (a, b) in enumerate(zip(original, replacement)):
            if a != b:
                # Show ~20 chars before and after the change point
                start = max(0, i - 20)
                # Find end of change
                j = i
                while (
                    j < min(len(original), len(replacement))
                    and original[j] != replacement[j]
                ):
                    j += 1
                end = min(len(original), j + 20)
                end_r = min(len(replacement), j + 20)
                prefix = "..." if start > 0 else ""
                suffix = "..." if end < len(original) else ""
                orig_snip = prefix + original[start:end] + suffix
                repl_snip = prefix + replacement[start:end_r] + suffix
                return f"{label}: '{orig_snip}' → '{repl_snip}' on page {page_1}"
        # No difference found in zip range — length differs
        trunc_orig = original[:60] + "..."
        trunc_repl = replacement[:60] + "..." if len(replacement) > 60 else replacement
        return f"{label}: '{trunc_orig}' → '{trunc_repl}' on page {page_1}"

    return f"{label}: '{original}' → '{replacement}' on page {page_1}"


def _severity_for_text_kind(kind: str) -> str:
    """Determine severity for text-level mutations."""
    # Most text mutations are material — they change the meaning of the statement
    return {
        "note_ref_wrong": "material",
        "year_swap": "material",
        "currency_swap": "material",
        "standard_ref_wrong": "minor",
        "label_swap_direction": "material",
        "label_swap_classification": "material",
        "label_swap_sign_word": "material",
        "restated_label": "material",
    }.get(kind, "material")


def random_inject(
    pdf_bytes: bytes,
    n_errors: int = 5,
    seed: int = 42,
    error_types: list[MutationKind] | None = None,
    source_document: str = "",
    min_value: int = 1000,
) -> InjectionResult:
    """Randomly inject N errors into a PDF (numeric + text mutations).

    Selects random numeric and text spans and applies random mutations.
    Prefers cross-page tie-break mutations (most useful for testing).
    When text mutation types are included, allocates a mix of numeric
    and text mutations.

    Args:
        pdf_bytes: Original PDF content
        n_errors: Number of errors to inject
        seed: Random seed for reproducibility
        error_types: Mutation kinds to choose from (default: all numeric)
        source_document: Name of the source document
        min_value: Minimum absolute value of numbers to target

    Returns:
        InjectionResult with mutated PDF and ground truth
    """
    rng = random.Random(seed)

    if error_types is None:
        error_types = ["magnitude", "offset", "transposition", "sign_flip", "tie_break"]

    # Split into numeric and text types
    requested_numeric = [t for t in error_types if t in NUMERIC_MUTATION_KINDS]
    requested_text = [t for t in error_types if t in TEXT_MUTATION_KINDS]

    # --- Phase 1: Build numeric mutations ---
    numeric_mutations: list[Mutation] = []
    if requested_numeric:
        spans_by_page = extract_numeric_spans(pdf_bytes)

        all_spans: list[NumericSpan] = []
        for page_spans in spans_by_page.values():
            for span in page_spans:
                if abs(span.value) >= min_value:
                    all_spans.append(span)

        cross_page_pairs = find_cross_page_numbers(spans_by_page)
        used_spans: set[tuple[int, int]] = set()

        # How many numeric vs text? Split proportionally, at least 1 of each if both requested
        if requested_text and all_spans:
            n_numeric = max(1, n_errors * len(requested_numeric) // len(error_types))
        else:
            n_numeric = n_errors

        # Allocate tie_break mutations
        if "tie_break" in requested_numeric and cross_page_pairs:
            n_tie_breaks = min(len(cross_page_pairs), n_numeric // 2 + 1)
            rng.shuffle(cross_page_pairs)
            for pair in cross_page_pairs[:n_tie_breaks]:
                target = rng.choice(pair)
                key = (target.page, target.span_index)
                if key in used_spans:
                    continue
                used_spans.add(key)
                numeric_mutations.append(
                    Mutation(
                        page=target.page,
                        span_index=target.span_index,
                        kind="tie_break",
                    )
                )

        # Fill remaining numeric slots
        remaining_numeric = n_numeric - len(numeric_mutations)
        non_tie_types = [t for t in requested_numeric if t != "tie_break"]
        if not non_tie_types and remaining_numeric > 0:
            non_tie_types = ["offset"]

        if all_spans and remaining_numeric > 0:
            eligible = [
                s for s in all_spans if (s.page, s.span_index) not in used_spans
            ]
            rng.shuffle(eligible)
            for span in eligible[:remaining_numeric]:
                kind = rng.choice(non_tie_types)
                m = Mutation(page=span.page, span_index=span.span_index, kind=kind)
                if kind == "magnitude":
                    m.factor = rng.choice([0.1, 10])
                elif kind == "offset":
                    pct = rng.uniform(0.05, 0.3)
                    m.delta = int(abs(span.value) * pct) * rng.choice([1, -1])
                    if m.delta == 0:
                        m.delta = rng.choice([1000, -1000])
                used_spans.add((span.page, span.span_index))
                numeric_mutations.append(m)

    # --- Phase 2: Build text mutations ---
    text_mutations: list[TextMutation] = []
    n_text = n_errors - len(numeric_mutations)

    if requested_text and n_text > 0:
        text_spans = extract_text_spans(pdf_bytes, kinds=set(requested_text))

        # Pool all available text targets
        all_text_targets: list[TextSpan] = []
        for kind in requested_text:
            all_text_targets.extend(text_spans.get(kind, []))

        if all_text_targets:
            rng.shuffle(all_text_targets)
            # Avoid duplicate pages/bboxes
            used_text: set[tuple[int, float, float, float, float]] = set()

            for ts in all_text_targets:
                if len(text_mutations) >= n_text:
                    break
                key = (ts.page, *ts.bbox)
                if key in used_text:
                    continue
                used_text.add(key)
                text_mutations.append(
                    TextMutation(
                        page=ts.page,
                        span_index=ts.span_index,
                        kind=ts.mutation_kind,
                        replacement=ts.replacement,
                    )
                )

    # --- Phase 3: Apply numeric mutations first, then text mutations ---
    # Apply numeric mutations
    if numeric_mutations:
        result = inject_errors(
            pdf_bytes, numeric_mutations, source_document=source_document
        )
        working_pdf = result.pdf_bytes
        ground_truth = result.ground_truth
    else:
        working_pdf = pdf_bytes
        ground_truth = []

    # Apply text mutations on top
    if text_mutations:
        text_result = inject_text_errors(
            working_pdf,
            text_mutations,
            source_document=source_document,
            id_offset=len(numeric_mutations),
        )
        working_pdf = text_result.pdf_bytes
        ground_truth.extend(text_result.ground_truth)

    final = InjectionResult(
        pdf_bytes=working_pdf,
        ground_truth=ground_truth,
        seed=seed,
        source_document=source_document,
    )
    return final


def _category_for_kind(kind: MutationKind) -> str:
    """Map mutation kind to error category for ground truth."""
    return {
        # Numeric
        "magnitude": "cross_footing",
        "offset": "cross_footing",
        "transposition": "cross_footing",
        "sign_flip": "cross_footing",
        "replace": "cross_footing",
        "tie_break": "note_ties",
        # Text — references
        "note_ref_wrong": "cross_reference",
        "standard_ref_wrong": "cross_reference",
        # Text — presentation
        "label_swap_direction": "presentation",
        "label_swap_classification": "presentation",
        "label_swap_sign_word": "presentation",
        "restated_label": "presentation",
        # Text — context
        "year_swap": "period_error",
        "currency_swap": "unit_error",
    }.get(kind, "other")


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


def list_text_spans(pdf_bytes: bytes, kinds: set[str] | None = None) -> str:
    """Pretty-print all text spans eligible for text mutations."""
    text_spans = extract_text_spans(pdf_bytes, kinds=kinds)
    lines: list[str] = []

    for kind in sorted(text_spans):
        spans = text_spans[kind]
        lines.append(f"\n=== {kind} ({len(spans)} targets) ===")
        for s in spans[:20]:  # Show first 20 per kind
            orig = s.text[:50] + "..." if len(s.text) > 50 else s.text
            repl = (
                s.replacement[:50] + "..." if len(s.replacement) > 50 else s.replacement
            )
            lines.append(f"  [{s.span_index:3d}] p{s.page + 1:3d}  '{orig}' → '{repl}'")
        if len(spans) > 20:
            lines.append(f"  ... and {len(spans) - 20} more")

    total = sum(len(v) for v in text_spans.values())
    lines.insert(
        0, f"Total: {total} text mutation targets across {len(text_spans)} kinds"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Visual uniformity check
# ---------------------------------------------------------------------------


def render_doped_pages(
    result: InjectionResult,
    dpi: int = 200,
) -> list[tuple[int, bytes]]:
    """Render the pages that were mutated as PNG images.

    Returns list of (page_1based, png_bytes) for each mutated page.
    """
    mutated_pages = sorted({item.page for item in result.ground_truth})  # 1-based

    doc = fitz.open(stream=result.pdf_bytes, filetype="pdf")
    images: list[tuple[int, bytes]] = []
    try:
        for page_1 in mutated_pages:
            page = doc[page_1 - 1]  # 0-based
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            pix = page.get_pixmap(matrix=mat)
            images.append((page_1, pix.tobytes("png")))
    finally:
        doc.close()

    return images


def visual_uniformity_check(
    result: InjectionResult,
    dpi: int = 200,
) -> str:
    """Render doped pages and ask Gemini Flash to check visual uniformity.

    Uses the `llm` CLI tool to send page images to Gemini Flash and ask
    whether any text looks visually out of place (different font, size,
    alignment, or color).

    Returns the LLM's assessment as a string.
    """
    import subprocess
    import tempfile
    import os

    images = render_doped_pages(result, dpi=dpi)

    if not images:
        return "No mutated pages to check."

    # Build the mutation summary for context
    mutation_summary = []
    for item in result.ground_truth:
        mutation_summary.append(
            f"  p{item.page}: [{item.mutation_kind}] '{item.original_text}' → '{item.mutated_text}'"
        )
    mutation_ctx = "\n".join(mutation_summary)

    # Write page images to temp files
    with tempfile.TemporaryDirectory() as tmpdir:
        attach_args = []
        for page_1, png_bytes in images:
            img_path = os.path.join(tmpdir, f"page_{page_1}.png")
            with open(img_path, "wb") as f:
                f.write(png_bytes)
            attach_args.extend(["-a", img_path])

        prompt = (
            "You are a visual quality checker for PDF documents. "
            "I have modified certain text in these financial statement pages "
            "using a redact-and-stamp technique. The stamped text should use "
            "the same embedded font as the original.\n\n"
            "Here are the modifications:\n"
            f"{mutation_ctx}\n\n"
            "For each modification, examine the page image and report:\n"
            "1. Is the modified text visually distinguishable from the "
            "surrounding original text? Look for: wrong font weight, "
            "different letter shapes, mismatched color, or misaligned baseline.\n"
            "2. Rate each as: PASS (indistinguishable from surrounding text), "
            "MARGINAL (minor spacing differences only), or FAIL (clearly "
            "different font/weight/color).\n"
            "3. Overall: would a reader notice any visual anomalies without "
            "knowing where to look?\n\n"
            "Note: word-spacing gaps from shorter/longer replacement words "
            "are expected and should not count as failures."
        )

        cmd = [
            "llm",
            "-n",
            "-m",
            "gemini-3-flash-preview",
            *attach_args,
            prompt,
        ]

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
            )
            if proc.returncode != 0:
                return f"LLM check failed: {proc.stderr}"
            return proc.stdout
        except subprocess.TimeoutExpired:
            return "LLM visual check timed out (120s)"
        except FileNotFoundError:
            return "llm CLI not found — install with: pip install llm"
