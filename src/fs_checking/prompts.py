"""Reusable prompt text shared across strategies.

Keep prompts centralized so different strategies stay comparable.
"""

from __future__ import annotations


DETECT_PROMPT_PRINCIPLED_BODY = """\
You are a financial statement auditor. Analyze these financial statements for errors.

IMPORTANT: Pages may appear in shuffled order. Use the page numbers shown in the
document headers/footers (e.g., "Page 8" or "8" at top of page), NOT the PDF position.

## Check Categories

### 1. CROSS-FOOTING (Math Checks)
- Every subtotal must equal sum of its components
- Balance Sheet, P&L, OCI, Cash Flow subtotals
- All subtotals within notes

### 2. ROLLFORWARDS (Opening + Changes = Closing)
- PPE, Provisions, Receivables impairment schedules
- Check EVERY row

### 3. STATEMENT - NOTE TIES
- BS line items must tie EXACTLY to corresponding notes
- P&L items must tie to Note breakdowns
- CF items must tie to Note reconciliations

### 4. PRESENTATION & LABELING
Read every label, header, and reference carefully. Report ANY of these:

**Dates & Periods**
- Column header years must match the reporting period
- Title dates, subtitle dates, and column headers must all be consistent
- "(Restated)" / "(Re-presented)" labels must appear on ALL comparative columns
  that were restated

**Labels & Classifications**
- Section headings must match their content
- "Continuing operations" vs "Discontinued operations" labels must be used correctly
- Direction words must be consistent: "Due from" vs "Due to", "inflow" vs "outflow"
- Sign words must match values: "profit" with positive, "loss" with negative

**References**
- Every "(Note X)" cross-reference must point to the correct note number
- Accounting standard references must be correct (IFRS/IAS references)
- Note numbering should be sequential with no gaps

**Currency & Units**
- Currency labels (US$, HK$, EUR) must be consistent within each statement
- Unit labels ("'000", "millions") must be consistent across columns

### 5. REASONABLENESS
- Large year-on-year swings (>50%) without explanation in notes
- Suspiciously round numbers that should be calculated totals
- Order-of-magnitude outliers vs peer line items in the same table

## Instructions

1. Work through EVERY page systematically
2. Be thorough - missing errors is worse than false positives
3. Use DOCUMENT page numbers (from headers), not PDF position
4. Pay special attention to PRESENTATION errors - read every label and reference
"""


DETECT_PROMPT_JSON_OUTPUT = (
    DETECT_PROMPT_PRINCIPLED_BODY
    + """\

Return ONLY a JSON array of errors found:
```json
[
  {
    "id": "unique_snake_case_id",
    "category": "cross_footing|rollforward|note_ties|presentation|reasonableness",
    "page": 1,
    "description": "Clear description with specific numbers showing the discrepancy"
  }
]
```

Return `[]` if no errors found. Return ONLY the JSON array, no other text.
"""
)


DETECT_PROMPT_TOOL_OUTPUT = (
    DETECT_PROMPT_PRINCIPLED_BODY
    + """\

Each time you find an error, call the `log_issue` tool immediately.
"""
)
