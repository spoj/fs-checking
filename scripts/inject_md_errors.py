#!/usr/bin/env python3
"""Manual error injection into unrendered markdown financial statements.

Injects 30 hand-crafted errors in 3 difficulty tiers (10 each):
  - EASY:   Large, obvious arithmetic breaks on face statements
  - MEDIUM: Cross-reference breaks between notes and face statements
  - HARD:   Subtle errors in note detail tables, small transpositions

Each error is defined explicitly with line number, old value, new value,
and a description for ground truth.

Usage:
    uv run python scripts/inject_md_errors.py
"""

import asyncio
import json
import math
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

INPUT_MD = Path("samples/ar2019/unrender/ar2019_fs.md")
OUTPUT_MD = Path("samples/ar2019/unrender/ar2019_fs.injected.md")
OUTPUT_PDF = Path("samples/ar2019/unrender/ar2019_fs.injected.pdf")
OUTPUT_GT = Path("samples/ar2019/unrender/ar2019_fs.injected.ground_truth.json")

# ─────────────────────────────────────────────────────────────
# Error definitions
# ─────────────────────────────────────────────────────────────

ERRORS = [
    # ══════════════════════════════════════════════════════════
    # EASY (10): Obvious arithmetic breaks on face statements
    # ══════════════════════════════════════════════════════════
    # E1. P&L: Gross profit doesn't tie (11,413,312 - 10,221,721 = 1,191,591)
    {
        "line": 194,
        "old": "1,191,591",
        "new": "1,191,951",
        "description": "P&L gross profit transposed: 1,191,591 -> 1,191,951. "
        "Turnover 11,413,312 minus COGS 10,221,721 should = 1,191,591.",
        "category": "transposition",
        "difficulty": "easy",
        "location": "Consolidated Profit and Loss Account",
    },
    # E2. P&L: Net profit for the year doesn't tie to components
    # 54,271 - 15,756 = 38,515? No: PBT 70,027 - Tax 15,756 = 54,271
    # Change taxation to break it
    {
        "line": 213,
        "old": "(15,756)",
        "new": "(15,576)",
        "description": "P&L taxation transposed: (15,756) -> (15,576). "
        "PBT 70,027 minus tax should = profit from continuing ops 54,271. "
        "With (15,576), it would be 54,451.",
        "category": "transposition",
        "difficulty": "easy",
        "location": "Consolidated Profit and Loss Account",
    },
    # E3. Balance Sheet: Current assets subtotal
    # Sum should be 2,859,843 (line 303)
    {
        "line": 303,
        "old": "2,859,843",
        "new": "2,895,843",
        "description": "BS current assets subtotal transposed: 2,859,843 -> 2,895,843. "
        "Sum of individual current asset items should = 2,859,843.",
        "category": "transposition",
        "difficulty": "easy",
        "location": "Consolidated Balance Sheet",
    },
    # E4. Balance Sheet: Total equity
    {
        "line": 327,
        "old": "2,112,786",
        "new": "2,112,876",
        "description": "BS total equity transposed: 2,112,786 -> 2,112,876. "
        "Sum of equity components should = 2,112,786.",
        "category": "transposition",
        "difficulty": "easy",
        "location": "Consolidated Balance Sheet",
    },
    # E5. Cash Flow: Net cash from operating activities
    # 317,102 - 1,129 - 35,119 = 280,854
    {
        "line": 433,
        "old": "280,854",
        "new": "280,584",
        "description": "CF operating cash inflow transposed: 280,854 -> 280,584. "
        "317,102 - 1,129 - 35,119 should = 280,854.",
        "category": "transposition",
        "difficulty": "easy",
        "location": "Consolidated Cash Flow Statement",
    },
    # E6. Balance Sheet: Non-current liabilities subtotal (line 336)
    {
        "line": 336,
        "old": "1,113,221",
        "new": "1,131,221",
        "description": "BS non-current liabilities subtotal transposed: 1,113,221 -> 1,131,221. "
        "Sum of NC liability items should = 1,113,221.",
        "category": "transposition",
        "difficulty": "easy",
        "location": "Consolidated Balance Sheet",
    },
    # E7. Comprehensive Income: Total comprehensive income
    # Net profit 54,271 + OCI (1,698) = 52,573 (line 260)
    {
        "line": 260,
        "old": "52,573",
        "new": "52,753",
        "description": "OCI total comprehensive income transposed: 52,573 -> 52,753. "
        "Net profit 54,271 minus total OCI 1,698 should = 52,573.",
        "category": "transposition",
        "difficulty": "easy",
        "location": "Consolidated Statement of Comprehensive Income",
    },
    # E8. Cash Flow: Net cash inflow before financing
    # Operating 280,854 + Investing (63,570) = 217,284
    {
        "line": 447,
        "old": "217,284",
        "new": "217,824",
        "description": "CF net inflow before financing transposed: 217,284 -> 217,824. "
        "Operating 280,854 plus investing (63,570) should = 217,284.",
        "category": "transposition",
        "difficulty": "easy",
        "location": "Consolidated Cash Flow Statement",
    },
    # E9. Balance Sheet: Current liabilities subtotal
    {
        "line": 316,
        "old": "2,573,857",
        "new": "2,573,587",
        "description": "BS current liabilities subtotal transposed: 2,573,857 -> 2,573,587. "
        "Sum of individual CL items should = 2,573,857.",
        "category": "transposition",
        "difficulty": "easy",
        "location": "Consolidated Balance Sheet",
    },
    # E10. Cash Flow: Financing activities total
    # Sum of financing items should = 103,610
    {
        "line": 460,
        "old": "103,610",
        "new": "103,160",
        "description": "CF financing activities total transposed: 103,610 -> 103,160. "
        "Sum of financing line items should = 103,610.",
        "category": "transposition",
        "difficulty": "easy",
        "location": "Consolidated Cash Flow Statement",
    },
    # ══════════════════════════════════════════════════════════
    # MEDIUM (10): Cross-reference breaks between notes & face
    # ══════════════════════════════════════════════════════════
    # M1. Note 3 Segment: Services turnover breaks segment total tie to P&L
    {
        "line": 1055,
        "old": "9,999,701",
        "new": "9,999,710",
        "description": "Note 3 Services turnover: 9,999,701 -> 9,999,710. "
        "Services + Products - Elimination should = P&L turnover 11,413,312. "
        "With 9,999,710 the segment total would be 11,413,321.",
        "category": "offset",
        "difficulty": "medium",
        "location": "Note 3 Segment Information",
    },
    # M2. Note 11: Intangible assets closing NBV breaks rollforward
    # Opening 2,321,294 + movements should = 2,298,948
    # Change amortization total to break rollforward
    {
        "line": 1381,
        "old": "(41,503)",
        "new": "(41,305)",
        "description": "Note 11 intangibles amortization total transposed: (41,503) -> (41,305). "
        "Component amortization sums to (41,503). Also breaks rollforward: "
        "2,321,294 + 4,886 + 24,686 - 41,503 - 10,322 - 93 = 2,298,948.",
        "category": "transposition",
        "difficulty": "medium",
        "location": "Note 11 Intangible Assets",
    },
    # M3. Note 18: Inventory finished goods breaks total tie to BS
    {
        "line": 1602,
        "old": "151,174",
        "new": "151,714",
        "description": "Note 18 finished goods transposed: 151,174 -> 151,714. "
        "Finished goods + raw materials should = BS inventories 156,644.",
        "category": "transposition",
        "difficulty": "medium",
        "location": "Note 18 Inventories",
    },
    # M4. Note 19: Trade receivable from related subtotal
    {
        "line": 1617,
        "old": "539,933",
        "new": "539,393",
        "description": "Note 19 trade due from subtotal transposed: 539,933 -> 539,393. "
        "Associated (57) + Other (539,876) should = 539,933.",
        "category": "transposition",
        "difficulty": "medium",
        "location": "Note 19 Due from/(to) Related Companies",
    },
    # M5. Note 21: Trade receivable ageing total breaks tie to face
    {
        "line": 1693,
        "old": "1,017,189",
        "new": "1,017,198",
        "description": "Note 21 ageing total: 1,017,189 -> 1,017,198. "
        "Ageing buckets sum to 1,017,189 which ties to BS trade receivable.",
        "category": "offset",
        "difficulty": "medium",
        "location": "Note 21 Trade and Other Receivables",
    },
    # M6. Note 23: Trade payables ageing total
    {
        "line": 1788,
        "old": "1,503,684",
        "new": "1,503,648",
        "description": "Note 23 payables ageing total transposed: 1,503,684 -> 1,503,648. "
        "Ageing buckets sum to 1,503,684 which ties to BS trade payables.",
        "category": "transposition",
        "difficulty": "medium",
        "location": "Note 23 Trade and Other Payables",
    },
    # M7. Note 28: Long-term liabilities total
    {
        "line": 2006,
        "old": "1,593,391",
        "new": "1,593,931",
        "description": "Note 28 LT liabilities total transposed: 1,593,391 -> 1,593,931. "
        "Sum of 5 LT liability items should = 1,593,391.",
        "category": "transposition",
        "difficulty": "medium",
        "location": "Note 28 Long-term Liabilities",
    },
    # M8. Note 31: Cash flow reconciliation — operating profit before WC
    {
        "line": 2311,
        "old": "417,535",
        "new": "417,353",
        "description": "Note 31 operating profit before WC changes transposed: 417,535 -> 417,353. "
        "Sum of PBT plus 14 adjustment items should = 417,535.",
        "category": "transposition",
        "difficulty": "medium",
        "location": "Note 31 Cash Flow Reconciliation",
    },
    # M9. Note 12: PPE closing NBV total breaks rollforward
    {
        "line": 1454,
        "old": "195,876",
        "new": "195,867",
        "description": "Note 12 PPE closing NBV transposed: 195,876 -> 195,867. "
        "Rollforward: 201,973 - 608 + 48,072 - 3,094 - 50,467 = 195,876. "
        "Also should tie to BS PPE.",
        "category": "transposition",
        "difficulty": "medium",
        "location": "Note 12 Property, Plant and Equipment",
    },
    # M10. Note 3: Geographic turnover total breaks tie to face
    {
        "line": 1158,
        "old": "11,413,312",
        "new": "11,413,321",
        "description": "Note 3 geographic turnover total: 11,413,312 -> 11,413,321. "
        "Sum of 4 regions should = P&L turnover 11,413,312.",
        "category": "offset",
        "difficulty": "medium",
        "location": "Note 3 Segment Information (Geographic)",
    },
    # ══════════════════════════════════════════════════════════
    # HARD (10): Subtle note-level detail errors
    # ══════════════════════════════════════════════════════════
    # H1. Note 11: Single intangible asset column (Cust relationships amortization)
    # Only breaks column crossfoot, not row total (row total already changed by M2)
    {
        "line": 1387,
        "old": "(122,202)",
        "new": "(122,022)",
        "description": "Note 11 accumulated amort for customer relationships transposed: "
        "(122,202) -> (122,022). Breaks the column check: cost 195,394 minus "
        "accum amort (122,202) should = NBV 73,192.",
        "category": "transposition",
        "difficulty": "hard",
        "location": "Note 11 Intangible Assets (detail column)",
    },
    # H2. Note 29: DBO movement — single actuarial item
    # 34,336 + 1,310 + 665 + 238 - 2,421 + 1,063 + 788 + 970 - 3,736 = 33,213
    {
        "line": 2126,
        "old": "(2,421)",
        "new": "(2,412)",
        "description": "Note 29 DBO actuarial gains from experience transposed: (2,421) -> (2,412). "
        "Breaks DBO rollforward: opening 34,336 + movements should = 33,213.",
        "category": "transposition",
        "difficulty": "hard",
        "location": "Note 29 Post-employment Benefit Obligations (DBO movement)",
    },
    # H3. Note 30: Deferred tax asset detail — provisions column
    # 5,623 + 1,605 + 0 + 308 = 7,536
    {
        "line": 2248,
        "old": "1,605",
        "new": "1,650",
        "description": "Note 30 DTA provisions P&L charge transposed: 1,605 -> 1,650. "
        "Breaks column check: 5,623 + 1,605 + 308 should = 7,536. "
        "Also breaks row total: component charges should sum to 7,237.",
        "category": "transposition",
        "difficulty": "hard",
        "location": "Note 30 Deferred Taxation (DTA detail)",
    },
    # H4. Note 21: Currency breakdown — single currency
    # Total should = 1,157,971
    {
        "line": 1746,
        "old": "95,042",
        "new": "95,024",
        "description": "Note 21 receivables RMB currency amount transposed: 95,042 -> 95,024. "
        "Sum of 8 currencies should = current portion 1,157,971.",
        "category": "transposition",
        "difficulty": "hard",
        "location": "Note 21 Trade and Other Receivables (currency breakdown)",
    },
    # H5. Note 28: Currency breakdown — single currency
    {
        "line": 2053,
        "old": "77,642",
        "new": "77,624",
        "description": "Note 28 LT liabilities RMB amount transposed: 77,642 -> 77,624. "
        "Sum of 11 currencies should = total LT liabilities 1,593,391.",
        "category": "transposition",
        "difficulty": "hard",
        "location": "Note 28 Long-term Liabilities (currency breakdown)",
    },
    # H6. Note 25: Share capital — shares issued (3-column cross-check)
    # 8,506,586 + 32,341 = 8,538,927
    {
        "line": 1844,
        "old": "32,341",
        "new": "32,431",
        "description": "Note 25 new shares issued transposed: 32,341 -> 32,431 ('000). "
        "Breaks: 8,506,586 + 32,341 should = 8,538,927. "
        "Also breaks parallel HK$ column: 106,332 + 404 = 106,736.",
        "category": "transposition",
        "difficulty": "hard",
        "location": "Note 25 Share Capital",
    },
    # H7. Note 41: Company BS — due from subsidiaries
    # Breaks company BS balance: assets should = equity + liabilities
    {
        "line": 2780,
        "old": "4,261,640",
        "new": "4,261,460",
        "description": "Note 41 company BS due from subsidiaries transposed: 4,261,640 -> 4,261,460. "
        "Breaks current assets total (should be 4,262,545) and ultimately "
        "the company balance sheet balance.",
        "category": "transposition",
        "difficulty": "hard",
        "location": "Note 41 Balance Sheet of the Company",
    },
    # H8. Note 29: Plan assets single investment category
    # 9,083 + 2,316 + 687 + 716 + 172 + 6,677 + 4,379 + 678 + 447 + 0 + 105 = 25,260
    {
        "line": 2185,
        "old": "6,677",
        "new": "6,767",
        "description": "Note 29 plan assets government debt transposed: 6,677 -> 6,767. "
        "Sum of 11 asset categories should = total plan assets 25,260.",
        "category": "transposition",
        "difficulty": "hard",
        "location": "Note 29 Post-employment Benefit Obligations (plan assets)",
    },
    # H9. Note 12: PPE single asset category depreciation
    # (27) + (15,403) + (19,159) + (15,001) + (877) = (50,467)
    {
        "line": 1453,
        "old": "(19,159)",
        "new": "(19,195)",
        "description": "Note 12 PPE furniture depreciation transposed: (19,159) -> (19,195). "
        "Component depreciation should sum to total (50,467). Also breaks "
        "column rollforward for furniture category.",
        "category": "transposition",
        "difficulty": "hard",
        "location": "Note 12 Property, Plant and Equipment (detail column)",
    },
    # H10. Note 3: Services sub-breakdown — supply chain solutions COP
    # 90,993 + 93,660 = 184,653 (Services COP)
    {
        "line": 1143,
        "old": "90,993",
        "new": "90,939",
        "description": "Note 3 SCS core operating profit transposed: 90,993 -> 90,939. "
        "SCS (90,993) + Logistics (93,660) should = Services COP 184,653. "
        "Also breaks tie to segment total in main table.",
        "category": "transposition",
        "difficulty": "hard",
        "location": "Note 3 Segment Information (Services breakdown)",
    },
]


def inject_errors(md_content: str, errors: list[dict]) -> tuple[str, list[dict]]:
    """Apply errors to markdown content. Returns (modified_content, ground_truth)."""
    lines = md_content.split("\n")
    total_lines = len(lines)
    ground_truth = []

    for i, err in enumerate(errors):
        line_idx = err["line"] - 1  # 0-indexed
        if line_idx >= total_lines:
            print(
                f"WARNING: Error {i + 1} line {err['line']} out of range",
                file=sys.stderr,
            )
            continue

        old_line = lines[line_idx]
        if err["old"] not in old_line:
            print(
                f"WARNING: Error {i + 1} - '{err['old']}' not found on line {err['line']}",
                file=sys.stderr,
            )
            print(f"  Line content: {old_line[:120]}", file=sys.stderr)
            continue

        # Apply the substitution
        new_line = old_line.replace(err["old"], err["new"], 1)
        lines[line_idx] = new_line

        # Estimate PDF page from markdown line position
        # 66 rendered pages / 3204 lines ≈ 48.5 lines per page
        est_page = max(1, round(err["line"] / 48.5))

        ground_truth.append(
            {
                "id": f"inject_{i + 1:03d}",
                "page": est_page,
                "category": err["category"],
                "severity": "material",
                "difficulty": err["difficulty"],
                "description": err["description"],
                "location": err["location"],
                "old_value": err["old"],
                "new_value": err["new"],
                "md_line": err["line"],
            }
        )

        tag = err["difficulty"].upper()[0]
        print(
            f"  [{tag}{i + 1:02d}] Line {err['line']}: {err['old']} -> {err['new']}",
            file=sys.stderr,
        )

    return "\n".join(lines), ground_truth


async def main():
    from unrender import md_to_html, render_html_to_pdf

    print(f"Input:  {INPUT_MD}", file=sys.stderr)
    md_content = INPUT_MD.read_text(encoding="utf-8")
    print(
        f"  {len(md_content.splitlines())} lines, {len(md_content):,} bytes",
        file=sys.stderr,
    )

    easy = [e for e in ERRORS if e["difficulty"] == "easy"]
    medium = [e for e in ERRORS if e["difficulty"] == "medium"]
    hard = [e for e in ERRORS if e["difficulty"] == "hard"]
    print(
        f"\nInjecting {len(ERRORS)} errors ({len(easy)} easy, {len(medium)} medium, {len(hard)} hard):",
        file=sys.stderr,
    )

    injected_md, ground_truth = inject_errors(md_content, ERRORS)

    # Save injected markdown
    OUTPUT_MD.write_text(injected_md, encoding="utf-8")
    print(f"\nMarkdown: {OUTPUT_MD}", file=sys.stderr)

    # Save ground truth
    gt_doc = {
        "document": str(INPUT_MD),
        "description": "30 hand-crafted errors: 10 easy, 10 medium, 10 hard",
        "issues": ground_truth,
    }
    OUTPUT_GT.write_text(json.dumps(gt_doc, indent=2), encoding="utf-8")
    print(f"Ground truth: {OUTPUT_GT}", file=sys.stderr)

    # Render to PDF
    html = md_to_html(injected_md)
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / "render.pdf"
        ok, err = await render_html_to_pdf(html, tmp)
        if ok:
            OUTPUT_PDF.write_bytes(tmp.read_bytes())
            import fitz

            doc = fitz.open(str(OUTPUT_PDF))
            print(f"PDF: {OUTPUT_PDF} ({len(doc)} pages)", file=sys.stderr)
            doc.close()
        else:
            print(f"Render failed: {err}", file=sys.stderr)
            raise SystemExit(1)

    print(f"\nDone. {len(ground_truth)} errors injected.", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
