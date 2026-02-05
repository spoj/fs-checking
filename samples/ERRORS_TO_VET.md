# Planted Errors - Vet for Detectability

For each error, mark:
- **[Y]** = Detectable within 4 corners of the document (cross-foot, tie-out, consistency)
- **[N]** = NOT detectable (needs external reference or compensating errors cancel out)
- **[X]** = Not inherently errors, just reformat to make test case

---

## Page 1: Profit and Loss Account

| # | Location | ERROR Value | CORRECT Value | Detectable? | Verification |
|---|----------|-------------|---------------|-------------|--------------|
| 1 | Report Title Year | 2022 | 2023 | **[Y] CONFIRMED** | Title says "2022" but columns are "2023/2022" - internal inconsistency |
| 2 | Missing "CONSOLIDATED" | absent | CONSOLIDATED | [Y] | It's not consistent with BS (which does have consolidation) | 

---

## Page 2: Statement of Comprehensive Income

| # | Location | ERROR Value | CORRECT Value | Detectable? | Verification |
|---|----------|-------------|---------------|-------------|--------------|
| 3 | Report Title Year | 2022 | 2023 | **[Y] CONFIRMED** | Same as #1 - title vs column headers |

---

## Page 3: Balance Sheet

| # | Location | ERROR Value | CORRECT Value | Detectable? | Verification |
|---|----------|-------------|---------------|-------------|--------------|
| 4 | PPE 2023 | 383,802 | 195,876 | [Y] | Not tie with PPE notes |
| 5 | ROU assets 2023 | 195,876 | 383,802 | [N] | |
| 6 | Label "Net current liabilities" | liabilities | assets/liabilities | **[Y] CONFIRMED** | 285,986 is POSITIVE but labeled "liabilities" |

---

## Page 4: Balance Sheet (Continued)

| # | Location | ERROR Value | CORRECT Value | Detectable? | Verification |
|---|----------|-------------|---------------|-------------|--------------|
| 7 | Missing "CONSOLIDATED" | absent | CONSOLIDATED | [Y] | It says consolidated in one of the staements. overall must be consolidated |

---

## Page 5: Cash Flow Statement

| # | Location | ERROR Value | CORRECT Value | Detectable? | Verification |
|---|----------|-------------|---------------|-------------|--------------|
| 8 | Net drawdown bank loans 2023 | 28,987 | 30,446 | **[Y] CONFIRMED** | CF shows 28,987, Note 31(b) shows 30,446 - tie fails |

---

## Page 7: Note 4 - Operating profit

| # | Location | ERROR Value | CORRECT Value | Detectable? | Verification |
|---|----------|-------------|---------------|-------------|--------------|
| 9 | Net exchange gains 2023 | 114,736 | 4,736 | [W] | a exchchange gain loss or huge magnitude should be warning |
| 10 | Cost of inventories sold 2023 | 11,322,355 | 10,221,721 | **[Y] CONFIRMED** | Note 4 shows 11,322,355, P&L Cost of sales shows 10,221,721 - tie fails |
| 11 | Depreciation PPE 2023 | 150,467 | 50,467 | **[Y] PARTIAL** | Note 4 = 150,467, Note 10 = 150,467 - they MATCH (both wrong!) |
| 12 | Depreciation PPE 2022 | 70,478 | 46,250 | **[Y] CONFIRMED** | Note 4 Depr 2022 = 70,478, but P&L Interest 2022 = 70,478 (same!), Note 31(a) Interest 2022 = 46,250, Depr 2022 = 70,478 - swap detectable via P&L tie |
| 13 | Amortization intangibles 2023 | 62,534 | 26,534 | **[Y] CONFIRMED** | P&L = 26,534, Note 4 = 62,534, Note 31(a) = 12,738 - three different values! |
| 14 | Note reference "Note 1" | Note 1 | Note 12 | **[Y] CONFIRMED** | Says "Note 1" but Note 1 doesn't exist / isn't PPE |

---

## Page 8: Note 10/12 - PPE

| # | Location | ERROR Value | CORRECT Value | Detectable? | Verification |
|---|----------|-------------|---------------|-------------|--------------|
| 15 | Note header number | Note 10 | Note 12 | **[Y] PARTIAL** | Sequential note check would reveal gap/mismatch |
| 16 | Depreciation Plant 2023 | (75,001) | (15,001) | **[N] REVISED** | Row sums correctly: 27+15,403+59,159+75,001+877 = 150,467 ✓ |
| 17 | Depreciation Total 2023 | (150,467) | (50,467) | **[N] REVISED** | Columns sum correctly - compensating errors |
| 18 | Closing NBV Fixtures 2023 | 25,400 | 65,400 | **[Y] CONFIRMED** | Rollforward: 65,344 + 20,021 - 59,159 + 280 - 526 = 25,960 ≠ 25,400 |
| 19 | Closing NBV Plant 2023 | 20,739 | 80,739 | **[Y] CONFIRMED** | Rollforward: 75,738 + 20,289 - 75,001 + 287 = 21,313 ≠ 20,739 |

---

## Page 10: Note 21 - Trade Receivables Aging

| # | Location | ERROR Value | CORRECT Value | Detectable? | Verification |
|---|----------|-------------|---------------|-------------|--------------|
| 20 | 181-360 days 2023 | 77,014 | 12,485 | [N] | Swapped - upper table still sums to 1,017,189 |
| 21 | Over 360 days 2023 | 109,278 | 9,278 | [N] | A large overdue, but probably hard to enforce a warning it depneds on the compnay|
| 22 | Not yet due (lower table) 2023 | 1,105,536 | 991,834 | [N] | Part of compensating set |
| 23 | 91-180 days (lower table) 2023 | (108,351) | 5,351 | **[Y] CONFIRMED** | Shows NEGATIVE (108,351) - aging shouldn't be negative |
| 24 | Over 180 days (lower table) 2023 | 26,786 | 20,004 | [N] | Part of compensating set |
| 25 | Total (lower table) 2023 | 1,023,971 | 1,017,189 | **[Y] CONFIRMED** | Upper table = 1,017,189, Lower table = 1,023,971 - don't match! |

---

## Page 11: Note 21 Continued

| # | Location | ERROR Value | CORRECT Value | Detectable? | Verification |
|---|----------|-------------|---------------|-------------|--------------|
| 26 | Reversal of provision 2023 | (999) | (888) | **[Y] CONFIRMED** | Rollforward: 51,280 + 15,023 - 53,725 - 999 + 160 = 11,739 ≠ 11,850 |
| 27 | Note reference | Note 3 | Note 4 | **[Y] PARTIAL** | Says "Note 3" but context suggests Note 4; text below says Note 4 |

---

## Page 12: Note 31 - Cash Flow Reconciliation

| # | Location | ERROR Value | CORRECT Value | Detectable? | Verification |
|---|----------|-------------|---------------|-------------|--------------|
| 28 | Interest expenses 2022 | 46,250 | 70,478 | **[Y] CONFIRMED** | Note 31(a) Interest = 46,250, P&L Interest = 70,478 - tie fails |
| 29 | Depreciation 2023 | 150,467 | 50,467 | **[N] REVISED** | Note 31(a) = 150,467, Note 4 = 150,467 - they match (both wrong) |
| 30 | Depreciation 2022 | 70,478 | 46,250 | **[Y] CONFIRMED** | Same as #12 - swap with interest detectable |
| 31 | Amort system dev 2023 | 28,765 | 14,969 | **[N] REVISED** | No independent source to verify breakout |
| 32 | Amort other intangibles 2023 | 12,738 | 26,534 | **[Y] CONFIRMED** | Note 31(a) = 12,738, P&L = 26,534, Note 4 = 62,534 - all different! |

---

## Summary of Detectable Errors

| # | Error | Detection Method |
|---|-------|------------------|
| 1, 3 | Header year 2022 vs 2023 | Title vs column header inconsistency |
| 6 | "Net current liabilities" label | Positive number labeled as liability |
| 8 | Bank loans 28,987 vs 30,446 | CF vs Note 31(b) tie-out |
| 10 | Cost of inventories 11.3M vs 10.2M | Note 4 vs P&L tie-out |
| 12, 28, 30 | Interest/Depreciation swap 2022 | P&L vs Note 31(a) tie-out |
| 13, 32 | Amortization inconsistency | P&L vs Note 4 vs Note 31(a) - three different values |
| 14 | "Note 1" reference | Invalid note reference |
| 15 | Note 10 vs Note 12 | Sequential note numbering check |
| 18, 19 | Closing NBV fixtures/plant | Rollforward doesn't balance |
| 23 | Negative aging (108,351) | Aging values shouldn't be negative |
| 25 | Two aging tables don't match | 1,017,189 vs 1,023,971 |
| 26 | Provision rollforward | Math doesn't balance (11,739 vs 11,850) |
| 27 | "Note 3" reference | Context indicates Note 4 |

**Total: 17 detectable errors out of 30 planted errors (57%)**

---

## Non-Detectable Errors (Compensating/Self-Consistent)

| # | Error | Why Not Detectable |
|---|-------|-------------------|
| 4, 5 | PPE/ROU swap | Both values exist, totals work |
| 9 | Exchange gains 114k vs 4k | No cross-reference |
| 11, 17, 29 | Depreciation 150k vs 50k | Note 4 and Note 10 and Note 31 all show 150k (consistent wrong value) |
| 16 | Plant depreciation 75k vs 15k | Row sums correctly |
| 20, 21, 22, 24 | Aging compensating errors | Upper table total is correct |
| 31 | Amort system dev breakout | No independent verification source |

**Total: 13 non-detectable errors (43%)**
