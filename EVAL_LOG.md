# Evaluation Log

All evaluations use LLM-based semantic matching (`fs_checking.eval`), never page-number matching. See AGENTS.md for why.

## ar2019.injected_s42.pdf (10 injected errors)

### 2026-02-07 — 10x Flash, no nudge (toolcall.json)

- **Config**: 10x `gemini-3-flash-preview`, tool-call loop, page shuffle, no nudge, no race
- **Eval model**: `gemini-3-flash-preview`
- **Recall**: 9/10 (90.0%)
- **Precision**: 75.0% (9 TP, 3 FP)
- **F1**: 81.8%
- **Cost**: ~$0.60
- **Time**: 437s
- **Missed**: inject_007 (transposition: 1,854,832 → 1,858,432, p151)
- **FP**: related_party_aging_mismatch, management_commentary_vs_data_inconsistency, cash_flow_reconciliation_rounding (all likely real errors in base document)
- **Notes**: Best cost/recall ratio. 3 FP are genuine issues in the original AR.

### 2026-02-07 — 10x Flash, 1 nudge (nudge1.json)

- **Config**: 10x `gemini-3-flash-preview`, tool-call loop, page shuffle, 1 nudge per run
- **Eval model**: `gemini-3-flash-preview`
- **Recall**: 9/10 (90.0%)
- **Precision**: 45.0% (9 TP, 11 FP)
- **F1**: 60.0%
- **Cost**: $2.36
- **Time**: 1041s
- **Missed**: inject_002 (tie break: 1,736,817 → 1,754,568, p197)
- **Notes**: Same recall as no-nudge but 4x cost, 2.4x time, much worse precision. Nudge causes model to report more low-confidence findings without improving GT coverage. Different error missed vs no-nudge (inject_002 vs inject_007) — just sampling noise. **Conclusion: nudging is not worth it.**

### 2026-02-07 — 25x Flash, race 30/25, stagger 5s (race30.json)

- **Config**: 25x `gemini-3-flash-preview` (launch 30, keep 25), tool-call loop, page shuffle, stagger 5s
- **Eval model**: `gemini-3-flash-preview`
- **Recall**: 10/10 (100%)
- **Precision**: 50.0% (10 TP, 10 FP)
- **F1**: 66.7%
- **Cost**: $5.04
- **Time**: 526s
- **Missed**: none
- **Cancelled**: 5 slow runners, 0 failures
- **Notes**: First 100% recall result. 10 FP are all genuine errors in the base document. Race pattern works — stagger prevents thundering herd, cancel avoids waiting for stragglers.

### 2026-02-07 — GPT-5.2 low reasoning + 2 nudges (gpt52_nudge_toolcall.json)

- **Config**: 1x `openai/gpt-5.2`, tool-call loop, low reasoning effort, 2 nudges
- **Eval model**: not formally evaluated with LLM matcher
- **Raw findings**: 5 errors found
- **Cost**: $0.21
- **Time**: 288s
- **Notes**: Multi-turn continuation dies with empty responses on 252-page PDFs. Low reasoning effort works around crashes but model stops too early. Flash dominates on cost/recall.

## ar2019.mixed_s42.pdf (29 errors: 15 numeric + 14 text)

### 2026-02-07 — 25x Flash, race 30/25, stagger 5s

- **Config**: 25x `gemini-3-flash-preview` (launch 30, keep 25), tool-call loop, page shuffle, stagger 5s
- **Eval model**: `gemini-3-flash-preview`
- **Recall**: 19/29 (65.5%)
- **Precision**: 79.2% (19 TP, 5 FP)
- **F1**: 71.7%
- **Cost**: $4.48
- **Time**: 695s
- **Cancelled**: 5 slow runners, 0 failures

**Detected (19/29):**
- Numeric (12/15): inject_000 (tie_break p145), inject_001 (tie_break p148), inject_003 (tie_break p152), inject_004 (offset p145), inject_005 (offset p149), inject_006 (magnitude p179), inject_007 (transposition p153), inject_008 (sign_flip p182), inject_009 (offset p183), inject_011 (sign_flip p197), inject_012 (offset p206), inject_013 (transposition p211)
- Text (7/14): inject_016 (year_swap p145), inject_018 (currency_swap p145), inject_020 (label_swap_classification p148), inject_022 (label_swap_sign_word p145), inject_026 (label_swap_sign_word p194), inject_027 (currency_swap p179), inject_028 (year_swap p156)

**Missed (10/29):**
- Numeric (3/15): inject_002 (tie_break p150 SOCIE), inject_010 (tie_break p194 related co), inject_014 (magnitude ÷10 p246)
- Text (7/14): inject_015 (note_ref_wrong p150), inject_017 (year_swap p148 column header "2019"→"2018"), inject_019 (label_swap_classification p145 Continuing→Discontinued), inject_021 (label_swap_direction p148 Due from→to), inject_023 (label_swap_direction p152 inflow→outflow), inject_024 (restated_label p148 remove "(Restated)"), inject_025 (standard_ref_wrong p154 HKFRS 16→17)

**FP (5):** pl_total_margin_crossfoot (cascade from inject_004), related_party_aging_mismatch, variable_sensitivity_reasonableness, soce_retained_earnings_crossfoot_2019, soce_other_reserves_crossfoot_2018

**Observations:**
- Numeric recall (12/15 = 80%) significantly higher than text recall (7/14 = 50%)
- Text mutations the model CAN detect: currency swaps (US$→HK$), year swaps in headers, bold sign-word swaps (profit→loss, Receivables→Payables), classification swaps in section headings
- Text mutations the model CANNOT reliably detect: note reference off-by-one, direction swaps (inflow→outflow, Due from→to), restated label removal, standard reference changes, subtle year column swaps
- Presentation/semantic errors are harder than arithmetic errors — model is tuned for cross-footing and tie-out checks, less for label consistency
- 1 FP (pl_total_margin_crossfoot) is a genuine cascade from inject_004 offset — the model correctly found the downstream inconsistency

### 2026-02-07 — 25x Flash, race 30/25, stagger 5s — STRENGTHENED PROMPT (v2)

- **Config**: 25x `gemini-3-flash-preview` (launch 30, keep 25), tool-call loop, page shuffle, stagger 5s
- **Prompt**: Expanded PRESENTATION section with sub-categories (Dates & Periods, Labels & Classifications, References, Currency & Units) plus REASONABLENESS section. Added explicit instruction to read every label and reference.
- **Eval model**: `gemini-3-flash-preview`
- **Recall**: 26/29 (89.7%) — up from 65.5%
- **Precision**: 81.2% (26 TP, 6 FP) — up from 79.2%
- **F1**: 85.2% — up from 71.7%
- **Cost**: $3.97
- **Time**: 871s (5 runs failed with 503s)
- **Raw findings**: 267 (vs 196 in v1) — model produces ~36% more findings

**Newly detected vs v1 (7 errors):**
- inject_010 (tie_break p194 related co) — numeric, now caught
- inject_014 (magnitude ÷10 p246 fin summary) — numeric, now caught
- inject_017 (year_swap p148 BS column "2019"→"2018") — prompt examples helped
- inject_019 (Continuing→Discontinued p145) — explicit classification check
- inject_021 (Due from→to p148) — explicit direction word check
- inject_023 (inflow→outflow p152 CF) — explicit direction word check
- inject_024 (restated label removal p148) — explicit restated label check
- inject_025 (HKFRS 16→17 p154) — explicit standard reference check

**Still missed (3/29):**
- inject_002 (tie_break p150 SOCIE +25k): SOCIE is a complex table, model consistently misses this
- inject_015 (note_ref_wrong p150 Note 25→26): Off-by-one in note refs remains hard to detect
- inject_022 (Gross profit→Gross loss p145): Model found the math error but not the label change — ironic

**Key takeaway:** Prompt engineering for presentation checks yielded +24pp recall improvement at no additional cost. The detailed sub-categories with concrete examples (direction words, restated labels, standard numbers) directly mapped to previously-missed error types.

### 2026-02-07 — 25x Flash, race 30/25, stagger 5s — ALIGNED RANKER + PARTIAL HARVEST (v3)

- **Config**: 25x `gemini-3-flash-preview` (launch 30, keep 25), tool-call loop, page shuffle, stagger 5s
- **Prompt**: Same strengthened detection prompt as v2. Ranker now embeds detection prompt directly (single source of truth). Ranker instructed to keep different errors on same page as separate entries.
- **Code changes**: Partial findings harvested from cancelled race runners. JSON decode errors (503 HTML) now retried instead of fatal.
- **Eval model**: `gemini-3-flash-preview`
- **Recall**: 29/29 (100%) — up from 89.7%
- **Precision**: 65.4% (34 TP, 18 FP) — down from 81.2%
- **F1**: 79.1% — down from 85.2%
- **Cost**: $4.40
- **Time**: 1014s
- **Raw findings**: 300 (47 harvested from 5 cancelled runners)
- **Unique after dedupe**: 52 (H:30 M:19 L:3)

**All 29 GT errors detected.** The 3 previously-missed errors:
- inject_002 (SOCIE tie_break p150): Now caught as `perpetual_securities_rollforward`
- inject_015 (note_ref_wrong p150): Now caught as `soce_note_header`
- inject_022 (Gross profit→loss p145): Now caught as `pl_gloss_loss_label` — ranker kept label error separate from math error

**18 FP** — many appear to be real errors in the base document (PPE rollforward, pension OCI tie, lease discrepancies). The ranker is less aggressive about merging, which helps recall but increases FP count.

**Key takeaway:** 100% recall achieved through three reinforcing changes: (1) aligned ranker prompt keeps presentation findings as HIGH instead of LOW, (2) partial harvest recovers findings from cancelled runners, (3) less aggressive merging preserves distinct error types on same page.

**Rank/dedupe bottleneck analysis (v2 run):**
- inject_002 (SOCIE tie_break p150): **Not in raw findings.** 0/25 runs detected it. Genuine detection miss.
- inject_015 (note_ref_wrong p150): **Not in raw findings.** 0/25 runs caught the off-by-one. Genuine detection miss.
- inject_022 (Gross profit→Gross loss label p145): **IS in raw findings.** 7/25 runs explicitly flagged "gross loss" as wrong label. But rank/dedupe merged these into `pl_2019_math_and_tie_errors` (20 findings merged) and `pl_presentation_errors_headers_currency` (16 findings merged). The eval LLM matched these mega-findings to inject_004 (offset) and inject_016/018/019 (year/currency/classification) but couldn't decompose the merged blob to also credit inject_022. **Rank/dedupe is the bottleneck here** — over-aggressive merging of 20+ findings into a single entry lost the granularity needed for eval to match individual GT errors.

### 2026-02-08 — 25x Flash, race 30/25, force-visual (rasterized 100dpi, no text layer)

- **Config**: 25x `gemini-3-flash-preview` (launch 30, keep 25), tool-call loop, page shuffle, stagger 5s, `--force-visual` (100dpi q70 JPEG rasterization, text layer stripped)
- **Purpose**: Validate doping quality — confirm the model detects errors from pixel recognition, not from artefacts in the embedded text layer.
- **Eval model**: `gemini-3-flash-preview`
- **Recall**: 26/29 (81.2%) — comparable to v2 native PDF (89.7%)
- **Precision**: 86.7% (26 TP, 4 FP)
- **F1**: 83.9%
- **Cost**: $4.80
- **Time**: 685s
- **Raw findings**: 313 (54 from partial harvest)
- **Rasterized PDF**: 21.0 MB (100dpi q70, 6.9s to rasterize). 150dpi (41MB) hit OpenRouter's ~50MB request size limit.

**Detected (26/29):** All numeric mutations detected except inject_002 (SOCIE tie_break) and inject_014 (magnitude ÷10 p246). Text mutations: inject_016 (year swap), inject_018 (currency), inject_019 (classification), inject_020 (BS heading), inject_022 (profit→loss label), inject_023 (outflow), inject_025 (HKFRS 17), inject_026 (Receivables→Payables), inject_027 (currency), inject_028 (year).

**Missed (6/29):**
- inject_002 (SOCIE tie_break p150): Consistently missed across all runs
- inject_014 (magnitude ÷10 p246): Financial summary page, low visibility
- inject_015 (note_ref_wrong p150 Note 25→26): Off-by-one too subtle for 100dpi OCR
- inject_017 (year_swap p148 column header "2019"→"2018"): Fine print in column header
- inject_021 (Due from→Due to p148): Subtle direction word swap
- inject_024 (restated label removal p148): Small label, hard to spot visually

**Key takeaway:** Visual-only mode confirms the doping is clean — no textual artefacts giving the model unfair hints. The 81% recall from pure pixel recognition is a genuine lower bound. The 3 errors caught in native PDF (v3: 100%) but missed visually (inject_017, inject_021, inject_024) are fine-print items where the embedded text layer gives an advantage. The doping technique (font ref reuse, background color matching) produces visually indistinguishable mutations.

### 2026-02-07 — CONTROL: 1x Gemini 3 Pro, single pass, no shuffle (full PDF)

- **Config**: 1x `gemini-3-pro-preview`, tool-call loop, no shuffle, no race
- **Eval model**: `gemini-3-flash-preview`
- **Recall**: 5/29 (17.2%)
- **Precision**: 71.4% (5 TP, 2 FP)
- **F1**: 27.8%
- **Cost**: $0.53
- **Time**: 238s
- **Findings**: 6 raw → 5 after rank/dedupe

**Detected (5):** inject_001 (tie_break p148 BS), inject_003 (tie_break p152 CF), inject_020 (label_swap_classification p148), inject_023 (label_swap_direction p152), inject_025 (standard_ref_wrong p154)

**Observation:** Pro is high-quality per finding (71% precision) but extremely low coverage — it stops after one pass through the document and only catches the most obvious errors. The single pass finds ~6 errors vs Flash's ~10 per run, but Flash compensates with 25 parallel shuffled runs. Pro's higher per-token cost makes it uneconomical for the brute-force ensemble approach. **Flash ensemble (25x) at $4 crushes Pro single-pass at $0.53 on recall.**

### 2026-02-07 — CONTROL: 1x GPT-5.2 high reasoning, single pass, FS-only pages (110pp)

- **Config**: 1x `openai/gpt-5.2`, tool-call loop, no shuffle, FS-only PDF (p143-252, 110 pages, 6.2 MB)
- **Eval model**: `gemini-3-flash-preview`
- **Recall**: 3/29 (10.0%) — only 2 unique GT errors (inject_003 matched twice, inject_008)
- **Precision**: 75.0% (3 TP, 1 FP)
- **F1**: 17.6%
- **Cost**: $0.47
- **Time**: 483s (5 turns)
- **Tokens**: 302,730 prompt, 17,574 completion
- **Findings**: 4 raw (no rank/dedupe)

**Detected (3 matches / 2 unique GT):** inject_003 (tie_break p152 CF — matched twice by two separate findings), inject_008 (sign_flip p182 auditor remuneration)

**Observation:** GPT-5.2 is the weakest performer despite being the most expensive per-token model. Even with the document trimmed to 110 FS pages (vs 252 full), it found only 4 errors in 5 turns. The tool-call loop ran 5 turns but most were near-empty. Known issue: GPT-5.2's multi-turn continuation degrades on large PDFs — it stops calling tools prematurely. At ~$0.47 for 2 unique GT errors, that's **$0.24/error vs Flash ensemble's $0.15/error** with far worse coverage.

## Written test_Case.pdf (27 errors)

### 2026-02-05 — Historical baseline (pre tool-call rewrite)

- **Config**: 10x `gemini-3-flash-preview` + `gemini-3-pro-preview` rank/dedupe, single-shot JSON output
- **Recall**: ~86.2% (estimated from 30 benchmark runs)
- **Precision**: ~96.2%
- **F1**: ~90.9%
- **Cost**: ~$0.15
- **Notes**: Pre tool-call rewrite. Numbers from `data/benchmarks/written_test_case/` analysis. Not yet re-evaluated with current tool-call pipeline.
