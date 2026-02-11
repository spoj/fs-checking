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

### 2026-02-08 — 25x Flash, race 30/25, visual FS-only 150dpi (110 pages)

- **Config**: 25x `gemini-3-flash-preview` (launch 30, keep 25), tool-call loop, page shuffle, stagger 5s. Input: FS pages only (p143-252, 110 pages), rasterized 150dpi q70, 14.8 MB.
- **Eval model**: `gemini-3-flash-preview`
- **Recall**: 26/29 (83.9%)
- **Precision**: 81.2% (26 TP, 6 FP)
- **F1**: 82.5%
- **Cost**: $3.67
- **Time**: 435s
- **Raw findings**: 315 (39 from partial harvest)

**Detected (26/29):** Same 26 as 100dpi full-doc visual run, PLUS inject_002 (SOCIE tie-break) and inject_024 (restated label removal) — both caught thanks to 150dpi sharpness. Lost inject_022 (Gross profit→loss label) which 100dpi had caught.

**Missed (5/29):** inject_014 (magnitude ÷10 p246 — outside FS pages), inject_015 (note ref off-by-one), inject_017 (year swap column header), inject_021 (Due from→to), inject_022 (Gross profit→loss label — math caught but not label)

**Key takeaway:** 150dpi on 110 FS pages is the best visual-only config: higher recall than 100dpi full-doc (83.9% vs 81.2%), faster (435s vs 685s), cheaper ($3.67 vs $4.80). The 5 remaining misses are genuinely subtle presentation errors.

### 2026-02-08 — 25x Flash, race 30/25, visual FS-only 150dpi (110 pages) — RUN 2

- **Config**: Same as run 1.
- **Recall**: 28/29 (97.1%) — up from 83.9%
- **Precision**: 73.9% (34 TP, 12 FP)
- **F1**: 84.0%
- **Cost**: $3.87
- **Time**: 566s
- **Raw findings**: 332 (42 from partial harvest)
- **Unique after dedupe**: 46 (H:25 M:16 L:5) — vs 28 in r1

**Only missed inject_015 (note ref off-by-one Note 25→26).** Caught inject_014 (magnitude ÷10 p246) and inject_022 (Gross profit→loss label) which r1 missed.

**Observation:** Large run-to-run variance (83.9% vs 97.1% recall). The ranker kept 46 vs 28 unique findings — less aggressive merging in r2 preserved more distinct errors. This confirms the ranker's merge behavior is the main source of eval variance, not detection coverage.

### 2026-02-08 — 25x Flash, race 30/25, RING shuffle, visual FS-only 150dpi (110 pages)

- **Config**: 25x `gemini-3-flash-preview`, race 30/25, stagger 5s, `--shuffle-mode ring` (circular offset, randomized start per seed), rasterized 110 FS pages at 150dpi q70
- **Eval model**: `gemini-3-flash-preview`
- **Recall**: 23/29 (79.3%)
- **Precision**: 85.2% (23 TP, 4 FP)
- **F1**: 82.1%
- **Cost**: $3.31
- **Time**: 530s
- **Raw findings**: 354 (36 from partial harvest)
- **Unique after dedupe**: 19 (H:6 M:12 L:1)

**Detected (23/29):** inject_000, inject_001, inject_003, inject_004, inject_005, inject_006, inject_008, inject_009, inject_010, inject_011, inject_012, inject_013, inject_014, inject_016, inject_017, inject_018, inject_019, inject_020, inject_021, inject_022, inject_023, inject_026, inject_027

**Missed (6/29):** inject_002 (tie_break SOCIE), inject_007 (transposition p153), inject_015 (note ref off-by-one — always missed), inject_024 (restated label), inject_025 (HKFRS ref), inject_028 (year swap p156)

**Observation:** Ring shuffle significantly underperforms random shuffle on recall (79.3% vs 83.9%-97.1%). The ranker was also very aggressive — 354 raw → 19 unique (vs 28-46 for random). Key hypothesis: random shuffle's diversity comes from breaking adjacency — each run samples a genuinely different subset of pages early (where attention is strongest). Ring mode preserves adjacency so runs that start near each other see very similar page sequences, reducing effective diversity. The lower cost ($3.31 vs $3.67-$3.87) is just from fewer turns per run, not an advantage. **Verdict: ring shuffle is worse than random shuffle. Random remains the default.**

### 2026-02-08 — Raw vs Ranked variance decomposition

Ran LLM eval on raw findings (pre-ranker) for ring, random r1, and random r2 to isolate how much recall the ranker destroys.

| Run | Raw findings | Raw GT hit | Ranked findings | Ranked GT hit | Ranker lost | Reduction % |
|-----|-------------|------------|-----------------|---------------|-------------|-------------|
| random r1 | 315 | 28/29 | 28 | 26/29 | 2 errors | 91% |
| random r2 | 332 | 28/29 | 46 | 28/29 | 0 errors | 86% |
| ring | 354 | 27/29 | 19 | 23/29 | 4 errors | 95% |

**Key finding: the ranker is the dominant source of recall variance, not detection.**

- Detection phase: ring loses only 1 GT error vs random (27 vs 28 raw). All three runs detect 27-28/29 before ranking.
- Ranker phase: ring loses 4 GT errors during dedup (27→23), random loses 0-2 (28→26 or 28→28). The ranker contributes ~80% of the recall gap.
- Ring's higher reduction rate (95% vs 86-91%) confirms adjacency preservation produces more similar findings across runs, causing over-aggressive merging.
- Random r2 ranker kept 46 findings (less aggressive) and lost 0 GT errors. Random r1 kept 28 (more aggressive) and lost 2. Ranker aggressiveness is the single biggest lever on recall.

### 2026-02-08 — Recall scaling curve: raw recall vs. number of detectors

Subsampled 5, 10, 15, 20, 25 runs from random r1 and r2 raw findings (same seed=42 subsample order). Evaluated each subset against GT with LLM eval. All numbers are **raw recall** (pre-ranker).

**NOTE: Previous eval recall calculations were buggy** — `TP/(TP+FN)` counted match pairs, not unique GT. Fixed to `unique_gt_matched / total_gt`. Some historical recall %s in earlier entries are inflated (GT hit counts were always correct).

| N runs | r1 GT hit | r1 recall | r2 GT hit | r2 recall | avg recall |
|--------|-----------|-----------|-----------|-----------|------------|
| 5      | 21/29     | 72.4%     | 17/29     | 58.6%     | 65.5%      |
| 10     | 26/29     | 89.7%     | 25/29     | 86.2%     | 87.9%      |
| 15     | 28/29     | 96.6%     | 26/29     | 89.7%     | 93.1%      |
| 20     | 28/29     | 96.6%     | 27/29     | 93.1%     | 94.8%      |
| 25     | 28/29     | 96.6%     | 28/29     | 96.6%     | 96.6%      |

**r1 misses by N:** n5: inject_009,010,014,015,017,021,024,025 (8) | n10: inject_015,017,021 (3) | n15+: inject_015 only

**r2 misses by N:** n5: inject_002,007,008,010,015,017,021,023,024,025,027,028 (12) | n10: inject_007,015,017,025 (4) | n15: inject_007,015,025 (3) | n20: inject_015,025 (2) | n25: inject_015 only

**Plot**: `samples/recall_vs_detectors.png`

**Key observations:**
1. Classic log curve — steep gains 5→15, diminishing 15→25. Both converge to 28/29.
2. **15 runs is the sweet spot** for cost/recall: avg 93.1% raw recall, captures most of the curve.
3. **inject_015 (Note 25→26 off-by-one) is a model blind spot** — never detected in any run across either set (0/60 runs). This is not a coverage problem but a detection capability limit.
4. **High variance at n=5**: r1 hit 21/29 vs r2 hit 17/29 (seed luck). Stabilizes by n=15.
5. **The ranker is the bottleneck, not detection.** At n=25, both sets achieve 28/29 raw recall. But the ranker previously dropped this to 26/29 (r1) or kept it at 28/29 (r2) depending on merge aggressiveness. Ranker prompt has been updated to be less aggressive; ranker no longer receives the PDF (prevents it from second-guessing detectors).

### 2026-02-08 — 25x Flash, race 30/25, revised ranker (no PDF, softer prompt, GPT-5.2), visual FS-only 150dpi

- **Config**: 25x `gemini-3-flash-preview`, race 30/25, stagger 5s, random shuffle, rasterized 110 FS pages at 150dpi q70. Ranker: `openai/gpt-5.2`, no PDF attached, softer merge prompt.
- **Eval model**: `gemini-3-flash-preview`
- **Recall**: 25/29 (86.2%)
- **Precision**: 55.6% (25/45 model findings correct)
- **F1**: 67.6%
- **Cost**: $3.42
- **Time**: 837s
- **Raw findings**: 224 (7 from partial harvest) from only 22 productive runs
- **Unique after dedupe**: 49 (H:20 M:29 L:0)

**Detected (25/29):** inject_000,001,003,004,005,006,007,008,009,010,011,012,013,016,018,019,020,021,022,023,024,025,026,027,028

**Missed (4/29):** inject_002 (SOCIE tie-break), inject_014 (magnitude ÷10 p246), inject_015 (note ref off-by-one — always missed), inject_017 (BS column header year swap)

**Issues identified:**
1. **4 empty runs claimed race slots.** Runs 2, 16, 25, 28 completed in <40s with 0 findings — likely API returned a non-tool-call response immediately. These took 4 of the 25 keeper slots, reducing effective detection runs to ~21. Need to filter out zero-finding runs from the race or not count them as "complete".
2. **GPT-5.2 ranker kept 49 findings (good — less aggressive)** but many FPs are cascade effects (secondary footing errors caused by primary injections). Precision dropped (55.6% vs previous 73-85%) but this is acceptable — the goal is recall.
3. **Lower raw count (224 vs 315-354)** due to empty runs reducing effective coverage.

**Verdict:** The ranker changes (no PDF + softer prompt) work as intended — 49 kept vs 19 previously. But the empty-run problem is a new regression that must be fixed before re-measuring. Need to add a minimum-findings threshold to count a run as "completed" in the race loop.

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

### 2026-02-08 — 5x GPT-5.2 ensemble, race 5/5, visual FS-only 150dpi (110 pages)

- **Config**: 5x `openai/gpt-5.2` (launch 5, keep 5), tool-call loop, random shuffle, stagger 3s, rasterized 110 FS pages at 150dpi q70. Ranker: `openai/gpt-5.2`.
- **Purpose**: Fair comparison of GPT-5.2 as detector — previous tests used old prompt and single runs.
- **Eval model**: `gemini-3-flash-preview`
- **Recall**: 15/29 (51.7%)
- **Precision**: 100% (32/32 raw findings correct, 18/18 ranked findings correct)
- **F1**: 68.2%
- **Cost**: $1.63
- **Time**: 973s
- **Raw findings**: 32 (4 productive runs: 9, 9, 8, 6 findings; 1 empty run with 0 findings)
- **Unique after dedupe**: 18 (H:10 M:8 L:0)
- **Ranker lost 0 GT errors** (15/29 raw → 15/29 ranked)

**Detected (15/29):** inject_003 (tie_break CF), inject_005 (offset BS), inject_006 (magnitude segment), inject_007 (transposition CF), inject_009 (offset dividends), inject_010 (tie_break related co), inject_011 (sign_flip cash), inject_012 (offset maturity), inject_013 (transposition deferred tax), inject_016 (year_swap PL title), inject_018 (currency_swap PL), inject_020 (label_swap_classification BS), inject_023 (label_swap_direction CF), inject_025 (standard_ref HKFRS), inject_026 (label_swap_sign_word receivables)

**Missed (14/29):** inject_000 (tie_break PL), inject_001 (tie_break BS), inject_002 (tie_break SOCIE), inject_004 (offset shareholders funds), inject_008 (sign_flip auditor), inject_014 (magnitude fin summary), inject_015 (note_ref off-by-one), inject_017 (year_swap column header), inject_019 (Continuing→Discontinued), inject_021 (Due from→to), inject_022 (Gross profit→loss), inject_024 (restated label removal), inject_027 (currency_swap p179), inject_028 (year_swap p156)

**Comparison with Flash at same N=5 (from scaling curve):**

| Metric | GPT-5.2 (5 runs) | Flash (5 runs avg) |
|--------|-------------------|--------------------|
| Raw GT hit | 15/29 | 19/29 (65.5%) |
| Raw findings | 32 | ~65 (est.) |
| Findings/run | 8.0 | ~13 |
| Cost | $1.63 | ~$0.75 |
| Cost/GT hit | $0.11 | $0.04 |
| Precision | 100% | ~80% |
| Time | 973s | ~400s |

**Key observations:**
1. **GPT-5.2 is dramatically better than previous test** (15/29 vs 2/29) — the strengthened prompt made a huge difference. Previous test used old prompt + no shuffle + native PDF.
2. **But Flash still dominates on recall/dollar.** At N=5, Flash averages 19/29 raw vs GPT-5.2's 15/29. Flash is 4x cheaper per GT error found ($0.04 vs $0.11).
3. **GPT-5.2 has perfect precision** (0 FP across 32 findings) vs Flash's ~80%. Every single finding matches a real injected error. GPT-5.2 is a more cautious, higher-quality detector — it just doesn't look at enough of the document.
4. **Per-run yield**: 8.0 findings/run (GPT-5.2) vs ~13/run (Flash). GPT-5.2 completes in 2 tool-call turns vs Flash's 5-8, confirming it stops scanning earlier.
5. **1/5 empty run** (20%) — same pattern as Flash. The model sometimes returns a non-tool-call response immediately.
6. **Numeric recall**: 9/15 (60%) — misses several tie-break and offset errors that Flash catches. **Text recall**: 6/14 (43%) — catches the obvious ones (year swap in title, currency swap, section mislabel, HKFRS ref) but misses subtler ones (direction words, restated labels).
7. **At $1.63 for 5 runs, scaling to 25 runs would cost ~$8** for an estimated ~22-24/29 raw recall (extrapolating the independent model). Flash achieves 28/29 raw at ~$4. Not competitive.

**Verdict:** The strengthened prompt rescued GPT-5.2 from "useless" (2/29) to "decent" (15/29), confirming prompt engineering matters across models. But Flash remains 4x more cost-effective for detection. GPT-5.2's strength is precision (100% vs ~80%) — potentially useful as a validator/filter, but not as the primary detector.

### 2026-02-08 — 5x GPT-5.2, STRUCTURED prompt, 45-min timeout, visual FS-only 150dpi (110 pages)

- **Config**: 5x `openai/gpt-5.2` (launch 5, keep 5), tool-call loop, random shuffle, stagger 3s, rasterized 110 FS pages at 150dpi q70. Ranker: `openai/gpt-5.2`. **Structured multi-pass prompt** (3-pass: face statements → notes → presentation sweep). Timeout 2700s (45 min).
- **Purpose**: Test whether a prescriptive multi-pass prompt can fix GPT-5.2's premature stopping (2 turns/run baseline).
- **Eval model**: `gemini-3-flash-preview`
- **Raw Recall**: 16/29 (55.2%)
- **Raw Precision**: 94.9% (37/39 matches, 2 FP)
- **Ranked Recall**: 17/29 (58.6%)
- **Ranked Precision**: 100% (20/20 model findings correct)
- **F1 (ranked)**: 73.9%
- **Cost**: $5.97
- **Time**: 2093s (35 min)
- **Raw findings**: 40 (all 5 runs productive: 14, 11, 8, 5, 2 findings)
- **Unique after dedupe**: 20 (H:8 M:12 L:0)
- **Per-run turns**: run_3=15, run_2=4, run_4=3, run_1=3, run_5=2 (highly variable)

**Detected (17/29, ranked):** inject_000 (tie_break PL), inject_001 (tie_break BS), inject_002 (tie_break SOCIE — ranked only), inject_003 (tie_break CF), inject_005 (offset BS), inject_009 (offset dividends), inject_010 (tie_break related co), inject_011 (sign_flip cash), inject_012 (offset maturity), inject_013 (transposition deferred tax), inject_016 (year_swap PL title), inject_018 (currency_swap PL), inject_019 (Continuing→Discontinued), inject_022 (Gross profit→loss label), inject_023 (outflow direction), inject_026 (Receivables→Payables), inject_027 (currency_swap segment)

**Missed (12/29):** inject_004 (offset shareholders), inject_006 (magnitude segment), inject_007 (transposition SOCIE), inject_008 (sign_flip auditor), inject_014 (magnitude fin summary), inject_015 (note_ref off-by-one), inject_017 (year_swap column header), inject_020 (label_swap_classification BS), inject_021 (Due from→to), inject_024 (restated label removal), inject_025 (standard_ref HKFRS), inject_028 (year_swap p156)

**Comparison with baseline (default prompt, same config):**

| Metric | Structured prompt | Default prompt | Delta |
|--------|-------------------|----------------|-------|
| Raw GT hit | 16/29 (55.2%) | 15/29 (51.7%) | +1 |
| Ranked GT hit | 17/29 (58.6%) | 15/29 (51.7%) | +2 |
| Raw findings | 40 | 32 | +25% |
| Findings/run | 8.0 | 8.0 | same |
| Max turns/run | 15 | 3 | +400% |
| Empty runs | 0/5 | 1/5 | better |
| Cost | $5.97 | $1.63 | +266% |
| Time | 2093s | 973s | +115% |
| Precision | 100% | 100% | same |

**Newly detected vs baseline (+4):** inject_000 (tie_break PL), inject_001 (tie_break BS), inject_002 (SOCIE tie-break), inject_022 (Gross profit→loss label)
**Lost vs baseline (−2):** inject_006 (magnitude segment), inject_025 (HKFRS ref) — sampling variance

**Key observations:**
1. **The structured prompt partially worked** — run_3 did 15 turns (vs max 3 baseline), confirming the multi-pass instructions kept GPT-5.2 going. But other runs (1, 4, 5) still stopped at 2-3 turns. The effect is inconsistent.
2. **Marginal recall improvement at 3.6x cost.** +2 ranked GT errors (17 vs 15) but cost jumped from $1.63 to $5.97. The extra turns generate more tokens but don't proportionally find more errors.
3. **Per-run yield unchanged at 8.0 findings/run** — even run_3 with 15 turns only found 14 errors (0.93/turn), same average as baseline's 8/run in 2 turns (4.0/turn). More turns ≠ more unique errors proportionally.
4. **Precision maintained at 100%** — GPT-5.2 never produces false positives regardless of prompt style. This is its consistent strength.
5. **The 4 face-statement errors (inject_000, 001, 002, 022)** that the structured prompt caught were explicitly targeted by Pass 1's arithmetic walkthrough. But Pass 3's presentation sweep added nothing vs baseline — direction words (inject_021), restated labels (inject_024), HKFRS refs (inject_025) still missed.

**Verdict:** The structured multi-pass prompt is not worth the 3.6x cost premium for +2 recall. GPT-5.2's limitation is not coverage (it can be forced to do more turns) but per-page detection quality — it misses errors even when looking at the right page. Flash at 5 runs costs $0.75 and achieves ~19/29 raw recall. **GPT-5.2 is definitively worse than Flash for detection, regardless of prompt style.**

## ar2019_fs.injected (unrendered markdown→PDF, 31 errors: 11 easy, 10 medium, 10 hard)

### Test Set Description

Source: `ar2019_fs.md` — 109-page FS section of ar2019 annual report, converted to GFM markdown via `unrender.py` (Gemini Flash, sliding window), then rendered to PDF via Playwright (66 pages). 30 hand-crafted errors injected via `inject_md_errors.py` + 1 pre-existing transcription error (LLM wrote "Due from" instead of "Due to" in current liabilities — confirmed original PDF is correct).

Difficulty tiers:
- **Easy (11)**: Obvious arithmetic breaks on face statements (P&L, BS, CF, OCI). Subtotals that don't tie to line items on the same page.
- **Medium (10)**: Cross-reference breaks between notes and face statements. Rollforward errors, ageing totals, segment ties.
- **Hard (10)**: Subtle detail-level errors in note tables. Single columns in multi-dimensional tables, currency breakdowns, share capital rollforwards, company-level BS.

### 2026-02-11 — Control run (clean unrendered PDF, no injected errors)

- **Config**: 25x `gemini-3-flash-preview` (launch 30, keep 25), stagger 5s, `gpt-5.2` rank/dedupe
- **Input**: `ar2019_fs.unrendered.pdf` (66 pages, clean)
- **Findings**: 45 (H:29, M:16)
- **Cost**: $2.90
- **Time**: 326s
- **Notes**: These 45 findings are the baseline noise — things the detector flags on the clean unrendered document. Includes legitimate observations about the transcription (e.g., "Due from" labeling error) and false alarms about formatting/presentation.

### 2026-02-11 — 25x Flash, race 30/25 (31 GT errors)

- **Config**: 25x `gemini-3-flash-preview` (launch 30, keep 25), stagger 5s, `gpt-5.2` rank/dedupe
- **Input**: `ar2019_fs.injected.pdf` (66 pages, 31 errors)
- **Eval model**: `gemini-3-flash-preview`
- **Recall**: 31/31 (100%)
- **Precision**: 72.1% (31/43 model findings correct)
- **F1**: 83.8%
- **FP**: 12 | **FN**: 0
- **Cost**: $3.05
- **Time**: 859s
- **Raw findings**: 249 (25 runs kept, 5 cancelled with 22 partial findings harvested)
- **Unique after dedupe**: 227 (H:88, M:139)

**Recall by difficulty tier:**

| Tier | Errors | Detected | Recall | Avg matches per error |
|------|--------|----------|--------|-----------------------|
| Easy | 11 | 11 | 100% | 2.5 |
| Medium | 10 | 10 | 100% | 1.3 |
| Hard | 10 | 10 | 100% | 1.1 |
| **Total** | **31** | **31** | **100%** | **1.6** |

**Detection redundancy drops sharply with difficulty**: Easy errors get flagged by ~2.5 deduplicated findings each (multiple cross-checks catch them). Hard errors average 1.1 — barely caught, usually by exactly one finding. This means hard errors are fragile — with fewer runs they'd likely be missed.

**False positives (12)**: All are observations about the unrendered document itself, not hallucinations:
- PPE HKFRS 16 restatement presentation (original is correct — false alarm)
- SOCE formatting/labeling (false alarm)
- Share premium vesting cross-statement tie (false alarm)
- Pension rollforward missing narrative line (false alarm)
- Purchase consideration 2018 rollforward (false alarm)
- Fair value Level 3 rollforward 2018 (false alarm)
- Segment turnover sub-breakdown tie (false alarm)
- Lease liability presentation classification (false alarm)
- Parent company reserves missing dividend line (false alarm)
- Currency translation reserves 2018 tie (false alarm)
- Company retained earnings rollforward (false alarm)
- SOCE 2018 header incomplete (false alarm)

Verified against original ar2019.pdf: all 12 FPs are false alarms by the detector (original document is correct). Only 1 pre-existing transcription error exists (the "Due from" labeling, now in GT as inject_031).

**Key observations:**
1. **100% recall on all 30 hand-crafted errors** confirms the ensemble approach works on markdown→PDF roundtripped documents.
2. **Hard errors are detectable but fragile** — avg 1.1 matches means they rely on a single detector run catching them. Scaling to fewer runs would degrade hard-tier recall first.
3. **The unrendered PDF is a cleaner test bed** than the original PDF — only 12 FPs vs 18+ on the original ar2019 mixed injection. The markdown→PDF pipeline produces consistent, text-layer-clean documents.
4. **Cost efficiency**: $3.05 for 31/31 recall = $0.10 per detected error. Comparable to the original ar2019 runs.

### 2026-02-12 — 25x Flash + GPT-5.2 validator (vision: Flash), race 30/25

- **Config**: 25x `gemini-3-flash-preview` (launch 30, keep 25), stagger 5s, `gpt-5.2` validator with `gemini-3-flash-preview` vision tool, reasoning high (validator), reasoning low (vision)
- **Input**: `ar2019_fs.injected.pdf` (66 pages, 31 errors)
- **Eval model**: `gemini-3-flash-preview`
- **Recall**: 30/31 (96.8%)
- **Precision**: 75.0% (30/40 model findings correct)
- **F1**: 84.5%
- **FP**: 10 | **FN**: 1
- **Cost**: $3.45
- **Time**: 1613s (27 min)
- **Raw findings**: 201 (25 runs kept, 5 cancelled with partial harvest)
- **Unique after validation**: 40 (H:19, M:18, L:3)
- **Validation details**: 198 verdicts, 35 vision calls, 15 turns

**Missed (1/31):**
- inject_024: Note 21 receivables RMB currency transposition (95,042→95,024). Hard tier. An 18-unit difference in a multi-currency breakdown table with 8 rows.

**False positives (10):**
- cf_total_increase_mismatch, cash_flow_net_increase_calculation_error, cash_gen_from_ops_reconciliation_mismatch — cascade/duplicate CF findings
- due_from_related_total_crossfoot_mismatch — duplicate of inject_014
- pl_total_margin_mismatch — cascade from inject_001 (gross profit)
- ppe_2019_furniture_nbv_mismatch — cascade from inject_029 (depreciation)
- note_ties_intangibles_accum_amort_mismatch — cascade from inject_012
- note_28_maturity_table_mismatch — cascade from inject_017
- net_pension_liability_rollforward_error_2018 — historical tie-in error (not in GT)
- financial_summary_duplicate_2018_labels — formatting observation (not an error)

**Comparison with text-only ranker (2026-02-11):**

| Metric | Text-only Ranker | GPT-5.2 Validator |
|--------|------------------|-------------------|
| Recall | 31/31 (100%) | 30/31 (96.8%) |
| Precision | 72.1% (31/43) | 75.0% (30/40) |
| F1 | 83.8% | 84.5% |
| FP | 12 | 10 |
| FN | 0 | 1 |
| Cost | $3.05 | $3.45 |
| Time | 859s | 1613s |
| Unique findings | 43 | 40 |

**Key observations:**
1. **GPT-5.2 validator lost 1 error (inject_024) that text-only ranker kept.** The validator's vision tool (Flash at low reasoning) could not confirm the 18-unit RMB transposition — likely Flash read the number as expected (95,042) instead of the actual mutated value (95,024), causing the validator to reject it.
2. **Slightly better precision** (75% vs 72%) — the validator correctly rejected 2 extra FPs that the text-only ranker kept. But the net effect is marginal.
3. **Nearly 2x slower** (1613s vs 859s) due to 35 vision calls, each sending the full 66-page PDF to Flash.
4. **Cost comparable** ($3.45 vs $3.05) — the vision calls add ~$0.40. Most cost is still in phase 1 detection.
5. **The previous Opus 4.6 validator result (98.8% F1, 100% recall, 97.7% precision) was with the audit-primed Flash system prompt + low reasoning fix.** This run uses the same fix, but GPT-5.2 as validator is less thorough than Opus — it made fewer vision calls and missed the subtlest error.

**Verdict:** GPT-5.2 validator does not improve on the text-only ranker for this test set. The recall regression (100%→96.8%) and 2x latency penalty are not justified by the marginal precision gain. For cost-effective runs, the text-only ranker remains the best phase 2 option. Opus 4.6 validator is the quality ceiling but at $13+ cost.

### 2026-02-12 — 25x Flash + Opus 4.6 validator (old design, vision: Flash), raw findings reuse

- **Config**: Reused 249 raw findings from the GPT-5.2 old-design validator run. `anthropic/claude-opus-4.6` validator with `gemini-3-flash-preview` vision tool, reasoning high (validator), reasoning low (vision). Old-design `validate_finding()` tool + `vision()` tool + verdict state machine.
- **Input**: `ar2019_fs.injected.pdf` (66 pages, 31 errors)
- **Eval model**: `gemini-3-flash-preview`
- **Recall**: 31/31 (100%)
- **Precision**: 97.7% (42/43 model findings correct)
- **F1**: 98.8%
- **FP**: 1 | **FN**: 0
- **Cost**: $13.42 (validator phase only)
- **Time**: 2327s (39 min)
- **Unique after validation**: 43 (H:10, M:21, L:12)

**FP (1):** note_21_incorrect_table_placement — layout observation about a payables aging table appearing in the receivables note section. Not an error.

**Key observations:**
1. **100% recall, 97.7% precision** — Opus 4.6 found all 31 GT errors including inject_024 (RMB currency transposition) which GPT-5.2 validator missed.
2. **43 output findings** (vs GPT-5.2's 35) — Opus is more thorough, keeping more granular findings. Many GT errors get matched by 2 separate findings.
3. **$13.42 is very expensive** for the validator phase alone. Total pipeline cost would be ~$16+ ($3 detection + $13 validation). Not practical for routine use.
4. **This is the quality ceiling** — best F1 (98.8%) achieved on this test set by any configuration.

### 2026-02-12 — Simplified validator: GPT-5.2 + vision (Flash), race 30/25

- **Config**: 25x `gemini-3-flash-preview` (launch 30, keep 25), stagger 5s. **Simplified validator**: `openai/gpt-5.2` validator with `gemini-3-flash-preview` vision tool. No `validate_finding()` tool — validator uses `vision()` to verify, then outputs a single JSON array. No verdict state machine, no priority tiers, no finding IDs.
- **Input**: `ar2019_fs.injected.pdf` (66 pages, 31 errors)
- **Eval model**: `gemini-3-flash-preview`
- **Recall**: 30/31 (96.8%)
- **Precision**: 93.5% (29/31 model findings correct)
- **F1**: 95.1%
- **FP**: 2 | **FN**: 1
- **Cost**: $3.14 (total, detection + validation)
- **Time**: 1240s (21 min)
- **Raw findings**: 270 (25 runs)
- **Output findings**: 31

**Missed (1/31):**
- inject_015: Trade receivables ageing total off by 9 (1,017,198 vs 1,017,189). Hard tier. 0/25 detection runs found it — consistent model blind spot.

**FP (2):**
- issue_029: Intangible assets cost rollforward from 2018 (historical, not in GT)
- issue_031: Blank table observation in impairment test section (presentation, not an error)

**Comparison with old-design GPT-5.2 validator (same test set):**

| Metric | Old design (GPT-5.2) | Simplified (GPT-5.2) | Delta |
|--------|---------------------|----------------------|-------|
| Recall | 87.1% (27/31) | 96.8% (30/31) | +9.7pp |
| Precision | 97.1% (34/35) | 93.5% (29/31) | -3.6pp |
| F1 | 91.8% | 95.1% | +3.3pp |
| FP | 1 | 2 | +1 |
| FN | 4 | 1 | -3 |
| Cost | $1.30 (val only) | $3.14 (total) | — |
| Time | 1456s (val only) | 1240s (total) | — |
| Output findings | 35 | 31 | -4 |

**Key observations:**
1. **Massive recall improvement** (87.1% → 96.8%) — the old-design validator lost 4 GT errors (inject_013, 017, 023, 024) through the verdict state machine's merge/reject logic. The simplified design preserves them.
2. **The old-design bug** where duplicate finding IDs caused verdict overwriting is eliminated — there are no IDs or verdicts in the simplified design.
3. **Slightly worse precision** (97.1% → 93.5%) — the simplified validator keeps 2 FPs that the old design's stricter rejection logic would have filtered. Acceptable tradeoff for +3 errors detected.
4. **Net F1 gain of 3.3pp** — the recall improvement outweighs the precision loss.

### 2026-02-12 — Simplified validator: Opus 4.6 + vision (Flash), raw findings reuse

- **Config**: Reused 270 raw findings from the simplified GPT-5.2 run. `anthropic/claude-opus-4.6` validator with `gemini-3-flash-preview` vision tool. Same simplified design (no `validate_finding()`, single JSON array output).
- **Input**: `ar2019_fs.injected.pdf` (66 pages, 31 errors)
- **Eval model**: `gemini-3-flash-preview`
- **Recall**: 30/31 (96.8%)
- **Precision**: 100% (35/35 model findings correct)
- **F1**: 98.4%
- **FP**: 0 | **FN**: 1
- **Cost**: $1.92 (validator phase only; total pipeline ~$4.95)
- **Time**: 1375s (23 min)
- **Output findings**: 35

**Missed (1/31):**
- inject_015: Same as GPT-5.2 simplified — detection-phase miss (0/25 runs), not a validator failure.

**Comparison with old-design Opus 4.6 validator (same test set):**

| Metric | Old design (Opus) | Simplified (Opus) | Delta |
|--------|-------------------|-------------------|-------|
| Recall | 100% (31/31) | 96.8% (30/31) | -3.2pp |
| Precision | 97.7% (42/43) | 100% (35/35) | +2.3pp |
| F1 | 98.8% | 98.4% | -0.4pp |
| FP | 1 | 0 | -1 |
| FN | 0 | 1 | +1 |
| Cost (val only) | $13.42 | $1.92 | **-85%** |
| Time | 2327s | 1375s | -41% |
| Output findings | 43 | 35 | -8 |

**Key observations:**
1. **7x cheaper** ($1.92 vs $13.42) — the simplified design requires far fewer tokens because the validator outputs a compact JSON array instead of calling `validate_finding()` for each raw finding individually.
2. **The 1 FN (inject_015) is a detection miss**, not a validator miss — 0/25 runs found it. The old-design run also used different raw findings (249 vs 270) from a different detection batch, which happened to include inject_015.
3. **Perfect precision** (100%) — Opus doesn't produce any false positives with the simplified design. The old design's 1 FP (table placement observation) was probably introduced by the vision tool + validate_finding interaction.
4. **F1 essentially tied** (98.8% vs 98.4%) — the old design's recall advantage comes from different detection inputs, not from better validation.

### Validator architecture comparison (all runs on ar2019_fs.injected, 31 GT errors)

| Config | Recall | Precision | F1 | FP | FN | Cost (total) | Time |
|--------|--------|-----------|-----|----|----|-------------|------|
| Text-only ranker (GPT-5.2) | 100% | 72.1% | 83.8% | 12 | 0 | $3.05 | 859s |
| Old-design GPT-5.2 validator | 87.1% | 97.1% | 91.8% | 1 | 4 | ~$4.35 | 1456s |
| Old-design Opus 4.6 validator | 100% | 97.7% | 98.8% | 1 | 0 | ~$16.45 | 2327s |
| **Simplified GPT-5.2 validator** | **96.8%** | **93.5%** | **95.1%** | **2** | **1** | **$3.14** | **1240s** |
| **Simplified Opus 4.6 validator** | **96.8%** | **100%** | **98.4%** | **0** | **1** | **~$4.95** | **1375s** |

**Verdict:** The simplified validator architecture is strictly better than the old design:
- GPT-5.2 simplified: +3.3pp F1 vs old GPT-5.2, at similar cost
- Opus simplified: -0.4pp F1 vs old Opus (detection variance, not validator), at **7x lower cost**
- The simplified design eliminates the verdict state machine bug and is ~50% less code

**Best configurations by use case:**
- **Cost-optimized**: Text-only ranker ($3.05, 100% recall, 72% precision). Best when FP tolerance is high.
- **Balanced**: Simplified GPT-5.2 validator ($3.14, 96.8% recall, 93.5% precision). Near-identical cost to text-only, much better precision.
- **Quality ceiling**: Simplified Opus 4.6 validator (~$4.95, 96.8% recall, 100% precision). Perfect precision, moderate premium.

## Written test_Case.pdf (27 errors)

### 2026-02-05 — Historical baseline (pre tool-call rewrite)

- **Config**: 10x `gemini-3-flash-preview` + `gemini-3-pro-preview` rank/dedupe, single-shot JSON output
- **Recall**: ~86.2% (estimated from 30 benchmark runs)
- **Precision**: ~96.2%
- **F1**: ~90.9%
- **Cost**: ~$0.15
- **Notes**: Pre tool-call rewrite. Numbers from `data/benchmarks/written_test_case/` analysis. Not yet re-evaluated with current tool-call pipeline.
