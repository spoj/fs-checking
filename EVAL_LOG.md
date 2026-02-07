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

## Written test_Case.pdf (27 errors)

### 2026-02-05 — Historical baseline (pre tool-call rewrite)

- **Config**: 10x `gemini-3-flash-preview` + `gemini-3-pro-preview` rank/dedupe, single-shot JSON output
- **Recall**: ~86.2% (estimated from 30 benchmark runs)
- **Precision**: ~96.2%
- **F1**: ~90.9%
- **Cost**: ~$0.15
- **Notes**: Pre tool-call rewrite. Numbers from `data/benchmarks/written_test_case/` analysis. Not yet re-evaluated with current tool-call pipeline.
