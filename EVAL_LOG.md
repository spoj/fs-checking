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

## Written test_Case.pdf (27 errors)

### 2026-02-05 — Historical baseline (pre tool-call rewrite)

- **Config**: 10x `gemini-3-flash-preview` + `gemini-3-pro-preview` rank/dedupe, single-shot JSON output
- **Recall**: ~86.2% (estimated from 30 benchmark runs)
- **Precision**: ~96.2%
- **F1**: ~90.9%
- **Cost**: ~$0.15
- **Notes**: Pre tool-call rewrite. Numbers from `data/benchmarks/written_test_case/` analysis. Not yet re-evaluated with current tool-call pipeline.
