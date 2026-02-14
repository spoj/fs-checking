# fs_checking — Agent Instructions

**This repo is public.** All commits are pushed to GitHub. Do not commit proprietary
data, internal documents, or client financial statements. Only commit code, configs,
ground truth JSON for public/synthetic test sets, and experiment logs. When in doubt, ask.

## Project Structure

```
src/fs_checking/
  api.py              # OpenRouter client (retry, cache control, cost tracking)
  agent_core.py       # Async agent loop
  pdf_utils.py        # PDF→images, page shuffling (PyMuPDF)
  detection.py        # Prompts, detection modes
  eval.py             # LLM-based evaluation (match findings to ground truth)
  error_inject.py     # PDF error injection (5 mutation types)
  strategies/
    ensemble/ensemble.py  # PRIMARY — Nx parallel detection + rank/dedupe
    baseline/baseline.py  # Single LLM pass (reference only)
    swarm/                # Placeholder (NotImplementedError)

scripts/
  cli.py              # `fs-check` — main detection CLI
  inject_cli.py       # `fs-inject` — error injection CLI
  verify_errors.py    # Verify injected errors are detectable

samples/
  Written test_Case.pdf                    # Synthetic IFRS FS, 27 known errors
  Written_test_Case.ground_truth.json      # 27-error GT
  ar2019.pdf                               # Real 252-page annual report
  ar2019.injected_s42.pdf                  # ar2019 with 10 injected errors
  ar2019.injected_s42.ground_truth.json    # 10-error GT for injected version
```

**CLI entry points** (pyproject.toml):
- `fs-check` → `scripts/cli.py:main`
- `fs-inject` → `scripts/inject_cli.py:main`

**Managed with `uv`** — always use `uv run` to execute, `uv add` to add deps.

## How the Ensemble Strategy Works

`src/fs_checking/strategies/ensemble/ensemble.py` is the primary detection pipeline:

1. **Detection phase**: Launch N parallel Flash runs, each seeing pages in a different shuffled order. Each run uses a multi-turn tool-call loop — the model calls `log_issue()` incrementally as it finds errors. Conversation is prefix-stable for caching (append-only).

2. **Race pattern** (optional): Launch more runs than needed (`--launch N`), keep the first K (`-n K`) to finish, cancel the rest. Use `--stagger S` to spread start times.

3. **Rank/dedupe phase**: Gemini Pro takes all raw findings + the PDF, deduplicates and ranks them into high/medium/low priority.

**Key run command**:
```bash
uv run python -m fs_checking.strategies.ensemble.ensemble \
  samples/ar2019.injected_s42.pdf \
  -o samples/ar2019.injected_s42.result.json \
  -n 10 --launch 15 --stagger 5
```

## Evaluation Protocol

**CRITICAL: Always use LLM-based evaluation, never page-number matching.**

Page-based recall matching is unreliable — models often reference a different page than the injection page (e.g., the cross-reference page, or a nearby page). The eval.py LLM matcher handles this correctly by comparing error descriptions semantically.

### Running Evaluation

```bash
uv run python -m fs_checking.eval \
  samples/ar2019.injected_s42.ground_truth.json \
  samples/ar2019.injected_s42.result.json
```

This outputs precision/recall/F1, matched pairs, false positives, and missed errors.

### Eval Log Requirement

**Every evaluation run MUST be recorded in `EVAL_LOG.md`** with:
- Date, test set, configuration (model, num_runs, launch, stagger)
- Recall, precision, F1, cost, time
- Which GT errors were hit/missed
- Key observations

This is how we track progress and avoid repeating failed experiments.

### Test Sets

| Test Set | File | GT Errors | Description |
|---|---|---|---|
| Written test_Case | `samples/Written test_Case.pdf` | 27 | Synthetic IFRS FS with planted errors |
| ar2019 injected (seed 42) | `samples/ar2019.injected_s42.pdf` | 10 | Real 252-page AR + 10 injected errors |

## Key Learnings

### What Works

1. **Tool-call loop beats single-shot JSON.** The model calls `log_issue()` incrementally as it scans, producing prefix-stable conversations that benefit from caching. More reliable than asking for a JSON array at the end.

2. **Page shuffling is essential for diversity.** Each run sees pages in a different random order. Without this, parallel runs find the same errors. Shuffling is lossless (PyMuPDF page reorder), and document page numbers in headers are preserved so the model reports correct references.

3. **Flash is the best value.** Gemini 3 Flash via OpenRouter: ~136K prompt tokens for 252-page PDF, ~$0.06/turn with caching. GPT-5.2 costs 5-8x more per run with lower recall.

4. **OpenRouter `usage.cost` is authoritative.** Always use this for cost tracking, not naive `prompt_tokens × rate`. The naive calc overestimates by ~4x because it doesn't account for cached token discounts.

5. **More runs = better recall, diminishing returns.** 10 runs → 9/10 (90% recall, ~$0.60). 25 runs → 10/10 (100% recall, ~$5.04). The marginal cost per additional error found increases steeply. All recall numbers are from LLM-based eval (see eval protocol).

6. **Race pattern reduces latency without waste.** Launch 30, keep 25, cancel 5 slow ones. Stagger start times to avoid thundering herd at the API.

### What Doesn't Work

1. **Nudging.** Injecting "continue reviewing remaining pages" when the model stops calling tools. Tested extensively — doesn't improve recall, roughly doubles cost. The model's first pass captures what it's going to find.

2. **GPT-5.2 for large documents.** Multi-turn continuation dies with empty responses on 252-page PDFs. Works fine for small documents. Even with workarounds (low reasoning effort + nudges), Flash dominates on cost/recall.

3. **Page-number matching for evaluation.** Models report errors on nearby or cross-referenced pages, not necessarily the exact page of the mutation. Must use LLM-based semantic matching.

### API Gotchas

- **Empty response retry**: OpenRouter sometimes returns HTTP 200 with empty body (transient proxy issue). The client retries these automatically.
- **Deep copy for cache control**: `add_cache_control` must deep-copy messages (nested content dicts), not shallow copy.
- **`asyncio.gather` with `return_exceptions=True`**: Individual failed runs shouldn't kill the ensemble. Exceptions are caught per-task in the race loop.

## Error Injection Tool

`fs-inject` / `src/fs_checking/error_inject.py` — Injects known numerical errors into PDF text layers via PyMuPDF redact+stamp. 

**5 mutation types**: tie_break, magnitude, transposition, offset, sign_flip.

```bash
uv run fs-inject samples/ar2019.pdf -o samples/ar2019.injected.pdf --seed 42 --count 10
```

Generates a ground truth JSON compatible with eval.py alongside the mutated PDF.

## Infrastructure

- **Proxy**: `HTTPS_PROXY=http://dosg:8888` (Singapore, for Gemini geo-restriction from HK)
- **OpenRouter**: All models accessed through OpenRouter API
- **Gemini Flash**: ~136K prompt tokens for 252-page PDF, first turn ~$0.06 with caching
- **Timeout**: Long-running ensemble jobs need `timeout=1800.0` on the client
