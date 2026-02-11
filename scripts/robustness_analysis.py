"""
Analyze per-error detection robustness across ensemble runs.

Strategy: For each raw finding, match it to a GT error using the injection details
(mutated values, original values, pages, descriptions). Then count per-run detection
frequency for each GT error.
"""

import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

SAMPLES = Path(__file__).parent.parent / "samples"


def load_json(path):
    with open(path) as f:
        return json.load(f)


def build_gt_matchers(gt):
    """Build matching rules from GT injection details."""
    matchers = []
    for detail in gt["_injection_details"]:
        gid = detail["id"]
        page = detail["page"]
        mutation = detail["mutation_kind"]
        orig_text = detail.get("original_text", "")
        mut_text = detail.get("mutated_text", "")
        orig_val = detail.get("original_value", 0)
        mut_val = detail.get("mutated_value", 0)

        matchers.append(
            {
                "id": gid,
                "page": page,
                "mutation": mutation,
                "orig_text": orig_text,
                "mut_text": mut_text,
                "orig_val": orig_val,
                "mut_val": mut_val,
            }
        )
    return matchers


def finding_matches_gt(finding, matcher):
    """Check if a raw finding matches a GT error based on values and context."""
    desc = finding.get("description", "").lower()
    f_page = finding.get("page", 0)
    f_actual = finding.get("actual")
    f_expected = finding.get("expected")

    gid = matcher["id"]
    mut_val = matcher["mut_val"]
    orig_val = matcher["orig_val"]
    mutation = matcher["mutation"]

    # Page proximity helper: original page or rasterized page (FS starts at p143)
    gt_page = matcher["page"]
    rast_page = gt_page - 142  # approximate rasterized page number

    def page_close(tolerance=5):
        return (
            abs(f_page - gt_page) <= tolerance or abs(f_page - rast_page) <= tolerance
        )

    # Strategy depends on mutation type

    if mutation in ("tie_break", "offset", "magnitude", "transposition"):
        # Numeric mutations: match by value
        # The finding should mention the mutated value (as actual) or the original (as expected)
        vals_to_check = set()
        if f_actual is not None:
            if isinstance(f_actual, (int, float)):
                vals_to_check.add(abs(f_actual))
            elif isinstance(f_actual, str):
                try:
                    vals_to_check.add(abs(float(f_actual.replace(",", ""))))
                except:
                    pass
        if f_expected is not None:
            if isinstance(f_expected, (int, float)):
                vals_to_check.add(abs(f_expected))
            elif isinstance(f_expected, str):
                try:
                    vals_to_check.add(abs(float(f_expected.replace(",", ""))))
                except:
                    pass

        # Also extract numbers from description
        numbers_in_desc = re.findall(r"[\d,]+", desc)
        for n in numbers_in_desc:
            try:
                vals_to_check.add(abs(float(n.replace(",", ""))))
            except:
                pass

        # Match if both the mutated and original values appear (exact or approximate)
        has_mut = abs(mut_val) in vals_to_check
        has_orig = abs(orig_val) in vals_to_check

        # Also check approximate match (within 0.1% — handles rounding in model output)
        if not has_orig and abs(orig_val) > 0:
            for v in vals_to_check:
                if abs(v - abs(orig_val)) / abs(orig_val) < 0.001:
                    has_orig = True
                    break
        if not has_mut and abs(mut_val) > 0:
            for v in vals_to_check:
                if abs(v - abs(mut_val)) / abs(mut_val) < 0.001:
                    has_mut = True
                    break

        if has_mut and has_orig:
            return True
        # For some cases, only the mutated value appears with nearby page
        # Use wider tolerance to handle rasterized page numbering (original p143-252 → rast p1-110)
        if has_mut and page_close():
            # Additional context checks
            return True

    elif mutation == "sign_flip":
        # inject_008: 4,511 -> (4,511) on p182
        # inject_011: 932,167 -> (932,167) on p197
        if abs(mut_val) != 0 and abs(orig_val) != 0:
            # Check if the absolute value is mentioned and sign issue noted
            abs_val = abs(orig_val)
            f_actual_num = None
            if f_actual is not None:
                try:
                    f_actual_num = abs(
                        float(
                            str(f_actual)
                            .replace(",", "")
                            .replace("(", "")
                            .replace(")", "")
                        )
                    )
                except (ValueError, TypeError):
                    pass
            if str(abs_val) in desc.replace(",", "") or (
                f_actual_num is not None and f_actual_num == abs_val
            ):
                if any(
                    w in desc
                    for w in ["sign", "bracket", "parenthes", "negative", "credit"]
                ):
                    if page_close():
                        return True

    elif mutation == "year_swap":
        # inject_016: 2019->2018 on p145 (PL title)
        # inject_017: 2019->2018 on p148 (BS column header)
        # inject_028: 2018->2019 on p156 (restatement table date)
        mut_text = matcher["mut_text"]
        orig_text = matcher["orig_text"]

        if gid == "inject_016":
            # PL title year error
            if (
                "profit and loss" in desc
                or "p&l" in desc
                or "title" in desc
                or "subtitle" in desc
            ):
                if "2018" in desc and ("2019" in desc or "year ended" in desc):
                    if page_close():
                        return True
        elif gid == "inject_017":
            # BS column header 2019->2018
            if (
                "column" in desc
                or "header" in desc
                or "balance sheet" in desc
                or "duplicate" in desc
            ):
                if "2018" in desc and page_close():
                    # Make sure it's about the column header, not the title
                    if (
                        "column" in desc
                        or "header" in desc
                        or "duplicate" in desc
                        or "both" in desc
                    ):
                        return True
        elif gid == "inject_028":
            # Restatement table: 2018->2019
            if (
                "1 january 2019" in desc
                or "january 2019" in desc
                or "restatement" in desc
                or "hkfrs 16" in desc
                or "1.1" in desc
                or "adoption" in desc
            ):
                if page_close(tolerance=6):
                    if "2018" in desc or "2019" in desc:
                        return True

    elif mutation == "currency_swap":
        # inject_018: US$->HK$ on p145 (PL)
        # inject_027: US$->HK$ on p179 (Note 3 turnover table)
        if "hk$" in desc or "hk$" in desc.replace("'", ""):
            if gid == "inject_018" and page_close():
                if (
                    "profit" in desc
                    or "p&l" in desc
                    or "income" in desc
                    or "statement" in desc
                ):
                    return True
            elif gid == "inject_027" and page_close():
                if (
                    "note 3" in desc
                    or "segment" in desc
                    or "turnover" in desc
                    or "supplementary" in desc
                ):
                    return True

    elif mutation == "label_swap_classification":
        # inject_019: Continuing->Discontinued on p145
        # inject_020: Non-current assets->liabilities on p148
        if gid == "inject_019":
            if "discontinued" in desc and page_close():
                if (
                    "continuing" in desc
                    or "mislabel" in desc
                    or "incorrect" in desc
                    or "header" in desc
                    or "section" in desc
                ):
                    return True
        elif gid == "inject_020":
            if "non-current" in desc and "liabilit" in desc and page_close():
                if "asset" in desc:
                    return True

    elif mutation == "label_swap_direction":
        # inject_021: Due from -> Due to on p148
        # inject_023: inflow -> outflow on p152
        if gid == "inject_021":
            if "due to" in desc and page_close():
                if "related" in desc or "asset" in desc or "current" in desc:
                    if "due from" in desc or "mislabel" in desc or "incorrect" in desc:
                        return True
        elif gid == "inject_023":
            if "outflow" in desc and page_close():
                if (
                    "inflow" in desc
                    or "operating" in desc
                    or "label" in desc
                    or "positive" in desc
                ):
                    return True

    elif mutation == "label_swap_sign_word":
        # inject_022: Gross profit -> Gross loss on p145
        # inject_026: Receivables -> Payables on p194
        if gid == "inject_022":
            if "gross loss" in desc and page_close():
                return True
        elif gid == "inject_026":
            if "payable" in desc and page_close():
                if (
                    "receivable" in desc
                    or "trade and other" in desc
                    or "note 21" in desc
                    or "mislabel" in desc
                    or "incorrect" in desc
                ):
                    return True

    elif mutation == "restated_label":
        # inject_024: removed (Restated) on p148
        if "restated" in desc and page_close():
            if (
                "missing" in desc
                or "label" in desc
                or "removed" in desc
                or "lack" in desc
            ):
                return True

    elif mutation == "note_ref_wrong":
        # inject_015: Note 25 -> Note 26 on p150
        if "note 26" in desc and page_close():
            if "note 25" in desc or "wrong" in desc or "incorrect" in desc:
                return True

    elif mutation == "standard_ref_wrong":
        # inject_025: HKFRS 16 -> HKFRS 17 on p154
        if "hkfrs 17" in desc and page_close():
            if (
                "hkfrs 16" in desc
                or "lease" in desc
                or "incorrect" in desc
                or "insurance" in desc
                or "wrong" in desc
            ):
                return True

    return False


def match_findings_to_gt(raw_findings, gt):
    """Match each raw finding to a GT error (or None if no match)."""
    matchers = build_gt_matchers(gt)

    result = []  # list of (finding_index, gt_id, run)

    for i, finding in enumerate(raw_findings):
        run = finding.get("_run", "unknown")
        matched_gts = []
        for m in matchers:
            if finding_matches_gt(finding, m):
                matched_gts.append(m["id"])
        result.append((i, matched_gts, run))

    return result


def analyze_run_frequencies(raw_findings, gt, label=""):
    """Compute per-GT-error detection frequency across runs."""
    matches = match_findings_to_gt(raw_findings, gt)

    # For each GT error, count unique runs that detected it
    gt_to_runs = defaultdict(set)
    total_matched = 0

    for _, gt_ids, run in matches:
        for gid in gt_ids:
            gt_to_runs[gid].add(run)
            total_matched += 1

    # Count all unique runs
    all_runs = sorted(set(f.get("_run", "?") for f in raw_findings))
    n_runs = len(all_runs)

    print(
        f"\n{label}: {len(raw_findings)} findings, {n_runs} runs, {total_matched} matches"
    )

    # Build frequency dict
    gt_issues = gt["issues"]
    freq = {}
    for e in gt_issues:
        gid = e["id"]
        freq[gid] = len(gt_to_runs.get(gid, set()))

    return freq, n_runs


def main():
    gt = load_json(SAMPLES / "ar2019.mixed_s42.ground_truth.json")
    gt_issues = gt["issues"]
    details = {d["id"]: d for d in gt["_injection_details"]}

    # Load r1 and r2 raw findings
    r1_raw_data = load_json(
        SAMPLES / "ar2019.mixed_s42.result_visual_fs150.raw_eval.json"
    )
    r1_findings = (
        r1_raw_data["high"] + r1_raw_data.get("medium", []) + r1_raw_data.get("low", [])
    )

    r2_raw_data = load_json(
        SAMPLES / "ar2019.mixed_s42.result_visual_fs150_r2.raw_eval.json"
    )
    r2_findings = (
        r2_raw_data["high"] + r2_raw_data.get("medium", []) + r2_raw_data.get("low", [])
    )

    freq_r1, n1 = analyze_run_frequencies(r1_findings, gt, "r1")
    freq_r2, n2 = analyze_run_frequencies(r2_findings, gt, "r2")

    # Cross-validate against known eval results
    # From EVAL_LOG: r1 raw hits 28/29 (missing inject_015), r2 raw hits 28/29 (missing inject_015)
    r1_detected = sum(1 for v in freq_r1.values() if v > 0)
    r2_detected = sum(1 for v in freq_r2.values() if v > 0)
    print(
        f"\nCross-validation: r1 detected {r1_detected}/29 GT errors, r2 detected {r2_detected}/29 GT errors"
    )
    print("Expected: both 28/29 (missing inject_015)")

    # Combined analysis
    n_total = n1 + n2
    combined_freq = {}
    for e in gt_issues:
        gid = e["id"]
        combined_freq[gid] = freq_r1.get(gid, 0) + freq_r2.get(gid, 0)

    # Sort by detection frequency descending
    sorted_errors = sorted(combined_freq.items(), key=lambda x: x[1], reverse=True)

    print(f"\n{'=' * 110}")
    print(f"PER-ERROR DETECTION FREQUENCY (combined r1+r2, {n_total} runs)")
    print(f"{'=' * 110}")
    print(
        f"{'GT Error':<15} {'Category':<16} {'Mutation':<22} {'r1':>4}/{n1:<3} {'r2':>4}/{n2:<3} {'Total':>6}/{n_total:<3} {'p(detect)':>9}"
    )
    print("-" * 110)

    for gid, total in sorted_errors:
        e = next(x for x in gt_issues if x["id"] == gid)
        d = details.get(gid, {})
        cat = e.get("category", "?")
        mutation = d.get("mutation_kind", "?")
        c1 = freq_r1.get(gid, 0)
        c2 = freq_r2.get(gid, 0)
        p = total / n_total if n_total > 0 else 0
        print(
            f"{gid:<15} {cat:<16} {mutation:<22} {c1:>4}/{n1:<3} {c2:>4}/{n2:<3} {total:>6}/{n_total:<3} {p:>8.1%}"
        )

    # ========================
    # DISTRIBUTION ANALYSIS
    # ========================
    rates = sorted([v / n_total for v in combined_freq.values()], reverse=True)
    counts_desc = sorted(combined_freq.values(), reverse=True)

    print(f"\n{'=' * 110}")
    print("DISTRIBUTION ANALYSIS")
    print(f"{'=' * 110}")

    # Basic stats
    nonzero_rates = [r for r in rates if r > 0]
    print(f"\nBasic statistics (all 29 errors):")
    print(f"  Mean p: {statistics.mean(rates):.1%}")
    print(f"  Median p: {statistics.median(rates):.1%}")
    print(f"  Stdev p: {statistics.stdev(rates):.1%}")
    print(f"  Min: {min(rates):.1%}, Max: {max(rates):.1%}")
    if nonzero_rates:
        print(f"\nNon-zero errors ({len(nonzero_rates)}):")
        print(f"  Mean p: {statistics.mean(nonzero_rates):.1%}")
        print(f"  Median p: {statistics.median(nonzero_rates):.1%}")

    # Difficulty tiers
    tiers = {
        "Trivial (>90%)": [],
        "Easy (70-90%)": [],
        "Medium (40-70%)": [],
        "Hard (10-40%)": [],
        "Very Hard (1-10%)": [],
        "Undetectable (0%)": [],
    }
    for gid, total in sorted_errors:
        p = total / n_total
        if p > 0.90:
            tiers["Trivial (>90%)"].append(gid)
        elif p > 0.70:
            tiers["Easy (70-90%)"].append(gid)
        elif p > 0.40:
            tiers["Medium (40-70%)"].append(gid)
        elif p > 0.10:
            tiers["Hard (10-40%)"].append(gid)
        elif p > 0.00:
            tiers["Very Hard (1-10%)"].append(gid)
        else:
            tiers["Undetectable (0%)"].append(gid)

    print(f"\nDifficulty tiers:")
    for tier, errors in tiers.items():
        if errors:
            print(f"  {tier}: {len(errors)} errors — {', '.join(errors)}")
        else:
            print(f"  {tier}: 0 errors")

    # Gini coefficient
    n = len(rates)
    sorted_asc = sorted(rates)
    total_sum = sum(sorted_asc)
    if total_sum > 0:
        gini = sum((2 * (i + 1) - n - 1) * sorted_asc[i] for i in range(n)) / (
            n * total_sum
        )
    else:
        gini = 0
    print(f"\nGini coefficient: {gini:.3f} (0=equal, 1=maximally unequal)")

    # Rank-frequency analysis (Zipf/power-law check)
    print(f"\nRank-frequency table (descending detection rate):")
    print(
        f"{'Rank':>4} {'GT Error':<15} {'Count':>6}/{n_total:<3} {'p':>7} {'ln(rank)':>8} {'ln(p)':>8}"
    )
    for rank, (gid, total) in enumerate(sorted_errors, 1):
        p = total / n_total
        lr = math.log(rank)
        lp = math.log(p) if p > 0 else float("-inf")
        print(
            f"{rank:>4} {gid:<15} {total:>6}/{n_total:<3} {p:>6.1%} {lr:>8.2f} {lp:>8.2f}"
        )

    # Log-log regression for power-law fit (excluding zeros)
    nonzero_pairs = [
        (rank, total / n_total)
        for rank, (gid, total) in enumerate(sorted_errors, 1)
        if total > 0
    ]
    if len(nonzero_pairs) >= 3:
        log_ranks = [math.log(r) for r, _ in nonzero_pairs]
        log_rates = [math.log(p) for _, p in nonzero_pairs]

        n_pts = len(log_ranks)
        mean_lr = sum(log_ranks) / n_pts
        mean_lp = sum(log_rates) / n_pts

        numerator = sum(
            (log_ranks[i] - mean_lr) * (log_rates[i] - mean_lp) for i in range(n_pts)
        )
        denominator = sum((log_ranks[i] - mean_lr) ** 2 for i in range(n_pts))

        if denominator > 0:
            slope = numerator / denominator
            intercept = mean_lp - slope * mean_lr

            # R² (coefficient of determination)
            ss_res = sum(
                (log_rates[i] - (slope * log_ranks[i] + intercept)) ** 2
                for i in range(n_pts)
            )
            ss_tot = sum((log_rates[i] - mean_lp) ** 2 for i in range(n_pts))
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

            print(
                f"\nLog-log regression (power-law fit): ln(p) = {slope:.3f} * ln(rank) + {intercept:.3f}"
            )
            print(f"  Slope (= -alpha): {slope:.3f}")
            print(f"  R²: {r_squared:.3f}")
            print(f"  A perfect power law would show R² close to 1.0")
            if r_squared > 0.85:
                print(f"  >> STRONG power-law fit (R²={r_squared:.2f})")
            elif r_squared > 0.6:
                print(f"  >> MODERATE power-law fit (R²={r_squared:.2f})")
            else:
                print(f"  >> WEAK power-law fit (R²={r_squared:.2f})")

    # Expected recall vs N (using per-run probabilities)
    print(f"\n{'=' * 110}")
    print("SCALING: Expected recall vs number of runs")
    print(f"{'=' * 110}")
    per_run_probs = [v / n_total for v in combined_freq.values()]

    print(f"\n{'N':>4} {'E[GT found]':>12} {'E[recall]':>10} {'E[cost @$0.15]':>15}")
    for N in [1, 2, 3, 5, 8, 10, 15, 20, 25, 30, 40, 50, 75, 100]:
        expected = sum(1 - (1 - p) ** N for p in per_run_probs)
        cost = N * 0.15
        print(f"{N:>4} {expected:>10.1f}/29 {expected / 29:>9.1%} {cost:>13.2f}")

    # Runs needed per error for 95% detection
    print(f"\nRuns needed per GT error for 95% and 99% detection probability:")
    print(f"{'GT Error':<15} {'p/run':>7} {'N(95%)':>8} {'N(99%)':>8} {'Category':<16}")
    for gid, total in sorted_errors:
        p = total / n_total
        e = next(x for x in gt_issues if x["id"] == gid)
        if 0 < p < 1:
            n95 = math.ceil(math.log(0.05) / math.log(1 - p))
            n99 = math.ceil(math.log(0.01) / math.log(1 - p))
        elif p >= 1.0:
            n95, n99 = 1, 1
        else:
            n95, n99 = "inf", "inf"
        print(f"{gid:<15} {p:>6.1%} {str(n95):>8} {str(n99):>8} {e['category']:<16}")


if __name__ == "__main__":
    main()
