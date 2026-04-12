import re
import json
import math
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
from scipy import stats


# ----------------------------
# Config grouping (x -> group)
# ----------------------------
def distribution_group_from_config(x: int) -> str:
    if 0 <= x <= 49:
        return "random"
    if 50 <= x <= 99:
        return "misinfo_gt"
    if 100 <= x <= 149:
        return "equal"
    if 150 <= x <= 199:
        return "real_gt"
    return "unknown"


# ----------------------------
# File discovery
# ----------------------------
# Accept:
#   hyp2-config-12s-0
#   hyp2-config-12s-0.json
RUN_FILE_RE = re.compile(r"^hyp2-config-(?P<x>\d+)s-(?P<seed>\d+)(?P<ext>\.json)?$")


def find_run_files(output_dir: Path) -> List[Tuple[Path, int, int]]:
    runs = []
    for p in output_dir.iterdir():
        if not p.is_file():
            continue
        m = RUN_FILE_RE.match(p.name)
        if not m:
            continue
        x = int(m.group("x"))
        seed = int(m.group("seed"))
        runs.append((p, x, seed))
    runs.sort(key=lambda t: (t[1], t[2], str(t[0])))
    return runs


# ----------------------------
# Metric computation per run
# ----------------------------
def compute_run_metric(run_data: Dict[str, Any]) -> Dict[str, float]:
    """
    Expects structure:
    {
      "static_post_information": {"regular_info_count": R_avail, "misinformation_count": M_avail},
      "agent_share_data": {"Agent_0": {"regular_information_shares": R_i, "misinformation_shares": M_i}, ...}
    }

    Returns system-level:
      D = (M_share_total/M_avail) - (R_share_total/R_avail)
    """
    spi = run_data.get("static_post_information", {})
    R_avail = float(spi.get("regular_info_count", 0.0))
    M_avail = float(spi.get("misinformation_count", 0.0))

    agent_share_data = run_data.get("agent_share_data", {})
    R_share_total = 0.0
    M_share_total = 0.0

    for _, a in agent_share_data.items():
        R_share_total += float(a.get("regular_information_shares", 0.0))
        M_share_total += float(a.get("misinformation_shares", 0.0))

    if R_avail <= 0 or M_avail <= 0:
        D = float("nan")
    else:
        D = (M_share_total / M_avail) - (R_share_total / R_avail)

    return {
        "R_avail": R_avail,
        "M_avail": M_avail,
        "R_share_total": R_share_total,
        "M_share_total": M_share_total,
        "D": D,
    }


# ----------------------------
# Stats helpers
# ----------------------------
def one_sided_one_sample_ttest_greater_than_zero(x: np.ndarray) -> Dict[str, float]:
    x = x[np.isfinite(x)]
    n = int(x.size)
    if n < 2:
        return {"n": n, "t": float("nan"), "p_one_sided": float("nan"), "mean": float(np.nan), "sd": float(np.nan)}

    t_stat, p_two_sided = stats.ttest_1samp(x, popmean=0.0, nan_policy="omit")

    if math.isnan(t_stat):
        p_one_sided = float("nan")
    else:
        # Convert 2-sided to 1-sided for H1: mean > 0
        if t_stat >= 0:
            p_one_sided = p_two_sided / 2.0
        else:
            p_one_sided = 1.0 - (p_two_sided / 2.0)

    return {
        "n": n,
        "t": float(t_stat),
        "p_one_sided": float(p_one_sided),
        "mean": float(np.mean(x)),
        "sd": float(np.std(x, ddof=1)),
    }


def mean_ci95(x: np.ndarray) -> Dict[str, float]:
    x = x[np.isfinite(x)]
    n = int(x.size)
    if n < 2:
        return {"n": n, "mean": float(np.nan), "ci95_low": float(np.nan), "ci95_high": float(np.nan)}
    m = float(np.mean(x))
    s = float(np.std(x, ddof=1))
    se = s / math.sqrt(n)
    tcrit = float(stats.t.ppf(0.975, df=n - 1))
    return {"n": n, "mean": m, "ci95_low": m - tcrit * se, "ci95_high": m + tcrit * se}


def welchs_anova_oneway(groups: Dict[str, np.ndarray]) -> Dict[str, float]:
    """
    Welch's one-way ANOVA (heteroscedastic).

    Returns F, df1, df2, p.
    """
    gvals = {k: v[np.isfinite(v)] for k, v in groups.items()}
    gvals = {k: v for k, v in gvals.items() if v.size >= 2}

    k = len(gvals)
    if k < 2:
        return {"k": k, "F": float("nan"), "df1": float("nan"), "df2": float("nan"), "p": float("nan")}

    means = {k: float(np.mean(v)) for k, v in gvals.items()}
    ns = {k: int(v.size) for k, v in gvals.items()}
    vars_ = {k: float(np.var(v, ddof=1)) for k, v in gvals.items()}

    # weights
    w = {k: ns[k] / vars_[k] for k in gvals.keys()}
    w_sum = sum(w.values())

    y_bar = sum(w[k] * means[k] for k in gvals.keys()) / w_sum

    numerator = sum(w[k] * (means[k] - y_bar) ** 2 for k in gvals.keys()) / (k - 1)

    a = sum((1.0 / (ns[k] - 1)) * (1.0 - (w[k] / w_sum)) ** 2 for k in gvals.keys())
    df1 = k - 1
    df2 = (df1 + 1) / (3 * a)

    F = numerator / (1.0 + (2 * (df1 - 1) * a))
    p = float(1.0 - stats.f.cdf(F, df1, df2))

    return {"k": k, "F": float(F), "df1": float(df1), "df2": float(df2), "p": p}


def pairwise_welch_ttests_holm(groups: Dict[str, np.ndarray]) -> List[Dict[str, float | str]]:
    names = sorted(groups.keys())
    pairs = []
    raw_ps = []

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a = groups[names[i]][np.isfinite(groups[names[i]])]
            b = groups[names[j]][np.isfinite(groups[names[j]])]

            if a.size < 2 or b.size < 2:
                t_stat, p_val = float("nan"), float("nan")
            else:
                t_stat, p_val = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")

            pairs.append({
                "group_a": names[i],
                "group_b": names[j],
                "n_a": int(a.size),
                "n_b": int(b.size),
                "mean_a": float(np.mean(a)) if a.size else float("nan"),
                "mean_b": float(np.mean(b)) if b.size else float("nan"),
                "t": float(t_stat),
                "p_raw": float(p_val),
            })
            raw_ps.append(p_val)

    # Holm correction (simple implementation)
    m = len(raw_ps)
    idx_ps = [(idx, p) for idx, p in enumerate(raw_ps) if p == p]  # filter NaN
    idx_ps.sort(key=lambda t: t[1])

    corrected = [float("nan")] * m
    prev = 0.0
    for rank, (idx, p) in enumerate(idx_ps, start=1):
        corr = min(1.0, (m - rank + 1) * p)
        corr = max(prev, corr)  # monotone
        corrected[idx] = corr
        prev = corr

    for i, c in enumerate(corrected):
        pairs[i]["p_holm"] = c

    return pairs


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Analyze hyp2 JSON outputs in-place (files in output/).")
    parser.add_argument("--output-dir", default="output", help="Folder containing hyp2-config-... JSON files.")
    parser.add_argument("--seeds", default=None,
                        help="Comma-separated list of seeds to include (optional), e.g. '0,1,2'. If omitted, include all.")
    parser.add_argument("--configs", default=None,
                        help="Config range(s) to include (optional). Examples: '0-199' or '0-49,150-199'. If omitted, include all.")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level for ANOVA post-hoc triggering.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        raise SystemExit(f"Output dir not found: {output_dir.resolve()}")

    seed_filter = None
    if args.seeds:
        seed_filter = set(int(s.strip()) for s in args.seeds.split(",") if s.strip())

    config_filter = None
    if args.configs:
        config_filter = set()
        parts = [p.strip() for p in args.configs.split(",") if p.strip()]
        for p in parts:
            if "-" in p:
                a, b = p.split("-", 1)
                a, b = int(a), int(b)
                lo, hi = min(a, b), max(a, b)
                for x in range(lo, hi + 1):
                    config_filter.add(x)
            else:
                config_filter.add(int(p))

    run_files = find_run_files(output_dir)

    runs = []
    errors = []

    for file_path, x, seed in run_files:
        if seed_filter is not None and seed not in seed_filter:
            continue
        if config_filter is not None and x not in config_filter:
            continue

        group = distribution_group_from_config(x)
        if group == "unknown":
            continue

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                run_data = json.load(f)
            metrics = compute_run_metric(run_data)
            runs.append({
                "file": str(file_path),
                "config": x,
                "seed": seed,
                "group": group,
                **metrics,
            })
        except Exception as e:
            errors.append({"file": str(file_path), "config": x, "seed": seed, "error": str(e)})

    # Build group arrays
    group_order = ["random", "misinfo_gt", "equal", "real_gt"]
    groups_np = {g: np.array([r["D"] for r in runs if r["group"] == g], dtype=float) for g in group_order}

    within = {}
    for g in group_order:
        within[g] = {
            "summary": mean_ci95(groups_np[g]),
            "t_test_one_sided_mean_gt_0": one_sided_one_sample_ttest_greater_than_zero(groups_np[g]),
        }

    anova = welchs_anova_oneway(groups_np)

    posthoc = None
    if anova["p"] == anova["p"] and anova["p"] < args.alpha:
        posthoc = pairwise_welch_ttests_holm(groups_np)

    results = {
        "input": {
            "output_dir": str(output_dir),
            "seeds_filter": sorted(list(seed_filter)) if seed_filter is not None else None,
            "configs_filter": sorted(list(config_filter)) if config_filter is not None else None,
            "alpha": args.alpha,
            "metric": "D = (M_share_total/M_avail) - (R_share_total/R_avail)",
        },
        "counts": {
            "runs_loaded": len(runs),
            "errors": len(errors),
            "runs_per_group": {g: int(np.isfinite(groups_np[g]).sum()) for g in group_order},
        },
        "within_group": within,
        "welch_anova": anova,
        "posthoc_pairwise_welch_ttests_holm": posthoc,
        "errors": errors,
    }

    out_dir = Path("analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_json = out_dir / "analysis_results.json"
    out_txt  = out_dir / "analysis_summary.txt" 

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    lines = []
    lines.append("Hyp2 analysis summary")
    lines.append("====================")
    lines.append(f"Output dir: {output_dir.resolve()}")
    lines.append(f"Runs loaded: {len(runs)} | Errors: {len(errors)}")
    lines.append("")
    lines.append("Metric per run:")
    lines.append("  D = (M_share_total/M_avail) - (R_share_total/R_avail)")
    lines.append("  Interpretation: D>0 => misinformation shared more per available post than regular info.")
    lines.append("")
    lines.append("Within-group one-sample t-tests (H1: mean(D) > 0)")
    for g in group_order:
        summ = within[g]["summary"]
        tt = within[g]["t_test_one_sided_mean_gt_0"]
        lines.append(
            f"- {g:9s} n={summ['n']:<4d} mean={summ['mean']:+.6f} "
            f"CI95=[{summ['ci95_low']:+.6f}, {summ['ci95_high']:+.6f}] "
            f"t={tt['t']:+.4f} p(one-sided)={tt['p_one_sided']:.6g}"
        )

    lines.append("")
    lines.append("Across-group Welch ANOVA on D")
    lines.append(f"- k={anova['k']} F={anova['F']:.6f} df1={anova['df1']} df2={anova['df2']:.3f} p={anova['p']:.6g}")

    if posthoc is not None:
        lines.append("")
        lines.append("Post-hoc pairwise Welch t-tests (two-sided) with Holm correction")
        for row in posthoc:
            lines.append(
                f"- {row['group_a']} vs {row['group_b']}: "
                f"mean_a={row['mean_a']:+.6f} mean_b={row['mean_b']:+.6f} "
                f"t={row['t']:+.4f} p_raw={row['p_raw']:.6g} p_holm={row['p_holm']:.6g}"
            )

    if errors:
        lines.append("")
        lines.append("Errors (first 20 shown):")
        for e in errors[:20]:
            lines.append(f"- {e['file']}: {e['error']}")

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Wrote: {out_json.resolve()}")
    print(f"Wrote: {out_txt.resolve()}")


if __name__ == "__main__":
    main()