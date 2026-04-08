import os
import re
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats


# ============================================================
# CONFIG
# ============================================================

INPUT_FOLDER = "output"
OUTPUT_FOLDER = "analysis_results"

# Filename example: hyp2-config-17s-0.json
# Extract the number before the 's' in "config-17s"
FILENAME_REGEX = re.compile(r"config-(\d+)s", re.IGNORECASE)

GROUP_ORDER = ["random", "misinfo_gt", "equal", "real_gt"]


# ============================================================
# GROUP MAPPING
# ============================================================

def get_distribution_group(config_num: int) -> str:
    if 0 <= config_num <= 49:
        return "random"
    elif 50 <= config_num <= 99:
        return "misinfo_gt"
    elif 100 <= config_num <= 149:
        return "equal"
    elif 150 <= config_num <= 199:
        return "real_gt"
    else:
        return "unknown"


# ============================================================
# UTILITIES
# ============================================================

def ensure_output_dir():
    Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)


def safe_div(numerator, denominator):
    if denominator is None or denominator == 0:
        return np.nan
    return numerator / denominator


def parse_config_from_filename(filename: str):
    match = FILENAME_REGEX.search(filename)
    if not match:
        return None
    return int(match.group(1))


def holm_correction(pvals):
    """
    Holm-Bonferroni correction.
    Returns adjusted p-values in original order.
    """
    pvals = np.array(pvals, dtype=float)
    m = len(pvals)
    order = np.argsort(pvals)
    sorted_p = pvals[order]

    adjusted = np.empty(m, dtype=float)
    for i, p in enumerate(sorted_p):
        adjusted[i] = (m - i) * p

    # enforce monotonicity
    for i in range(1, m):
        adjusted[i] = max(adjusted[i], adjusted[i - 1])

    adjusted = np.minimum(adjusted, 1.0)

    result = np.empty(m, dtype=float)
    result[order] = adjusted
    return result


# ============================================================
# STAT TESTS
# ============================================================

def one_sample_ttest_greater(sample, popmean=0.0):
    """
    One-sample one-tailed t-test: H1 = mean(sample) > popmean
    """
    sample = np.asarray(sample, dtype=float)
    sample = sample[np.isfinite(sample)]

    n = len(sample)
    if n < 2:
        return {
            "n": n,
            "mean": np.nan,
            "sd": np.nan,
            "t_stat": np.nan,
            "p_value": np.nan,
            "cohen_d": np.nan,
        }

    mean_val = np.mean(sample)
    sd_val = np.std(sample, ddof=1)

    t_stat, p_two = stats.ttest_1samp(sample, popmean=popmean, nan_policy='omit')

    if np.isnan(t_stat):
        p_one = np.nan
    elif t_stat > 0:
        p_one = p_two / 2
    else:
        p_one = 1 - (p_two / 2)

    cohen_d = np.nan
    if sd_val > 0:
        cohen_d = (mean_val - popmean) / sd_val

    return {
        "n": n,
        "mean": mean_val,
        "sd": sd_val,
        "t_stat": t_stat,
        "p_value": p_one,
        "cohen_d": cohen_d,
    }


def wilcoxon_greater_zero(sample):
    """
    One-sided Wilcoxon signed-rank test: H1 = median(sample) > 0
    """
    sample = np.asarray(sample, dtype=float)
    sample = sample[np.isfinite(sample)]

    nonzero = sample[sample != 0]
    n = len(nonzero)

    if n < 1:
        return {
            "n": n,
            "stat": np.nan,
            "p_value": np.nan,
            "median": np.nan,
        }

    try:
        stat, p = stats.wilcoxon(nonzero, alternative="greater", zero_method="wilcox")
    except Exception:
        stat, p = np.nan, np.nan

    return {
        "n": n,
        "stat": stat,
        "p_value": p,
        "median": np.median(sample) if len(sample) > 0 else np.nan,
    }


def welch_anova(df, value_col, group_col):
    """
    Manual Welch's ANOVA
    """
    groups = []
    for g, sub in df.groupby(group_col):
        vals = sub[value_col].dropna().values
        if len(vals) >= 2:
            groups.append((g, vals))

    k = len(groups)
    if k < 2:
        return {"k": k, "F": np.nan, "df1": np.nan, "df2": np.nan, "p_value": np.nan}

    means = np.array([np.mean(v) for _, v in groups], dtype=float)
    vars_ = np.array([np.var(v, ddof=1) for _, v in groups], dtype=float)
    ns = np.array([len(v) for _, v in groups], dtype=float)

    vars_ = np.where(vars_ == 0, 1e-12, vars_)
    w = ns / vars_

    y_bar_w = np.sum(w * means) / np.sum(w)
    numerator = np.sum(w * (means - y_bar_w) ** 2) / (k - 1)

    term = np.sum(((1 - (w / np.sum(w))) ** 2) / (ns - 1))
    denominator = 1 + (2 * (k - 2) / (k**2 - 1)) * term

    F = numerator / denominator
    df1 = k - 1
    df2 = (k**2 - 1) / (3 * term) if term > 0 else np.inf
    p_value = 1 - stats.f.cdf(F, df1, df2)

    return {"k": k, "F": F, "df1": df1, "df2": df2, "p_value": p_value}


def eta_squared_anova(df, value_col, group_col):
    clean = df[[value_col, group_col]].dropna()
    if clean.empty:
        return np.nan

    grand_mean = clean[value_col].mean()
    ss_between = 0.0
    ss_total = ((clean[value_col] - grand_mean) ** 2).sum()

    for g, sub in clean.groupby(group_col):
        ss_between += len(sub) * (sub[value_col].mean() - grand_mean) ** 2

    if ss_total == 0:
        return np.nan

    return ss_between / ss_total


def kruskal_wallis(df, value_col, group_col):
    groups = []
    for g, sub in df.groupby(group_col):
        vals = sub[value_col].dropna().values
        if len(vals) >= 1:
            groups.append(vals)

    if len(groups) < 2:
        return {"H": np.nan, "p_value": np.nan, "k": len(groups)}

    H, p = stats.kruskal(*groups)
    return {"H": H, "p_value": p, "k": len(groups)}


def epsilon_squared_kruskal(df, value_col, group_col):
    clean = df[[value_col, group_col]].dropna()
    n = len(clean)
    k = clean[group_col].nunique()

    if n == 0 or k < 2:
        return np.nan

    H, _ = stats.kruskal(*[sub[value_col].values for _, sub in clean.groupby(group_col)])
    return (H - k + 1) / (n - k) if (n - k) > 0 else np.nan


def pairwise_welch_tests(df, value_col, group_col):
    group_names = [g for g in GROUP_ORDER if g in df[group_col].unique()]
    results = []

    pvals = []
    pairs = []

    for i in range(len(group_names)):
        for j in range(i + 1, len(group_names)):
            g1 = group_names[i]
            g2 = group_names[j]

            x1 = df.loc[df[group_col] == g1, value_col].dropna().values
            x2 = df.loc[df[group_col] == g2, value_col].dropna().values

            if len(x1) < 2 or len(x2) < 2:
                t_stat, p_val = np.nan, np.nan
            else:
                t_stat, p_val = stats.ttest_ind(x1, x2, equal_var=False, nan_policy='omit')

            pairs.append((g1, g2, t_stat))
            pvals.append(1.0 if np.isnan(p_val) else p_val)

    adj_pvals = holm_correction(pvals)

    for (g1, g2, t_stat), p_raw, p_adj in zip(pairs, pvals, adj_pvals):
        results.append({
            "group1": g1,
            "group2": g2,
            "t_stat": t_stat,
            "p_raw": p_raw,
            "p_holm": p_adj,
        })

    return pd.DataFrame(results)


def pairwise_mannwhitney(df, value_col, group_col):
    group_names = [g for g in GROUP_ORDER if g in df[group_col].unique()]
    results = []

    pvals = []
    pairs = []

    for i in range(len(group_names)):
        for j in range(i + 1, len(group_names)):
            g1 = group_names[i]
            g2 = group_names[j]

            x1 = df.loc[df[group_col] == g1, value_col].dropna().values
            x2 = df.loc[df[group_col] == g2, value_col].dropna().values

            if len(x1) < 1 or len(x2) < 1:
                u_stat, p_val = np.nan, np.nan
            else:
                try:
                    u_stat, p_val = stats.mannwhitneyu(x1, x2, alternative="two-sided")
                except Exception:
                    u_stat, p_val = np.nan, np.nan

            pairs.append((g1, g2, u_stat))
            pvals.append(1.0 if np.isnan(p_val) else p_val)

    adj_pvals = holm_correction(pvals)

    for (g1, g2, u_stat), p_raw, p_adj in zip(pairs, pvals, adj_pvals):
        results.append({
            "group1": g1,
            "group2": g2,
            "u_stat": u_stat,
            "p_raw": p_raw,
            "p_holm": p_adj,
        })

    return pd.DataFrame(results)


# ============================================================
# PLOTTING
# ============================================================

def make_boxplot(df, value_col, title, ylabel, filename):
    data = []
    labels = []

    for g in GROUP_ORDER:
        vals = df.loc[df["distribution_group"] == g, value_col].dropna().values
        if len(vals) > 0:
            data.append(vals)
            labels.append(g)

    if not data:
        return

    plt.figure(figsize=(10, 6))
    plt.boxplot(data, labels=labels, showmeans=True)
    plt.axhline(0, linestyle="--")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel("Distribution Group")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, filename), dpi=300)
    plt.close()


def make_histograms_by_group(df, value_col, prefix):
    for g in GROUP_ORDER:
        vals = df.loc[df["distribution_group"] == g, value_col].dropna().values
        if len(vals) == 0:
            continue

        plt.figure(figsize=(8, 5))
        plt.hist(vals, bins=min(15, max(5, len(vals) // 3)))
        plt.axvline(0, linestyle="--")
        plt.title(f"{value_col} histogram - {g}")
        plt.xlabel(value_col)
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_FOLDER, f"{prefix}_{g}.png"), dpi=300)
        plt.close()


# ============================================================
# JSON PARSING
# ============================================================

def extract_run_data(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    filename = os.path.basename(json_path)
    config_num = parse_config_from_filename(filename)
    if config_num is None:
        raise ValueError(f"Could not parse config number from filename: {filename}")

    distribution_group = get_distribution_group(config_num)

    static_info = data.get("static_post_information", {})
    agent_share_data = data.get("agent_share_data", {})

    if not isinstance(agent_share_data, dict):
        raise ValueError(f"'agent_share_data' is not a dictionary in {filename}")

    regular_info_count = static_info.get("regular_info_count", None)
    misinformation_count = static_info.get("misinformation_count", None)

    if regular_info_count is None or misinformation_count is None:
        raise ValueError(f"Missing regular_info_count or misinformation_count in {filename}")

    agent_records = []
    raw_diffs = []
    norm_diffs = []

    # Per agent-class accumulators for run-level subgroup summaries
    class_values = {}

    for agent_name, agent_info in agent_share_data.items():
        agent_class = agent_info.get("agent_classification", "unknown")
        r = agent_info.get("regular_information_shares", 0)
        m = agent_info.get("misinformation_shares", 0)

        raw_diff = m - r
        norm_diff = safe_div(m, misinformation_count) - safe_div(r, regular_info_count)

        raw_diffs.append(raw_diff)
        if np.isfinite(norm_diff):
            norm_diffs.append(norm_diff)

        class_values.setdefault(agent_class, []).append(norm_diff)

        agent_records.append({
            "filename": filename,
            "config_num": config_num,
            "distribution_group": distribution_group,
            "agent_name": agent_name,
            "agent_classification": agent_class,
            "regular_info_count": regular_info_count,
            "misinformation_count": misinformation_count,
            "regular_information_shares": r,
            "misinformation_shares": m,
            "raw_diff": raw_diff,
            "norm_diff": norm_diff,
        })

    raw_diffs = np.array(raw_diffs, dtype=float)
    norm_diffs = np.array(norm_diffs, dtype=float)

    run_record = {
        "filename": filename,
        "config_num": config_num,
        "distribution_group": distribution_group,
        "regular_info_count": regular_info_count,
        "misinformation_count": misinformation_count,
        "total_posts": (regular_info_count or 0) + (misinformation_count or 0),
        "num_agents": len(agent_records),

        # Raw metrics (secondary)
        "run_mean_raw_diff": np.mean(raw_diffs) if len(raw_diffs) > 0 else np.nan,
        "run_median_raw_diff": np.median(raw_diffs) if len(raw_diffs) > 0 else np.nan,
        "run_pct_positive_raw": np.mean(raw_diffs > 0) if len(raw_diffs) > 0 else np.nan,
        "run_std_raw_diff": np.std(raw_diffs, ddof=1) if len(raw_diffs) > 1 else np.nan,

        # Normalized metrics (PRIMARY)
        "run_mean_norm_diff": np.mean(norm_diffs) if len(norm_diffs) > 0 else np.nan,
        "run_median_norm_diff": np.median(norm_diffs) if len(norm_diffs) > 0 else np.nan,
        "run_pct_positive_norm": np.mean(norm_diffs > 0) if len(norm_diffs) > 0 else np.nan,
        "run_std_norm_diff": np.std(norm_diffs, ddof=1) if len(norm_diffs) > 1 else np.nan,
    }

    # Add per-agent-class run-level summaries (very useful extension)
    # Example columns:
    # normal_run_mean_norm_diff, gullible_run_mean_norm_diff, stubborn_run_mean_norm_diff
    for cls, vals in class_values.items():
        vals = np.array(vals, dtype=float)
        vals = vals[np.isfinite(vals)]

        run_record[f"{cls}_run_mean_norm_diff"] = np.mean(vals) if len(vals) > 0 else np.nan
        run_record[f"{cls}_run_median_norm_diff"] = np.median(vals) if len(vals) > 0 else np.nan
        run_record[f"{cls}_run_pct_positive_norm"] = np.mean(vals > 0) if len(vals) > 0 else np.nan
        run_record[f"{cls}_n_agents"] = len(vals)

    return run_record, agent_records


# ============================================================
# ANALYSIS HELPERS
# ============================================================

def run_within_group_tests(df, metric_col, metric_label):
    rows = []

    for g in GROUP_ORDER:
        sub = df.loc[df["distribution_group"] == g, metric_col].dropna().values

        t_res = one_sample_ttest_greater(sub, popmean=0.0)
        w_res = wilcoxon_greater_zero(sub)

        rows.append({
            "metric": metric_label,
            "distribution_group": g,
            "n_runs": len(sub),
            "sample_mean": np.mean(sub) if len(sub) > 0 else np.nan,
            "sample_median": np.median(sub) if len(sub) > 0 else np.nan,
            "sample_sd": np.std(sub, ddof=1) if len(sub) > 1 else np.nan,
            "ttest_t": t_res["t_stat"],
            "ttest_p_one_tailed": t_res["p_value"],
            "ttest_cohen_d": t_res["cohen_d"],
            "wilcoxon_stat": w_res["stat"],
            "wilcoxon_p_one_tailed": w_res["p_value"],
        })

    return pd.DataFrame(rows)


def run_across_group_tests(df, metric_col, metric_label):
    welch = welch_anova(df, metric_col, "distribution_group")
    eta2 = eta_squared_anova(df, metric_col, "distribution_group")

    kw = kruskal_wallis(df, metric_col, "distribution_group")
    eps2 = epsilon_squared_kruskal(df, metric_col, "distribution_group")

    return pd.DataFrame([{
        "metric": metric_label,
        "welch_F": welch["F"],
        "welch_df1": welch["df1"],
        "welch_df2": welch["df2"],
        "welch_p": welch["p_value"],
        "eta_squared_descriptive": eta2,
        "kruskal_H": kw["H"],
        "kruskal_p": kw["p_value"],
        "epsilon_squared": eps2,
    }])


def write_summary_report(df, primary_within_mean, primary_within_median,
                         primary_across_mean, primary_across_median,
                         pairwise_mean, pairwise_median):
    report_path = os.path.join(OUTPUT_FOLDER, "analysis_summary.txt")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("HYP2 SIMULATION ANALYSIS SUMMARY\n")
        f.write("=" * 80 + "\n\n")

        f.write("RUN COUNTS BY DISTRIBUTION GROUP\n")
        f.write("-" * 80 + "\n")
        counts = df["distribution_group"].value_counts()
        for g in GROUP_ORDER:
            f.write(f"{g}: {counts.get(g, 0)} runs\n")
        f.write("\n")

        f.write("PRIMARY DESCRIPTIVE SUMMARY (NORMALIZED METRICS)\n")
        f.write("-" * 80 + "\n")
        desc = df.groupby("distribution_group")[[
            "run_mean_norm_diff",
            "run_median_norm_diff",
            "run_pct_positive_norm"
        ]].agg(["mean", "median", "std", "count"])
        f.write(desc.to_string())
        f.write("\n\n")

        f.write("WITHIN-GROUP TESTS: RUN MEAN NORMALIZED DIFFERENCE\n")
        f.write("-" * 80 + "\n")
        f.write(primary_within_mean.to_string(index=False))
        f.write("\n\n")

        f.write("WITHIN-GROUP TESTS: RUN MEDIAN NORMALIZED DIFFERENCE\n")
        f.write("-" * 80 + "\n")
        f.write(primary_within_median.to_string(index=False))
        f.write("\n\n")

        f.write("ACROSS-GROUP TESTS: RUN MEAN NORMALIZED DIFFERENCE\n")
        f.write("-" * 80 + "\n")
        f.write(primary_across_mean.to_string(index=False))
        f.write("\n\n")

        f.write("ACROSS-GROUP TESTS: RUN MEDIAN NORMALIZED DIFFERENCE\n")
        f.write("-" * 80 + "\n")
        f.write(primary_across_median.to_string(index=False))
        f.write("\n\n")

        f.write("PAIRWISE POST-HOC (PRACTICAL): WELCH T-TESTS ON RUN MEAN NORMALIZED DIFFERENCE\n")
        f.write("-" * 80 + "\n")
        f.write(pairwise_mean.to_string(index=False))
        f.write("\n\n")

        f.write("PAIRWISE ROBUST COMPARISON (PRACTICAL): MANN-WHITNEY ON RUN MEDIAN NORMALIZED DIFFERENCE\n")
        f.write("-" * 80 + "\n")
        f.write(pairwise_median.to_string(index=False))
        f.write("\n\n")

        f.write("NOTES\n")
        f.write("-" * 80 + "\n")
        f.write("Primary metric: norm_diff = (misinformation_shares / misinformation_count) - (regular_information_shares / regular_info_count)\n")
        f.write("Positive values indicate greater misinformation sharing relative to availability.\n")
        f.write("Run-level summaries are the inferential unit to avoid pseudo-replication from within-run agent dependence.\n")
        f.write("Welch's ANOVA is the primary across-group parametric comparison.\n")
        f.write("Kruskal-Wallis provides a robust nonparametric confirmation.\n")
        f.write("Pairwise Welch and Mann-Whitney are practical post-hoc approximations.\n")


# ============================================================
# MAIN
# ============================================================

def main():
    ensure_output_dir()

    input_path = Path(INPUT_FOLDER)
    if not input_path.exists():
        raise FileNotFoundError(f"Input folder not found: {INPUT_FOLDER}")

    json_files = sorted([p for p in input_path.iterdir() if p.is_file() and p.suffix.lower() == ".json"])

    if not json_files:
        raise FileNotFoundError(f"No .json files found in {INPUT_FOLDER}")

    run_records = []
    all_agent_records = []
    skipped = []

    for file_path in json_files:
        try:
            run_record, agent_records = extract_run_data(str(file_path))

            if run_record["distribution_group"] == "unknown":
                skipped.append((file_path.name, "Config number out of expected range"))
                continue

            run_records.append(run_record)
            all_agent_records.extend(agent_records)

        except Exception as e:
            skipped.append((file_path.name, str(e)))

    if not run_records:
        raise RuntimeError("No valid simulation runs were processed.")

    run_df = pd.DataFrame(run_records)
    agent_df = pd.DataFrame(all_agent_records)

    # Save raw tables
    run_df.to_csv(os.path.join(OUTPUT_FOLDER, "run_level_summary.csv"), index=False)
    agent_df.to_csv(os.path.join(OUTPUT_FOLDER, "agent_level_summary.csv"), index=False)

    if skipped:
        pd.DataFrame(skipped, columns=["filename", "reason"]).to_csv(
            os.path.join(OUTPUT_FOLDER, "skipped_files.csv"), index=False
        )

    # ========================================================
    # PRIMARY ANALYSIS (NORMALIZED)
    # ========================================================

    # Within-group
    within_mean_norm = run_within_group_tests(run_df, "run_mean_norm_diff", "run_mean_norm_diff")
    within_median_norm = run_within_group_tests(run_df, "run_median_norm_diff", "run_median_norm_diff")

    within_mean_norm.to_csv(os.path.join(OUTPUT_FOLDER, "within_group_run_mean_norm_diff.csv"), index=False)
    within_median_norm.to_csv(os.path.join(OUTPUT_FOLDER, "within_group_run_median_norm_diff.csv"), index=False)

    # Across-group
    across_mean_norm = run_across_group_tests(run_df, "run_mean_norm_diff", "run_mean_norm_diff")
    across_median_norm = run_across_group_tests(run_df, "run_median_norm_diff", "run_median_norm_diff")

    across_mean_norm.to_csv(os.path.join(OUTPUT_FOLDER, "across_group_run_mean_norm_diff.csv"), index=False)
    across_median_norm.to_csv(os.path.join(OUTPUT_FOLDER, "across_group_run_median_norm_diff.csv"), index=False)

    # Pairwise
    pairwise_mean_norm = pairwise_welch_tests(run_df, "run_mean_norm_diff", "distribution_group")
    pairwise_median_norm = pairwise_mannwhitney(run_df, "run_median_norm_diff", "distribution_group")

    pairwise_mean_norm.to_csv(os.path.join(OUTPUT_FOLDER, "pairwise_welch_run_mean_norm_diff.csv"), index=False)
    pairwise_median_norm.to_csv(os.path.join(OUTPUT_FOLDER, "pairwise_mannwhitney_run_median_norm_diff.csv"), index=False)

    # ========================================================
    # SECONDARY ANALYSIS (RAW)
    # ========================================================
    within_mean_raw = run_within_group_tests(run_df, "run_mean_raw_diff", "run_mean_raw_diff")
    within_median_raw = run_within_group_tests(run_df, "run_median_raw_diff", "run_median_raw_diff")

    across_mean_raw = run_across_group_tests(run_df, "run_mean_raw_diff", "run_mean_raw_diff")
    across_median_raw = run_across_group_tests(run_df, "run_median_raw_diff", "run_median_raw_diff")

    within_mean_raw.to_csv(os.path.join(OUTPUT_FOLDER, "within_group_run_mean_raw_diff.csv"), index=False)
    within_median_raw.to_csv(os.path.join(OUTPUT_FOLDER, "within_group_run_median_raw_diff.csv"), index=False)
    across_mean_raw.to_csv(os.path.join(OUTPUT_FOLDER, "across_group_run_mean_raw_diff.csv"), index=False)
    across_median_raw.to_csv(os.path.join(OUTPUT_FOLDER, "across_group_run_median_raw_diff.csv"), index=False)

    # ========================================================
    # OPTIONAL: AGENT-CLASS DESCRIPTIVE SUMMARIES
    # ========================================================
    if "agent_classification" in agent_df.columns:
        class_desc = (
            agent_df.groupby(["distribution_group", "agent_classification"])["norm_diff"]
            .agg(["mean", "median", "std", "count"])
            .reset_index()
        )
        class_desc.to_csv(os.path.join(OUTPUT_FOLDER, "agent_class_descriptive_norm_diff.csv"), index=False)

    # ========================================================
    # PLOTS
    # ========================================================
    make_boxplot(
        run_df,
        "run_mean_norm_diff",
        "Run Mean of Normalized Sharing Difference by Distribution Group",
        "Run Mean Normalized Difference",
        "boxplot_run_mean_norm_diff.png"
    )

    make_boxplot(
        run_df,
        "run_median_norm_diff",
        "Run Median of Normalized Sharing Difference by Distribution Group",
        "Run Median Normalized Difference",
        "boxplot_run_median_norm_diff.png"
    )

    make_boxplot(
        run_df,
        "run_mean_raw_diff",
        "Run Mean of Raw Sharing Difference by Distribution Group",
        "Run Mean Raw Difference",
        "boxplot_run_mean_raw_diff.png"
    )

    # Histograms for assumption checking
    make_histograms_by_group(run_df, "run_mean_norm_diff", "hist_run_mean_norm_diff")
    make_histograms_by_group(run_df, "run_median_norm_diff", "hist_run_median_norm_diff")

    # ========================================================
    # REPORT
    # ========================================================
    write_summary_report(
        run_df,
        within_mean_norm,
        within_median_norm,
        across_mean_norm,
        across_median_norm,
        pairwise_mean_norm,
        pairwise_median_norm
    )

    # ========================================================
    # CONSOLE SUMMARY
    # ========================================================
    print("=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"Valid runs processed: {len(run_df)}")
    print(f"Agent records processed: {len(agent_df)}")
    print(f"Results saved to: {OUTPUT_FOLDER}")
    if skipped:
        print(f"Skipped files: {len(skipped)} (see skipped_files.csv)")
    print("\nPrimary output files:")
    print("- run_level_summary.csv")
    print("- agent_level_summary.csv")
    print("- analysis_summary.txt")
    print("- within_group_run_mean_norm_diff.csv")
    print("- within_group_run_median_norm_diff.csv")
    print("- across_group_run_mean_norm_diff.csv")
    print("- across_group_run_median_norm_diff.csv")
    print("- pairwise_welch_run_mean_norm_diff.csv")
    print("- pairwise_mannwhitney_run_median_norm_diff.csv")
    print("=" * 70)


if __name__ == "__main__":
    main()