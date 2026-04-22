"""
hyp12_analysis.py
─────────────────────────────────────────────────────────────────────────────
Analysis script for Hypothesis 12:
  "Two posts of different beliefs compete for interactions —
   one real news post vs. a three-wave misinformation campaign."

WHAT THIS SCRIPT DOES
──────────────────────
1. Loads and joins every input config + output JSON pair in INPUT_DIR / OUTPUT_DIR.
2. Computes per-run metrics:
     R  — interaction ratio  (S_RN / S_Misinfo)
     W  — real news win rate among contested agents
          (C_RN / (C_RN + C_Misinfo))
3. Runs statistical tests:
     (a) One-sample two-tailed t-test: H0: mu_R = 1
     (b) Two-way ANOVA (Factor B × Factor C) on R  →  reports eta²
     (c) Two-way ANOVA (Factor B × Factor C) on W  →  reports eta²
     (d) Post-hoc Tukey HSD on significant main effects
4. Saves figures to OUTPUT_DIR/figures/:
     - Boxplot of R by Factor B and Factor C
     - Boxplot of W by Factor B and Factor C
     - Mean R heatmap (Factor B × Factor C)
     - Mean W heatmap (Factor B × Factor C)
5. Writes a plain-text results summary to OUTPUT_DIR/hyp12_results.txt

USAGE
──────
    python3 hyp12_analysis.py

    Set INPUT_DIR and OUTPUT_DIR below if your folder layout differs.
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_1samp, f_oneway, tukey_hsd

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

INPUT_DIR  = "input"
OUTPUT_DIR = "output"
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")

ALPHA = 0.05   # significance threshold throughout

# Display ordering for factors (used in plots and tables)
FACTOR_B_ORDER = ["weaker", "equal", "stronger"]
FACTOR_C_ORDER = ["real-news-first", "simultaneous", "between-waves"]

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 1 — DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_data(input_dir, output_dir):
    """
    Pairs every hyp12_*.json input file with its corresponding output file,
    extracts the fields needed for analysis, and returns a flat DataFrame
    where each row is one simulation run.

    Output filenames follow the pattern:
      input/hyp12_B-equal_C-simultaneous_s06.json        (config)
      output/hyp12_B-equal_C-simultaneous_s06s-{seed}.json  (one per simulation seed)

    Each input file may have multiple output files (one per simulation seed).
    Every output file is loaded as a separate row, all sharing the same input metadata.
    """
    records = []
    missing_outputs = []

    input_files = sorted(
        f for f in os.listdir(input_dir)
        if f.startswith("hyp12_") and f.endswith(".json")
    )

    # Index all output files by their input stem for fast lookup.
    # Output filename pattern: {input_stem}s-{seed}.json
    # e.g. hyp12_B-equal_C-simultaneous_s06s-36.json
    output_files = [
        f for f in os.listdir(output_dir)
        if f.startswith("hyp12_") and f.endswith(".json")
    ]
    # Map: input_stem -> list of matching output filenames
    from collections import defaultdict
    stem_to_outputs = defaultdict(list)
    for of in output_files:
        # Strip the trailing s-{seed}.json suffix to recover the input stem.
        # The suffix always looks like  s-<digits>.json
        # so we split on the last occurrence of "s-"
        base = of[:-5]          # remove .json
        idx  = base.rfind("s-")
        if idx != -1:
            stem = base[:idx]   # e.g. hyp12_B-equal_C-simultaneous_s06
            stem_to_outputs[stem].append(of)

    for fname in input_files:
        in_path = os.path.join(input_dir, fname)
        stem    = fname[:-5]    # strip .json

        matched = stem_to_outputs.get(stem, [])
        if not matched:
            missing_outputs.append(fname)
            continue

        with open(in_path) as f:
            inp = json.load(f)
        meta = inp["Metadata"]

        for out_fname in sorted(matched):
            out_path = os.path.join(output_dir, out_fname)
            with open(out_path) as f:
                out = json.load(f)

            mvr = out["misinfo_vs_realnews"]
            ca  = out["contested_agents"]

            # Extract the simulation seed from the output filename suffix
            run_seed = int(out_fname[:-5].rsplit("s-", 1)[-1])

            records.append({
                # ── experimental factors ─────────────────────────────────────
                "input_file":  fname,
                "output_file": out_fname,
                "config_seed": meta["seed"],      # seed baked into the config
                "run_seed":    run_seed,           # seed used by Main.py for this run
                "factor_B":    meta["factor_B"],
                "factor_C":    meta["factor_C"],

                # ── post parameters (nuisance covariates) ────────────────────
                "misinfo_belief":      meta["misinfo_belief"],
                "misinfo_direction":   meta["misinfo_direction"],
                "misinfo_interest_s1": meta["misinfo_interest_s1"],
                "real_news_interest":  meta["real_news_interest"],
                "interest_delta":      meta["interest_delta"],
                "resurgence_gap_min":  meta["resurgence_gap_min"],
                "rn_head_start_min":   meta.get("rn_head_start_min"),

                # ── raw interaction counts ───────────────────────────────────
                "total_misinfo_interactions":  mvr["total_misinfo_interactions"],
                "total_realnews_interactions": mvr["total_realnews_interactions"],

                # ── per-spawn interaction counts ─────────────────────────────
                # JSON always serialises dict keys as strings, so use "0".."3"
                "interactions_s1": out["posts"]["0"]["total_interactions"],
                "interactions_s2": out["posts"]["1"]["total_interactions"],
                "interactions_s3": out["posts"]["2"]["total_interactions"],
                "interactions_rn": out["posts"]["3"]["total_interactions"],

                # ── contested agent outcomes ─────────────────────────────────
                "chose_misinfo_only":  ca["chose_misinfo_only"],
                "chose_realnews_only": ca["chose_realnews_only"],
                "chose_both":          ca["chose_both"],
                "chose_neither":       ca["chose_neither"],
            })

    if missing_outputs:
        warnings.warn(
            f"{len(missing_outputs)} input file(s) had no matching output and were skipped:\n"
            + "\n".join(f"  {f}" for f in missing_outputs[:10])
            + ("\n  ..." if len(missing_outputs) > 10 else "")
        )

    df = pd.DataFrame(records)

    if df.empty:
        warnings.warn(
            "No output files were matched. The DataFrame is empty.\n"
            "Check that OUTPUT_DIR contains files named identically to the input files."
        )
        return df

    # Enforce ordered categoricals for correct sorting in plots and tables
    df["factor_B"] = pd.Categorical(df["factor_B"], categories=FACTOR_B_ORDER, ordered=True)
    df["factor_C"] = pd.Categorical(df["factor_C"], categories=FACTOR_C_ORDER, ordered=True)

    return df


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 2 — METRIC COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(df):
    """
    Adds R (interaction ratio) and W (real news win rate) columns to df.

    R = S_RN / S_Misinfo
        - If both are 0: R = 1.0 (parity — neither side spread at all)
        - If S_Misinfo = 0 but S_RN > 0: R = NaN, run excluded from R analysis
          (degenerate run, misinformation never spread)

    W = C_RN / (C_RN + C_Misinfo)
        - If denominator = 0: W = NaN, run excluded from W analysis
          (no agent was genuinely contested)
    """
    s_rn    = df["total_realnews_interactions"].astype(float)
    s_mis   = df["total_misinfo_interactions"].astype(float)

    # R computation
    both_zero = (s_rn == 0) & (s_mis == 0)
    mis_zero  = (s_mis == 0) & (s_rn > 0)

    R = s_rn / s_mis.replace(0, np.nan)
    R[both_zero] = 1.0
    R[mis_zero]  = np.nan

    df["R"] = R

    # W computation
    c_rn  = df["chose_realnews_only"].astype(float)
    c_mis = df["chose_misinfo_only"].astype(float)
    denom = c_rn + c_mis
    df["W"] = (c_rn / denom.replace(0, np.nan))

    # Flag excluded runs
    df["R_valid"] = ~df["R"].isna()
    df["W_valid"] = ~df["W"].isna()

    return df


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 3 — STATISTICAL TESTS
# ─────────────────────────────────────────────────────────────────────────────

def two_way_anova_with_eta2(df, response, factor_a_col, factor_b_col):
    """
    Manual two-way ANOVA with eta-squared effect sizes.

    Returns a dict with keys:
      grand_mean, SS_A, SS_B, SS_AB, SS_error, SS_total,
      df_A, df_B, df_AB, df_error,
      MS_A, MS_B, MS_AB, MS_error,
      F_A, F_B, F_AB,
      p_A, p_B, p_AB,
      eta2_A, eta2_B, eta2_AB
    """
    from scipy.stats import f as f_dist

    data   = df[[response, factor_a_col, factor_b_col]].dropna()
    y      = data[response].values
    a_vals = data[factor_a_col].values
    b_vals = data[factor_b_col].values

    grand_mean = y.mean()
    n_total    = len(y)

    a_levels   = np.unique(a_vals)
    b_levels   = np.unique(b_vals)
    n_a        = len(a_levels)
    n_b        = len(b_levels)

    # ── cell means and marginal means ────────────────────────────────────────
    cell_means = {}
    cell_ns    = {}
    for a in a_levels:
        for b in b_levels:
            mask = (a_vals == a) & (b_vals == b)
            cell_means[(a, b)] = y[mask].mean() if mask.sum() > 0 else grand_mean
            cell_ns[(a, b)]    = mask.sum()

    a_means = {a: y[a_vals == a].mean() for a in a_levels}
    b_means = {b: y[b_vals == b].mean() for b in b_levels}

    # ── sums of squares ───────────────────────────────────────────────────────
    # SS_A  : effect of factor A
    SS_A = sum(
        cell_ns[(a, b)] * (a_means[a] - grand_mean) ** 2
        for a in a_levels for b in b_levels
    )
    # SS_B  : effect of factor B
    SS_B = sum(
        cell_ns[(a, b)] * (b_means[b] - grand_mean) ** 2
        for a in a_levels for b in b_levels
    )
    # SS_AB : interaction
    SS_AB = sum(
        cell_ns[(a, b)] * (
            cell_means[(a, b)] - a_means[a] - b_means[b] + grand_mean
        ) ** 2
        for a in a_levels for b in b_levels
    )
    # SS_total
    SS_total = sum((yi - grand_mean) ** 2 for yi in y)
    # SS_error (within cells)
    SS_error = SS_total - SS_A - SS_B - SS_AB

    # ── degrees of freedom ────────────────────────────────────────────────────
    df_A     = n_a - 1
    df_B     = n_b - 1
    df_AB    = df_A * df_B
    df_error = n_total - n_a * n_b

    # ── mean squares ─────────────────────────────────────────────────────────
    MS_A     = SS_A     / df_A
    MS_B     = SS_B     / df_B
    MS_AB    = SS_AB    / df_AB
    MS_error = SS_error / df_error if df_error > 0 else np.nan

    # ── F statistics and p-values ─────────────────────────────────────────────
    F_A  = MS_A  / MS_error if MS_error else np.nan
    F_B  = MS_B  / MS_error if MS_error else np.nan
    F_AB = MS_AB / MS_error if MS_error else np.nan

    p_A  = 1 - f_dist.cdf(F_A,  df_A,  df_error) if not np.isnan(F_A)  else np.nan
    p_B  = 1 - f_dist.cdf(F_B,  df_B,  df_error) if not np.isnan(F_B)  else np.nan
    p_AB = 1 - f_dist.cdf(F_AB, df_AB, df_error) if not np.isnan(F_AB) else np.nan

    # ── eta-squared (proportion of total variance) ────────────────────────────
    eta2_A  = SS_A  / SS_total
    eta2_B  = SS_B  / SS_total
    eta2_AB = SS_AB / SS_total

    return dict(
        grand_mean=grand_mean, n=n_total,
        SS_A=SS_A, SS_B=SS_B, SS_AB=SS_AB, SS_error=SS_error, SS_total=SS_total,
        df_A=df_A, df_B=df_B, df_AB=df_AB, df_error=df_error,
        MS_A=MS_A, MS_B=MS_B, MS_AB=MS_AB, MS_error=MS_error,
        F_A=F_A, F_B=F_B, F_AB=F_AB,
        p_A=p_A, p_B=p_B, p_AB=p_AB,
        eta2_A=eta2_A, eta2_B=eta2_B, eta2_AB=eta2_AB,
    )


def run_posthoc_tukey(df, response, factor_col, valid_col):
    """
    Runs Tukey HSD pairwise comparisons for a given factor.
    Returns a dict of {(level_i, level_j): {"statistic": ..., "pvalue": ...}}.

    Skips levels with fewer than 2 observations (Tukey HSD requirement)
    and warns if any levels were dropped.
    """
    sub    = df[df[valid_col]][[response, factor_col]].dropna()
    levels = sub[factor_col].cat.categories.tolist()

    # Filter out any level that has fewer than 2 observations
    groups_raw = [(lv, sub.loc[sub[factor_col] == lv, response].values)
                  for lv in levels]
    skipped = [lv for lv, g in groups_raw if len(g) < 2]
    groups_ok = [(lv, g) for lv, g in groups_raw if len(g) >= 2]

    if skipped:
        warnings.warn(
            f"Post-hoc Tukey HSD for {response} ~ {factor_col}: "
            f"skipped level(s) {skipped} — fewer than 2 observations."
        )

    if len(groups_ok) < 2:
        warnings.warn(
            f"Post-hoc Tukey HSD for {response} ~ {factor_col}: "
            f"fewer than 2 testable groups remaining after filtering. Skipping."
        )
        return {}

    ok_levels = [lv for lv, _ in groups_ok]
    ok_groups = [g  for _, g  in groups_ok]

    result = tukey_hsd(*ok_groups)

    comparisons = {}
    for i, li in enumerate(ok_levels):
        for j, lj in enumerate(ok_levels):
            if i < j:
                comparisons[(li, lj)] = {
                    "statistic": result.statistic[i, j],
                    "pvalue":    result.pvalue[i, j],
                }
    return comparisons


def run_statistical_tests(df):
    """
    Runs all three statistical procedures and returns a results dict.
    """
    results = {}

    # ── (a) One-sample t-test on R vs. 1.0 ───────────────────────────────────
    R_vals = df.loc[df["R_valid"], "R"].values
    t_stat, p_val = ttest_1samp(R_vals, popmean=1.0)
    results["ttest_R"] = {
        "n":      len(R_vals),
        "mean":   float(np.mean(R_vals)),
        "sd":     float(np.std(R_vals, ddof=1)),
        "median": float(np.median(R_vals)),
        "t":      float(t_stat),
        "p":      float(p_val),
        "significant": p_val < ALPHA,
    }

    # ── (b) Two-way ANOVA on R ────────────────────────────────────────────────
    df_R = df[df["R_valid"]].copy()
    anova_R = two_way_anova_with_eta2(df_R, "R", "factor_B", "factor_C")
    results["anova_R"] = anova_R

    # ── (c) Two-way ANOVA on W ────────────────────────────────────────────────
    df_W = df[df["W_valid"]].copy()
    anova_W = two_way_anova_with_eta2(df_W, "W", "factor_B", "factor_C")
    results["anova_W"] = anova_W

    # ── (d) Post-hoc Tukey HSD where significant ──────────────────────────────
    results["posthoc"] = {}

    for response, anova, valid_col in [
        ("R", anova_R, "R_valid"),
        ("W", anova_W, "W_valid"),
    ]:
        results["posthoc"][response] = {}

        # Factor B main effect
        if anova["p_A"] < ALPHA:
            results["posthoc"][response]["factor_B"] = run_posthoc_tukey(
                df, response, "factor_B", valid_col
            )

        # Factor C main effect
        if anova["p_B"] < ALPHA:
            results["posthoc"][response]["factor_C"] = run_posthoc_tukey(
                df, response, "factor_C", valid_col
            )

    return results


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 4 — FIGURES
# ─────────────────────────────────────────────────────────────────────────────

def make_figures(df, figures_dir):
    os.makedirs(figures_dir, exist_ok=True)

    sns.set_theme(style="whitegrid", font_scale=1.1)
    PALETTE = "Set2"

    # ── helper: add significance line ────────────────────────────────────────
    def hline_at_1(ax):
        ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0,
                   label="Parity (R = 1)")

    # ── Figure 1: Boxplot of R by Factor B and Factor C ───────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    df_R = df[df["R_valid"]]
    sns.boxplot(
        data=df_R, x="factor_B", y="R", hue="factor_C",
        order=FACTOR_B_ORDER, hue_order=FACTOR_C_ORDER,
        palette=PALETTE, ax=ax, width=0.6, fliersize=3,
    )
    hline_at_1(ax)
    ax.set_xlabel("Factor B — Real News Interest Level", labelpad=8)
    ax.set_ylabel("Interaction Ratio (R)", labelpad=8)
    ax.set_title("Hypothesis 12 — Interaction Ratio R by Factor B and Factor C")
    ax.legend(title="Factor C — Timing", bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.tight_layout()
    fig.savefig(os.path.join(figures_dir, "hyp12_boxplot_R.png"), dpi=150)
    plt.close(fig)

    # ── Figure 2: Boxplot of W by Factor B and Factor C ───────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    df_W = df[df["W_valid"]]
    sns.boxplot(
        data=df_W, x="factor_B", y="W", hue="factor_C",
        order=FACTOR_B_ORDER, hue_order=FACTOR_C_ORDER,
        palette=PALETTE, ax=ax, width=0.6, fliersize=3,
    )
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1.0,
               label="Parity (W = 0.5)")
    ax.set_xlabel("Factor B — Real News Interest Level", labelpad=8)
    ax.set_ylabel("Real News Win Rate (W)", labelpad=8)
    ax.set_title("Hypothesis 12 — Real News Win Rate W by Factor B and Factor C")
    ax.legend(title="Factor C — Timing", bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.tight_layout()
    fig.savefig(os.path.join(figures_dir, "hyp12_boxplot_W.png"), dpi=150)
    plt.close(fig)

    # ── Figure 3: Mean R heatmap (Factor B × Factor C) ────────────────────────
    pivot_R = (
        df_R.groupby(["factor_B", "factor_C"])["R"]
        .mean()
        .unstack("factor_C")
        .reindex(index=FACTOR_B_ORDER, columns=FACTOR_C_ORDER)
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.heatmap(
        pivot_R, annot=True, fmt=".3f", cmap="RdYlGn",
        center=1.0, linewidths=0.5, ax=ax,
        cbar_kws={"label": "Mean R"},
    )
    ax.set_xlabel("Factor C — Timing", labelpad=8)
    ax.set_ylabel("Factor B — Interest Level", labelpad=8)
    ax.set_title("Hypothesis 12 — Mean Interaction Ratio R\n(Factor B × Factor C)")
    plt.tight_layout()
    fig.savefig(os.path.join(figures_dir, "hyp12_heatmap_R.png"), dpi=150)
    plt.close(fig)

    # ── Figure 4: Mean W heatmap (Factor B × Factor C) ────────────────────────
    pivot_W = (
        df_W.groupby(["factor_B", "factor_C"])["W"]
        .mean()
        .unstack("factor_C")
        .reindex(index=FACTOR_B_ORDER, columns=FACTOR_C_ORDER)
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.heatmap(
        pivot_W, annot=True, fmt=".3f", cmap="RdYlGn",
        center=0.5, linewidths=0.5, ax=ax,
        cbar_kws={"label": "Mean W"},
    )
    ax.set_xlabel("Factor C — Timing", labelpad=8)
    ax.set_ylabel("Factor B — Interest Level", labelpad=8)
    ax.set_title("Hypothesis 12 — Mean Real News Win Rate W\n(Factor B × Factor C)")
    plt.tight_layout()
    fig.savefig(os.path.join(figures_dir, "hyp12_heatmap_W.png"), dpi=150)
    plt.close(fig)

    print(f"Figures saved to: {figures_dir}")


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 5 — RESULTS SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def sig_stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


def format_anova_table(anova, factor_a_label, factor_b_label):
    lines = []
    lines.append(
        f"  {'Source':<30} {'SS':>10} {'df':>5} {'MS':>10} "
        f"{'F':>10} {'p':>10} {'eta2':>8} {'sig':>5}"
    )
    lines.append("  " + "-" * 82)

    rows = [
        (factor_a_label, "A",  anova["SS_A"],  anova["df_A"],  anova["MS_A"],
         anova["F_A"],  anova["p_A"],  anova["eta2_A"]),
        (factor_b_label, "B",  anova["SS_B"],  anova["df_B"],  anova["MS_B"],
         anova["F_B"],  anova["p_B"],  anova["eta2_B"]),
        (f"{factor_a_label} × {factor_b_label}", "AB",
         anova["SS_AB"], anova["df_AB"], anova["MS_AB"],
         anova["F_AB"], anova["p_AB"], anova["eta2_AB"]),
        ("Error", "err", anova["SS_error"], anova["df_error"], anova["MS_error"],
         None, None, None),
    ]

    for label, _, ss, df_, ms, f, p, eta2 in rows:
        f_str   = f"{f:>10.4f}"   if f    is not None else f"{'':>10}"
        p_str   = f"{p:>10.4f}"   if p    is not None else f"{'':>10}"
        eta_str = f"{eta2:>8.4f}" if eta2 is not None else f"{'':>8}"
        sig_str = sig_stars(p)    if p    is not None else ""
        lines.append(
            f"  {label:<30} {ss:>10.4f} {df_:>5} {ms:>10.4f} "
            f"{f_str} {p_str} {eta_str} {sig_str:>5}"
        )

    lines.append(f"  {'Total':<30} {anova['SS_total']:>10.4f} "
                 f"{anova['df_A']+anova['df_B']+anova['df_AB']+anova['df_error']:>5}")
    return "\n".join(lines)


def format_posthoc_table(comparisons):
    lines = []
    lines.append(f"  {'Comparison':<40} {'Statistic':>12} {'p-value':>12} {'sig':>5}")
    lines.append("  " + "-" * 72)
    for (li, lj), vals in comparisons.items():
        lines.append(
            f"  {li:<20} vs {lj:<17} "
            f"{vals['statistic']:>12.4f} {vals['pvalue']:>12.4f} "
            f"{sig_stars(vals['pvalue']):>5}"
        )
    return "\n".join(lines)


def write_results(df, results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "hyp12_results.txt")

    lines = []
    sep = "=" * 80

    lines += [
        sep,
        "HYPOTHESIS 12 — RESULTS SUMMARY",
        "Two posts of different beliefs compete for interactions",
        sep, "",
    ]

    # ── Dataset overview ──────────────────────────────────────────────────────
    n_total = len(df)
    n_R     = df["R_valid"].sum()
    n_W     = df["W_valid"].sum()

    lines += [
        "DATASET OVERVIEW",
        "-" * 40,
        f"  Total simulation runs loaded : {n_total}",
        f"  Runs valid for R analysis    : {n_R}",
        f"  Runs valid for W analysis    : {n_W}",
        f"  Runs excluded from R (degenerate misinfo) : {n_total - n_R}",
        f"  Runs excluded from W (no contested agents): {n_total - n_W}",
        f"  NOTE: A run is excluded from W when no agent received all four posts.",
        f"        A high exclusion rate indicates the two campaigns had limited",
        f"        audience overlap — agents were mostly exposed to one side only.",
        "",
    ]

    # ── Descriptive statistics ────────────────────────────────────────────────
    lines += ["DESCRIPTIVE STATISTICS — Interaction Ratio R", "-" * 40]
    desc_R = (
        df[df["R_valid"]]
        .groupby(["factor_B", "factor_C"], observed=True)["R"]
        .agg(["mean", "median", "std", "count"])
        .rename(columns={"mean": "Mean", "median": "Median",
                         "std": "SD", "count": "N"})
    )
    lines.append(desc_R.to_string())
    lines.append("")

    lines += ["DESCRIPTIVE STATISTICS — Real News Win Rate W", "-" * 40]
    desc_W = (
        df[df["W_valid"]]
        .groupby(["factor_B", "factor_C"], observed=True)["W"]
        .agg(["mean", "median", "std", "count"])
        .rename(columns={"mean": "Mean", "median": "Median",
                         "std": "SD", "count": "N"})
    )
    lines.append(desc_W.to_string())
    lines.append("")

    # ── (a) One-sample t-test ─────────────────────────────────────────────────
    tt = results["ttest_R"]
    lines += [
        "STATISTICAL TEST (a) — One-Sample t-Test on R (H0: mu_R = 1.0)",
        "-" * 40,
        f"  N      = {tt['n']}",
        f"  Mean R = {tt['mean']:.4f}  (SD = {tt['sd']:.4f},  Median = {tt['median']:.4f})",
        f"  t      = {tt['t']:.4f}",
        f"  p      = {tt['p']:.4f}  {sig_stars(tt['p'])}",
        f"  Result : {'SIGNIFICANT' if tt['significant'] else 'NOT SIGNIFICANT'} "
        f"at alpha = {ALPHA}",
        f"  Interpretation: Mean R {'< 1 — misinformation dominated' if tt['mean'] < 1 else '> 1 — real news dominated' if tt['mean'] > 1 else '= 1 — parity'}",
        "",
    ]

    # ── (b) Two-way ANOVA on R ────────────────────────────────────────────────
    anova_R = results["anova_R"]
    lines += [
        "STATISTICAL TEST (b) — Two-Way ANOVA on R (Factor B × Factor C)",
        "-" * 40,
        f"  N = {anova_R['n']}   Grand mean R = {anova_R['grand_mean']:.4f}",
        "",
        format_anova_table(anova_R, "Factor B (interest)", "Factor C (timing)"),
        "",
        "  Significance codes: *** p<0.001  ** p<0.01  * p<0.05  ns = not significant",
        "",
    ]

    # ── (c) Two-way ANOVA on W ────────────────────────────────────────────────
    anova_W = results["anova_W"]
    lines += [
        "STATISTICAL TEST (c) — Two-Way ANOVA on W (Factor B × Factor C)",
        "-" * 40,
        f"  N = {anova_W['n']}   Grand mean W = {anova_W['grand_mean']:.4f}",
        "",
        format_anova_table(anova_W, "Factor B (interest)", "Factor C (timing)"),
        "",
        "  Significance codes: *** p<0.001  ** p<0.01  * p<0.05  ns = not significant",
        "",
    ]

    # ── (d) Post-hoc Tukey HSD ────────────────────────────────────────────────
    lines += ["STATISTICAL TEST (d) — Post-Hoc Tukey HSD", "-" * 40]

    ph = results["posthoc"]
    any_posthoc = False

    for response in ["R", "W"]:
        for factor_key, factor_label in [("factor_B", "Factor B"), ("factor_C", "Factor C")]:
            if factor_key in ph.get(response, {}):
                any_posthoc = True
                lines += [
                    f"  {response} ~ {factor_label} pairwise comparisons:",
                    format_posthoc_table(ph[response][factor_key]),
                    "",
                ]

    if not any_posthoc:
        lines += ["  No significant main effects — post-hoc tests not conducted.", ""]

    # ── Condition means summary table ─────────────────────────────────────────
    lines += [
        "CONDITION MEANS SUMMARY",
        "-" * 40,
    ]
    summary = (
        df[df["R_valid"] & df["W_valid"]]
        .groupby(["factor_B", "factor_C"], observed=True)
        .agg(
            R_mean=("R", "mean"),
            R_sd=("R", "std"),
            W_mean=("W", "mean"),
            W_sd=("W", "std"),
            n=("R", "count"),
        )
        .round(4)
    )
    lines.append(summary.to_string())
    lines += ["", sep, "END OF RESULTS", sep]

    text = "\n".join(lines)

    with open(path, "w") as f:
        f.write(text)

    print(text)
    print(f"\nResults saved to: {path}")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("Loading data...")
    df = load_data(INPUT_DIR, OUTPUT_DIR)
    print(f"  Loaded {len(df)} runs.")

    if len(df) == 0:
        print("No data found. Check that INPUT_DIR and OUTPUT_DIR are correct "
              "and that output files exist.")
        return

    print("Computing metrics...")
    df = compute_metrics(df)
    print(f"  R valid: {df['R_valid'].sum()}  |  W valid: {df['W_valid'].sum()}")

    print("Running statistical tests...")
    results = run_statistical_tests(df)

    print("Generating figures...")
    make_figures(df, FIGURES_DIR)

    print("Writing results summary...")
    write_results(df, results, OUTPUT_DIR)


if __name__ == "__main__":
    main()