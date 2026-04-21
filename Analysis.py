"""
analyze.py
Run this once after all seed batches have been summarized.
Reads the summary CSV and produces two ANOVA reports (growth rate and decay rate)
and two heatmaps.

Usage:
    python analyze.py
    python analyze.py --summary summary.csv --out_dir results
"""

import os
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

warnings.filterwarnings("ignore")

DISTRIBUTION_ORDER = [
    "4-1-1", "3-2-1", "3-1-2",
    "1-4-1", "2-3-1", "1-3-2",
    "1-1-4", "2-1-3", "1-2-3",
    "2-2-2",
]

DIST_LABELS = {
    "4-1-1": "4R-1C-1B", "3-2-1": "3R-2C-1B", "3-1-2": "3R-1C-2B",
    "1-4-1": "1R-4C-1B", "2-3-1": "2R-3C-1B", "1-3-2": "1R-3C-2B",
    "1-1-4": "1R-1C-4B", "2-1-3": "2R-1C-3B", "1-2-3": "1R-2C-3B",
    "2-2-2": "2R-2C-2B",
}

BELIEF_ORDER = [-4, -3, -2, -1, 0, 1, 2, 3, 4]

METRICS = [
    ("growth_rate", "Growth Rate", "Mean Growth Rate\n(interactions/timestep during rise phase)"),
    ("decay_rate",  "Decay Rate",  "Mean Decay Rate\n(interactions/timestep during decline phase)"),
]


# ─────────────────────────────────────────────
# ANOVA
# ─────────────────────────────────────────────

def interpret_eta(eta):
    if eta < 0.01:   return "negligible"
    elif eta < 0.06: return "small"
    elif eta < 0.14: return "medium"
    else:            return "large"


def run_anova(df, metric):
    df = df.copy()
    df["transformed"] = np.sqrt(df[metric])
    df["belief"] = df["post_belief_value"].astype(str).astype("category")
    df["dist"]   = df["agent_distribution"].astype("category")

    model = ols("transformed ~ C(belief) * C(dist)", data=df).fit()
    anova_table = anova_lm(model, typ=2)

    ss_total = anova_table["sum_sq"].sum() + model.ssr
    anova_table["eta_squared"] = anova_table["sum_sq"] / ss_total

    return anova_table


def format_anova_report(anova_table, df, metric_label):
    lines = []
    lines.append("=" * 60)
    lines.append("TWO-WAY ANOVA RESULTS")
    lines.append(f"Response variable : sqrt({metric_label})")
    lines.append(f"Total observations: {len(df)}")
    lines.append(f"Seeds included    : {sorted(df['seed'].unique())}")
    lines.append("=" * 60)

    row_map = {
        "C(belief)":         "Post Belief Value",
        "C(dist)":           "Agent Distribution",
        "C(belief):C(dist)": "Interaction (Belief x Distribution)",
        "Residual":          "Residual",
    }

    for idx, row in anova_table.iterrows():
        label = row_map.get(idx, idx)
        lines.append(f"\n[{label}]")
        lines.append(f"  Sum of Squares : {row['sum_sq']:.4f}")
        lines.append(f"  df             : {int(row['df'])}")

        f_val = row.get("F", float("nan"))
        if not np.isnan(f_val):
            lines.append(f"  F-statistic    : {f_val:.4f}")

        p_val = row.get("PR(>F)", float("nan"))
        if not np.isnan(p_val):
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "(not significant)"
            lines.append(f"  p-value        : {p_val:.4e}  {sig}")

        eta = row.get("eta_squared", float("nan"))
        if not np.isnan(eta):
            lines.append(f"  eta2 (effect)  : {eta:.4f}  ({interpret_eta(eta)})")

    lines.append("\n" + "=" * 60)
    lines.append("Significance codes: *** p<0.001  ** p<0.01  * p<0.05")
    lines.append("Effect size (eta2): <0.01 negligible | 0.01-0.06 small |")
    lines.append("                    0.06-0.14 medium  | >0.14 large")
    lines.append("=" * 60)
    return "\n".join(lines)


# ─────────────────────────────────────────────
# HEATMAP
# ─────────────────────────────────────────────

def generate_heatmap(df, metric, title, out_path):
    present_beliefs = [b for b in BELIEF_ORDER if b in df["post_belief_value"].values]
    present_dists   = [d for d in DISTRIBUTION_ORDER if d in df["agent_distribution"].values]

    pivot = df.pivot_table(
        values=metric,
        index="agent_distribution",
        columns="post_belief_value",
        aggfunc="mean",
    ).reindex(index=present_dists, columns=present_beliefs)

    pivot.index = [DIST_LABELS.get(d, d) for d in pivot.index]

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(
        pivot, ax=ax, cmap="YlOrRd", annot=True, fmt=".2f",
        linewidths=0.5, linecolor="#dddddd",
        cbar_kws={"label": "Mean Rate (interactions/timestep)"},
    )

    ax.set_title(title, fontsize=14, pad=16)
    ax.set_xlabel("Post Belief Value", fontsize=11)
    ax.set_ylabel("Agent Distribution (Red-Centrist-Blue ratio)", fontsize=11)

    n_cols     = len(present_beliefs)
    red_end    = sum(1 for b in present_beliefs if b <= -2)
    blue_start = sum(1 for b in present_beliefs if b < 2)

    for x, color in [(red_end, "#d62728"), (blue_start, "#1f77b4")]:
        ax.axvline(x=x, color=color, linewidth=1.5, linestyle="--", alpha=0.6)

    ax.text(red_end / 2,                -0.2, "Red camp", ha="center", color="#d62728",  fontsize=9, transform=ax.transData)
    ax.text((red_end + blue_start) / 2, -0.2, "Centrist", ha="center", color="#555555",  fontsize=9, transform=ax.transData)
    ax.text((blue_start + n_cols) / 2,  -0.2, "Blue camp", ha="center", color="#1f77b4", fontsize=9, transform=ax.transData)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Heatmap saved to '{out_path}'")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Final ANOVA + heatmaps from the summary CSV.")
    parser.add_argument("--summary", default="summary.csv", help="Summary CSV produced by summarize.py")
    parser.add_argument("--out_dir", default="analysis",     help="Output directory for results")
    args = parser.parse_args()

    if not os.path.exists(args.summary):
        print(f"Summary file '{args.summary}' not found. Run summarize.py first.")
        return

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.summary)
    print(f"\nLoaded {len(df)} records from '{args.summary}'")
    print(f"  Seeds included  : {sorted(df['seed'].unique())}")
    print(f"  Distributions   : {sorted(df['agent_distribution'].unique())}")
    print(f"  Belief values   : {sorted(df['post_belief_value'].unique())}")
    print(f"  Growth rate range : {df['growth_rate'].min():.4f} - {df['growth_rate'].max():.4f}")
    print(f"  Decay rate range  : {df['decay_rate'].min():.4f} - {df['decay_rate'].max():.4f}")

    # Warn if not all 30 seeds are present
    missing = set(range(30)) - set(df['seed'].unique())
    if missing:
        print(f"\n  Warning: seeds {sorted(missing)} are not yet in the summary.")
        print(  "  You can still run analyze.py on partial data for a preliminary look.")

    full_report = []

    for i, (metric, metric_label, heatmap_title) in enumerate(METRICS, start=1):
        print(f"\n[{i*2-1}/{len(METRICS)*2}] Running ANOVA for {metric_label}...")
        anova_table = run_anova(df, metric)
        report = format_anova_report(anova_table, df, metric_label)
        print("\n" + report)
        full_report.append(report)

        print(f"\n[{i*2}/{len(METRICS)*2}] Generating heatmap for {metric_label}...")
        heatmap_path = os.path.join(args.out_dir, f"heatmap_{metric}.png")
        generate_heatmap(df, metric, heatmap_title, heatmap_path)

    # Save both reports to a single file
    report_path = os.path.join(args.out_dir, "anova_results.txt")
    with open(report_path, "w") as f:
        f.write("\n\n".join(full_report))
    print(f"\n  Full ANOVA report saved to '{report_path}'")

    print("\nDone. Results are in '{}'.".format(args.out_dir))


if __name__ == "__main__":
    main()