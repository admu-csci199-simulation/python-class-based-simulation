"""
hyp8_analysis.py
Extracts virality data from hyp8 simulation JSON files, runs a two-way ANOVA,
and generates a heatmap. All in one script.

Usage:
    python hyp8_analysis.py
    python hyp8_analysis.py --data_dir output --out_dir results --n 100 --p 10
"""

import os
import json
import re
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

AGENT_DISTRIBUTIONS = [
    [4, 1, 1],
    [3, 2, 1],
    [3, 1, 2],
    [1, 4, 1],
    [2, 3, 1],
    [1, 3, 2],
    [1, 1, 4],
    [2, 1, 3],
    [1, 2, 3],
    [2, 2, 2],
]

DISTRIBUTION_ORDER = [f"{d[0]}-{d[1]}-{d[2]}" for d in AGENT_DISTRIBUTIONS]

DIST_LABELS = {
    "4-1-1": "4R-1C-1B", "3-2-1": "3R-2C-1B", "3-1-2": "3R-1C-2B",
    "1-4-1": "1R-4C-1B", "2-3-1": "2R-3C-1B", "1-3-2": "1R-3C-2B",
    "1-1-4": "1R-1C-4B", "2-1-3": "2R-1C-3B", "1-2-3": "1R-2C-3B",
    "2-2-2": "2R-2C-2B",
}

BELIEF_ORDER = [-4, -3, -2, -1, 0, 1, 2, 3, 4]


# ─────────────────────────────────────────────
# STEP 1: EXTRACTION
# ─────────────────────────────────────────────

def get_camp(belief_value):
    if belief_value <= -2:
        return "red"
    elif belief_value >= 2:
        return "blue"
    return "centrist"


def compute_total_shares(interaction_array):
    return sum(sum(timestep) for timestep in interaction_array)


def parse_filename(filename):
    match = re.match(r"hyp8-config-(\d+)s-(\d+)\.json", filename)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def extract_records(data_dir, p):
    records = []
    files = sorted(os.listdir(data_dir))

    for filename in files:
        parsed = parse_filename(filename)
        if parsed is None:
            continue

        test_case, seed = parsed
        dist_index = test_case // p
        if dist_index >= len(AGENT_DISTRIBUTIONS):
            print(f"  Warning: test case {test_case} maps to dist index {dist_index}, out of range. Skipping.")
            continue

        distribution = AGENT_DISTRIBUTIONS[dist_index]
        dist_label = f"{distribution[0]}-{distribution[1]}-{distribution[2]}"

        filepath = os.path.join(data_dir, filename)
        with open(filepath, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                print(f"  Warning: Could not parse {filename}: {e}. Skipping.")
                continue

        static_info = data.get("static_post_information", {})
        shared_data = data.get("post_shared_data", {})

        for post_id, interactions in shared_data.items():
            post_meta = static_info.get(post_id, {})
            belief_value = post_meta.get("post_beliefValue")
            post_camp = post_meta.get("post_camp") or (get_camp(belief_value) if belief_value is not None else None)
            total_shares = compute_total_shares(interactions)

            records.append({
                "run_id":             f"{test_case}s{seed}",
                "test_case":          test_case,
                "seed":               seed,
                "dist_index":         dist_index,
                "agent_distribution": dist_label,
                "red_ratio":          distribution[0],
                "centrist_ratio":     distribution[1],
                "blue_ratio":         distribution[2],
                "post_id":            post_id,
                "post_belief_value":  belief_value,
                "post_camp":          post_camp,
                "total_shares":       total_shares,
            })

    return records


# ─────────────────────────────────────────────
# STEP 2: ANOVA
# ─────────────────────────────────────────────

def interpret_eta(eta):
    if eta < 0.01:   return "negligible"
    elif eta < 0.06: return "small"
    elif eta < 0.14: return "medium"
    else:            return "large"


def run_anova(df):
    df = df.copy()
    df["log_shares"] = np.log1p(df["total_shares"])
    df["belief"] = df["post_belief_value"].astype(str).astype("category")
    df["dist"] = df["agent_distribution"].astype("category")

    model = ols("log_shares ~ C(belief) * C(dist)", data=df).fit()
    anova_table = anova_lm(model, typ=2)

    ss_total = anova_table["sum_sq"].sum() + model.ssr
    anova_table["eta_squared"] = anova_table["sum_sq"] / ss_total

    return anova_table


def format_anova_report(anova_table):
    lines = []
    lines.append("=" * 60)
    lines.append("TWO-WAY ANOVA RESULTS")
    lines.append("Response variable: log(total_shares + 1)")
    lines.append("=" * 60)

    row_map = {
        "C(belief)":         "Post Belief Value",
        "C(dist)":           "Agent Distribution",
        "C(belief):C(dist)": "Interaction (Belief × Distribution)",
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
            lines.append(f"  η² (effect)    : {eta:.4f}  ({interpret_eta(eta)})")

    lines.append("\n" + "=" * 60)
    lines.append("Significance codes: *** p<0.001  ** p<0.01  * p<0.05")
    lines.append("Effect size (η²):   <0.01 negligible | 0.01-0.06 small |")
    lines.append("                    0.06-0.14 medium  | >0.14 large")
    lines.append("=" * 60)
    return "\n".join(lines)


# ─────────────────────────────────────────────
# STEP 3: HEATMAP
# ─────────────────────────────────────────────

def generate_heatmap(df, out_path):
    present_beliefs = [b for b in BELIEF_ORDER if b in df["post_belief_value"].values]
    present_dists   = [d for d in DISTRIBUTION_ORDER if d in df["agent_distribution"].values]

    pivot = df.pivot_table(
        values="total_shares",
        index="agent_distribution",
        columns="post_belief_value",
        aggfunc="mean",
    ).reindex(index=present_dists, columns=present_beliefs)

    pivot.index = [DIST_LABELS.get(d, d) for d in pivot.index]

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(
        pivot, ax=ax, cmap="YlOrRd", annot=True, fmt=".0f",
        linewidths=0.5, linecolor="#dddddd",
        cbar_kws={"label": "Mean Total Shares"},
    )

    ax.set_title("Mean Post Virality\nby Post Belief Value × Agent Distribution", fontsize=14, pad=16)
    ax.set_xlabel("Post Belief Value", fontsize=11)
    ax.set_ylabel("Agent Distribution (Red-Centrist-Blue ratio)", fontsize=11)

    n_cols    = len(present_beliefs)
    red_end   = sum(1 for b in present_beliefs if b <= -2)
    blue_start = sum(1 for b in present_beliefs if b < 2)

    for x, color in [(red_end, "#d62728"), (blue_start, "#1f77b4")]:
        ax.axvline(x=x, color=color, linewidth=1.5, linestyle="--", alpha=0.6)

    ax.text(red_end / 2,                  -0.6, "Red camp", ha="center", color="#d62728",  fontsize=9, transform=ax.transData)
    ax.text((red_end + blue_start) / 2,   -0.6, "Centrist", ha="center", color="#555555",  fontsize=9, transform=ax.transData)
    ax.text((blue_start + n_cols) / 2,    -0.6, "Blue camp", ha="center", color="#1f77b4", fontsize=9, transform=ax.transData)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Heatmap saved to '{out_path}'")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Hyp8 virality analysis: extract, ANOVA, heatmap.")
    parser.add_argument("--data_dir", default="output",  help="Directory containing JSON files")
    parser.add_argument("--out_dir",  default="analysis", help="Output directory for results")
    parser.add_argument("--n",        type=int, default=100, help="Total number of test cases (default: 100)")
    parser.add_argument("--p",        type=int, default=10,  help="Test cases per distribution (default: n/10)")
    args = parser.parse_args()

    p = args.p or args.n // 10
    os.makedirs(args.out_dir, exist_ok=True)

    # ── Extract ──────────────────────────────
    print(f"\n[1/3] Extracting records from '{args.data_dir}'...")
    print(f"      n={args.n}, p={p}")
    records = extract_records(args.data_dir, p)

    if not records:
        print("No records found. Check your data directory and file naming.")
        return

    df = pd.DataFrame(records)
    print(f"      {len(df)} post-observations extracted.")
    print(f"      Seeds found       : {sorted(df['seed'].unique())}")
    print(f"      Distributions     : {sorted(df['agent_distribution'].unique())}")
    print(f"      Belief values     : {sorted(df['post_belief_value'].unique())}")
    print(f"      Shares range      : {df['total_shares'].min()} – {df['total_shares'].max()}")

    # ── ANOVA ────────────────────────────────
    print("\n[2/3] Running two-way ANOVA...")
    anova_table = run_anova(df)
    report = format_anova_report(anova_table)
    print("\n" + report)

    report_path = os.path.join(args.out_dir, "anova_results.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\n      ANOVA report saved to '{report_path}'")

    # ── Heatmap ──────────────────────────────
    print("\n[3/3] Generating heatmap...")
    generate_heatmap(df, os.path.join(args.out_dir, "heatmap.png"))

    print("\nDone. Results are in the '{}' directory.".format(args.out_dir))


if __name__ == "__main__":
    main()