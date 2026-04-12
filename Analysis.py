"""
Virality Analysis Script
========================
Hypothesis: The interest value of a post affects its virality
(rise and fall of interactions over time).

Usage:
    python analyze_virality.py [--output_dir OUTPUT_DIR] [--n_agents N]

Reads all files matching 'hyp9-config*.json' from ./output/
Produces summary plots and a statistical report. No intermediate CSVs.

Parameters:
    n_agents : total number of agents in simulation (default: 300)
"""

import os
import re
import json
import argparse
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from collections import defaultdict

from scipy import stats
from scipy.stats import (
    spearmanr, pearsonr, kruskal, mannwhitneyu,
    linregress, shapiro
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
INPUT_GLOB   = "hyp9-config*.json"
INPUT_DIR    = Path("output")
OUTPUT_DIR   = Path("analysis_output")
N_AGENTS_DEFAULT = 300
TIME_SLICES  = 2880

# Interest grouping boundaries (inclusive)
# Low: -10 to 0 | Mid: 1 to 10 | High: 11 to 21
GROUP_BINS   = [-11, 0, 10, 21]
GROUP_LABELS = ["Low (-10–0)", "Mid (1–10)", "High (11–21)"]


# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────

def load_all_files(input_dir: Path, glob: str) -> list[dict]:
    files = sorted(input_dir.glob(glob))
    if not files:
        raise FileNotFoundError(
            f"No files matching '{glob}' found in '{input_dir}'."
        )
    print(f"Found {len(files)} file(s):")
    runs = []
    for f in files:
        # print(f"  {f.name}")
        with open(f) as fh:
            runs.append(json.load(fh))
    return runs


def extract_virality_features(runs: list[dict], n_agents: int) -> list[dict]:
    """
    For every post in every run, compute virality metrics from its
    interaction time series (post_shared_data), anchored at post_time.
    Returns a flat list of record dicts.
    """
    records = []

    for run_idx, run in enumerate(runs):
        static  = run.get("static_post_information", {})
        shared  = run.get("post_shared_data", {})

        for post_id, s_data in shared.items():
            if post_id not in static:
                continue

            interest_val = static[post_id]["interest_value"]
            post_time    = int(static[post_id]["post_time"])

            series = np.array(s_data, dtype=float)  # length 2880

            # Slice from post_time onward
            active = series[post_time:]
            if active.size == 0 or active.sum() == 0:
                # Post never interacted — still record as zero-virality
                records.append({
                    "run": run_idx,
                    "post_id": post_id,
                    "interest_value": interest_val,
                    "post_time": post_time,
                    "total_interactions": 0.0,
                    "total_interactions_norm": 0.0,
                    "peak_interactions": 0.0,
                    "peak_interactions_norm": 0.0,
                    "time_to_peak": np.nan,
                    "growth_rate": 0.0,
                    "decay_rate": 0.0,
                    "active_duration": 0,
                    "interest_group": _group_label(interest_val),
                })
                continue

            total  = float(active.sum())
            peak   = float(active.max())
            t_peak = int(np.argmax(active))          # relative to post_time

            # Growth rate: slope from post_time to peak (if peak > 0)
            growth_rate = 0.0
            if t_peak > 0:
                g_x = np.arange(t_peak + 1, dtype=float)
                g_y = active[: t_peak + 1]
                if g_x.std() > 0:
                    growth_rate = float(linregress(g_x, g_y).slope)

            # Decay rate: slope from peak to last nonzero
            decay_rate = 0.0
            nonzero_after = np.where(active[t_peak:] > 0)[0]
            if nonzero_after.size > 1:
                d_seg = active[t_peak: t_peak + nonzero_after[-1] + 1]
                d_x   = np.arange(len(d_seg), dtype=float)
                if d_x.std() > 0:
                    decay_rate = float(linregress(d_x, d_seg).slope)

            # Active duration: number of time slices with > 0 interactions
            active_duration = int((active > 0).sum())

            records.append({
                "run": run_idx,
                "post_id": post_id,
                "interest_value": interest_val,
                "post_time": post_time,
                "total_interactions": total,
                "total_interactions_norm": total / n_agents,
                "peak_interactions": peak,
                "peak_interactions_norm": peak / n_agents,
                "time_to_peak": float(t_peak),
                "growth_rate": growth_rate,
                "decay_rate": decay_rate,
                "active_duration": active_duration,
                "interest_group": _group_label(interest_val),
            })

    return records


def _group_label(iv: float) -> str:
    if iv <= 0:
        return GROUP_LABELS[0]
    elif iv <= 10:
        return GROUP_LABELS[1]
    else:
        return GROUP_LABELS[2]


def records_to_arrays(records: list[dict]) -> dict:
    """Convert list of dicts to dict of numpy arrays for easy slicing."""
    keys = [
        "interest_value", "total_interactions", "total_interactions_norm",
        "peak_interactions", "peak_interactions_norm", "time_to_peak",
        "growth_rate", "decay_rate", "active_duration"
    ]
    out = {k: np.array([r[k] for r in records], dtype=float) for k in keys}
    out["interest_group"] = np.array([r["interest_group"] for r in records])
    return out


# ──────────────────────────────────────────────
# STATISTICAL TESTS
# ──────────────────────────────────────────────

def run_statistics(arrays: dict, report_lines: list[str]):

    iv    = arrays["interest_value"]
    total = arrays["total_interactions_norm"]
    peak  = arrays["peak_interactions_norm"]
    t2p   = arrays["time_to_peak"]
    grow  = arrays["growth_rate"]
    decay = arrays["decay_rate"]
    dur   = arrays["active_duration"]
    grp   = arrays["interest_group"]

    def section(title):
        sep = "=" * 60
        report_lines.extend(["", sep, f"  {title}", sep])

    def line(text=""):
        report_lines.append(text)

    # ── 1. Descriptive stats per group ──────────────────────────
    section("1. DESCRIPTIVE STATISTICS BY INTEREST GROUP")
    metrics = {
        "Total Interactions (norm)": total,
        "Peak Interactions (norm)":  peak,
        "Growth Rate":               grow,
        "Decay Rate":                decay,
        "Active Duration (slices)":  dur,
    }
    for gname in GROUP_LABELS:
        mask = grp == gname
        line(f"\n  ── {gname}  (n={mask.sum()}) ──")
        for mname, marr in metrics.items():
            sub = marr[mask]
            if len(sub) == 0:
                continue
            line(f"    {mname}:")
            line(f"      mean={sub.mean():.4f}  median={np.median(sub):.4f}"
                 f"  std={sub.std():.4f}  min={sub.min():.4f}  max={sub.max():.4f}")

    # ── 2. Normality check (Shapiro-Wilk on full sample) ────────
    section("2. NORMALITY CHECK (Shapiro-Wilk)")
    line("  (Determines whether parametric or non-parametric tests are primary)")
    for mname, marr in metrics.items():
        if len(marr) >= 3:
            stat, p = shapiro(marr[:5000])   # shapiro limit
            line(f"  {mname}: W={stat:.4f}, p={p:.4e}  "
                 f"→ {'NORMAL' if p > 0.05 else 'NOT normal'} (α=0.05)")

    # ── 3. Correlation: interest_value vs virality metrics ──────
    section("3. CORRELATION — Interest Value vs Virality Metrics")
    line("  Spearman (rank) and Pearson (linear) correlations.")
    line("  Primary test: Spearman (robust to non-normality).\n")
    corr_targets = {
        "Total Interactions (norm)": total,
        "Peak Interactions (norm)":  peak,
        "Time to Peak":              t2p,
        "Growth Rate":               grow,
        "Decay Rate":                decay,
        "Active Duration":           dur,
    }
    for mname, marr in corr_targets.items():
        valid = ~np.isnan(marr) & ~np.isnan(iv)
        iv_v, m_v = iv[valid], marr[valid]
        if len(iv_v) < 3:
            line(f"  {mname}: insufficient data")
            continue
        sp_r, sp_p = spearmanr(iv_v, m_v)
        pe_r, pe_p = pearsonr(iv_v, m_v)
        sig_sp = "***" if sp_p < 0.001 else ("**" if sp_p < 0.01 else ("*" if sp_p < 0.05 else "ns"))
        sig_pe = "***" if pe_p < 0.001 else ("**" if pe_p < 0.01 else ("*" if pe_p < 0.05 else "ns"))
        line(f"  {mname}:")
        line(f"    Spearman r={sp_r:+.4f}  p={sp_p:.4e}  {sig_sp}")
        line(f"    Pearson  r={pe_r:+.4f}  p={pe_p:.4e}  {sig_pe}")

    # ── 4. Kruskal-Wallis (group comparison) ────────────────────
    section("4. KRUSKAL-WALLIS TEST — Group Differences in Virality")
    line("  Tests whether at least one interest group differs significantly.")
    line("  H0: All groups have the same distribution of virality.\n")

    groups_data = {gname: [] for gname in GROUP_LABELS}
    for gname in GROUP_LABELS:
        mask = grp == gname
        groups_data[gname] = total[mask]

    valid_groups = [v for v in groups_data.values() if len(v) >= 2]
    if len(valid_groups) >= 2:
        kw_stat, kw_p = kruskal(*valid_groups)
        line(f"  Kruskal-Wallis H={kw_stat:.4f}, p={kw_p:.4e}")
        line(f"  → {'SIGNIFICANT difference' if kw_p < 0.05 else 'No significant difference'} across interest groups (α=0.05)")
    else:
        line("  Insufficient groups for Kruskal-Wallis test.")

    # ── 5. Mann-Whitney U (pairwise) ────────────────────────────
    section("5. MANN-WHITNEY U — Pairwise Group Comparisons")
    line("  Bonferroni-corrected α = 0.05 / 3 = 0.0167\n")
    pairs = [
        (GROUP_LABELS[0], GROUP_LABELS[1]),
        (GROUP_LABELS[0], GROUP_LABELS[2]),
        (GROUP_LABELS[1], GROUP_LABELS[2]),
    ]
    bonferroni_alpha = 0.05 / len(pairs)
    for g1, g2 in pairs:
        a, b = groups_data[g1], groups_data[g2]
        if len(a) < 2 or len(b) < 2:
            line(f"  {g1} vs {g2}: insufficient data")
            continue
        u_stat, u_p = mannwhitneyu(a, b, alternative="two-sided")
        sig = "SIGNIFICANT" if u_p < bonferroni_alpha else "not significant"
        line(f"  {g1} vs {g2}:")
        line(f"    U={u_stat:.1f}, p={u_p:.4e}  → {sig} (Bonferroni α={bonferroni_alpha:.4f})")

    # ── 6. Linear Regression ─────────────────────────────────────
    section("6. LINEAR REGRESSION — Interest Value → Virality Metrics")
    line("  OLS regression with significance of slope.\n")
    reg_targets = {
        "Total Interactions (norm)": total,
        "Peak Interactions (norm)":  peak,
        "Growth Rate":               grow,
    }
    for mname, marr in reg_targets.items():
        valid = ~np.isnan(marr) & ~np.isnan(iv)
        iv_v, m_v = iv[valid].reshape(-1, 1), marr[valid]
        if len(iv_v) < 3:
            continue
        slope, intercept, r_val, p_val, se = linregress(iv_v.ravel(), m_v)
        r2 = r_val ** 2
        line(f"  {mname}:")
        line(f"    slope={slope:.6f}  intercept={intercept:.6f}")
        line(f"    R²={r2:.4f}  p={p_val:.4e}  SE={se:.6f}")
        sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns"))
        line(f"    Significance: {sig}")

    # ── 7. Summary ───────────────────────────────────────────────
    section("7. SUMMARY & INTERPRETATION")
    line("  Significance codes:  *** p<0.001  ** p<0.01  * p<0.05  ns=not significant")
    line("")
    line("  The central hypothesis is that interest_value positively correlates")
    line("  with post virality (total interactions, peak, growth rate).")
    line("")
    line("  Key findings are reported in sections 3–6 above.")
    line("  Refer to the generated plots for visual confirmation.")


# ──────────────────────────────────────────────
# PLOTTING
# ──────────────────────────────────────────────

PALETTE = {
    GROUP_LABELS[0]: "#e74c3c",
    GROUP_LABELS[1]: "#f39c12",
    GROUP_LABELS[2]: "#2ecc71",
}

def plot_all(arrays: dict, records: list[dict], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    iv    = arrays["interest_value"]
    total = arrays["total_interactions_norm"]
    peak  = arrays["peak_interactions_norm"]
    grow  = arrays["growth_rate"]
    decay = arrays["decay_rate"]
    dur   = arrays["active_duration"]
    t2p   = arrays["time_to_peak"]
    grp   = arrays["interest_group"]

    # ── Fig 1: Scatter matrix — IV vs virality metrics ──────────
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Interest Value vs Virality Metrics", fontsize=14, fontweight="bold")

    scatter_pairs = [
        ("Total Interactions\n(norm by agents)", total),
        ("Peak Interactions\n(norm by agents)",  peak),
        ("Growth Rate",                           grow),
        ("Decay Rate",                            decay),
        ("Active Duration\n(time slices)",        dur),
        ("Time to Peak\n(slices from post_time)", t2p),
    ]
    colors = np.array([PALETTE[g] for g in grp])
    for ax, (ylabel, yarr) in zip(axes.ravel(), scatter_pairs):
        valid = ~np.isnan(yarr)
        ax.scatter(iv[valid], yarr[valid], c=colors[valid], alpha=0.5, s=18, edgecolors="none")
        # trend line
        if valid.sum() > 2:
            slope, intercept, *_ = linregress(iv[valid], yarr[valid])
            xr = np.linspace(iv[valid].min(), iv[valid].max(), 100)
            ax.plot(xr, slope * xr + intercept, "k--", linewidth=1.2, alpha=0.7)
        ax.set_xlabel("Interest Value", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.tick_params(labelsize=8)

    # Legend
    from matplotlib.patches import Patch
    handles = [Patch(color=PALETTE[g], label=g) for g in GROUP_LABELS]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9,
               title="Interest Group", title_fontsize=9, bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(out_dir / "fig1_scatter_iv_vs_virality.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig1_scatter_iv_vs_virality.png")

    # ── Fig 2: Box plots — virality by interest group ────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Virality Distribution by Interest Group", fontsize=13, fontweight="bold")

    box_metrics = [
        ("Total Interactions (norm)", total),
        ("Peak Interactions (norm)",  peak),
        ("Active Duration (slices)",  dur),
    ]
    for ax, (title, marr) in zip(axes, box_metrics):
        data_by_group = [marr[grp == g] for g in GROUP_LABELS]
        bp = ax.boxplot(data_by_group, patch_artist=True, medianprops=dict(color="black", linewidth=2))
        for patch, gname in zip(bp["boxes"], GROUP_LABELS):
            patch.set_facecolor(PALETTE[gname])
            patch.set_alpha(0.75)
        ax.set_xticklabels(GROUP_LABELS, fontsize=8, rotation=10)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(title, fontsize=8)
        ax.tick_params(labelsize=8)
    plt.tight_layout()
    fig.savefig(out_dir / "fig2_boxplots_by_group.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig2_boxplots_by_group.png")

    # ── Fig 3: Mean interaction timeseries by group (aggregated) ─
    # Collect all time series per group
    group_series = defaultdict(list)
    for r in records:
        gname = r["interest_group"]
        series = None  # will be extracted below

    # Re-read series from records requires raw data; store it during extract
    # We pass raw_series through records (added below in extract step)
    if any("_raw_series" in r for r in records):
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.set_title("Mean Interaction Rate Over Time by Interest Group", fontsize=12, fontweight="bold")
        for gname in GROUP_LABELS:
            segs = [np.array(r["_raw_series"], dtype=float)
                    for r in records if r["interest_group"] == gname and "_raw_series" in r]
            if not segs:
                continue
            max_len = max(len(s) for s in segs)
            padded  = np.array([np.pad(s, (0, max_len - len(s))) for s in segs])
            mean_ts = padded.mean(axis=0)
            ax.plot(mean_ts, label=gname, color=PALETTE[gname], linewidth=1.2, alpha=0.85)
        ax.set_xlabel("Time Slice (from post_time=0)", fontsize=10)
        ax.set_ylabel("Mean Interactions (raw count)", fontsize=10)
        ax.legend(fontsize=9)
        plt.tight_layout()
        fig.savefig(out_dir / "fig3_mean_timeseries_by_group.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("  Saved fig3_mean_timeseries_by_group.png")

    # ── Fig 4: Correlation heatmap ────────────────────────────────
    metric_names = ["Interest Value", "Total (norm)", "Peak (norm)",
                    "Growth Rate", "Decay Rate", "Active Duration", "Time to Peak"]
    metric_arrays = [iv, total, peak, grow, decay, dur, t2p]

    corr_matrix = np.full((len(metric_names), len(metric_names)), np.nan)
    pval_matrix = np.full_like(corr_matrix, np.nan)
    for i, a in enumerate(metric_arrays):
        for j, b in enumerate(metric_arrays):
            valid = ~np.isnan(a) & ~np.isnan(b)
            if valid.sum() > 2:
                r, p = spearmanr(a[valid], b[valid])
                corr_matrix[i, j] = r
                pval_matrix[i, j] = p

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(corr_matrix, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="Spearman ρ")
    ax.set_xticks(range(len(metric_names)))
    ax.set_yticks(range(len(metric_names)))
    ax.set_xticklabels(metric_names, rotation=35, ha="right", fontsize=8)
    ax.set_yticklabels(metric_names, fontsize=8)
    ax.set_title("Spearman Correlation Heatmap (Virality Metrics)", fontsize=11, fontweight="bold")
    for i in range(len(metric_names)):
        for j in range(len(metric_names)):
            val = corr_matrix[i, j]
            if not np.isnan(val):
                sig = "*" if pval_matrix[i, j] < 0.05 else ""
                ax.text(j, i, f"{val:.2f}{sig}", ha="center", va="center", fontsize=7,
                        color="black" if abs(val) < 0.6 else "white")
    plt.tight_layout()
    fig.savefig(out_dir / "fig4_spearman_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig4_spearman_heatmap.png")

    # ── Fig 5: Distribution of interest values across all posts ───
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(iv, bins=range(-10, 23), color="#3498db", edgecolor="white", alpha=0.85)
    ax.set_xlabel("Interest Value", fontsize=10)
    ax.set_ylabel("Post Count", fontsize=10)
    ax.set_title("Distribution of Interest Values Across All Posts & Runs", fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_dir / "fig5_interest_value_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig5_interest_value_distribution.png")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Virality hypothesis analysis")
    parser.add_argument("--output_dir", default="analysis", help="Where to save results")
    parser.add_argument("--n_agents",   type=int, default=N_AGENTS_DEFAULT, help="Total agents in simulation")
    parser.add_argument("--input_dir",  default="output", help="Folder containing hyp9-config*.json files")
    args = parser.parse_args()

    out_dir   = Path(args.output_dir)
    inp_dir   = Path(args.input_dir)
    n_agents  = args.n_agents

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Virality Hypothesis Analysis")
    print(f"  n_agents = {n_agents}")
    print(f"{'='*60}\n")

    # Load data
    print("Loading simulation files...")
    runs = load_all_files(inp_dir, INPUT_GLOB)
    print(f"  Total runs loaded: {len(runs)}\n")

    # Extract features (and attach raw series for timeseries plot)
    print("Extracting virality features...")
    records = []
    for run_idx, run in enumerate(runs):
        static = run.get("static_post_information", {})
        shared = run.get("post_shared_data", {})
        for post_id, s_data in shared.items():
            if post_id not in static:
                continue
            interest_val = static[post_id]["interest_value"]
            post_time    = int(static[post_id]["post_time"])
            series       = np.array(s_data, dtype=float)
            active       = series[post_time:]

            if active.size == 0 or active.sum() == 0:
                records.append({
                    "run": run_idx, "post_id": post_id,
                    "interest_value": interest_val, "post_time": post_time,
                    "total_interactions": 0.0,
                    "total_interactions_norm": 0.0,
                    "peak_interactions": 0.0,
                    "peak_interactions_norm": 0.0,
                    "time_to_peak": np.nan,
                    "growth_rate": 0.0, "decay_rate": 0.0,
                    "active_duration": 0,
                    "interest_group": _group_label(interest_val),
                    "_raw_series": active.tolist(),
                })
                continue

            total  = float(active.sum())
            peak   = float(active.max())
            t_peak = int(np.argmax(active))

            growth_rate = 0.0
            if t_peak > 0:
                g_x = np.arange(t_peak + 1, dtype=float)
                g_y = active[:t_peak + 1]
                if g_x.std() > 0:
                    growth_rate = float(linregress(g_x, g_y).slope)

            decay_rate = 0.0
            nonzero_after = np.where(active[t_peak:] > 0)[0]
            if nonzero_after.size > 1:
                d_seg = active[t_peak: t_peak + nonzero_after[-1] + 1]
                d_x   = np.arange(len(d_seg), dtype=float)
                if d_x.std() > 0:
                    decay_rate = float(linregress(d_x, d_seg).slope)

            active_duration = int((active > 0).sum())

            records.append({
                "run": run_idx, "post_id": post_id,
                "interest_value": interest_val, "post_time": post_time,
                "total_interactions": total,
                "total_interactions_norm": total / n_agents,
                "peak_interactions": peak,
                "peak_interactions_norm": peak / n_agents,
                "time_to_peak": float(t_peak),
                "growth_rate": growth_rate, "decay_rate": decay_rate,
                "active_duration": active_duration,
                "interest_group": _group_label(interest_val),
                "_raw_series": active.tolist(),
            })

    print(f"  Total post-run records: {len(records)}\n")

    arrays = records_to_arrays(records)

    # Statistical report
    print("Running statistical tests...")
    report_lines = [
        "=" * 60,
        "  VIRALITY HYPOTHESIS ANALYSIS — STATISTICAL REPORT",
        f"  n_agents = {n_agents}",
        f"  Total records analysed: {len(records)}",
        "=" * 60,
    ]
    run_statistics(arrays, report_lines)
    report_text = "\n".join(report_lines)

    report_path = out_dir / "statistical_report.txt"
    report_path.write_text(report_text)
    print(f"\nStatistical report saved → {report_path}")

    # Plots
    print("\nGenerating plots...")
    plot_all(arrays, records, out_dir)

    print(f"\nAll outputs saved to: {out_dir}/")
    print("Done.\n")


if __name__ == "__main__":
    main()