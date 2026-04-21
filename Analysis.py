"""
analyze_summary.py
==================
Runs Spearman correlation and linear regression on the completed summary CSV
produced by extract_chunk.py. Does not touch any raw JSON files.

Usage:
    python analyze_summary.py
    python analyze_summary.py --summary summary.csv --output_dir analysis_output
"""

import csv
import argparse
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr, linregress

warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────────
SUMMARY_FILE = Path("summary.csv")
OUTPUT_DIR   = Path("analysis")


# ── LOAD SUMMARY CSV ──────────────────────────────────────────

def load_summary(summary_path: Path) -> dict:
    """
    Reads the summary CSV into numpy arrays.
    Returns a dict keyed by column name.
    Skips rows where growth_rate and decay_rate are both zero
    (posts that never became active) only for metric columns —
    interest_value is always kept so counts are reported honestly.
    """
    iv_list, grow_list, decay_list, peak_list = [], [], [], []

    print(f"  Reading {summary_path} ...")
    with open(summary_path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            iv_list.append(float(row["interest_value"]))
            grow_list.append(float(row["growth_rate"]))
            decay_list.append(float(row["decay_rate"]))
            peak_list.append(float(row["peak_interactions_norm"]))

    return {
        "interest_value": np.array(iv_list,   dtype=np.float64),
        "growth_rate":    np.array(grow_list,  dtype=np.float64),
        "decay_rate":     np.array(decay_list, dtype=np.float64),
        "peak_norm":      np.array(peak_list,  dtype=np.float64),
    }


# ── STATISTICAL ANALYSIS ──────────────────────────────────────

def run_analysis(arrays: dict, out_dir: Path):
    iv    = arrays["interest_value"]
    grow  = arrays["growth_rate"]
    decay = arrays["decay_rate"]

    n_total        = len(iv)
    n_active_grow  = int((grow  != 0).sum())
    n_active_decay = int((decay != 0).sum())

    # Virality metrics — only use rows where the metric is nonzero
    # (zero means the post was never active, not a true zero effect)
    METRICS = {
        "Growth Rate": grow,
        "Decay Rate":  decay,
    }

    report_lines = [
        "=" * 60,
        "  VIRALITY HYPOTHESIS ANALYSIS - STATISTICAL REPORT",
        f"  Total records in summary : {n_total}",
        f"  Posts with nonzero growth rate  : {n_active_grow}",
        f"  Posts with nonzero decay rate   : {n_active_decay}",
        "=" * 60,
    ]

    def section(title):
        report_lines.extend(["", "=" * 60, f"  {title}", "=" * 60])

    def line(text=""):
        report_lines.append(text)

    def sig(p):
        return "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))

    # ── 1. Spearman Correlation ───────────────────────────────────
    section("1. SPEARMAN CORRELATION - Interest Value vs Virality Metrics")
    line("  Tests whether interest_value and each virality metric are monotonically")
    line("  related. Rank-based and robust to skewed interaction distributions.")
    line("  Only posts where the metric is nonzero are included (active posts only).")
    line("  H0: No monotonic relationship between interest_value and the metric.")
    line("")
    for mname, marr in METRICS.items():
        mask = marr != 0
        iv_v, m_v = iv[mask], marr[mask]
        if len(iv_v) < 3:
            line(f"  {mname}: insufficient active posts ({mask.sum()})")
            continue
        sp_r, sp_p = spearmanr(iv_v, m_v)
        line(f"  {mname}  (n={len(iv_v)}):")
        line(f"    rho={sp_r:+.4f}  p={sp_p:.4e}  {sig(sp_p)}")

    # ── 2. Linear Regression ─────────────────────────────────────
    section("2. LINEAR REGRESSION - Interest Value -> Virality Metrics")
    line("  Slope = expected change in metric per 1-unit increase in interest_value.")
    line("  Only posts where the metric is nonzero are included (active posts only).")
    line("  H0: slope = 0 (no linear effect).")
    line("")
    for mname, marr in METRICS.items():
        mask = marr != 0
        iv_v, m_v = iv[mask], marr[mask]
        if len(iv_v) < 3:
            line(f"  {mname}: insufficient active posts ({mask.sum()})")
            continue
        slope, intercept, r_val, p_val, se = linregress(iv_v, m_v)
        line(f"  {mname}  (n={len(iv_v)}):")
        line(f"    slope={slope:.6f}  intercept={intercept:.6f}")
        line(f"    R2={r_val**2:.4f}  p={p_val:.4e}  SE={se:.6f}  {sig(p_val)}")

    # ── 3. Summary ───────────────────────────────────────────────
    section("3. SUMMARY")
    line("  Significance codes: *** p<0.001  ** p<0.01  * p<0.05  ns=not significant")
    line("  Growth rate > 0 means the post gained interactions rapidly before peak.")
    line("  Decay rate < 0 means the post lost interactions after peak (expected).")
    line("  A positive rho on decay rate means slower decay (less negative slope)")
    line("  for higher interest-value posts, i.e. they sustain engagement longer.")
    line("  Refer to generated plots for visual confirmation.")

    report_path = out_dir / "statistical_report.txt"
    report_path.write_text("\n".join(report_lines))
    print(f"  Statistical report -> {report_path}")

    return METRICS, iv


# ── PLOTS ─────────────────────────────────────────────────────

def make_plots(arrays: dict, out_dir: Path):
    iv    = arrays["interest_value"]
    grow  = arrays["growth_rate"]
    decay = arrays["decay_rate"]

    METRICS = {
        "Growth Rate": grow,
        "Decay Rate":  decay,
    }

    print("  Generating plots ...")

    # Fig 1 — Scatter: IV vs growth rate and decay rate, active posts only
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Interest Value vs Virality Metrics (active posts only)\n"
                 "scatter + linear regression",
                 fontsize=12, fontweight="bold")

    for ax, (ylabel, marr) in zip(axes, METRICS.items()):
        mask  = marr != 0
        iv_v  = iv[mask]
        m_v   = marr[mask]
        ax.scatter(iv_v, m_v, color="#3498db", alpha=0.3, s=8, edgecolors="none")
        if mask.sum() > 2:
            sl, ic, r_val, *_ = linregress(iv_v, m_v)
            xr = np.linspace(iv_v.min(), iv_v.max(), 100)
            ax.plot(xr, sl * xr + ic, color="#e74c3c", linewidth=1.8,
                    label=f"R²={r_val**2:.3f}")
            ax.legend(fontsize=9, loc="upper left")
        ax.set_xlabel("Interest Value", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.tick_params(labelsize=8)

    plt.tight_layout()
    fig.savefig(out_dir / "fig1_scatter_regression.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("    fig1_scatter_regression.png")

    # Fig 2 — Spearman rho bar chart
    rho_vals, rho_pvals, rho_labels = [], [], []
    for mname, marr in METRICS.items():
        mask = marr != 0
        iv_v, m_v = iv[mask], marr[mask]
        if len(iv_v) < 3:
            continue
        r, p = spearmanr(iv_v, m_v)
        rho_vals.append(r)
        rho_pvals.append(p)
        rho_labels.append(mname)

    fig, ax = plt.subplots(figsize=(8, 4))
    bar_colors = ["#2ecc71" if r >= 0 else "#e74c3c" for r in rho_vals]
    bars = ax.barh(rho_labels, rho_vals, color=bar_colors, edgecolor="white", alpha=0.85)
    ax.axvline(0, color="black", linewidth=0.8)
    for bar, p in zip(bars, rho_pvals):
        s = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        x = bar.get_width()
        ax.text(x + (0.01 if x >= 0 else -0.01),
                bar.get_y() + bar.get_height() / 2,
                s, va="center", ha="left" if x >= 0 else "right", fontsize=10)
    ax.set_xlabel("Spearman rho", fontsize=10)
    ax.set_title("Spearman Correlation: Interest Value vs Virality Metrics",
                 fontsize=11, fontweight="bold")
    ax.set_xlim(-1, 1)
    plt.tight_layout()
    fig.savefig(out_dir / "fig2_spearman_bar.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("    fig2_spearman_bar.png")

    print(f"\n  All outputs saved to: {out_dir}/")


# ── MAIN ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run virality analysis on the completed summary CSV."
    )
    parser.add_argument("--summary",    default=str(SUMMARY_FILE),
                        help="Path to summary CSV produced by extract_chunk.py")
    parser.add_argument("--output_dir", default=str(OUTPUT_DIR),
                        help="Where to save the report and plots")
    args = parser.parse_args()

    summary_path = Path(args.summary)
    out_dir      = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not summary_path.exists():
        raise FileNotFoundError(
            f"Summary file not found: {summary_path}\n"
            "Run extract_chunk.py first to build it."
        )

    print(f"\n{'='*60}")
    print(f"  analyze_summary.py")
    print(f"  Summary : {summary_path}")
    print(f"  Output  : {out_dir}")
    print(f"{'='*60}\n")

    arrays = load_summary(summary_path)
    print(f"  Loaded {len(arrays['interest_value'])} records\n")

    run_analysis(arrays, out_dir)
    make_plots(arrays, out_dir)

    print("\nDone.\n")


if __name__ == "__main__":
    main()