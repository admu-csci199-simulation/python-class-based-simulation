"""
Virality Analysis Script (Streaming / Memory-Safe)
====================================================
Hypothesis: The interest value of a post affects its virality
(rise and fall of interactions over time).

ARCHITECTURE — Two phases:
  Phase 1 (stream):  Read one JSON file at a time → extract compact feature
                     rows → append to a running list of small arrays → discard
                     raw JSON immediately. Memory stays flat regardless of size.
  Phase 2 (analyze): Load only the feature accumulator (~MBs) → run all
                     statistical tests → produce plots and report.

  The feature accumulator (features.npy) is written to --output_dir and is
  kept after the run so you can re-run Phase 2 alone with --skip_extraction.

Usage:
    # Full run
    python analyze_virality.py

    # Skip re-extraction if features.npy already exists
    python analyze_virality.py --skip_extraction

    # Override defaults
    python analyze_virality.py --input_dir output --output_dir results --n_agents 300
"""

import gc
import json
import argparse
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr, linregress

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────
INPUT_GLOB       = "hyp9-config*.json"
INPUT_DIR        = Path("output")
OUTPUT_DIR       = Path("analysis")
N_AGENTS_DEFAULT = 300
FEATURES_FILE    = "features.npy"   # compact accumulator written during Phase 1



# Feature column layout stored in the .npy array (all float64)
COL = {
    "interest_value":           0,
    "post_time":                1,
    "total_interactions_norm":  2,
    "peak_interactions_norm":   3,
    "time_to_peak":             4,   # NaN if post never active
    "growth_rate":              5,
    "decay_rate":               6,
    "active_duration":          7,
}
N_COLS = len(COL)


# ──────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────

def _extract_features_from_run(run, n_agents):
    """
    Extract one feature row per post from a single loaded run dict.
    Returns float64 array of shape (n_posts, N_COLS).
    The heavy time-series arrays are created and discarded inside this
    function before returning — only the 9-column summary survives.
    """
    static = run.get("static_post_information", {})
    shared = run.get("post_shared_data", {})
    rows   = []

    for post_id, s_data in shared.items():
        if post_id not in static:
            continue

        iv        = float(static[post_id]["interest_value"])
        post_time = int(static[post_id]["post_time"])

        # float32 during extraction halves RAM vs float64
        series = np.asarray(s_data, dtype=np.float32)
        active = series[post_time:].astype(np.float64)
        del series   # free full-length array immediately

        if active.size == 0 or active.sum() == 0:
            rows.append([iv, post_time, 0.0, 0.0, np.nan, 0.0, 0.0, 0.0])
            continue

        total  = float(active.sum())
        peak   = float(active.max())
        t_peak = int(np.argmax(active))

        growth_rate = 0.0
        if t_peak > 0:
            g_x = np.arange(t_peak + 1, dtype=np.float64)
            g_y = active[:t_peak + 1]
            if g_x.std() > 0:
                growth_rate = float(linregress(g_x, g_y).slope)

        decay_rate = 0.0
        tail = active[t_peak:]
        nonzero_after = np.where(tail > 0)[0]
        if nonzero_after.size > 1:
            d_seg = tail[:nonzero_after[-1] + 1]
            d_x   = np.arange(len(d_seg), dtype=np.float64)
            if d_x.std() > 0:
                decay_rate = float(linregress(d_x, d_seg).slope)

        active_duration = float((active > 0).sum())

        rows.append([
            iv, float(post_time),
            total / n_agents, peak / n_agents,
            float(t_peak), growth_rate, decay_rate,
            active_duration,
        ])

    return np.array(rows, dtype=np.float64) if rows else np.empty((0, N_COLS), dtype=np.float64)


# ──────────────────────────────────────────────────────────────
# PHASE 1 — STREAMING EXTRACTION
# ──────────────────────────────────────────────────────────────

def phase1_extract(inp_dir, out_dir, n_agents):
    """
    Stream through every matching JSON file one at a time.
    Peak memory ~= one JSON file + accumulated feature rows.
    """
    files = sorted(inp_dir.glob(INPUT_GLOB))
    if not files:
        raise FileNotFoundError(
            f"No files matching '{INPUT_GLOB}' found in '{inp_dir}'."
        )

    print(f"\nPhase 1 - Streaming extraction")
    print(f"  Found {len(files)} file(s) in '{inp_dir}'")

    chunks = []
    total_posts = 0

    for i, fpath in enumerate(files, 1):
        print(f"  [{i:>4}/{len(files)}] {fpath.name} ... ", end="", flush=True)

        with open(fpath, "r") as fh:
            run = json.load(fh)

        chunk = _extract_features_from_run(run, n_agents)

        del run      # release raw JSON before loading the next file
        gc.collect()

        chunks.append(chunk)
        total_posts += len(chunk)
        print(f"{len(chunk)} posts  (running total: {total_posts})")

    print(f"\n  Stacking {total_posts} feature rows ...")
    features = np.vstack(chunks) if chunks else np.empty((0, N_COLS), dtype=np.float64)
    del chunks
    gc.collect()

    out_path = out_dir / FEATURES_FILE
    np.save(out_path, features)
    print(f"  Saved -> {out_path}  ({features.nbytes / 1e6:.1f} MB,  shape={features.shape})")
    return out_path


# ──────────────────────────────────────────────────────────────
# PHASE 2 — STATISTICS & PLOTS
# ──────────────────────────────────────────────────────────────

def phase2_analyze(features_path, out_dir, n_agents):
    print(f"\nPhase 2 - Analysis")
    print(f"  Loading {features_path} ...")
    F = np.load(features_path)
    print(f"  Shape: {F.shape}  ({F.nbytes / 1e6:.1f} MB)")

    iv    = F[:, COL["interest_value"]]
    total = F[:, COL["total_interactions_norm"]]
    peak  = F[:, COL["peak_interactions_norm"]]
    t2p   = F[:, COL["time_to_peak"]]
    grow  = F[:, COL["growth_rate"]]
    decay = F[:, COL["decay_rate"]]
    dur   = F[:, COL["active_duration"]]

    # Virality metrics to test against interest_value
    # Growth rate and decay rate directly capture the rise/fall shape of the
    # interaction curve, which is the core definition of virality being tested.
    METRICS = {
        "Growth Rate": grow,
        "Decay Rate":  decay,
    }

    report_lines = [
        "=" * 60,
        "  VIRALITY HYPOTHESIS ANALYSIS - STATISTICAL REPORT",
        f"  n_agents = {n_agents}",
        f"  Total records analysed: {len(F)}",
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
    line("  related. Rank-based, so robust to skewed interaction count distributions.")
    line("  H0: No monotonic relationship between interest_value and the metric.")
    line("")
    for mname, marr in METRICS.items():
        valid = ~np.isnan(marr) & ~np.isnan(iv)
        iv_v, m_v = iv[valid], marr[valid]
        if len(iv_v) < 3:
            line(f"  {mname}: insufficient data")
            continue
        sp_r, sp_p = spearmanr(iv_v, m_v)
        line(f"  {mname}:")
        line(f"    rho={sp_r:+.4f}  p={sp_p:.4e}  {sig(sp_p)}")

    # ── 2. Linear Regression ─────────────────────────────────────
    section("2. LINEAR REGRESSION - Interest Value -> Virality Metrics")
    line("  Quantifies direction, magnitude, and goodness-of-fit of the linear")
    line("  relationship. Slope = expected change in metric per 1-unit IV increase.")
    line("  H0: slope = 0 (interest_value has no linear effect on the metric).")
    line("")
    for mname, marr in METRICS.items():
        valid = ~np.isnan(marr) & ~np.isnan(iv)
        if valid.sum() < 3:
            continue
        slope, intercept, r_val, p_val, se = linregress(iv[valid], marr[valid])
        line(f"  {mname}:")
        line(f"    slope={slope:.6f}  intercept={intercept:.6f}")
        line(f"    R2={r_val**2:.4f}  p={p_val:.4e}  SE={se:.6f}  {sig(p_val)}")

    # ── 3. Summary ───────────────────────────────────────────────
    section("3. SUMMARY")
    line("  Significance codes: *** p<0.001  ** p<0.01  * p<0.05  ns=not significant")
    line("  Refer to generated plots for visual confirmation.")

    report_path = out_dir / "statistical_report.txt"
    report_path.write_text("\n".join(report_lines))
    print(f"  Statistical report -> {report_path}")

    # ── PLOTS ────────────────────────────────────────────────────
    print("  Generating plots ...")

    # Fig 1 — Scatter: IV vs growth rate and decay rate with regression line
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Interest Value vs Virality Metrics\n(scatter + linear regression)",
                 fontsize=13, fontweight="bold")
    for ax, (ylabel, yarr) in zip(axes.ravel(), METRICS.items()):
        valid = ~np.isnan(yarr)
        ax.scatter(iv[valid], yarr[valid], color="#3498db", alpha=0.35, s=12, edgecolors="none")
        if valid.sum() > 2:
            sl, ic, r_val, *_ = linregress(iv[valid], yarr[valid])
            xr = np.linspace(iv[valid].min(), iv[valid].max(), 100)
            ax.plot(xr, sl * xr + ic, color="#e74c3c", linewidth=1.5,
                    label=f"R²={r_val**2:.3f}")
            ax.legend(fontsize=8, loc="upper left")
        ax.set_xlabel("Interest Value", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.tick_params(labelsize=8)
    plt.tight_layout()
    fig.savefig(out_dir / "fig1_scatter_regression.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("    fig1_scatter_regression.png")

    # Fig 2 — Spearman rho bar chart (IV vs each metric)
    rho_vals, rho_pvals, rho_labels = [], [], []
    for mname, marr in METRICS.items():
        valid = ~np.isnan(marr) & ~np.isnan(iv)
        if valid.sum() < 3:
            continue
        r, p = spearmanr(iv[valid], marr[valid])
        rho_vals.append(r)
        rho_pvals.append(p)
        rho_labels.append(mname)

    fig, ax = plt.subplots(figsize=(10, 5))
    bar_colors = ["#2ecc71" if r >= 0 else "#e74c3c" for r in rho_vals]
    bars = ax.barh(rho_labels, rho_vals, color=bar_colors, edgecolor="white", alpha=0.85)
    ax.axvline(0, color="black", linewidth=0.8)
    for bar, p in zip(bars, rho_pvals):
        s = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        x = bar.get_width()
        ax.text(x + (0.01 if x >= 0 else -0.01), bar.get_y() + bar.get_height() / 2,
                s, va="center", ha="left" if x >= 0 else "right", fontsize=9)
    ax.set_xlabel("Spearman rho", fontsize=10)
    ax.set_title("Spearman Correlation: Interest Value vs Virality Metrics", fontsize=11, fontweight="bold")
    ax.set_xlim(-1, 1)
    plt.tight_layout()
    fig.savefig(out_dir / "fig2_spearman_bar.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("    fig2_spearman_bar.png")

    print(f"\n  All outputs saved to: {out_dir}/")


# ──────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Virality hypothesis analysis (streaming)")
    parser.add_argument("--input_dir",       default=str(INPUT_DIR))
    parser.add_argument("--output_dir",      default=str(OUTPUT_DIR))
    parser.add_argument("--n_agents",        type=int, default=N_AGENTS_DEFAULT)
    parser.add_argument("--skip_extraction", action="store_true",
                        help="Skip Phase 1 and reuse existing features.npy")
    args = parser.parse_args()

    inp_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Virality Hypothesis Analysis  (n_agents={args.n_agents})")
    print(f"{'='*60}")

    features_path = out_dir / FEATURES_FILE

    if args.skip_extraction:
        if not features_path.exists():
            raise FileNotFoundError(
                f"--skip_extraction set but {features_path} does not exist. "
                "Run without --skip_extraction first."
            )
        print(f"\nSkipping extraction - reusing {features_path}")
    else:
        phase1_extract(inp_dir, out_dir, args.n_agents)

    phase2_analyze(features_path, out_dir, args.n_agents)
    print("\nDone.\n")


if __name__ == "__main__":
    main()