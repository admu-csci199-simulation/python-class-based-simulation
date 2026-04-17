"""
Hyp7Analysis.py
---------------
Statistical analysis for Hypothesis 7:
  "The time since a post is published affects the rise or fall of a post's virality."

Expects output JSON files in OUTPUT_FOLDER matching HYP_PREFIX:
  { "<postID>": [interactions_per_minute_delta, ...] }  length = MINUTES

Analysis performed on the single aggregated curve (all posts, all runs):
  1. Spearman correlation  — does time-delta significantly predict interaction count?
  2. Rise / fall phases    — peak minute, rise slope, decay slope
  3. Virality half-life    — minutes to drop to 50 % of peak
  4. Exponential decay fit — decay rate k on the tail
"""

import os, json, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import stats
from scipy.optimize import curve_fit

# ── Config ────────────────────────────────────────────────────────────────────
OUTPUT_FOLDER  = "output"
RESULTS_FOLDER = os.path.join(OUTPUT_FOLDER, "hyp7_analysis")
HYP_PREFIX     = "hyp7"
MINUTES        = 24 * 60
# ─────────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_and_aggregate(output_folder: str, prefix: str) -> np.ndarray:
    """Sum all post interaction arrays across every output file into one curve."""
    aggregated   = np.zeros(MINUTES, dtype=np.float64)
    files_loaded = 0

    for fname in os.listdir(output_folder):
        if not (fname.startswith(prefix) and fname.endswith(".json")):
            continue
        with open(os.path.join(output_folder, fname)) as f:
            data = json.load(f)
        for arr in data.values():
            a = np.array(arr[:MINUTES], dtype=np.float64)
            if len(a) < MINUTES:
                a = np.pad(a, (0, MINUTES - len(a)))
            aggregated += a
        files_loaded += 1

    if not files_loaded:
        sys.exit(f"[ERROR] No output files found in '{output_folder}' with prefix '{prefix}'.")

    print(f"[INFO] Loaded {files_loaded} output file(s).")
    return aggregated


# ══════════════════════════════════════════════════════════════════════════════
#  ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def spearman_correlation(curve: np.ndarray):
    minutes    = np.arange(MINUTES, dtype=float)
    nonzero    = curve > 0
    sp_r, sp_p = stats.spearmanr(minutes[nonzero], curve[nonzero])
    return sp_r, sp_p


def rise_fall_phases(curve: np.ndarray):
    minutes    = np.arange(MINUTES, dtype=float)
    peak_idx   = int(np.argmax(curve))
    peak_count = curve[peak_idx]

    # Rise: minute 0 → peak
    if peak_idx > 0:
        rise_slope, *_ = stats.linregress(minutes[:peak_idx + 1], curve[:peak_idx + 1])
    else:
        rise_slope = 0.0

    # Decay: peak → last nonzero minute
    nonzero   = curve > 0
    tail_mask = nonzero & (minutes > peak_idx)
    if tail_mask.sum() >= 2:
        decay_slope, *_ = stats.linregress(minutes[tail_mask], curve[tail_mask])
    else:
        decay_slope = 0.0

    return peak_idx, peak_count, rise_slope, decay_slope


def virality_half_life(curve: np.ndarray, peak_idx: int, peak_count: float) -> float:
    """Minutes after peak for the curve to drop to <= 50 % of peak."""
    half_target = peak_count * 0.5
    tail        = curve[peak_idx:]
    below       = np.where(tail <= half_target)[0]
    return float(below[0]) if len(below) else float("nan")


def exponential_decay_fit(curve: np.ndarray, peak_idx: int, peak_count: float):
    """Fit y = A * exp(-k * t) on the tail after the peak."""
    minutes   = np.arange(MINUTES, dtype=float)
    nonzero   = curve > 0
    tail_mask = nonzero & (minutes > peak_idx)

    if tail_mask.sum() < 3:
        return float("nan"), float("nan"), float("nan")

    t_rel  = minutes[tail_mask] - peak_idx
    y_tail = curve[tail_mask]

    def exp_decay(t, A, k):
        return A * np.exp(-k * t)

    try:
        popt, _  = curve_fit(exp_decay, t_rel, y_tail,
                             p0=(peak_count, 0.01), maxfev=5000,
                             bounds=([0, 0], [np.inf, np.inf]))
        A_fit, k_fit = popt
        y_pred   = exp_decay(t_rel, *popt)
        ss_res   = np.sum((y_tail - y_pred) ** 2)
        ss_tot   = np.sum((y_tail - y_tail.mean()) ** 2)
        r2       = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        return A_fit, k_fit, r2
    except Exception as e:
        print(f"[WARN] Exponential fit failed: {e}")
        return float("nan"), float("nan"), float("nan")


# ══════════════════════════════════════════════════════════════════════════════
#  REPORTING
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(curve, sp_r, sp_p, peak_idx, peak_count,
                  rise_slope, decay_slope, half_life, A_fit, k_fit, r2):
    print("\n" + "="*57)
    print("  HYPOTHESIS 7 — VIRALITY VS TIME-SINCE-POSTING")
    print("="*57)
    print(f"  Total interactions        : {int(curve.sum()):,}")
    print()
    print(f"  [SPEARMAN CORRELATION]")
    print(f"    r = {sp_r:.4f}   p = {sp_p:.2e}")
    if sp_p < 0.05:
        direction = "decreases" if sp_r < 0 else "increases"
        print(f"    → Significant: virality {direction} with time.")
    else:
        print(f"    → Not significant at p < 0.05.")
    print()
    print(f"  [RISE / FALL PHASES]")
    print(f"    Peak at               : {peak_idx} min  ({peak_idx/60:.2f} h)")
    print(f"    Peak interaction count: {peak_count:.1f}")
    print(f"    Rise slope            : {rise_slope:+.4f} interactions/min")
    print(f"    Decay slope           : {decay_slope:+.4f} interactions/min")
    print()
    print(f"  [VIRALITY HALF-LIFE]")
    if not np.isnan(half_life):
        print(f"    {half_life:.0f} min after peak  ({half_life/60:.2f} h)")
    else:
        print(f"    Not reached within simulation window.")
    print()
    print(f"  [EXPONENTIAL DECAY FIT  y = A·e^(-k·t)]")
    if not np.isnan(k_fit):
        print(f"    A = {A_fit:.2f}   k = {k_fit:.5f}   R² = {r2:.4f}")
        print(f"    → Half-life of decay: {np.log(2)/k_fit:.1f} min  "
              f"({np.log(2)/k_fit/60:.2f} h)")
    else:
        print(f"    Fit could not be computed.")
    print()

    # Overall verdict
    sig      = sp_p < 0.05 and sp_r < 0
    good_fit = (not np.isnan(r2)) and r2 > 0.5
    if sig and good_fit:
        verdict = "SUPPORTED — time is a significant negative predictor of virality with a good exponential decay fit."
    elif sig:
        verdict = "PARTIALLY SUPPORTED — time is a significant predictor but decay shape is irregular."
    else:
        verdict = "NOT SUPPORTED — time does not significantly predict virality in this data."
    print(f"  Verdict: {verdict}")
    print("="*57 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
#  PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

def plot_results(curve, peak_idx, peak_count, k_fit, A_fit, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    hours   = np.arange(MINUTES) / 60
    minutes = np.arange(MINUTES, dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Hypothesis 7 — Virality vs Time Since Posting",
                 fontsize=14, fontweight="bold")

    # ── Left: full aggregated virality curve ──────────────────────────────────
    ax = axes[0]
    nonzero = curve > 0
    ax.plot(hours, curve, color="#7CB9E8", linewidth=1.2, alpha=0.9, label="Interactions")
    ax.axvline(peak_idx / 60, color="red", linestyle="--", linewidth=1.2,
               label=f"Peak @ {peak_idx} min")
    ax.set_title("Aggregated Virality Curve")
    ax.set_xlabel("Hours after posting")
    ax.set_ylabel("Total interactions")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax.legend()

    # ── Middle: log-scale to show decay clearly ───────────────────────────────
    ax = axes[1]
    ax.semilogy(hours[nonzero], curve[nonzero],
                color="#7CB9E8", linewidth=1.2, alpha=0.9, label="Interactions (log scale)")
    ax.axvline(peak_idx / 60, color="red", linestyle="--", linewidth=1.2,
               label=f"Peak @ {peak_idx} min")

    if not np.isnan(k_fit):
        tail_mask = nonzero & (minutes > peak_idx)
        t_rel     = minutes[tail_mask] - peak_idx
        fit_vals  = A_fit * np.exp(-k_fit * t_rel)
        ax.semilogy(hours[tail_mask], fit_vals, color="#E07B39", linewidth=2,
                    linestyle="--", label=f"Exp fit  k={k_fit:.4f}")

    ax.set_title("Log-Scale Virality + Decay Fit")
    ax.set_xlabel("Hours after posting")
    ax.set_ylabel("Interactions (log)")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax.legend()

    # ── Right: log-log tail to check power-law vs exponential ─────────────────
    ax = axes[2]
    tail_mask = nonzero & (minutes > peak_idx)
    if tail_mask.sum() > 2:
        log_h = np.log(hours[tail_mask])
        log_c = np.log(curve[tail_mask])
        ax.scatter(log_h, log_c, s=4, color="#555", alpha=0.5, label="log-log tail")
        slope, intercept, r, *_ = stats.linregress(log_h, log_c)
        ax.plot(log_h, intercept + slope * log_h, color="#E07B39", linewidth=2,
                label=f"slope={slope:.2f}  R²={r**2:.3f}")
    ax.set_title("Log-Log Tail (Power-Law Check)")
    ax.set_xlabel("log(hours after posting)")
    ax.set_ylabel("log(interactions)")
    ax.legend()

    plt.tight_layout()
    path = os.path.join(out_dir, "hyp7_virality.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[INFO] Plot saved → {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    curve      = load_and_aggregate(OUTPUT_FOLDER, HYP_PREFIX)
    sp_r, sp_p = spearman_correlation(curve)

    peak_idx, peak_count, rise_slope, decay_slope = rise_fall_phases(curve)
    half_life        = virality_half_life(curve, peak_idx, peak_count)
    A_fit, k_fit, r2 = exponential_decay_fit(curve, peak_idx, peak_count)

    print_summary(curve, sp_r, sp_p, peak_idx, peak_count,
                  rise_slope, decay_slope, half_life, A_fit, k_fit, r2)
    plot_results(curve, peak_idx, peak_count, k_fit, A_fit, RESULTS_FOLDER)


if __name__ == "__main__":
    main()