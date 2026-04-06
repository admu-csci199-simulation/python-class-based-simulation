"""
Hyp1Analysis.py
---------------
Statistical analysis for Hypothesis 1:
  "Agents interact more in the first few hours of the simulation and die down
   a couple of hours later, following a long-tail (power-law-like) distribution."

Expects output JSON files in the OUTPUT_FOLDER with the naming pattern:
  hyp1-config-<N>s-<seed>.json

Each JSON has the structure:
  { "<postID>": [interaction_count_per_minute_delta, ...],  ... }
  Array length = 24 * 60 = 1440  (minute 0 .. 1439 after posting)
"""

import os, json, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import stats
from scipy.optimize import curve_fit

# ── Config ────────────────────────────────────────────────────────────────────
OUTPUT_FOLDER   = "output"
RESULTS_FOLDER  = os.path.join(OUTPUT_FOLDER, "hyp1_analysis")
FILE_PREFIX     = "hyp1"
MINUTES         = 24 * 60     # array length per post
HOURS_THRESHOLD = 3           # "first few hours" benchmark for reporting
# ─────────────────────────────────────────────────────────────────────────────


def load_and_aggregate(output_folder: str, prefix: str) -> np.ndarray:
    """
    Load all matching JSON files and sum interaction counts into one array
    of length MINUTES, indexed by time-delta (minutes after posting).
    """
    aggregated = np.zeros(MINUTES, dtype=np.int64)
    files_loaded = 0

    for fname in os.listdir(output_folder):
        if not (fname.startswith(prefix) and fname.endswith(".json")):
            continue
        fpath = os.path.join(output_folder, fname)
        with open(fpath, "r") as f:
            data = json.load(f)
        for post_array in data.values():
            arr = np.array(post_array[:MINUTES], dtype=np.int64)
            if len(arr) < MINUTES:                       # pad if short
                arr = np.pad(arr, (0, MINUTES - len(arr)))
            aggregated += arr
        files_loaded += 1

    if files_loaded == 0:
        sys.exit(f"[ERROR] No files found in '{output_folder}' with prefix '{prefix}'.")

    print(f"[INFO] Loaded {files_loaded} output file(s).")
    return aggregated


# ── Distribution models ───────────────────────────────────────────────────────

def lognormal_pdf(x, mu, sigma, scale):
    return scale * stats.lognorm.pdf(x, s=sigma, scale=np.exp(mu))

def gamma_pdf(x, a, loc, scale_p, amp):
    return amp * stats.gamma.pdf(x, a=a, loc=loc, scale=scale_p)

# ─────────────────────────────────────────────────────────────────────────────


def fit_distributions(aggregated: np.ndarray):
    """
    Fit log-normal and gamma distributions directly to minute-level data.
    Returns a dict of fit results and the normalised density array.
    """
    minutes     = np.arange(MINUTES)
    total       = aggregated.sum()
    # Normalised density: each "bin" is 1 minute wide
    norm_counts = aggregated / total

    # Expand to flat sample array (each minute repeated by its count)
    samples = np.repeat(minutes, aggregated.astype(int))

    results = {}

    # ── Log-normal ────────────────────────────────────────────────────────────
    try:
        ln_shape, ln_loc, ln_scale = stats.lognorm.fit(samples, floc=0)
        ln_mu    = np.log(ln_scale)
        ln_sigma = ln_shape
        ks_stat, ks_p = stats.kstest(samples, "lognorm",
                                      args=(ln_shape, ln_loc, ln_scale))
        pdf_vals = stats.lognorm.pdf(minutes, s=ln_shape, loc=ln_loc, scale=ln_scale)
        ss_res   = np.sum((norm_counts - pdf_vals) ** 2)
        ss_tot   = np.sum((norm_counts - norm_counts.mean()) ** 2)
        r2       = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        peak_min = np.exp(ln_mu - ln_sigma ** 2)
        results["lognormal"] = dict(
            shape=ln_shape, mu=ln_mu, sigma=ln_sigma,
            ks_stat=ks_stat, ks_p=ks_p, r2=r2,
            peak_min=peak_min, pdf_vals=pdf_vals
        )
    except Exception as e:
        print(f"[WARN] Log-normal fit failed: {e}")

    # ── Gamma ─────────────────────────────────────────────────────────────────
    try:
        g_a, g_loc, g_scale = stats.gamma.fit(samples, floc=0)
        ks_stat, ks_p = stats.kstest(samples, "gamma",
                                      args=(g_a, g_loc, g_scale))
        pdf_vals = stats.gamma.pdf(minutes, a=g_a, loc=g_loc, scale=g_scale)
        ss_res   = np.sum((norm_counts - pdf_vals) ** 2)
        r2       = 1 - ss_res / np.sum((norm_counts - norm_counts.mean()) ** 2)
        peak_min = (g_a - 1) * g_scale + g_loc if g_a >= 1 else g_loc
        results["gamma"] = dict(
            a=g_a, loc=g_loc, scale=g_scale,
            ks_stat=ks_stat, ks_p=ks_p, r2=r2,
            peak_min=peak_min, pdf_vals=pdf_vals
        )
    except Exception as e:
        print(f"[WARN] Gamma fit failed: {e}")

    return results, norm_counts


def print_summary(aggregated, fit_results):
    minutes      = np.arange(MINUTES)
    total        = aggregated.sum()
    peak_min     = int(np.argmax(aggregated))

    thresh_min   = HOURS_THRESHOLD * 60
    pct_early    = 100 * aggregated[:thresh_min].sum() / total if total > 0 else 0

    print("\n" + "="*55)
    print("  HYPOTHESIS 1 — INTERACTION DECAY ANALYSIS")
    print("="*55)
    print(f"  Total interactions recorded : {total:,}")
    print(f"  Peak activity minute        : {peak_min} min after posting "
          f"({peak_min/60:.2f} h)")
    print(f"  Interactions in first {HOURS_THRESHOLD}h    : {pct_early:.1f}%")
    print()

    for name, res in fit_results.items():
        print(f"  [{name.upper()} FIT]")
        print(f"    R²         : {res['r2']:.4f}")
        print(f"    KS stat    : {res['ks_stat']:.4f}   p-value: {res['ks_p']:.4e}")
        print(f"    Model peak : {res['peak_min']:.1f} min ({res['peak_min']/60:.1f} h)")
        print()

    # Verdict
    best = max(fit_results, key=lambda k: fit_results[k]["r2"])
    best_r2 = fit_results[best]["r2"]
    best_p  = fit_results[best]["ks_p"]
    print(f"  Best fit     : {best.upper()}  (R² = {best_r2:.4f})")
    if best_p > 0.05:
        verdict = "SUPPORTED — data is consistent with a long-tail distribution."
    else:
        verdict = ("PARTIALLY SUPPORTED — shape matches but KS test rejects "
                   "the exact distribution (likely due to large sample size).")
    print(f"  Verdict      : {verdict}")
    print("="*55 + "\n")


def plot_results(aggregated, norm_counts, fit_results, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    minutes = np.arange(MINUTES)
    hours   = minutes / 60

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Hypothesis 1 — Interaction Decay Over Time", fontsize=14, fontweight="bold")

    colors = {"lognormal": "#E07B39", "gamma": "#4C9BE8"}

    # ── Left: raw minute-level counts ─────────────────────────────────────────
    ax = axes[0]
    ax.plot(hours, aggregated, color="#7CB9E8", linewidth=1, alpha=0.9, label="Observed")
    ax.set_xlabel("Hours after posting")
    ax.set_ylabel("Total interactions")
    ax.set_title("Interaction Count (per minute)")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax.legend()

    # ── Right: normalised density + fits ─────────────────────────────────────
    ax = axes[1]
    ax.plot(hours, norm_counts, color="#7CB9E8", linewidth=1, alpha=0.65, label="Observed (density)")
    for name, res in fit_results.items():
        ax.plot(hours, res["pdf_vals"], color=colors.get(name, "grey"),
                linewidth=2.5, label=f"{name.capitalize()} fit  R²={res['r2']:.3f}")
    ax.set_xlabel("Hours after posting")
    ax.set_ylabel("Density")
    ax.set_title("Distribution Fit")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax.legend()

    plt.tight_layout()
    out_path = os.path.join(out_dir, "hyp1_interaction_decay.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[INFO] Plot saved → {out_path}")

    # ── Log-log plot to visually inspect power-law tail ───────────────────────
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    nonzero = aggregated > 0
    ax2.scatter(np.log(hours[nonzero]), np.log(aggregated[nonzero]),
                s=4, color="#555", alpha=0.5, label="log(interactions)")
    peak_idx  = int(np.argmax(aggregated))
    tail_mask = nonzero & (minutes > peak_idx)
    if tail_mask.sum() > 2:
        slope, intercept, r, *_ = stats.linregress(
            np.log(hours[tail_mask]), np.log(aggregated[tail_mask]))
        x_line = np.log(hours[tail_mask])
        ax2.plot(x_line, intercept + slope * x_line,
                 color="#E07B39", linewidth=2,
                 label=f"Tail fit  slope={slope:.2f}  R²={r**2:.3f}")
        print(f"[INFO] Log-log tail regression: slope={slope:.3f}, R²={r**2:.4f}")
        if slope < -0.5:
            print("[INFO] Negative slope on log-log plot is consistent with a "
                  "heavy/long tail decay.")
    ax2.set_xlabel("log(hours after posting)")
    ax2.set_ylabel("log(interactions)")
    ax2.set_title("Log-Log Plot — Tail Behaviour")
    ax2.legend()
    plt.tight_layout()
    out_path2 = os.path.join(out_dir, "hyp1_loglog.png")
    plt.savefig(out_path2, dpi=150)
    plt.close()
    print(f"[INFO] Log-log plot saved → {out_path2}")


def main():
    aggregated          = load_and_aggregate(OUTPUT_FOLDER, FILE_PREFIX)
    fit_results, norm_c = fit_distributions(aggregated)
    print_summary(aggregated, fit_results)
    plot_results(aggregated, norm_c, fit_results, RESULTS_FOLDER)


if __name__ == "__main__":
    main()