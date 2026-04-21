"""
Analysis: Virality Decay Hypothesis (Hypothesis 7)
===================================================
"The time since a post is published affects the rise or fall of a post's virality."

Metrics computed per post per run:
  - Rise slope        : linear slope from posting time to peak (interactions/min)
  - Decay slope       : linear slope from peak to end (interactions/min)
  - Virality half-life: hours after peak for rate to fall to 50% of peak value
  - Exponential decay : fit A * exp(-lambda * t) to post-peak interaction rate;
                        1/lambda gives characteristic decay time (hours)

The fitted decay time is compared against the literature decay rate of λ = 1/5.4 /hour (τ = 5.4 hours).

Expected file layout (relative to this script):
  input/   hyp7-config-{config_num}s-{seed_num}.json  — config + Metadata
  output/  hyp7-config-{config_num}s-{seed_num}.json  — simulationData

Run:
  python hyp7_analysis.py
"""

import json
import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.optimize import curve_fit
from scipy.stats import ttest_1samp

# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR   = os.path.join(SCRIPT_DIR, "input")
OUTPUT_DIR  = os.path.join(SCRIPT_DIR, "output")
PLOTS_DIR   = os.path.join(SCRIPT_DIR, "plots")
NUM_CONFIGS = 100
NUM_SEEDS   = 100

SMOOTHING_WINDOW    = 30     # minutes; applied before fitting to reduce noise
LITERATURE_DECAY_HRS = 5.4  # literature decay rate λ_lit = 1/5.4 /hour → τ_lit = 5.4 hours

# ─────────────────────────────────────────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_runs(input_dir, output_dir, num_configs, num_seeds):
    """
    Output files are flat dicts: { "0": [interactions at elapsed_time=0, ...], ... }
    The array is indexed by elapsed time (minutes since posting), not absolute time.
    Metadata is read from the matching input config file.
    """
    runs = []
    skipped_missing  = 0
    skipped_bad      = 0

    for config_num in range(num_configs):
        for seed_num in range(100, num_seeds+100):
            config_path = os.path.join(input_dir,  f"hyp7-config-{config_num}.json")
            output_path = os.path.join(output_dir, f"hyp7-config-{config_num}s-{seed_num}.json")

            if not os.path.exists(config_path) or not os.path.exists(output_path):
                skipped_missing += 1
                continue

            with open(config_path) as f:
                config = json.load(f)
            with open(output_path) as f:
                sim = json.load(f)

            meta = config.get("Metadata", {})

            # Output is { "0": [...], "1": [...], ... } — one key per postID
            if "0" not in sim:
                skipped_bad += 1
                continue

            runs.append({
                "run_id":                 f"{config_num}s-{seed_num}",
                "posting_time":           meta.get("posting_time",   0),
                "is_misinfo":             meta.get("is_misinfo",     False),
                "belief_value":           meta.get("belief_value",   0),
                "interest_value":         meta.get("interest_value", 0),
                # Array is already relative to posting time (elapsed minutes)
                "interactions_over_time": sim["0"],
                "total_interactions":     sum(sim["0"]),
            })

    print(f"Loaded:          {len(runs)} runs")
    print(f"Skipped missing: {skipped_missing} (config or output file not found)")
    print(f"Skipped bad:     {skipped_bad} (output missing post 0)")
    return runs


# ─────────────────────────────────────────────────────────────────────────────
#  METRIC COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def smooth(arr, window):
    """Rolling-average smooth to reduce per-minute noise before fitting."""
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="same")


def exp_decay(t, A, lam):
    """Exponential decay model: A * exp(-lambda * t)."""
    return A * np.exp(-lam * t)


def compute_metrics(run):
    """
    Given one run, return a dict of all four virality metrics.
    All time units are converted to hours for interpretability.
    Returns None if the post had no interactions or fitting failed.
    """
    raw    = np.array(run["interactions_over_time"], dtype=float)

    if raw.sum() == 0:
        return None

    # Array is already indexed by elapsed time (minutes since posting)
    rate        = raw
    # Convert interactions/min → interactions/hour so all fitted λ values
    # and slopes are in consistent per-hour units, matching the literature
    rate_smooth = smooth(rate, SMOOTHING_WINDOW) * 60

    # ── peak ──────────────────────────────────────────────────────────────────
    peak_idx  = int(np.argmax(rate_smooth))    # minutes after posting
    peak_val  = rate_smooth[peak_idx]

    if peak_val == 0:
        return None

    # ── rise slope ────────────────────────────────────────────────────────────
    # Linear fit over [0, peak_idx] in hours
    if peak_idx >= 2:
        rise_t = np.arange(peak_idx + 1) / 60
        rise_y = rate_smooth[:peak_idx + 1]
        rise_coeffs = np.polyfit(rise_t, rise_y, 1)
        rise_slope  = rise_coeffs[0]   # interactions per hour
    else:
        rise_slope = np.nan

    # ── decay slope ───────────────────────────────────────────────────────────
    # Linear fit over [peak_idx, end] in hours
    decay_segment = rate_smooth[peak_idx:]
    if len(decay_segment) >= 2:
        decay_t = np.arange(len(decay_segment)) / 60
        decay_coeffs = np.polyfit(decay_t, decay_segment, 1)
        decay_slope  = decay_coeffs[0]   # interactions per hour (expected < 0)
    else:
        decay_slope = np.nan

    # ── virality half-life ────────────────────────────────────────────────────
    # First minute after peak where rate falls to <= 50% of peak
    half_target = peak_val * 0.5
    half_life   = np.nan
    for i, val in enumerate(decay_segment):
        if val <= half_target:
            half_life = i / 60   # hours after peak
            break

    # ── exponential decay fit ─────────────────────────────────────────────────
    # Fit A * exp(-lambda * t) to the post-peak portion
    # t is in hours so lambda is in units of 1/hour; 1/lambda = characteristic time (hours)
    fit_lambda    = np.nan
    fit_A         = np.nan
    fit_r_squared = np.nan

    if len(decay_segment) >= 10:
        fit_t = np.arange(len(decay_segment)) / 60
        fit_y = decay_segment
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                popt, _ = curve_fit(
                    exp_decay, fit_t, fit_y,
                    p0=[peak_val, 1.0 / LITERATURE_DECAY_HRS],   # initialise near literature value
                    bounds=([0, 1e-6], [np.inf, np.inf]),
                    maxfev=5000
                )
            fit_A, fit_lambda = popt

            # R² of the fit
            y_pred    = exp_decay(fit_t, fit_A, fit_lambda)
            ss_res    = np.sum((fit_y - y_pred) ** 2)
            ss_tot    = np.sum((fit_y - fit_y.mean()) ** 2)
            fit_r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

        except RuntimeError:
            pass   # fitting did not converge; leave as nan

    return {
        "peak_val":       peak_val,
        "peak_time_hrs":  peak_idx / 60,
        "rise_slope":     rise_slope,       # interactions / hour
        "decay_slope":    decay_slope,      # interactions / hour (negative)
        "half_life_hrs":  half_life,        # hours after peak
        "fit_A":          fit_A,
        "fit_lambda":     fit_lambda,       # 1/hour
        "fit_decay_time": 1.0 / fit_lambda if (not np.isnan(fit_lambda) and fit_lambda > 0) else np.nan,
        "fit_r_squared":  fit_r_squared,
        # carry through for the figure
        "rate_smooth":    rate_smooth,
        "peak_idx":       peak_idx,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  FIGURE — Exponential Decay Fit
# ─────────────────────────────────────────────────────────────────────────────

def plot_decay_fit(runs, metrics_list, plots_dir):
    """
    Two-panel figure for the paper:

    Left  — mean smoothed interaction rate across all runs (relative to posting
            time) with the mean fitted exponential decay curve overlaid, plus
            the literature decay curve for comparison.

    Right — distribution of fitted characteristic decay times (1/lambda) across
            all runs, with the literature value marked as a vertical line.
    """
    valid = [
        (r, m) for r, m in zip(runs, metrics_list)
        if m is not None and not np.isnan(m["fit_lambda"]) and m["fit_lambda"] > 0
    ]
    if not valid:
        print("  [skip] decay fit figure: no valid fits.")
        return

    # ── align all rate curves relative to posting time ──────────────────────
    # Trim to the length of the shortest curve so we can stack them
    min_len = min(len(m["rate_smooth"]) for _, m in valid)
    T_hrs   = np.arange(min_len) / 60

    stacked = np.array([m["rate_smooth"][:min_len] for _, m in valid])
    mean_rate = stacked.mean(axis=0)

    # ── mean fitted parameters ───────────────────────────────────────────────
    fit_As      = np.array([m["fit_A"]      for _, m in valid])
    fit_lambdas = np.array([m["fit_lambda"] for _, m in valid])
    mean_A      = fit_As.mean()
    mean_lam    = fit_lambdas.mean()

    # ── find mean peak to anchor decay curve ─────────────────────────────────
    mean_peak_idx = int(np.mean([m["peak_idx"] for _, m in valid]))

    decay_t_from_peak = np.arange(min_len - mean_peak_idx) / 60  # hours after peak

    # Simulation fitted curve (anchored at mean peak value)
    mean_peak_val = mean_rate[mean_peak_idx]
    sim_decay_curve   = mean_peak_val * np.exp(-mean_lam    * decay_t_from_peak)
    lit_decay_curve   = mean_peak_val * np.exp(-(1.0 / LITERATURE_DECAY_HRS) * decay_t_from_peak)

    decay_times = np.array([m["fit_decay_time"] for _, m in valid])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── left: mean rate + fitted curves ──────────────────────────────────────
    ax = axes[0]

    # faint individual run curves
    for _, m in valid[:min(200, len(valid))]:   # cap at 200 for legibility
        ax.plot(T_hrs[:len(m["rate_smooth"])], m["rate_smooth"][:min_len],
                color="#aaaaaa", alpha=0.07, linewidth=0.6)

    ax.plot(T_hrs, mean_rate, color="#333333", linewidth=2.0,
            label="Mean interaction rate (simulation)")

    # overlay decay curves starting from mean peak
    peak_t_hrs = mean_peak_idx / 60
    fit_t_abs  = peak_t_hrs + decay_t_from_peak

    ax.plot(fit_t_abs, sim_decay_curve,
            color="#e6550d", linewidth=2.2, linestyle="--",
            label=f"Fitted decay  (λ = {mean_lam:.4f} /h,  τ = {1/mean_lam:.2f} h)")
    ax.plot(fit_t_abs, lit_decay_curve,
            color="#3182bd", linewidth=2.2, linestyle=":",
            label=f"Literature decay  (λ = 1/5.4 /h,  τ = {LITERATURE_DECAY_HRS} h)")

    ax.axvline(peak_t_hrs, color="#333333", linewidth=0.9,
               linestyle="-.", alpha=0.6, label=f"Mean peak ({peak_t_hrs:.2f} h)")

    ax.set_xlabel("Time since posting (hours)")
    ax.set_ylabel(f"Interactions per minute\n({SMOOTHING_WINDOW}-min rolling avg)")
    ax.set_title("Interaction Rate with Exponential Decay Fit\nvs Literature Decay Curve")
    ax.legend(fontsize=8)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(6))

    # ── right: distribution of fitted decay times ─────────────────────────────
    ax2 = axes[1]
    ax2.hist(decay_times, bins=40, color="#e6550d", alpha=0.75, edgecolor="white",
             linewidth=0.5, label="Fitted τ = 1/λ per run")
    ax2.axvline(mean_lam and 1/mean_lam, color="#e6550d", linewidth=2.0,
                linestyle="--", label=f"Mean fitted τ = {1/mean_lam:.2f} h")
    ax2.axvline(LITERATURE_DECAY_HRS,    color="#3182bd", linewidth=2.0,
                linestyle=":",  label=f"Literature τ = {LITERATURE_DECAY_HRS} h")

    ax2.set_xlabel("Characteristic decay time τ = 1/λ (hours)")
    ax2.set_ylabel("Number of runs")
    ax2.set_title("Distribution of Fitted Decay Times\nvs Literature Value")
    ax2.legend(fontsize=8)

    fig.suptitle("Fig — Virality Decay: Exponential Fit vs Literature", fontsize=13, fontweight="bold")
    fig.tight_layout()

    path = os.path.join(plots_dir, "hyp7_decay_fit.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
#  SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def generate_summary(runs, metrics_list):
    valid = [m for m in metrics_list if m is not None]
    n_valid = len(valid)
    n_total = len(runs)

    def _stat(key):
        vals = [m[key] for m in valid if not np.isnan(m[key])]
        if not vals:
            return "N/A"
        return f"{np.mean(vals):.4f}  (SD {np.std(vals):.4f},  median {np.median(vals):.4f},  n={len(vals)})"

    lines = []

    def section(title):
        lines.append("")
        lines.append("=" * 65)
        lines.append(f"  {title}")
        lines.append("=" * 65)

    def row(label, value):
        lines.append(f"  {label:<48} {value}")

    lines.append("=" * 65)
    lines.append("  HYPOTHESIS 7 — VIRALITY DECAY  |  SUMMARY REPORT")
    lines.append("=" * 65)
    row("Total runs", n_total)
    row("Runs with sufficient data for metrics", n_valid)

    # ── 1. Rise slope ─────────────────────────────────────────────────────────
    section("1. RISE SLOPE  (interactions / hour, posting time → peak)")
    row("Mean  (SD,  median)", _stat("rise_slope"))
    rise_vals = [m["rise_slope"] for m in valid if not np.isnan(m["rise_slope"])]
    if rise_vals:
        positive_pct = sum(1 for v in rise_vals if v > 0) / len(rise_vals)
        row("Fraction of runs with positive rise slope", f"{positive_pct:.1%}")

    # ── 2. Decay slope ────────────────────────────────────────────────────────
    section("2. DECAY SLOPE  (interactions / hour, peak → end)")
    row("Mean  (SD,  median)", _stat("decay_slope"))
    decay_slope_vals = [m["decay_slope"] for m in valid if not np.isnan(m["decay_slope"])]
    if decay_slope_vals:
        negative_pct = sum(1 for v in decay_slope_vals if v < 0) / len(decay_slope_vals)
        row("Fraction of runs with negative decay slope", f"{negative_pct:.1%}")

    # ── 3. Virality half-life ─────────────────────────────────────────────────
    section("3. VIRALITY HALF-LIFE  (hours after peak to reach 50% of peak rate)")
    row("Mean  (SD,  median)", _stat("half_life_hrs"))
    hl_vals = [m["half_life_hrs"] for m in valid if not np.isnan(m["half_life_hrs"])]
    row("Runs where half-life was reached before simulation end",
        f"{len(hl_vals)} / {n_valid}  ({len(hl_vals)/n_valid:.1%})")

    # ── 4. Exponential decay fit ──────────────────────────────────────────────
    section("4. EXPONENTIAL DECAY FIT   f(t) = A · exp(−λt)")

    fit_vals = [m for m in valid
                if not np.isnan(m["fit_lambda"]) and m["fit_lambda"] > 0]
    n_fit = len(fit_vals)
    row("Runs where fit converged", f"{n_fit} / {n_valid}  ({n_fit/n_valid:.1%})")

    if fit_vals:
        lambdas     = np.array([m["fit_lambda"]     for m in fit_vals])
        decay_times = np.array([m["fit_decay_time"] for m in fit_vals])
        r2_vals     = np.array([m["fit_r_squared"]  for m in fit_vals
                                if not np.isnan(m["fit_r_squared"])])

        row("Mean λ  (SD,  median)  [1/hour]",
            f"{lambdas.mean():.4f}  (SD {lambdas.std():.4f},  median {np.median(lambdas):.4f})")
        row("Mean τ = 1/λ  (SD,  median)  [hours]",
            f"{decay_times.mean():.4f}  (SD {decay_times.std():.4f},  median {np.median(decay_times):.4f})")
        if len(r2_vals):
            row("Mean R² of fit  (SD)",
                f"{r2_vals.mean():.4f}  (SD {r2_vals.std():.4f})")

        # ── comparison to literature ──────────────────────────────────────────
        section(f"5. COMPARISON TO LITERATURE  (λ_lit = 1/5.4 /hour,  τ_lit = {LITERATURE_DECAY_HRS} h)")

        lit_lam = 1.0 / LITERATURE_DECAY_HRS
        diff    = decay_times - LITERATURE_DECAY_HRS
        row("Literature decay rate λ_lit",          f"1/5.4 = {lit_lam:.4f} /hour")
        row("Literature decay time τ_lit",          f"{LITERATURE_DECAY_HRS} hours")
        row("Mean difference  τ_sim − τ_lit",       f"{diff.mean():.4f} h  (SD {diff.std():.4f})")
        row("Median difference τ_sim − τ_lit",      f"{np.median(diff):.4f} h")

        faster = sum(1 for d in diff if d < 0)
        slower = sum(1 for d in diff if d > 0)
        row("Runs where simulation decays FASTER than literature",
            f"{faster} / {n_fit}  ({faster/n_fit:.1%})")
        row("Runs where simulation decays SLOWER than literature",
            f"{slower} / {n_fit}  ({slower/n_fit:.1%})")

        # one-sample t-test: is the mean τ_sim significantly different from τ_lit?
        t_stat, p_val = ttest_1samp(decay_times, LITERATURE_DECAY_HRS)
        row(f"One-sample t-test  (H₀: τ_sim = τ_lit = {LITERATURE_DECAY_HRS} h,  i.e. λ_sim = 1/5.4 /h)",
            f"t = {t_stat:.3f},  p = {p_val:.4f}")
        row("Interpretation",
            "significant difference" if p_val < 0.05 else "no significant difference")

    lines.append("")
    lines.append("=" * 65)
    lines.append("  END OF REPORT")
    lines.append("=" * 65)
    lines.append("")

    print("\n".join(lines))


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    runs = load_runs(INPUT_DIR, OUTPUT_DIR, NUM_CONFIGS, NUM_SEEDS)

    if not runs:
        print(f"No runs loaded. Check that '{INPUT_DIR}' and '{OUTPUT_DIR}' exist "
              f"and contain hyp7-config-{{config_num}}s-{{seed_num}}.json files.")
        return

    metrics_list = [compute_metrics(r) for r in runs]

    plot_decay_fit(runs, metrics_list, PLOTS_DIR)
    generate_summary(runs, metrics_list)


if __name__ == "__main__":
    main()