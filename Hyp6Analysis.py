"""
Hyp6Analysis.py
---------------
Statistical analysis for Hypothesis 6:
  "Real information spreads because the original poster tends to have a massive
   following (broadcast), whereas misinformation thrives on agent-to-agent
   sharing resembling a virus-like infection (viral cascade)."

Expected output JSON format:
  {
    "<postID>": {
      "isMisinformation": true | false,
      "interactions": {
        "1": count,
        "2": count,
        ...
      }
    },
    ...
  }

Note: keys in "interactions" are layer numbers (strings).
      Layer 0 (the OP) is never recorded as an interaction.
"""

import os, json, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import stats

# ── Config ────────────────────────────────────────────────────────────────────
OUTPUT_FOLDER  = "output"
RESULTS_FOLDER = os.path.join(OUTPUT_FOLDER, "hyp6_analysis")
HYP_PREFIX     = "hyp6-config"
MIN_TOTAL      = 5
# ─────────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_posts(output_folder: str, prefix: str):
    """
    Load all matching output JSONs and average layer counts across runs.

    Returns:
      real_posts    : { postID: { layer_int: avg_count, ... } }
      misinfo_posts : same structure
    """
    accum: dict[str, dict] = {}
    files_loaded = 0

    for fname in sorted(os.listdir(output_folder)):
        if not (fname.startswith(prefix) and fname.endswith(".json")):
            continue
        with open(os.path.join(output_folder, fname)) as f:
            data = json.load(f)

        # Strip the seed suffix (e.g. "hyp6-config-0s-5.json" -> "hyp6-config-0")
        # so that the same config run across different seeds shares one accum entry,
        # but posts from different configs (which reuse IDs like "0", "1") stay separate.
        cfg_key = fname.rsplit("s-", 1)[0]

        for pid, entry in data.items():
            is_m   = entry["isMisinformation"]
            # JSON keys are strings; convert layer keys to int, values to float
            layers = {int(k): float(v) for k, v in entry["interactions"].items()}

            accum_key = f"{cfg_key}__{pid}"

            if accum_key not in accum:
                accum[accum_key] = {"is_misinfo": is_m, "layers": {}, "runs": 0}

            accum[accum_key]["runs"] += 1
            for layer, count in layers.items():
                accum[accum_key]["layers"][layer] = accum[accum_key]["layers"].get(layer, 0.0) + count

        files_loaded += 1

    if not files_loaded:
        sys.exit(f"[ERROR] No files found in '{output_folder}' with prefix '{prefix}'.")
    print(f"[INFO] Loaded {files_loaded} output file(s).")

    # Diagnostic: show unique config keys and runs distribution
    unique_cfgs  = len({k.rsplit("__", 1)[0] for k in accum})
    runs_counts  = [rec["runs"] for rec in accum.values()]
    print(f"[INFO] Unique configs found  : {unique_cfgs}")
    print(f"[INFO] Unique (config, post) : {len(accum)}")
    if runs_counts:
        print(f"[INFO] Seeds per post — min: {min(runs_counts)}  "
              f"max: {max(runs_counts)}  mean: {sum(runs_counts)/len(runs_counts):.1f}")
        if min(runs_counts) == max(runs_counts):
            print(f"[INFO] Every post averaged over exactly {min(runs_counts)} seed(s). ✓")
        else:
            print(f"[WARN] Uneven seed counts — some posts have fewer runs than others. "
                  f"Check that all config files ran the same number of seeds.")

    # Average counts across runs
    for rec in accum.values():
        n = rec["runs"]
        for layer in rec["layers"]:
            rec["layers"][layer] /= n

    real_posts    = {k: rec["layers"] for k, rec in accum.items() if not rec["is_misinfo"]}
    misinfo_posts = {k: rec["layers"] for k, rec in accum.items() if rec["is_misinfo"]}
    print(f"[INFO] Posts before MIN_TOTAL filter — real: {len(real_posts)},  misinfo: {len(misinfo_posts)}")
    return real_posts, misinfo_posts


# ══════════════════════════════════════════════════════════════════════════════
#  PER-POST METRICS
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(posts: dict) -> list[dict]:
    """
    For each post compute:
      mean_layer      — probability-weighted mean layer depth
      max_layer       — deepest layer that received ≥1 interaction
      cascade_ratio   — interactions(layer≥2) / interactions(layer 1)
                        NaN if layer 1 has zero interactions (no direct OP spread)
      layer1_fraction — fraction of all interactions at layer 1
    """
    records = []

    for pid, layers in posts.items():
        total = sum(layers.values())
        if total < MIN_TOTAL:
            continue

        layer_keys = sorted(layers.keys())
        counts     = np.array([layers[l] for l in layer_keys], dtype=float)
        layer_arr  = np.array(layer_keys, dtype=float)

        mean_layer = float(np.sum(layer_arr * counts) / counts.sum())
        max_layer  = int(max(layer_keys))

        l1_count   = layers.get(1, 0.0)
        deep_count = sum(v for l, v in layers.items() if l >= 2)

        layer1_fraction = l1_count / total if total > 0 else 0.0

        # Use NaN explicitly when layer 1 is absent rather than inf,
        # so mw_test's clean() drops it with a clear reason
        if l1_count > 0:
            cascade_ratio = deep_count / l1_count
        else:
            cascade_ratio = float("nan")
            print(f"[WARN] Post {pid} has no layer-1 interactions — "
                  f"cascade_ratio set to NaN.")

        records.append({
            "postID"          : pid,
            "total"           : total,
            "mean_layer"      : mean_layer,
            "max_layer"       : max_layer,
            "cascade_ratio"   : cascade_ratio,
            "layer1_fraction" : layer1_fraction,
        })

    return records


# ══════════════════════════════════════════════════════════════════════════════
#  AGGREGATED LAYER TOTALS
# ══════════════════════════════════════════════════════════════════════════════

def aggregate_layer_totals(posts: dict) -> dict[int, float]:
    combined: dict[int, float] = {}
    for layers in posts.values():
        for layer, count in layers.items():
            combined[layer] = combined.get(layer, 0.0) + count
    return combined


# ══════════════════════════════════════════════════════════════════════════════
#  STATISTICAL TESTS
# ══════════════════════════════════════════════════════════════════════════════

def mw_test(real_records, misinfo_records, key):
    """Mann-Whitney U test on a metric between the two groups."""
    def clean(records):
        return [x[key] for x in records
                if not np.isnan(x.get(key, float("nan")))
                and not np.isinf(x.get(key, float("nan")))]
    r_vals = clean(real_records)
    m_vals = clean(misinfo_records)
    if len(r_vals) < 2 or len(m_vals) < 2:
        return float("nan"), float("nan"), float("nan"), float("nan")
    u, p = stats.mannwhitneyu(r_vals, m_vals, alternative="two-sided")
    return np.median(r_vals), np.median(m_vals), u, p


# ══════════════════════════════════════════════════════════════════════════════
#  REPORTING
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(real_records, misinfo_records):
    metrics = [
        ("mean_layer",      "Mean layer depth",        "misinfo expected HIGHER"),
        ("max_layer",       "Max layer reached",       "misinfo expected HIGHER"),
        ("cascade_ratio",   "Cascade ratio (≥L2/L1)",  "misinfo expected HIGHER"),
        ("layer1_fraction", "Layer-1 fraction",        "real expected HIGHER"),
    ]

    print("\n" + "="*65)
    print("  HYPOTHESIS 6 — BROADCAST vs VIRAL SPREADING")
    print("="*65)
    print(f"  Real posts   : {len(real_records)}")
    print(f"  Misinfo posts: {len(misinfo_records)}")
    print()

    # Warn early if a group is too small for comparison
    if len(real_records) < 2:
        print(f"  [WARN] Only {len(real_records)} real post(s) found — "
              f"Mann-Whitney tests require >=2 per group. "
              f"Check that your configs generate real news posts.")
    if len(misinfo_records) < 2:
        print(f"  [WARN] Only {len(misinfo_records)} misinfo post(s) found — "
              f"Mann-Whitney tests require >=2 per group.")

    def safe_median(records, key):
        vals = [x[key] for x in records
                if not np.isnan(x.get(key, float("nan")))
                and not np.isinf(x.get(key, float("nan")))]
        return np.median(vals) if vals else float("nan")

    sig_flags = []
    for key, label, expectation in metrics:
        med_r = safe_median(real_records,    key)
        med_m = safe_median(misinfo_records, key)
        print(f"  [{label}]  ({expectation})")
        med_r_str = f"{med_r:.3f}" if not np.isnan(med_r) else "n/a"
        med_m_str = f"{med_m:.3f}" if not np.isnan(med_m) else "n/a"
        print(f"    Median — real: {med_r_str}   misinfo: {med_m_str}")
        _, _, u, p = mw_test(real_records, misinfo_records, key)
        sig = (not np.isnan(p)) and p < 0.05
        sig_flags.append(sig)
        if not np.isnan(p):
            print(f"    Mann-Whitney U={u:.1f}  p={p:.2e}  "
                  f"{'✓ significant' if sig else '✗ not significant'}")
        else:
            print(f"    Mann-Whitney — insufficient data for test.")
        print()

    n_sig = sum(sig_flags)
    print(f"  {n_sig}/{len(sig_flags)} metrics statistically significant (p < 0.05).")
    if n_sig >= 3:
        verdict = "STRONGLY SUPPORTED — misinfo consistently shows deeper, more cascading spread."
    elif n_sig >= 2:
        verdict = "PARTIALLY SUPPORTED — some metrics align with the hypothesis but not all."
    else:
        verdict = "NOT SUPPORTED — no consistent statistical difference between spreading patterns."
    print(f"  Verdict: {verdict}")
    print("="*65 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
#  PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

def plot_layer_distribution(real_posts, misinfo_posts, out_dir):
    """Bar chart: fraction of total interactions per layer for each group."""
    def fractions(posts):
        totals = aggregate_layer_totals(posts)
        grand  = sum(totals.values())
        layers = sorted(totals.keys())
        return layers, [totals[l] / grand for l in layers]

    r_layers, r_fracs = fractions(real_posts)
    m_layers, m_fracs = fractions(misinfo_posts)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Hypothesis 6 — Interaction Distribution by Layer",
                 fontsize=13, fontweight="bold")

    for ax, layers, fracs, label, color in zip(
        axes,
        [r_layers,    m_layers],
        [r_fracs,     m_fracs],
        ["Real News", "Misinformation"],
        ["#4C9BE8",   "#E07B39"]
    ):
        ax.bar(layers, fracs, color=color, alpha=0.85, edgecolor="none")
        ax.set_title(label)
        ax.set_xlabel("Layer (hop count from original poster)")
        ax.set_ylabel("Fraction of total interactions")
        ax.set_xticks(layers)
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1))

    plt.tight_layout()
    path = os.path.join(out_dir, "hyp6_layer_distribution.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[INFO] Layer distribution saved → {path}")


def plot_metric_boxplots(real_records, misinfo_records, out_dir):
    """Side-by-side boxplots for all per-post metrics."""
    metrics = [
        ("mean_layer",      "Mean Layer Depth"),
        ("max_layer",       "Max Layer Reached"),
        ("cascade_ratio",   "Cascade Ratio (≥L2 / L1)"),
        ("layer1_fraction", "Layer-1 Fraction"),
    ]

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))
    fig.suptitle("Hypothesis 6 — Per-Post Metric Comparison",
                 fontsize=13, fontweight="bold")

    for ax, (key, label) in zip(axes, metrics):
        def clean(records):
            return [x[key] for x in records
                    if not np.isnan(x.get(key, float("nan")))
                    and not np.isinf(x.get(key, float("nan")))]
        r_vals = clean(real_records)
        m_vals = clean(misinfo_records)

        bp = ax.boxplot([r_vals, m_vals],
                        labels=["Real", "Misinfo"],
                        patch_artist=True,
                        medianprops=dict(color="black", linewidth=2))
        bp["boxes"][0].set_facecolor("#4C9BE8")
        bp["boxes"][1].set_facecolor("#E07B39")
        ax.set_title(label)
        ax.set_ylabel(label)

    plt.tight_layout()
    path = os.path.join(out_dir, "hyp6_boxplots.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[INFO] Boxplots saved → {path}")


def plot_mean_depth_histogram(real_records, misinfo_records, out_dir):
    """Overlapping histogram of mean layer depth for both groups."""
    r_vals = [x["mean_layer"] for x in real_records]
    m_vals = [x["mean_layer"] for x in misinfo_records]

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, max(max(r_vals, default=0), max(m_vals, default=0)) + 1, 30)
    ax.hist(r_vals, bins=bins, color="#4C9BE8", alpha=0.65,
            label=f"Real News  (median={np.median(r_vals):.2f})")
    ax.hist(m_vals, bins=bins, color="#E07B39", alpha=0.65,
            label=f"Misinformation  (median={np.median(m_vals):.2f})")
    ax.axvline(np.median(r_vals), color="#2255AA", linestyle="--", linewidth=1.5)
    ax.axvline(np.median(m_vals), color="#AA4400", linestyle="--", linewidth=1.5)
    ax.set_title("Hypothesis 6 — Distribution of Mean Layer Depth per Post",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Mean layer depth")
    ax.set_ylabel("Post count")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(out_dir, "hyp6_mean_depth_hist.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[INFO] Mean depth histogram saved → {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(RESULTS_FOLDER, exist_ok=True)

    real_posts, misinfo_posts = load_posts(OUTPUT_FOLDER, HYP_PREFIX)

    real_records    = compute_metrics(real_posts)
    misinfo_records = compute_metrics(misinfo_posts)

    print_summary(real_records, misinfo_records)

    plot_layer_distribution(real_posts, misinfo_posts, RESULTS_FOLDER)
    plot_metric_boxplots(real_records, misinfo_records, RESULTS_FOLDER)
    plot_mean_depth_histogram(real_records, misinfo_records, RESULTS_FOLDER)


if __name__ == "__main__":
    main()