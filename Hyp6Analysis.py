"""
Hyp6Analysis.py
---------------
Statistical analysis for Hypothesis 6:
  "Real information spreads because the original poster tends to have a massive
   following (broadcast), whereas misinformation thrives on agent-to-agent
   sharing resembling a virus-like infection (viral cascade)."

Broadcast (real news) signature:
  - Interactions concentrated at layer 1 (OP → direct followers)
  - Low cascade ratio, high layer-1 fraction, low mean depth

Viral (misinformation) signature:
  - Interactions spread across many layers
  - High cascade ratio, low layer-1 fraction, high mean depth

Expected output JSON format (one file per simulation run):
  {
    "<postID>": {
      "isMisinformation": true | false,
      "layers": {
        "0": count,
        "1": count,
        "2": count,
        ...
      }
    },
    ...
  }

Analyses:
  1. Layer distribution   — fraction of interactions per layer (real vs misinfo)
  2. Mean layer depth     — probability-weighted mean layer; Mann-Whitney U test
  3. Max layer reached    — deepest layer with ≥1 interaction
  4. Cascade ratio        — interactions(layer≥2) / interactions(layer 1)
  5. Layer-1 fraction     — share of all interactions occurring at layer 1
  6. Statistical verdict
"""

import os, json, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import stats

# ── Config ────────────────────────────────────────────────────────────────────
OUTPUT_FOLDER  = "output"
RESULTS_FOLDER = os.path.join(OUTPUT_FOLDER, "hyp6_analysis")
HYP_PREFIX     = "hyp6"
MIN_TOTAL      = 5      # skip posts with fewer total interactions
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
    # { pid: { "is_misinfo": bool, "layers": { int: float }, "runs": int } }
    accum: dict[str, dict] = {}
    files_loaded = 0

    for fname in sorted(os.listdir(output_folder)):
        if not (fname.startswith(prefix) and fname.endswith(".json")):
            continue
        with open(os.path.join(output_folder, fname)) as f:
            data = json.load(f)

        for pid, entry in data.items():
            is_m   = entry["isMisinformation"]
            layers = {int(k): float(v) for k, v in entry["interactions"].items()}

            if pid not in accum:
                accum[pid] = {"is_misinfo": is_m, "layers": {}, "runs": 0}

            accum[pid]["runs"] += 1
            for layer, count in layers.items():
                accum[pid]["layers"][layer] = accum[pid]["layers"].get(layer, 0.0) + count

        files_loaded += 1

    if not files_loaded:
        sys.exit(f"[ERROR] No files found in '{output_folder}' with prefix '{prefix}'.")
    print(f"[INFO] Loaded {files_loaded} output file(s).")

    # Average counts across runs
    for rec in accum.values():
        for layer in rec["layers"]:
            rec["layers"][layer] /= rec["runs"]

    real_posts    = {pid: rec["layers"] for pid, rec in accum.items() if not rec["is_misinfo"]}
    misinfo_posts = {pid: rec["layers"] for pid, rec in accum.items() if rec["is_misinfo"]}
    print(f"[INFO] Posts — real: {len(real_posts)},  misinfo: {len(misinfo_posts)}")
    return real_posts, misinfo_posts


# ══════════════════════════════════════════════════════════════════════════════
#  PER-POST METRICS
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(posts: dict[str, dict]) -> list[dict]:
    """
    For each post compute:
      mean_layer      — probability-weighted mean layer depth
      max_layer       — deepest layer that received ≥1 interaction
      cascade_ratio   — interactions(layer≥2) / interactions(layer 1)
                        (0 = pure broadcast, >> 1 = deep viral cascade)
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
        cascade_ratio   = deep_count / l1_count if l1_count > 0 else float("inf")

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

def aggregate_layer_totals(posts: dict[str, dict]) -> dict[int, float]:
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

    sig_flags = []
    for key, label, expectation in metrics:
        med_r, med_m, u, p = mw_test(real_records, misinfo_records, key)
        sig = (not np.isnan(p)) and p < 0.05
        sig_flags.append(sig)
        print(f"  [{label}]  ({expectation})")
        print(f"    Median — real: {med_r:.3f}   misinfo: {med_m:.3f}")
        print(f"    Mann-Whitney U={u:.1f}  p={p:.2e}  "
              f"{'✓ significant' if sig else '✗ not significant'}")
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
        [r_layers,            m_layers],
        [r_fracs,             m_fracs],
        ["Real News",         "Misinformation"],
        ["#4C9BE8",           "#E07B39"]
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