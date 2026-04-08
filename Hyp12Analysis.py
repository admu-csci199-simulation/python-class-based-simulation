"""
hyp12Analysis.py  (labelled Hyp12 per project convention)
---------------------------------------------------------
Statistical analysis for Hypothesis 2:
  "Two posts of different beliefs compete for interactions."

Competition is real if:
  (a) the two posts share a meaningful audience (Jaccard overlap > 0), AND
  (b) one post gaining interactions correlates with the other losing them
      (negative cross-correlation = temporal suppression).

If both posts only reach agents from their own ideological camp, they are
spreading through separate echo chambers — not competing.

Expected output JSON format (produced by the modified Main.py):
  {
    "<postID>": {
      "isMisinformation" : bool,
      "beliefValue"      : int,
      "interactions"     : [count per minute delta, ...],   // length = MINUTES
      "interactionsByBeliefCamp": {
        "red":      [...],
        "centrist": [...],
        "blue":     [...]
      },
      "agentsReached": [agent_id, ...]
    },
    ...
  }

Output filenames follow Tester.py's pattern:
  hyp12-config-{GROUP}-{index}s-{seed}.json

Analyses:
  1. Audience overlap    — Jaccard similarity of agentsReached sets
  2. Winner analysis     — which post got more total interactions, and why
  3. Camp segregation    — are the two posts drawing from the same agent pool?
  4. Centrist contest    — centrist agents are the swing vote; who wins them?
  5. Temporal suppression— cross-correlation of the two interaction curves
  6. Per-group summaries — all metrics broken down by config group (A–E)
  7. Statistical tests   — Wilcoxon signed-rank on Jaccard; binomial on win rate
"""

import os, json, sys, re
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import stats
from scipy.signal import correlate

# ── Config ────────────────────────────────────────────────────────────────────
OUTPUT_FOLDER  = "output"
RESULTS_FOLDER = os.path.join(OUTPUT_FOLDER, "hyp12_analysis")
HYP_PREFIX     = "hyp12"
MINUTES        = 24 * 60
MIN_TOTAL      = 3          # skip configs where both posts have negligible reach
GROUPS         = list("ABCDE")
# ─────────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def parse_filename(fname: str):
    """
    Extract (group, config_index, seed) from
    'hyp12-config-{GROUP}-{index}s-{seed}.json'
    Returns None if the pattern doesn't match.
    """
    m = re.match(r"hyp12-config-([A-Z])-(\d+)s-(\d+)\.json", fname)
    if not m:
        return None
    return m.group(1), int(m.group(2)), int(m.group(3))


def load_all(output_folder: str, prefix: str) -> dict:
    """
    Load and average all output files.

    Returns a nested dict:
      { group: { config_index: averaged_entry } }

    averaged_entry = {
        post0: { "isMisinformation", "beliefValue",
                 "interactions"      : np.ndarray,
                 "interactionsByBeliefCamp": { camp: np.ndarray },
                 "agentsReached"     : set  },
        post1: { ... }
    }
    """
    # Accumulator: group -> config_idx -> pid -> { arrays, sets, meta, runs }
    accum = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {
        "isMisinformation": None,
        "beliefValue"     : None,
        "interactions"    : np.zeros(MINUTES),
        "interactionsByBeliefCamp": {
            "red"     : np.zeros(MINUTES),
            "centrist": np.zeros(MINUTES),
            "blue"    : np.zeros(MINUTES),
        },
        "agentsReached": set(),
        "runs"          : 0,
    })))

    files_loaded = 0
    for fname in sorted(os.listdir(output_folder)):
        if not fname.startswith(prefix):
            continue
        parsed = parse_filename(fname)
        if parsed is None:
            continue
        group, cfg_idx, seed = parsed

        with open(os.path.join(output_folder, fname)) as f:
            data = json.load(f)

        for pid_str, entry in data.items():
            pid = int(pid_str)
            rec = accum[group][cfg_idx][pid]

            rec["isMisinformation"] = entry["isMisinformation"]
            rec["beliefValue"]      = entry["beliefValue"]

            arr = np.array(entry["interactions"][:MINUTES], dtype=np.float64)
            if len(arr) < MINUTES:
                arr = np.pad(arr, (0, MINUTES - len(arr)))
            rec["interactions"] += arr

            for camp in ("red", "centrist", "blue"):
                camp_arr = np.array(
                    entry["interactionsByBeliefCamp"][camp][:MINUTES], dtype=np.float64)
                if len(camp_arr) < MINUTES:
                    camp_arr = np.pad(camp_arr, (0, MINUTES - len(camp_arr)))
                rec["interactionsByBeliefCamp"][camp] += camp_arr

            rec["agentsReached"].update(entry["agentsReached"])
            rec["runs"] += 1

        files_loaded += 1

    if not files_loaded:
        sys.exit(f"[ERROR] No files found in '{output_folder}' with prefix '{prefix}'.")
    print(f"[INFO] Loaded {files_loaded} output file(s).")

    # Average interaction arrays across runs; keep agentsReached as union set
    result = {}
    for group, cfgs in accum.items():
        result[group] = {}
        for cfg_idx, posts in cfgs.items():
            result[group][cfg_idx] = {}
            for pid, rec in posts.items():
                n = max(rec["runs"], 1)
                result[group][cfg_idx][pid] = {
                    "isMisinformation": rec["isMisinformation"],
                    "beliefValue"     : rec["beliefValue"],
                    "interactions"    : rec["interactions"] / n,
                    "interactionsByBeliefCamp": {
                        c: rec["interactionsByBeliefCamp"][c] / n
                        for c in ("red", "centrist", "blue")
                    },
                    "agentsReached"   : rec["agentsReached"],  # union across seeds
                }

    total_configs = sum(len(v) for v in result.values())
    print(f"[INFO] Groups loaded: { {g: len(v) for g, v in result.items()} }")
    print(f"[INFO] Total configs : {total_configs}")
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  PER-CONFIG METRICS
# ══════════════════════════════════════════════════════════════════════════════

def jaccard(set_a: set, set_b: set) -> float:
    """Jaccard similarity: |A∩B| / |A∪B|. Returns 0 if both empty."""
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def camp_fractions(post_rec: dict) -> dict[str, float]:
    """Fraction of total interactions that came from each belief camp."""
    total = sum(
        post_rec["interactionsByBeliefCamp"][c].sum()
        for c in ("red", "centrist", "blue")
    )
    if total == 0:
        return {"red": 0.0, "centrist": 0.0, "blue": 0.0}
    return {
        c: post_rec["interactionsByBeliefCamp"][c].sum() / total
        for c in ("red", "centrist", "blue")
    }


def centrist_share(post_rec: dict, total_centrist: float) -> float:
    """Fraction of ALL centrist interactions in this config that went to this post."""
    if total_centrist == 0:
        return 0.0
    return post_rec["interactionsByBeliefCamp"]["centrist"].sum() / total_centrist


def peak_cross_correlation(curve0: np.ndarray, curve1: np.ndarray):
    """
    Normalised cross-correlation of the two interaction time-series.
    Returns (peak_correlation, lag_at_peak).
    A negative peak_correlation means the two curves are anti-correlated
    (when one rises, the other tends to fall) → evidence of suppression.
    A near-zero value means the two curves are independent.
    """
    # Use only the portion where at least one curve is nonzero
    active = (curve0 + curve1) > 0
    if active.sum() < 4:
        return float("nan"), 0

    c0 = curve0[active] - curve0[active].mean()
    c1 = curve1[active] - curve1[active].mean()

    norm = np.sqrt((c0 ** 2).sum() * (c1 ** 2).sum())
    if norm == 0:
        return 0.0, 0

    xcorr  = correlate(c0, c1, mode="full") / norm
    lags   = np.arange(-(len(c0) - 1), len(c0))
    peak_idx = int(np.argmax(np.abs(xcorr)))
    return float(xcorr[peak_idx]), int(lags[peak_idx])


def compute_pair_metrics(cfg_posts: dict) -> dict | None:
    """
    Compute all competition metrics for one config (one pair of posts).
    Returns None if the config doesn't have exactly 2 posts or both are too small.
    """
    pids = sorted(cfg_posts.keys())
    if len(pids) != 2:
        return None

    p0, p1 = cfg_posts[pids[0]], cfg_posts[pids[1]]

    total0 = p0["interactions"].sum()
    total1 = p1["interactions"].sum()

    if total0 + total1 < MIN_TOTAL:
        return None

    # Jaccard
    jac = jaccard(p0["agentsReached"], p1["agentsReached"])

    # Winner
    winner = 0 if total0 >= total1 else 1

    # Camp fractions per post
    cf0 = camp_fractions(p0)
    cf1 = camp_fractions(p1)

    # Centrist contest
    total_centrist = (
        p0["interactionsByBeliefCamp"]["centrist"].sum() +
        p1["interactionsByBeliefCamp"]["centrist"].sum()
    )
    centrist0 = centrist_share(p0, total_centrist)
    centrist1 = centrist_share(p1, total_centrist)

    # Cross-correlation
    xcorr_peak, xcorr_lag = peak_cross_correlation(
        p0["interactions"], p1["interactions"])

    # Belief distance between the two posts
    belief_distance = abs(p0["beliefValue"] - p1["beliefValue"])

    return {
        "postID0"         : pids[0],
        "postID1"         : pids[1],
        "beliefValue0"    : p0["beliefValue"],
        "beliefValue1"    : p1["beliefValue"],
        "beliefDistance"  : belief_distance,
        "isMisinfo0"      : p0["isMisinformation"],
        "isMisinfo1"      : p1["isMisinformation"],
        "total0"          : total0,
        "total1"          : total1,
        "winner"          : winner,                 # 0 or 1
        "jaccard"         : jac,
        "overlap_count"   : len(p0["agentsReached"] & p1["agentsReached"]),
        "camp_fractions0" : cf0,
        "camp_fractions1" : cf1,
        "centrist_share0" : centrist0,
        "centrist_share1" : centrist1,
        "xcorr_peak"      : xcorr_peak,
        "xcorr_lag"       : xcorr_lag,
        "curves"          : (p0["interactions"], p1["interactions"]),  # for plotting
    }


# ══════════════════════════════════════════════════════════════════════════════
#  STATISTICAL TESTS
# ══════════════════════════════════════════════════════════════════════════════

def wilcoxon_greater_than_zero(values: list[float], label: str):
    """
    Wilcoxon signed-rank test: are values significantly > 0?
    (one-sample, testing against median=0)
    """
    vals = [v for v in values if not np.isnan(v)]
    if len(vals) < 4:
        return
    stat, p = stats.wilcoxon(vals, alternative="greater")
    sig = "✓ significant" if p < 0.05 else "✗ not significant"
    print(f"    Wilcoxon (>{0}) — stat={stat:.1f}  p={p:.3e}  {sig}  "
          f"(median={np.median(vals):.4f}, n={len(vals)})")


def binomial_win_rate(wins_post0: int, total: int, label: str):
    """
    Binomial test: is post 0's win rate significantly different from 50%?
    (tests whether competition has a systematic winner)
    """
    if total == 0:
        return
    p = stats.binomtest(wins_post0, total, p=0.5, alternative="two-sided").pvalue
    rate = wins_post0 / total
    sig = "✓ significant" if p < 0.05 else "✗ not significant"
    print(f"    Binomial win-rate test — post0 wins {wins_post0}/{total} "
          f"({100*rate:.1f}%)  p={p:.3e}  {sig}")


# ══════════════════════════════════════════════════════════════════════════════
#  REPORTING
# ══════════════════════════════════════════════════════════════════════════════

GROUP_DESCRIPTIONS = {
    "A": "Max opposition, balanced pop, simultaneous  (baseline)",
    "B": "Moderate opposition, balanced pop, simultaneous",
    "C": "Max opposition, red-heavy pop, simultaneous",
    "D": "Max opposition, balanced pop, staggered posting",
    "E": "Max opposition, balanced pop, one misinfo post",
}


def print_group_summary(group: str, metrics: list[dict]) -> None:
    if not metrics:
        print(f"\n  [Group {group}] No valid configs.")
        return

    jaccards   = [m["jaccard"]    for m in metrics]
    xcorrs     = [m["xcorr_peak"] for m in metrics if not np.isnan(m["xcorr_peak"])]
    wins0      = sum(1 for m in metrics if m["winner"] == 0)
    centrist0  = [m["centrist_share0"] for m in metrics]

    print(f"\n  {'─'*60}")
    print(f"  Group {group} — {GROUP_DESCRIPTIONS.get(group, '')}")
    print(f"  {'─'*60}")
    print(f"  Configs analysed     : {len(metrics)}")
    print()
    print(f"  [AUDIENCE OVERLAP — Jaccard similarity]")
    print(f"    Mean   : {np.mean(jaccards):.4f}")
    print(f"    Median : {np.median(jaccards):.4f}")
    print(f"    Range  : [{min(jaccards):.4f}, {max(jaccards):.4f}]")
    wilcoxon_greater_than_zero(jaccards, group)
    print()
    print(f"  [WINNER ANALYSIS]")
    binomial_win_rate(wins0, len(metrics), group)
    print()
    print(f"  [CENTRIST CONTEST — post 0's share of centrist interactions]")
    print(f"    Mean   : {np.mean(centrist0):.3f}  "
          f"(0.5 = split evenly, >0.5 = post 0 dominates centrists)")
    print()
    print(f"  [TEMPORAL SUPPRESSION — peak cross-correlation]")
    if xcorrs:
        neg = sum(1 for x in xcorrs if x < 0)
        print(f"    Mean   : {np.mean(xcorrs):+.4f}")
        print(f"    Negative (suppression): {neg}/{len(xcorrs)}  "
              f"({100*neg/len(xcorrs):.1f}%)")
    else:
        print(f"    Insufficient data.")


def print_full_summary(all_metrics: dict[str, list[dict]]) -> None:
    all_jaccards = [m["jaccard"] for g in all_metrics.values() for m in g]
    all_xcorrs   = [m["xcorr_peak"] for g in all_metrics.values()
                    for m in g if not np.isnan(m["xcorr_peak"])]

    print("\n" + "="*62)
    print("  HYPOTHESIS 2 — COMPETITION BETWEEN OPPOSING-BELIEF POSTS")
    print("="*62)

    for group in GROUPS:
        if group in all_metrics:
            print_group_summary(group, all_metrics[group])

    print(f"\n  {'═'*60}")
    print(f"  OVERALL  ({sum(len(v) for v in all_metrics.values())} configs)")
    print(f"  {'═'*60}")
    print(f"  Mean Jaccard        : {np.mean(all_jaccards):.4f}")
    neg_xcorr = sum(1 for x in all_xcorrs if x < 0)
    print(f"  Negative xcorr rate : {neg_xcorr}/{len(all_xcorrs)}  "
          f"({100*neg_xcorr/max(len(all_xcorrs),1):.1f}%)")
    print()

    # Overall verdict
    mean_jac   = np.mean(all_jaccards)
    neg_rate   = neg_xcorr / max(len(all_xcorrs), 1)
    _, jac_p   = stats.wilcoxon(all_jaccards, alternative="greater") \
                 if len(all_jaccards) >= 4 else (None, 1.0)

    if mean_jac > 0.05 and jac_p < 0.05 and neg_rate > 0.5:
        verdict = ("SUPPORTED — posts share a significant audience and show "
                   "temporal suppression patterns.")
    elif mean_jac > 0.05 and jac_p < 0.05:
        verdict = ("PARTIALLY SUPPORTED — posts compete for the same audience "
                   "but suppression is not consistent.")
    elif neg_rate > 0.5:
        verdict = ("PARTIALLY SUPPORTED — temporal suppression observed but "
                   "audience overlap is low (echo chamber spreading).")
    else:
        verdict = ("NOT SUPPORTED — posts spread through separate communities "
                   "with no evidence of suppression.")
    print(f"  Verdict: {verdict}")
    print("="*62 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
#  PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

COLORS = {
    "post0": "#4C9BE8",   # blue
    "post1": "#E07B39",   # orange
    "centrist": "#7CB97C",
}
GROUP_COLORS = {"A": "#4C9BE8", "B": "#7CB97C", "C": "#E07B39",
                "D": "#9B59B6", "E": "#E74C3C"}


def plot_jaccard_by_group(all_metrics: dict, out_dir: str) -> None:
    groups   = [g for g in GROUPS if g in all_metrics and all_metrics[g]]
    data     = [[m["jaccard"] for m in all_metrics[g]] for g in groups]
    colors   = [GROUP_COLORS[g] for g in groups]

    fig, ax = plt.subplots(figsize=(9, 5))
    bp = ax.boxplot(data, labels=groups, patch_artist=True,
                    medianprops=dict(color="black", linewidth=2))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_title("Hypothesis 2 — Audience Overlap (Jaccard) by Config Group",
                 fontweight="bold")
    ax.set_xlabel("Config Group")
    ax.set_ylabel("Jaccard Similarity")

    # Annotate group descriptions
    for i, g in enumerate(groups):
        ax.text(i + 1, ax.get_ylim()[0] - 0.02,
                GROUP_DESCRIPTIONS.get(g, ""), ha="center", fontsize=6,
                color="grey", rotation=8)

    plt.tight_layout()
    path = os.path.join(out_dir, "hyp12_jaccard_by_group.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[INFO] Jaccard plot saved → {path}")


def plot_camp_segregation(all_metrics: dict, out_dir: str) -> None:
    """
    Stacked bar chart: for each group, average fraction of interactions
    that each post drew from each belief camp.
    """
    groups = [g for g in GROUPS if g in all_metrics and all_metrics[g]]
    camps  = ["red", "centrist", "blue"]
    camp_colors = {"red": "#E07B39", "centrist": "#7CB97C", "blue": "#4C9BE8"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Hypothesis 2 — Camp Segregation by Group",
                 fontsize=13, fontweight="bold")

    for ax, post_key, post_label in zip(
        axes,
        ["camp_fractions0", "camp_fractions1"],
        ["Post 0", "Post 1"]
    ):
        bottoms = np.zeros(len(groups))
        for camp in camps:
            vals = [
                np.mean([m[post_key][camp] for m in all_metrics[g]])
                for g in groups
            ]
            ax.bar(groups, vals, bottom=bottoms,
                   color=camp_colors[camp], alpha=0.85,
                   label=camp.capitalize())
            bottoms += np.array(vals)

        ax.set_title(post_label)
        ax.set_xlabel("Config Group")
        ax.set_ylabel("Avg fraction of interactions")
        ax.set_ylim(0, 1)
        ax.legend(loc="upper right", fontsize=8)
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))

    plt.tight_layout()
    path = os.path.join(out_dir, "hyp12_camp_segregation.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[INFO] Camp segregation plot saved → {path}")


def plot_centrist_contest(all_metrics: dict, out_dir: str) -> None:
    """
    For each group, scatter post0's centrist share vs post1's.
    Points above the diagonal = post0 won centrists.
    """
    fig, axes = plt.subplots(1, len(GROUPS), figsize=(4 * len(GROUPS), 4))
    fig.suptitle("Hypothesis 2 — Centrist Contest (share of centrist interactions)",
                 fontsize=12, fontweight="bold")

    for ax, group in zip(axes, GROUPS):
        if group not in all_metrics or not all_metrics[group]:
            ax.set_title(f"Group {group}\n(no data)")
            ax.axis("off")
            continue
        c0 = [m["centrist_share0"] for m in all_metrics[group]]
        c1 = [m["centrist_share1"] for m in all_metrics[group]]
        ax.scatter(c0, c1, color=GROUP_COLORS[group], s=20, alpha=0.7)
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)   # diagonal
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_title(f"Group {group}", fontsize=9)
        ax.set_xlabel("Post 0 centrist share", fontsize=8)
        ax.set_ylabel("Post 1 centrist share", fontsize=8)
        ax.set_aspect("equal")

    plt.tight_layout()
    path = os.path.join(out_dir, "hyp12_centrist_contest.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[INFO] Centrist contest plot saved → {path}")


def plot_sample_curves(all_metrics: dict, out_dir: str, n_samples: int = 3) -> None:
    """
    For each group, plot n_samples side-by-side interaction curves
    (post 0 vs post 1) to visually show whether they suppress each other.
    """
    hours = np.arange(MINUTES) / 60
    groups = [g for g in GROUPS if g in all_metrics and all_metrics[g]]

    fig, axes = plt.subplots(
        len(groups), n_samples,
        figsize=(5 * n_samples, 3.5 * len(groups)),
        squeeze=False
    )
    fig.suptitle("Hypothesis 2 — Sample Interaction Curves (Post 0 vs Post 1)",
                 fontsize=13, fontweight="bold")

    for row, group in enumerate(groups):
        samples = all_metrics[group][:n_samples]
        for col in range(n_samples):
            ax = axes[row][col]
            if col < len(samples):
                m = samples[col]
                c0, c1 = m["curves"]
                ax.plot(hours, c0, color=COLORS["post0"], linewidth=1.2,
                        label=f"Post 0 (b={m['beliefValue0']})")
                ax.plot(hours, c1, color=COLORS["post1"], linewidth=1.2,
                        label=f"Post 1 (b={m['beliefValue1']})")
                ax.set_title(
                    f"G{group} | J={m['jaccard']:.2f} | "
                    f"xcorr={m['xcorr_peak']:+.2f}", fontsize=7)
                ax.set_xlabel("Hours", fontsize=7)
                ax.set_ylabel("Interactions", fontsize=7)
                ax.tick_params(labelsize=6)
                ax.legend(fontsize=6)
                ax.xaxis.set_major_locator(ticker.MultipleLocator(6))
            else:
                ax.axis("off")

    plt.tight_layout()
    path = os.path.join(out_dir, "hyp12_sample_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[INFO] Sample curves saved → {path}")


def plot_xcorr_distribution(all_metrics: dict, out_dir: str) -> None:
    """
    Histogram of peak cross-correlation values per group.
    Negative values indicate temporal suppression.
    """
    groups = [g for g in GROUPS if g in all_metrics and all_metrics[g]]

    fig, ax = plt.subplots(figsize=(9, 5))
    for group in groups:
        vals = [m["xcorr_peak"] for m in all_metrics[group]
                if not np.isnan(m["xcorr_peak"])]
        if vals:
            ax.hist(vals, bins=20, alpha=0.55, color=GROUP_COLORS[group],
                    label=f"Group {group}", edgecolor="none")

    ax.axvline(0, color="black", linewidth=1.2, linestyle="--",
               label="Zero (no suppression)")
    ax.set_title("Hypothesis 2 — Cross-Correlation Distribution by Group",
                 fontweight="bold")
    ax.set_xlabel("Peak cross-correlation (negative = suppression)")
    ax.set_ylabel("Config count")
    ax.legend(fontsize=8)
    plt.tight_layout()
    path = os.path.join(out_dir, "hyp12_xcorr_distribution.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[INFO] Cross-correlation distribution saved → {path}")


def plot_jaccard_vs_belief_distance(all_metrics: dict, out_dir: str) -> None:
    """
    Scatter: belief distance between the two posts vs Jaccard overlap.
    Tests whether more ideologically distant posts share less audience.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    for group in GROUPS:
        if group not in all_metrics:
            continue
        x = [m["beliefDistance"] for m in all_metrics[group]]
        y = [m["jaccard"]        for m in all_metrics[group]]
        ax.scatter(x, y, color=GROUP_COLORS[group], s=18, alpha=0.6,
                   label=f"Group {group}")

    # Overall regression
    all_x = [m["beliefDistance"] for g in all_metrics.values() for m in g]
    all_y = [m["jaccard"]        for g in all_metrics.values() for m in g]
    if len(set(all_x)) > 1:
        slope, intercept, r, p, _ = stats.linregress(all_x, all_y)
        xs = np.linspace(min(all_x), max(all_x), 100)
        ax.plot(xs, intercept + slope * xs, "k--", linewidth=1.5,
                label=f"Overall fit  r={r:.2f}  p={p:.3e}")

    ax.set_title("Hypothesis 2 — Belief Distance vs Audience Overlap",
                 fontweight="bold")
    ax.set_xlabel("Belief distance |b0 − b1|")
    ax.set_ylabel("Jaccard similarity")
    ax.legend(fontsize=8)
    plt.tight_layout()
    path = os.path.join(out_dir, "hyp12_distance_vs_jaccard.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[INFO] Distance vs Jaccard plot saved → {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(RESULTS_FOLDER, exist_ok=True)

    raw = load_all(OUTPUT_FOLDER, HYP_PREFIX)

    # Compute per-config pair metrics, grouped
    all_metrics: dict[str, list[dict]] = defaultdict(list)
    for group, cfgs in raw.items():
        for cfg_idx, cfg_posts in cfgs.items():
            m = compute_pair_metrics(cfg_posts)
            if m is not None:
                all_metrics[group].append(m)

    print_full_summary(all_metrics)

    plot_jaccard_by_group(all_metrics, RESULTS_FOLDER)
    plot_camp_segregation(all_metrics, RESULTS_FOLDER)
    plot_centrist_contest(all_metrics, RESULTS_FOLDER)
    plot_xcorr_distribution(all_metrics, RESULTS_FOLDER)
    plot_jaccard_vs_belief_distance(all_metrics, RESULTS_FOLDER)
    plot_sample_curves(all_metrics, RESULTS_FOLDER)


if __name__ == "__main__":
    main()