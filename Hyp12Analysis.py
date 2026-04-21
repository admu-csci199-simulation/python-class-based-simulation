"""
Analysis: Competing Beliefs Hypothesis (Hypothesis 12)
=======================================================
Two posts of opposing beliefs compete for interactions in a balanced network.

Expected file layout (relative to this script):
  output/  hyp12-config-{config_num}s-{seed_num}.json  — simulationData for each run

Run:
  python hyp12_analysis.py
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from collections import defaultdict

# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR   = os.path.join(SCRIPT_DIR, "input")
OUTPUT_DIR  = os.path.join(SCRIPT_DIR, "output")
PLOTS_DIR   = os.path.join(SCRIPT_DIR, "plots")
NUM_CONFIGS = 100
NUM_SEEDS   = 100

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────────────────

SMOOTHING_WINDOW = 30   # minutes; rolling window for the interaction-rate curves

CAMP_COLOURS = {
    "post_0_red":  "#d62728",   # post 0 is the red-leaning post
    "post_1_blue": "#1f77b4",   # post 1 is the blue-leaning post
}

# ─────────────────────────────────────────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_runs(input_dir, output_dir, num_configs, num_seeds):
    """
    Joins Metadata from the input config file with simulationData from the
    output file on (config_num, seed_num).
    Returns a list of dicts, one per run that loaded cleanly.
    """
    runs = []
    skipped_missing = 0
    skipped_bad_posts = 0

    for config_num in range(num_configs):
        for seed_num in range(num_seeds):
            config_path = os.path.join(input_dir,  f"hyp12-config-{config_num}.json")
            output_path = os.path.join(output_dir, f"hyp12-config-{config_num}s-{seed_num}.json")

            if not os.path.exists(config_path) or not os.path.exists(output_path):
                skipped_missing += 1
                continue

            with open(config_path) as f:
                config = json.load(f)
            with open(output_path) as f:
                sim = json.load(f)

            meta = config.get("Metadata", {})

            # simulationData stores postID as string keys after json round-trip
            posts = sim.get("posts", {})
            if "0" not in posts or "1" not in posts:
                skipped_bad_posts += 1
                continue

            runs.append({
                "run_id": f"{config_num}s{seed_num}",
            # ── metadata knobs ──
            "red_belief":      meta.get("red_belief",      0),
            "blue_belief":     meta.get("blue_belief",     0),
            "polarity_gap":    meta.get("polarity_gap",    0),
            "belief_symmetry": meta.get("belief_symmetry", False),
            "red_interest":    meta.get("red_interest",    0),
            "blue_interest":   meta.get("blue_interest",   0),
            "interest_delta":  meta.get("interest_delta",  0),
            "time_offset":     meta.get("time_offset",     0),
            "is_misinfo":      meta.get("is_misinfo",      False),
            # ── simulation output ──
            "p0_interactions_over_time": posts["0"]["interactions_over_time"],
            "p1_interactions_over_time": posts["1"]["interactions_over_time"],
            "p0_total":                  posts["0"]["total_interactions"],
            "p1_total":                  posts["1"]["total_interactions"],
            "p0_by_camp":                posts["0"]["interactions_by_belief_camp"],
            "p1_by_camp":                posts["1"]["interactions_by_belief_camp"],
            "contested":                 sim.get("contested_agents", {}),
        })

    print(f"Loaded:          {len(runs)} runs")
    print(f"Skipped missing: {skipped_missing} (config or output file not found)")
    print(f"Skipped bad:     {skipped_bad_posts} (output missing post 0 or post 1)")
    return runs


# ─────────────────────────────────────────────────────────────────────────────
#  DERIVED METRICS
# ─────────────────────────────────────────────────────────────────────────────

def rolling_sum(arr, window):
    """Simple rolling window sum over a 1-D list."""
    arr = np.array(arr, dtype=float)
    result = np.convolve(arr, np.ones(window), mode="same")
    return result


def winner(run):
    """
    Returns 'post_0', 'post_1', or 'tie' based on total_interactions.
    """
    if run["p0_total"] > run["p1_total"]:
        return "post_0"
    elif run["p1_total"] > run["p0_total"]:
        return "post_1"
    return "tie"


def contested_winner(run):
    """
    Winner among contested agents only (saw both posts, had to choose).
    Returns 'post_0', 'post_1', 'tie', or 'no_contest' if no data.
    """
    c = run["contested"]
    c0 = c.get("chose_post_0", 0)
    c1 = c.get("chose_post_1", 0)
    if c0 == 0 and c1 == 0:
        return "no_contest"
    if c0 > c1:
        return "post_0"
    if c1 > c0:
        return "post_1"
    return "tie"


# ─────────────────────────────────────────────────────────────────────────────
#  FIGURE 1 — Interaction Race (cumulative curves, all runs)
# ─────────────────────────────────────────────────────────────────────────────

def plot_interaction_race(runs, plots_dir):
    """
    For each run, plot the CUMULATIVE interaction curve of post 0 (red) and
    post 1 (blue) on the shared absolute simulation timeline.

    All runs are overlaid with low alpha so the envelope of typical behaviour
    is visible. The mean across runs is drawn thick on top.
    """
    T = len(runs[0]["p0_interactions_over_time"])
    time_axis = np.arange(T) / 60  # convert minutes → hours

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    # ── left panel: every individual run (faint) + mean (bold) ──
    ax = axes[0]
    p0_curves, p1_curves = [], []

    for run in runs:
        p0_cum = np.cumsum(run["p0_interactions_over_time"])
        p1_cum = np.cumsum(run["p1_interactions_over_time"])
        p0_curves.append(p0_cum)
        p1_curves.append(p1_cum)
        ax.plot(time_axis, p0_cum, color=CAMP_COLOURS["post_0_red"],  alpha=0.08, linewidth=0.7)
        ax.plot(time_axis, p1_cum, color=CAMP_COLOURS["post_1_blue"], alpha=0.08, linewidth=0.7)

    mean_p0 = np.mean(p0_curves, axis=0)
    mean_p1 = np.mean(p1_curves, axis=0)
    ax.plot(time_axis, mean_p0, color=CAMP_COLOURS["post_0_red"],  linewidth=2.2, label="Post 0 (red-leaning) — mean")
    ax.plot(time_axis, mean_p1, color=CAMP_COLOURS["post_1_blue"], linewidth=2.2, label="Post 1 (blue-leaning) — mean")

    ax.set_title("Cumulative Interactions Over Time\n(all runs overlaid)")
    ax.set_xlabel("Simulation time (hours)")
    ax.set_ylabel("Cumulative interactions")
    ax.legend(fontsize=8)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(6))

    # ── right panel: smoothed interaction RATE, mean only ──
    ax2 = axes[1]
    all_p0_rate = np.mean(
        [rolling_sum(r["p0_interactions_over_time"], SMOOTHING_WINDOW) for r in runs], axis=0
    )
    all_p1_rate = np.mean(
        [rolling_sum(r["p1_interactions_over_time"], SMOOTHING_WINDOW) for r in runs], axis=0
    )
    ax2.plot(time_axis, all_p0_rate, color=CAMP_COLOURS["post_0_red"],  linewidth=2, label="Post 0 (red-leaning)")
    ax2.plot(time_axis, all_p1_rate, color=CAMP_COLOURS["post_1_blue"], linewidth=2, label="Post 1 (blue-leaning)")

    ax2.set_title(f"Interaction Rate Over Time\n({SMOOTHING_WINDOW}-min rolling window, mean across runs)")
    ax2.set_xlabel("Simulation time (hours)")
    ax2.set_ylabel(f"Interactions per {SMOOTHING_WINDOW}-min window")
    ax2.legend(fontsize=8)
    ax2.xaxis.set_major_locator(mticker.MultipleLocator(6))

    fig.suptitle("Fig 1 — The Interaction Race", fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(plots_dir, "fig1_interaction_race.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
#  FIGURE 2 — Win Rate by Belief Pair
# ─────────────────────────────────────────────────────────────────────────────

def plot_win_rate_by_belief_pair(runs, plots_dir):
    """
    For each unique (red_belief, blue_belief) pair, compute the fraction of
    runs where post 0 won, post 1 won, or they tied.

    Reveals whether asymmetric belief pairs produce a consistent winner and
    whether wider polarity gaps amplify the competition.
    """
    # group runs by their belief pair label
    groups = defaultdict(lambda: {"post_0": 0, "post_1": 0, "tie": 0, "n": 0})
    for run in runs:
        label = f"({run['red_belief']}, {run['blue_belief']})"
        w = winner(run)
        groups[label][w] += 1
        groups[label]["n"] += 1

    # sort by polarity gap then red belief for a sensible x-axis order
    unique_pairs = sorted(
        groups.keys(),
        key=lambda lbl: (
            # extract ints from "(a, b)" string
            abs(int(lbl.split(",")[1].strip(" )"))),     # gap magnitude
            int(lbl.split("(")[1].split(",")[0])          # red belief
        )
    )

    n_pairs = len(unique_pairs)
    x = np.arange(n_pairs)
    bar_w = 0.28

    fig, ax = plt.subplots(figsize=(max(8, n_pairs * 1.2), 5))

    p0_fracs, p1_fracs, tie_fracs, counts = [], [], [], []
    for lbl in unique_pairs:
        g = groups[lbl]
        n = g["n"]
        counts.append(n)
        p0_fracs.append(g["post_0"] / n if n else 0)
        p1_fracs.append(g["post_1"] / n if n else 0)
        tie_fracs.append(g["tie"]   / n if n else 0)

    bars0 = ax.bar(x - bar_w, p0_fracs, bar_w, label="Post 0 wins (red-leaning)", color=CAMP_COLOURS["post_0_red"])
    bars1 = ax.bar(x,          p1_fracs, bar_w, label="Post 1 wins (blue-leaning)", color=CAMP_COLOURS["post_1_blue"])
    bars2 = ax.bar(x + bar_w,  tie_fracs, bar_w, label="Tie", color="#7f7f7f")

    # annotate with run counts
    for xi, n in zip(x, counts):
        ax.text(xi, 1.02, f"n={n}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(unique_pairs, rotation=30, ha="right", fontsize=9)
    ax.set_xlabel("Belief pair (red post belief, blue post belief)")
    ax.set_ylabel("Fraction of runs")
    ax.set_ylim(0, 1.12)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.axhline(0.5, linestyle="--", color="black", linewidth=0.8, alpha=0.5)
    ax.legend(fontsize=8)
    ax.set_title("Fig 2 — Win Rate by Belief Pair\n(does belief proximity to centre give a structural advantage?)")

    fig.tight_layout()
    path = os.path.join(plots_dir, "fig2_win_rate_by_belief_pair.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
#  FIGURE 3 — Interest Delta Effect (symmetric pairs only)
# ─────────────────────────────────────────────────────────────────────────────

def plot_interest_delta_effect(runs, plots_dir):
    """
    Restrict to symmetric belief pairs (red_belief == -blue_belief) so belief
    is not a confound.  Within that subset, group runs by interest_delta and
    show win rates.

    If interest edge reliably decides outcomes, the bars should flip
    predictably across delta = -3 / 0 / +3.
    """
    sym_runs = [r for r in runs if r["belief_symmetry"]]

    if len(sym_runs) == 0:
        return

    delta_groups = defaultdict(lambda: {"post_0": 0, "post_1": 0, "tie": 0, "n": 0})
    for run in sym_runs:
        d = run["interest_delta"]
        w = winner(run)
        delta_groups[d][w] += 1
        delta_groups[d]["n"] += 1

    deltas = sorted(delta_groups.keys())
    x = np.arange(len(deltas))
    bar_w = 0.28

    fig, ax = plt.subplots(figsize=(7, 5))

    p0_fracs = [delta_groups[d]["post_0"] / delta_groups[d]["n"] for d in deltas]
    p1_fracs = [delta_groups[d]["post_1"] / delta_groups[d]["n"] for d in deltas]
    tie_fracs = [delta_groups[d]["tie"]   / delta_groups[d]["n"] for d in deltas]
    counts    = [delta_groups[d]["n"] for d in deltas]

    ax.bar(x - bar_w, p0_fracs, bar_w, label="Post 0 wins (red-leaning)", color=CAMP_COLOURS["post_0_red"])
    ax.bar(x,          p1_fracs, bar_w, label="Post 1 wins (blue-leaning)", color=CAMP_COLOURS["post_1_blue"])
    ax.bar(x + bar_w,  tie_fracs, bar_w, label="Tie", color="#7f7f7f")

    for xi, n in zip(x, counts):
        ax.text(xi, 1.02, f"n={n}", ha="center", va="bottom", fontsize=8)

    labels = [
        f"δ={d}\n({'red edge' if d < 0 else 'blue edge' if d > 0 else 'equal'})"
        for d in deltas
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_xlabel("Interest delta (blue interest − red interest)")
    ax.set_ylabel("Fraction of runs")
    ax.set_ylim(0, 1.12)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.axhline(0.5, linestyle="--", color="black", linewidth=0.8, alpha=0.5)
    ax.legend(fontsize=8)
    ax.set_title(
        "Fig 3 — Effect of Interest Edge on Win Rate\n"
        "(symmetric belief pairs only — belief is not a confound)"
    )

    fig.tight_layout()
    path = os.path.join(plots_dir, "fig3_interest_delta_effect.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
#  FIGURE 4 — Contested Agent Breakdown
# ─────────────────────────────────────────────────────────────────────────────

def plot_contested_agents(runs, plots_dir):
    """
    Two panels:

    Left  — stacked bar of the four contested-agent outcomes aggregated across
            all runs. This is the cleanest view of the "fight": only agents
            who received BOTH posts are counted.

    Right — scatter of (chose_post_0 / total_contested) vs
            (p0_total / (p0_total + p1_total)): does the contested-agent
            ratio track the overall interaction ratio?  High correlation
            would confirm that the fight among contested agents explains
            the overall outcome.
    """
    totals = {"chose_post_0": 0, "chose_post_1": 0, "chose_both": 0, "chose_neither": 0}
    scatter_x, scatter_y, scatter_col = [], [], []

    for run in runs:
        c = run["contested"]
        for key in totals:
            totals[key] += c.get(key, 0)

        contested_total = c.get("chose_post_0", 0) + c.get("chose_post_1", 0)
        if contested_total > 0:
            cx = c["chose_post_0"] / contested_total
        else:
            cx = None

        overall_total = run["p0_total"] + run["p1_total"]
        if overall_total > 0:
            cy = run["p0_total"] / overall_total
        else:
            cy = None

        if cx is not None and cy is not None:
            scatter_x.append(cx)
            scatter_y.append(cy)
            # colour dot by winner for extra information
            w = winner(run)
            scatter_col.append(
                CAMP_COLOURS["post_0_red"] if w == "post_0" else
                CAMP_COLOURS["post_1_blue"] if w == "post_1" else
                "#7f7f7f"
            )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── left: stacked bar ──
    ax = axes[0]
    labels = ["Chose post 0\n(red-leaning)", "Chose post 1\n(blue-leaning)", "Chose both", "Chose neither"]
    values = [
        totals["chose_post_0"],
        totals["chose_post_1"],
        totals["chose_both"],
        totals["chose_neither"],
    ]
    colours = [
        CAMP_COLOURS["post_0_red"],
        CAMP_COLOURS["post_1_blue"],
        "#9467bd",
        "#bcbd22",
    ]
    bars = ax.bar(labels, values, color=colours, edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.01,
                str(val), ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Agent count (summed across all runs)")
    ax.set_title("Contested Agent Outcomes\n(agents who received both posts)")

    # ── right: scatter ──
    ax2 = axes[1]
    if scatter_x:
        ax2.scatter(scatter_x, scatter_y, c=scatter_col, alpha=0.65, edgecolors="none", s=45)
        # diagonal reference line: perfect correlation
        ax2.plot([0, 1], [0, 1], "k--", linewidth=0.9, alpha=0.5, label="Perfect correlation")
        # correlation coefficient
        if len(scatter_x) > 1:
            corr = np.corrcoef(scatter_x, scatter_y)[0, 1]
            ax2.text(0.05, 0.93, f"r = {corr:.3f}", transform=ax2.transAxes,
                     fontsize=10, va="top")
        ax2.set_xlabel("Post 0 share among contested agents\n(chose_post_0 / (chose_post_0 + chose_post_1))")
        ax2.set_ylabel("Post 0 share of total interactions\n(p0_total / (p0_total + p1_total))")
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.legend(fontsize=8)
    else:
        ax2.text(0.5, 0.5, "No contested-agent data", ha="center", va="center",
                 transform=ax2.transAxes, fontsize=11, color="grey")

    ax2.set_title("Contested vs Overall Ratio\n(do contested agents drive the overall outcome?)")

    fig.suptitle("Fig 4 — Contested Agent Breakdown", fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(plots_dir, "fig4_contested_agents.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
#  SUMMARY REPORT
# ─────────────────────────────────────────────────────────────────────────────

def generate_summary(runs):
    """
    Computes all cross-cutting statistics from the loaded runs and prints
    them to the command line.

    Sections:
      1. Overall results       — win counts, mean interactions, margin
      2. Interaction dynamics  — time-to-peak, peak rate, when lead locks in
      3. By belief pair        — win rate per (red_belief, blue_belief) pair
      4. Interest delta effect — win rate by delta, symmetric pairs only
      5. First-mover effect    — win rate by time_offset
      6. Contested agents      — chose_post_0/1/both/neither totals + correlation
      7. Cross-camp reach      — how much each post pulled from opposing camp
    """
    n = len(runs)
    lines = []

    def section(title):
        lines.append("")
        lines.append("=" * 60)
        lines.append(f"  {title}")
        lines.append("=" * 60)

    def row(label, value):
        lines.append(f"  {label:<42} {value}")

    # ── header ────────────────────────────────────────────────────────────────
    lines.append("=" * 60)
    lines.append("  HYPOTHESIS 12 — COMPETING BELIEFS  |  SUMMARY REPORT")
    lines.append("=" * 60)
    row("Runs analysed", n)

    # ── 1. Overall results ────────────────────────────────────────────────────
    section("1. OVERALL RESULTS")

    wins_p0 = sum(1 for r in runs if winner(r) == "post_0")
    wins_p1 = sum(1 for r in runs if winner(r) == "post_1")
    ties    = sum(1 for r in runs if winner(r) == "tie")

    p0_totals = [r["p0_total"] for r in runs]
    p1_totals = [r["p1_total"] for r in runs]
    margins   = [r["p0_total"] - r["p1_total"] for r in runs]

    row("Post 0 (red-leaning) wins",  f"{wins_p0}  ({wins_p0/n:.1%})")
    row("Post 1 (blue-leaning) wins", f"{wins_p1}  ({wins_p1/n:.1%})")
    row("Ties",                        f"{ties}  ({ties/n:.1%})")
    row("Mean total interactions — post 0", f"{np.mean(p0_totals):.1f}  (SD {np.std(p0_totals):.1f})")
    row("Mean total interactions — post 1", f"{np.mean(p1_totals):.1f}  (SD {np.std(p1_totals):.1f})")
    row("Mean interaction margin (p0 − p1)", f"{np.mean(margins):.1f}  (SD {np.std(margins):.1f})")
    row("Median interaction margin",          f"{np.median(margins):.1f}")

    # ── 2. Interaction dynamics ───────────────────────────────────────────────
    section("2. INTERACTION DYNAMICS")

    p0_peaks, p1_peaks = [], []
    p0_peak_times, p1_peak_times = [], []
    crossover_times = []  # first minute where cumulative lead flips

    for r in runs:
        p0_arr = np.array(r["p0_interactions_over_time"])
        p1_arr = np.array(r["p1_interactions_over_time"])

        p0_peaks.append(float(np.max(p0_arr)))
        p1_peaks.append(float(np.max(p1_arr)))
        p0_peak_times.append(int(np.argmax(p0_arr)))
        p1_peak_times.append(int(np.argmax(p1_arr)))

        # crossover: first tick where the cumulative leader changes
        p0_cum = np.cumsum(p0_arr)
        p1_cum = np.cumsum(p1_arr)
        diff = p0_cum - p1_cum
        sign_changes = np.where(np.diff(np.sign(diff)))[0]
        if len(sign_changes) > 0:
            crossover_times.append(int(sign_changes[0]))

    row("Mean peak interaction rate — post 0",       f"{np.mean(p0_peaks):.2f} interactions/min")
    row("Mean peak interaction rate — post 1",       f"{np.mean(p1_peaks):.2f} interactions/min")
    row("Mean time-to-peak — post 0 (hours)",        f"{np.mean(p0_peak_times)/60:.2f}")
    row("Mean time-to-peak — post 1 (hours)",        f"{np.mean(p1_peak_times)/60:.2f}")
    if crossover_times:
        row("Runs with a lead crossover",            f"{len(crossover_times)}  ({len(crossover_times)/n:.1%})")
        row("Mean crossover time (hours)",           f"{np.mean(crossover_times)/60:.2f}")
    else:
        row("Runs with a lead crossover",            "0  (0.0%)")

    # ── 3. Win rate by belief pair ────────────────────────────────────────────
    section("3. WIN RATE BY BELIEF PAIR")

    pair_groups = defaultdict(lambda: {"post_0": 0, "post_1": 0, "tie": 0, "n": 0})
    for r in runs:
        label = f"({r['red_belief']:+d}, {r['blue_belief']:+d})"
        pair_groups[label][winner(r)] += 1
        pair_groups[label]["n"] += 1

    lines.append(f"  {'Belief pair':<18} {'n':>4}  {'P0 wins':>8}  {'P1 wins':>8}  {'Ties':>6}")
    lines.append("  " + "-" * 50)
    for label in sorted(pair_groups):
        g = pair_groups[label]
        nn = g["n"]
        lines.append(
            f"  {label:<18} {nn:>4}  "
            f"{g['post_0']:>4} ({g['post_0']/nn:.0%})  "
            f"{g['post_1']:>4} ({g['post_1']/nn:.0%})  "
            f"{g['tie']:>3} ({g['tie']/nn:.0%})"
        )

    # ── 4. Interest delta effect (symmetric pairs only) ───────────────────────
    section("4. INTEREST DELTA EFFECT  (symmetric belief pairs only)")

    sym_runs = [r for r in runs if r["belief_symmetry"]]
    if not sym_runs:
        lines.append("  No symmetric-belief runs found.")
    else:
        delta_groups = defaultdict(lambda: {"post_0": 0, "post_1": 0, "tie": 0, "n": 0})
        for r in sym_runs:
            delta_groups[r["interest_delta"]][winner(r)] += 1
            delta_groups[r["interest_delta"]]["n"] += 1

        lines.append(f"  {'Delta':>6}  {'Edge':>12}  {'n':>4}  {'P0 wins':>8}  {'P1 wins':>8}  {'Ties':>6}")
        lines.append("  " + "-" * 56)
        for d in sorted(delta_groups):
            g = delta_groups[d]
            nn = g["n"]
            edge = "red edge" if d < 0 else "blue edge" if d > 0 else "equal"
            lines.append(
                f"  {d:>+6}  {edge:>12}  {nn:>4}  "
                f"{g['post_0']:>4} ({g['post_0']/nn:.0%})  "
                f"{g['post_1']:>4} ({g['post_1']/nn:.0%})  "
                f"{g['tie']:>3} ({g['tie']/nn:.0%})"
            )

    # ── 5. First-mover effect (by time_offset) ────────────────────────────────
    section("5. FIRST-MOVER EFFECT  (red posts first by time_offset minutes)")

    offset_groups = defaultdict(lambda: {"post_0": 0, "post_1": 0, "tie": 0, "n": 0})
    for r in runs:
        offset_groups[r["time_offset"]][winner(r)] += 1
        offset_groups[r["time_offset"]]["n"] += 1

    lines.append(f"  {'Offset (min)':>12}  {'n':>4}  {'P0 wins':>8}  {'P1 wins':>8}  {'Ties':>6}")
    lines.append("  " + "-" * 50)
    for offset in sorted(offset_groups):
        g = offset_groups[offset]
        nn = g["n"]
        lines.append(
            f"  {offset:>12}  {nn:>4}  "
            f"{g['post_0']:>4} ({g['post_0']/nn:.0%})  "
            f"{g['post_1']:>4} ({g['post_1']/nn:.0%})  "
            f"{g['tie']:>3} ({g['tie']/nn:.0%})"
        )

    # ── 6. Contested agents ───────────────────────────────────────────────────
    section("6. CONTESTED AGENTS  (saw both posts, had to choose)")

    total_c0      = sum(r["contested"].get("chose_post_0",  0) for r in runs)
    total_c1      = sum(r["contested"].get("chose_post_1",  0) for r in runs)
    total_both    = sum(r["contested"].get("chose_both",    0) for r in runs)
    total_neither = sum(r["contested"].get("chose_neither", 0) for r in runs)
    total_contested = total_c0 + total_c1 + total_both + total_neither

    row("Total contested agents (across all runs)", total_contested)
    if total_contested:
        row("  Chose post 0 only (red-leaning)",  f"{total_c0}  ({total_c0/total_contested:.1%})")
        row("  Chose post 1 only (blue-leaning)", f"{total_c1}  ({total_c1/total_contested:.1%})")
        row("  Chose both",                        f"{total_both}  ({total_both/total_contested:.1%})")
        row("  Chose neither",                     f"{total_neither}  ({total_neither/total_contested:.1%})")

    # correlation: contested ratio vs overall ratio
    cx_vals, cy_vals = [], []
    for r in runs:
        c = r["contested"]
        denom_c = c.get("chose_post_0", 0) + c.get("chose_post_1", 0)
        denom_o = r["p0_total"] + r["p1_total"]
        if denom_c > 0 and denom_o > 0:
            cx_vals.append(c["chose_post_0"] / denom_c)
            cy_vals.append(r["p0_total"] / denom_o)

    if len(cx_vals) > 1:
        corr = np.corrcoef(cx_vals, cy_vals)[0, 1]
        row("Correlation: contested ratio vs overall ratio", f"r = {corr:.3f}")

    # agreement: does contested winner match overall winner?
    agree = sum(1 for r in runs if contested_winner(r) == winner(r) and winner(r) != "tie")
    eligible = sum(1 for r in runs if winner(r) != "tie" and contested_winner(r) != "no_contest")
    if eligible:
        row("Contested winner matches overall winner", f"{agree}/{eligible}  ({agree/eligible:.1%})")

    # ── 7. Cross-camp reach ───────────────────────────────────────────────────
    section("7. CROSS-CAMP REACH  (mean interactions per camp, across all runs)")

    for post_key, label in [("p0_by_camp", "Post 0 (red-leaning)"), ("p1_by_camp", "Post 1 (blue-leaning)")]:
        red_vals      = [r[post_key].get("red",      0) for r in runs]
        centrist_vals = [r[post_key].get("centrist", 0) for r in runs]
        blue_vals     = [r[post_key].get("blue",     0) for r in runs]
        lines.append(f"  {label}:")
        lines.append(f"    Red camp      : {np.mean(red_vals):.1f}  (SD {np.std(red_vals):.1f})")
        lines.append(f"    Centrist camp : {np.mean(centrist_vals):.1f}  (SD {np.std(centrist_vals):.1f})")
        lines.append(f"    Blue camp     : {np.mean(blue_vals):.1f}  (SD {np.std(blue_vals):.1f})")

    lines.append("")
    lines.append("=" * 60)
    lines.append("  END OF REPORT")
    lines.append("=" * 60)
    lines.append("")

    print("\n".join(lines))


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    print(f"Script dir:  {SCRIPT_DIR}")
    print(f"Input dir:   {INPUT_DIR}  (exists: {os.path.exists(INPUT_DIR)})")
    print(f"Output dir:  {OUTPUT_DIR}  (exists: {os.path.exists(OUTPUT_DIR)})")

    # Print the first few files found in each folder so we can verify the naming pattern
    for label, folder in [("input", INPUT_DIR), ("output", OUTPUT_DIR)]:
        if os.path.exists(folder):
            files = os.listdir(folder)[:5]
            print(f"  First files in {label}/: {files}")
        else:
            print(f"  {label}/ folder not found")

    runs = load_runs(INPUT_DIR, OUTPUT_DIR, NUM_CONFIGS, NUM_SEEDS)

    if not runs:
        print(f"No runs loaded. Check that '{INPUT_DIR}' and '{OUTPUT_DIR}' exist and contain the expected files.")
        print(f"Expected pattern: {INPUT_DIR}/hyp1-config-{{config_num}}s-{{seed_num}}.json")
        return

    plot_interaction_race(runs, PLOTS_DIR)
    plot_win_rate_by_belief_pair(runs, PLOTS_DIR)
    plot_interest_delta_effect(runs, PLOTS_DIR)
    plot_contested_agents(runs, PLOTS_DIR)
    generate_summary(runs)


if __name__ == "__main__":
    main()