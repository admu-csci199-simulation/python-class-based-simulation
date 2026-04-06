import os
import json
import re
import numpy as np
import pandas as pd
from scipy import stats

# -----------------------------
# CONFIG
# -----------------------------

OUTPUT_DIR = "output"

CONFIG_MAP = {
    "equal": {"red": 1000, "centrist": 1000, "blue": 1000},
    "red": {"red": 2400, "centrist": 300, "blue": 300},
    "centrist": {"red": 300, "centrist": 2400, "blue": 300},
    "blue": {"red": 300, "centrist": 300, "blue": 2400},
}

GROUPS = ["red", "centrist", "blue"]

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------

def get_expected_ratios(config):
    total = sum(config.values())
    return {k: v / total for k, v in config.items()}

def compute_alignment(group, counts):
    total_seen = sum(counts)
    if total_seen == 0:
        return None
    idx = GROUPS.index(group)
    aligned = counts[idx]
    return aligned / total_seen

# -----------------------------
# PARSE FILES
# -----------------------------

rows = []

for filename in os.listdir(OUTPUT_DIR):
    if not filename.endswith(".json"):
        continue

    # Example: hyp3-red12s-45.json
    parts = filename.replace(".json", "").split("-")
    config_part = parts[1]

    # Extract config name (remove trailing number)
    config_name = re.match(r"(equal|red|centrist|blue)", config_part).group(1)

    config = CONFIG_MAP[config_name]
    expected_ratios = get_expected_ratios(config)

    filepath = os.path.join(OUTPUT_DIR, filename)

    with open(filepath, "r") as f:
        data = json.load(f)

    for group in GROUPS:
        counts = data[group]
        alignment = compute_alignment(group, counts)

        if alignment is None:
            continue

        expected = expected_ratios[group]
        difference = alignment - expected
        lift = alignment / expected if expected > 0 else None

        rows.append({
            "config": config_name,
            "group": group,
            "alignment": alignment,
            "expected": expected,
            "difference": difference,
            "lift": lift
        })

# Convert to DataFrame
df = pd.DataFrame(rows)

# -----------------------------
# ANALYSIS
# -----------------------------

print("\n=== SUMMARY STATISTICS ===\n")

summary = df.groupby(["config", "group"]).agg(
    mean_alignment=("alignment", "mean"),
    mean_expected=("expected", "mean"),
    mean_difference=("difference", "mean"),
    mean_lift=("lift", "mean"),
    std_difference=("difference", "std"),
    count=("difference", "count")
).reset_index()

print(summary)

# -----------------------------
# T-TESTS
# -----------------------------

print("\n=== T-TEST RESULTS (H0: difference = 0) ===\n")

ttest_results = []

for (config, group), subset in df.groupby(["config", "group"]):
    diffs = subset["difference"].dropna()

    t_stat, p_value = stats.ttest_1samp(diffs, 0)

    ttest_results.append({
        "config": config,
        "group": group,
        "mean_diff": diffs.mean(),
        "t_stat": t_stat,
        "p_value": p_value
    })

ttest_df = pd.DataFrame(ttest_results)
print(ttest_df)

# -----------------------------
# INTERPRETATION HELPER
# -----------------------------

print("\n=== INTERPRETATION ===\n")

for _, row in ttest_df.iterrows():
    significance = "SIGNIFICANT" if row["p_value"] < 0.05 else "NOT SIGNIFICANT"
    direction = "aligned MORE than expected" if row["mean_diff"] > 0 else "aligned LESS than expected"

    print(f"[{row['config']} | {row['group']}] → {direction} ({significance})")

# -----------------------------
# OPTIONAL: SAVE RESULTS
# -----------------------------

summary.to_csv("summary_results.csv", index=False)
ttest_df.to_csv("ttest_results.csv", index=False)

print("\nResults saved to summary_results.csv and ttest_results.csv")