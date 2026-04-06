import os
import json
import numpy as np
import pandas as pd
from scipy import stats
import re

# -----------------------------
# CONFIG
# -----------------------------

OUTPUT_DIR = "output"

GROUPS = ["red", "centrist", "blue"]
CENTRIST_INDEX = 1

TOTAL_AGENTS = 300
DISAPPEAR_THRESHOLD = 0.05 * TOTAL_AGENTS  # 5% threshold (15 agents)

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------

def extract_config(filename):
    # hyp4-red12s-45.json → red
    match = re.match(r"hyp4-(blue|centrist|red|equal)", filename)
    if match:
        return match.group(1)
    return None

def compute_slope(time_steps, values):
    # Linear regression slope
    if len(values) < 2:
        return None
    slope, _ = np.polyfit(time_steps, values, 1)
    return slope

# -----------------------------
# PARSE FILES
# -----------------------------

rows = []

for filename in os.listdir(OUTPUT_DIR):
    if not filename.endswith(".json") or not filename.startswith("hyp4"):
        continue

    config = extract_config(filename)
    if config is None:
        print(f"Skipping invalid filename: {filename}")
        continue

    filepath = os.path.join(OUTPUT_DIR, filename)

    with open(filepath, "r") as f:
        data = json.load(f)

    # Sort time steps numerically
    time_steps = sorted([int(t) for t in data.keys()])
    centrist_counts = [data[str(t)][CENTRIST_INDEX] for t in time_steps]

    if len(centrist_counts) == 0:
        continue

    initial = centrist_counts[0]
    final = centrist_counts[-1]

    delta = final - initial
    slope = compute_slope(time_steps, centrist_counts)

    disappeared = final <= DISAPPEAR_THRESHOLD

    rows.append({
        "config": config,
        "initial_centrist": initial,
        "final_centrist": final,
        "delta": delta,
        "slope": slope,
        "disappeared": disappeared
    })

# Convert to DataFrame
df = pd.DataFrame(rows)

# -----------------------------
# SUMMARY STATISTICS
# -----------------------------

print("\n=== SUMMARY STATISTICS ===\n")

summary = df.groupby("config").agg(
    mean_initial=("initial_centrist", "mean"),
    mean_final=("final_centrist", "mean"),
    mean_delta=("delta", "mean"),
    std_delta=("delta", "std"),
    mean_slope=("slope", "mean"),
    disappearance_rate=("disappeared", "mean"),
    count=("delta", "count")
).reset_index()

print(summary)

# -----------------------------
# T-TESTS (DELTA)
# -----------------------------

print("\n=== T-TEST (ΔC < 0) ===\n")

ttest_delta_results = []

for config, subset in df.groupby("config"):
    deltas = subset["delta"].dropna()

    t_stat, p_value = stats.ttest_1samp(deltas, 0)

    ttest_delta_results.append({
        "config": config,
        "mean_delta": deltas.mean(),
        "t_stat": t_stat,
        "p_value": p_value
    })

ttest_delta_df = pd.DataFrame(ttest_delta_results)
print(ttest_delta_df)

# -----------------------------
# T-TESTS (SLOPE)
# -----------------------------

print("\n=== T-TEST (Slope < 0) ===\n")

ttest_slope_results = []

for config, subset in df.groupby("config"):
    slopes = subset["slope"].dropna()

    t_stat, p_value = stats.ttest_1samp(slopes, 0)

    ttest_slope_results.append({
        "config": config,
        "mean_slope": slopes.mean(),
        "t_stat": t_stat,
        "p_value": p_value
    })

ttest_slope_df = pd.DataFrame(ttest_slope_results)
print(ttest_slope_df)

# -----------------------------
# INTERPRETATION
# -----------------------------

print("\n=== INTERPRETATION ===\n")

for _, row in ttest_delta_df.iterrows():
    significance = "SIGNIFICANT" if row["p_value"] < 0.05 else "NOT SIGNIFICANT"

    if row["mean_delta"] < 0:
        direction = "Centrist population DECREASED"
    elif row["mean_delta"] > 0:
        direction = "Centrist population INCREASED"
    else:
        direction = "NO CHANGE"

    print(f"[{row['config']}] ΔC → {direction} ({significance})")

print("\n--- Slope Interpretation ---\n")

for _, row in ttest_slope_df.iterrows():
    significance = "SIGNIFICANT" if row["p_value"] < 0.05 else "NOT SIGNIFICANT"

    if row["mean_slope"] < 0:
        direction = "DOWNWARD TREND"
    elif row["mean_slope"] > 0:
        direction = "UPWARD TREND"
    else:
        direction = "FLAT TREND"

    print(f"[{row['config']}] slope → {direction} ({significance})")

# -----------------------------
# SAVE RESULTS
# -----------------------------

summary.to_csv("hyp4_summary.csv", index=False)
ttest_delta_df.to_csv("hyp4_ttest_delta.csv", index=False)
ttest_slope_df.to_csv("hyp4_ttest_slope.csv", index=False)

print("\nResults saved to CSV files.")