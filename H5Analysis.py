import os
import json
import numpy as np
import pandas as pd
from scipy import stats

# -----------------------------
# CONFIG
# -----------------------------

OUTPUT_DIR = "output"

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------

def compute_depth(layers):
    return np.count_nonzero(layers)


def compute_breadth(layers):
    return max(layers) if len(layers) > 0 else 0


def compute_shape_ratio(depth, breadth):
    return breadth / depth if depth > 0 else 0


def compute_branching_factor(layers):
    layers = np.array(layers)

    ratios = []
    for i in range(len(layers) - 1):
        if layers[i] > 0:
            ratios.append(layers[i + 1] / layers[i])

    if len(ratios) == 0:
        return 0

    return np.mean(ratios)


def compute_effective_depth(layers):
    layers = np.array(layers)
    total = layers.sum()

    if total == 0:
        return 0

    indices = np.arange(len(layers))
    return np.sum(indices * layers) / total


# -----------------------------
# PARSE FILES
# -----------------------------

rows = []

for filename in os.listdir(OUTPUT_DIR):
    if not filename.endswith(".json") or not filename.startswith("hyp5"):
        continue

    filepath = os.path.join(OUTPUT_DIR, filename)

    with open(filepath, "r") as f:
        data = json.load(f)

    # ---- MISINFORMATION ----
    for post_id, layers in data.get("misinformation", {}).items():
        depth = compute_depth(layers)
        breadth = compute_breadth(layers)
        shape_ratio = compute_shape_ratio(depth, breadth)
        branching = compute_branching_factor(layers)
        eff_depth = compute_effective_depth(layers)

        rows.append({
            "type": "misinformation",
            "depth": depth,
            "breadth": breadth,
            "shape_ratio": shape_ratio,
            "branching_factor": branching,
            "effective_depth": eff_depth
        })

    # ---- REAL ----
    for post_id, layers in data.get("regular", {}).items():
        depth = compute_depth(layers)
        breadth = compute_breadth(layers)
        shape_ratio = compute_shape_ratio(depth, breadth)
        branching = compute_branching_factor(layers)
        eff_depth = compute_effective_depth(layers)

        rows.append({
            "type": "real",
            "depth": depth,
            "breadth": breadth,
            "shape_ratio": shape_ratio,
            "branching_factor": branching,
            "effective_depth": eff_depth
        })

# Convert to DataFrame
df = pd.DataFrame(rows)

# -----------------------------
# SUMMARY STATISTICS
# -----------------------------

print("\n=== SUMMARY STATISTICS ===\n")

summary = df.groupby("type").agg(
    mean_depth=("depth", "mean"),
    mean_breadth=("breadth", "mean"),
    mean_shape_ratio=("shape_ratio", "mean"),
    mean_branching=("branching_factor", "mean"),
    mean_effective_depth=("effective_depth", "mean"),
    count=("depth", "count")
).reset_index()

print(summary)

# -----------------------------
# T-TESTS
# -----------------------------

print("\n=== T-TEST RESULTS (MISINFO vs REAL) ===\n")

metrics = [
    "depth",
    "breadth",
    "shape_ratio",
    "branching_factor",
    "effective_depth"
]

ttest_results = []

misinfo_df = df[df["type"] == "misinformation"]
real_df = df[df["type"] == "real"]

for metric in metrics:
    misinfo_vals = misinfo_df[metric].dropna()
    real_vals = real_df[metric].dropna()

    t_stat, p_value = stats.ttest_ind(misinfo_vals, real_vals, equal_var=False)

    ttest_results.append({
        "metric": metric,
        "misinfo_mean": misinfo_vals.mean(),
        "real_mean": real_vals.mean(),
        "t_stat": t_stat,
        "p_value": p_value
    })

ttest_df = pd.DataFrame(ttest_results)
print(ttest_df)

# -----------------------------
# INTERPRETATION
# -----------------------------

print("\n=== INTERPRETATION ===\n")

for _, row in ttest_df.iterrows():
    significance = "SIGNIFICANT" if row["p_value"] < 0.05 else "NOT SIGNIFICANT"

    if row["misinfo_mean"] < row["real_mean"]:
        direction = "MISINFORMATION is LOWER"
    elif row["misinfo_mean"] > row["real_mean"]:
        direction = "MISINFORMATION is HIGHER"
    else:
        direction = "NO DIFFERENCE"

    print(f"[{row['metric']}] → {direction} than REAL ({significance})")

# -----------------------------
# SAVE RESULTS
# -----------------------------

summary.to_csv("hyp5_summary.csv", index=False)
ttest_df.to_csv("hyp5_ttest_results.csv", index=False)

print("\nResults saved to hyp5_summary.csv and hyp5_ttest_results.csv")