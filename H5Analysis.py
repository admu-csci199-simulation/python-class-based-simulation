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

def compute_metrics(layers):
    """
    layers: list of counts per layer
    returns: total, depth, breadth, shape_ratio
    """

    layers = np.array(layers)

    total = layers.sum()

    # Depth = number of layers with nonzero values
    nonzero_layers = np.count_nonzero(layers)
    depth = nonzero_layers

    # Breadth = max layer size
    breadth = layers.max() if len(layers) > 0 else 0

    # Shape ratio = breadth / depth
    shape_ratio = breadth / depth if depth > 0 else 0

    return total, depth, breadth, shape_ratio

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

    # ---- MISINFORMATION POSTS ----
    for post_id, layers in data.get("misinformation", {}).items():
        total, depth, breadth, shape_ratio = compute_metrics(layers)

        rows.append({
            "type": "misinformation",
            "post_id": post_id,
            "total": total,
            "depth": depth,
            "breadth": breadth,
            "shape_ratio": shape_ratio
        })

    # ---- REAL POSTS ----
    for post_id, layers in data.get("regular", {}).items():
        total, depth, breadth, shape_ratio = compute_metrics(layers)

        rows.append({
            "type": "real",
            "post_id": post_id,
            "total": total,
            "depth": depth,
            "breadth": breadth,
            "shape_ratio": shape_ratio
        })

# Convert to DataFrame
df = pd.DataFrame(rows)

# -----------------------------
# SUMMARY STATISTICS
# -----------------------------

print("\n=== SUMMARY STATISTICS ===\n")

summary = df.groupby("type").agg(
    mean_total=("total", "mean"),
    std_total=("total", "std"),
    mean_depth=("depth", "mean"),
    mean_breadth=("breadth", "mean"),
    mean_shape_ratio=("shape_ratio", "mean"),
    count=("total", "count")
).reset_index()

print(summary)

# -----------------------------
# T-TESTS (MISINFO vs REAL)
# -----------------------------

print("\n=== T-TEST RESULTS (MISINFO vs REAL) ===\n")

metrics = ["total", "depth", "breadth", "shape_ratio"]

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