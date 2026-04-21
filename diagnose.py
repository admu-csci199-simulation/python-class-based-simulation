"""
diagnose.py
Inspects the summary CSV and a sample raw JSON file to diagnose why
decay rates may be coming out as zero.

Usage:
    python diagnose.py --summary summary.csv --data_dir output
"""

import os
import re
import json
import math
import argparse
import csv
from scipy.stats import linregress

SMOOTH_WINDOW = 5


def inspect_summary(summary_path):
    print("\n=== SUMMARY CSV INSPECTION ===")
    growth_rates = []
    decay_rates  = []

    with open(summary_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            growth_rates.append(float(row["growth_rate"]))
            decay_rates.append(float(row["decay_rate"]))

    total = len(decay_rates)
    zero_decay  = sum(1 for v in decay_rates  if v == 0.0)
    zero_growth = sum(1 for v in growth_rates if v == 0.0)

    print(f"Total records      : {total}")
    print(f"Decay  == 0.0      : {zero_decay}  ({100*zero_decay/total:.1f}%)")
    print(f"Growth == 0.0      : {zero_growth} ({100*zero_growth/total:.1f}%)")

    nonzero_decay = [v for v in decay_rates if v > 0.0]
    if nonzero_decay:
        print(f"Decay  non-zero range : {min(nonzero_decay):.6f} – {max(nonzero_decay):.6f}")
    else:
        print("Decay  non-zero range : NO NON-ZERO VALUES FOUND")

    nonzero_growth = [v for v in growth_rates if v > 0.0]
    if nonzero_growth:
        print(f"Growth non-zero range : {min(nonzero_growth):.6f} – {max(nonzero_growth):.6f}")
    else:
        print("Growth non-zero range : NO NON-ZERO VALUES FOUND")


def inspect_json(json_path, n_posts=3):
    print(f"\n=== RAW JSON INSPECTION: {json_path} ===")
    with open(json_path, "r") as f:
        data = json.load(f)

    static_info = data.get("static_post_information", {})
    shared_data = data.get("post_shared_data", {})

    for i, (post_id, interactions) in enumerate(shared_data.items()):
        if i >= n_posts:
            break

        cumulative = [sum(t) for t in interactions]
        series = [0] + [max(0, cumulative[i] - cumulative[i-1]) for i in range(1, len(cumulative))]
        total  = sum(series)
        nonzero_steps = [(idx, v) for idx, v in enumerate(series) if v > 0]

        print(f"\n  Post: {post_id}")
        print(f"  Belief value   : {static_info.get(post_id, {}).get('post_beliefValue')}")
        print(f"  Total shares   : {total}")
        print(f"  Active steps   : {len(nonzero_steps)} / {len(series)}")

        if not nonzero_steps:
            print("  → Never shared, skipping timeseries breakdown.")
            continue

        first_active = nonzero_steps[0][0]
        last_active  = nonzero_steps[-1][0]

        # Smooth for peak detection
        half = SMOOTH_WINDOW // 2
        n = len(series)
        smoothed = []
        for idx in range(n):
            window = series[max(0, idx - half): min(n, idx + half + 1)]
            smoothed.append(sum(window) / len(window))

        peak_idx = max(range(first_active, last_active + 1), key=lambda idx: smoothed[idx])

        print(f"  First active   : t={first_active}")
        print(f"  Peak           : t={peak_idx}  (value={series[peak_idx]})")
        print(f"  Last active    : t={last_active}")

        growth_segment = series[first_active: peak_idx + 1]
        decay_segment  = series[peak_idx: last_active + 1]

        print(f"  Growth segment : {len(growth_segment)} steps, values={growth_segment[:10]}{'...' if len(growth_segment)>10 else ''}")
        print(f"  Decay segment  : {len(decay_segment)} steps,  values={decay_segment[:10]}{'...' if len(decay_segment)>10 else ''}")

        log_decay = [math.log1p(v) for v in decay_segment]
        print(f"  Log decay seg  : {[round(v,3) for v in log_decay[:10]]}{'...' if len(log_decay)>10 else ''}")
        print(f"  Unique log vals: {len(set(log_decay))}")

        if len(log_decay) >= 2 and len(set(log_decay)) > 1:
            x = list(range(len(log_decay)))
            result = linregress(x, log_decay)
            print(f"  OLS slope      : {result.slope:.6f}  (decay_rate = {abs(result.slope):.6f})")
        else:
            print(f"  OLS skipped    : segment too short or constant → decay_rate = 0.0")


def find_sample_json(data_dir):
    for fname in sorted(os.listdir(data_dir)):
        if re.match(r"hyp8-config-\d+s-\d+\.json", fname):
            return os.path.join(data_dir, fname)
    return None


def main():
    parser = argparse.ArgumentParser(description="Diagnose decay rate issues.")
    parser.add_argument("--summary",  default="summary.csv", help="Summary CSV to inspect")
    parser.add_argument("--data_dir", default="output",      help="Directory containing raw JSONs")
    parser.add_argument("--json",     default=None,          help="Specific JSON file to inspect (optional)")
    args = parser.parse_args()

    if os.path.exists(args.summary):
        inspect_summary(args.summary)
    else:
        print(f"Summary file '{args.summary}' not found, skipping CSV inspection.")

    json_path = args.json or find_sample_json(args.data_dir)
    if json_path and os.path.exists(json_path):
        inspect_json(json_path)
    else:
        print(f"\nNo JSON files found in '{args.data_dir}', skipping raw data inspection.")


if __name__ == "__main__":
    main()