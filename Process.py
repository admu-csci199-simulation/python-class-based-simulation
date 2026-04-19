"""
summarize.py
Run this after each batch of seeds. Reads raw JSON files for a given seed range,
extracts the minimal data needed for analysis, and appends to a summary CSV.
The raw JSONs can be deleted/archived after this runs successfully.

Usage:
    python summarize.py --seed_start 0  --seed_end 9
    python summarize.py --seed_start 10 --seed_end 19
    python summarize.py --seed_start 20 --seed_end 29

    # Optional overrides:
    python summarize.py --seed_start 0 --seed_end 9 --data_dir output --summary summary.csv --n 100 --p 10
"""

import os
import json
import re
import argparse
import csv

AGENT_DISTRIBUTIONS = [
    [4, 1, 1], [3, 2, 1], [3, 1, 2],
    [1, 4, 1], [2, 3, 1], [1, 3, 2],
    [1, 1, 4], [2, 1, 3], [1, 2, 3],
    [2, 2, 2],
]

FIELDNAMES = [
    "run_id", "test_case", "seed", "dist_index",
    "agent_distribution", "red_ratio", "centrist_ratio", "blue_ratio",
    "post_id", "post_belief_value", "post_camp",
    "growth_rate", "decay_rate",
]

SMOOTH_WINDOW = 5  # rolling average window for peak detection


def get_camp(belief_value):
    if belief_value <= -2:   return "red"
    elif belief_value >= 2:  return "blue"
    return "centrist"


def compute_rates(interaction_array):
    """
    Compute growth rate and decay rate from a post's interaction timeseries.

    - Timeseries: sum of red+centrist+blue interactions per timestep.
    - Smooth with a rolling average to avoid noise spikes being misidentified as peak.
    - First active timestep: first index where interactions > 0.
    - Peak: global maximum of the smoothed series from first active timestep onward.
    - Last active timestep: last index where interactions > 0.

    Growth rate = total interactions from first_active to peak
                  divided by number of timesteps in that window.
    Decay rate  = total interactions from peak to last_active
                  divided by number of timesteps in that window.

    Both are 0 if the post was never shared.
    Decay rate is 0 if peak == last_active (no decay phase).
    """
    # Collapse to a 1D timeseries
    series = [sum(t) for t in interaction_array]

    total = sum(series)
    if total == 0:
        return 0.0, 0.0

    # Smoothed series for peak detection
    smoothed = []
    half = SMOOTH_WINDOW // 2
    n = len(series)
    for i in range(n):
        window = series[max(0, i - half): min(n, i + half + 1)]
        smoothed.append(sum(window) / len(window))

    # First and last active timestep
    first_active = next(i for i, v in enumerate(series) if v > 0)
    last_active  = max(i for i, v in enumerate(series) if v > 0)

    # Global peak index within the active window (using smoothed series)
    peak_idx = max(range(first_active, last_active + 1), key=lambda i: smoothed[i])

    # Growth rate
    growth_steps = peak_idx - first_active
    if growth_steps > 0:
        growth_interactions = sum(series[first_active: peak_idx + 1])
        growth_rate = growth_interactions / growth_steps
    else:
        growth_rate = 0.0  # posted and immediately peaked (single timestep)

    # Decay rate
    decay_steps = last_active - peak_idx
    if decay_steps > 0:
        decay_interactions = sum(series[peak_idx: last_active + 1])
        decay_rate = decay_interactions / decay_steps
    else:
        decay_rate = 0.0  # no decay phase

    return growth_rate, decay_rate


def parse_filename(filename):
    match = re.match(r"hyp8-config-(\d+)s-(\d+)\.json", filename)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def process_batch(data_dir, summary_path, seed_start, seed_end, p):
    # Check which (test_case, seed) pairs are already in the summary
    existing = set()
    if os.path.exists(summary_path):
        with open(summary_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing.add((int(row["test_case"]), int(row["seed"])))

    files = sorted(os.listdir(data_dir))
    new_records = []
    processed_files = []

    for filename in files:
        parsed = parse_filename(filename)
        if parsed is None:
            continue

        test_case, seed = parsed

        # Only process seeds in the requested batch
        if not (seed_start <= seed <= seed_end):
            continue

        if (test_case, seed) in existing:
            print(f"  Skipping {filename} (already in summary)")
            continue

        dist_index = test_case // p
        if dist_index >= len(AGENT_DISTRIBUTIONS):
            print(f"  Warning: test case {test_case} out of range. Skipping.")
            continue

        distribution = AGENT_DISTRIBUTIONS[dist_index]
        dist_label = f"{distribution[0]}-{distribution[1]}-{distribution[2]}"

        filepath = os.path.join(data_dir, filename)
        with open(filepath, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                print(f"  Warning: Could not parse {filename}: {e}. Skipping.")
                continue

        static_info = data.get("static_post_information", {})
        shared_data = data.get("post_shared_data", {})

        for post_id, interactions in shared_data.items():
            post_meta    = static_info.get(post_id, {})
            belief_value = post_meta.get("post_beliefValue")
            post_camp    = post_meta.get("post_camp") or (get_camp(belief_value) if belief_value is not None else None)
            growth_rate, decay_rate = compute_rates(interactions)

            new_records.append({
                "run_id":             f"{test_case}s{seed}",
                "test_case":          test_case,
                "seed":               seed,
                "dist_index":         dist_index,
                "agent_distribution": dist_label,
                "red_ratio":          distribution[0],
                "centrist_ratio":     distribution[1],
                "blue_ratio":         distribution[2],
                "post_id":            post_id,
                "post_belief_value":  belief_value,
                "post_camp":          post_camp,
                "growth_rate":        growth_rate,
                "decay_rate":         decay_rate,
            })

        processed_files.append(filename)

    if not new_records:
        print("  No new records to add.")
        return 0

    # Append to summary CSV (create with header if it doesn't exist yet)
    write_header = not os.path.exists(summary_path)
    with open(summary_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerows(new_records)

    print(f"  {len(new_records)} records appended to '{summary_path}'")
    print(f"  Files processed: {processed_files}")
    return len(new_records)


def main():
    parser = argparse.ArgumentParser(description="Summarize a batch of seeds into the running summary CSV.")
    parser.add_argument("--seed_start", type=int, required=True, help="First seed in this batch (inclusive)")
    parser.add_argument("--seed_end",   type=int, required=True, help="Last seed in this batch (inclusive)")
    parser.add_argument("--data_dir",   default="output",      help="Directory containing JSON files")
    parser.add_argument("--summary",    default="summary.csv", help="Running summary CSV to append to")
    parser.add_argument("--n",          type=int, default=100, help="Total number of test cases")
    parser.add_argument("--p",          type=int, default=10,  help="Test cases per distribution")
    args = parser.parse_args()

    p = args.p or args.n // 10

    print(f"\nSummarizing seeds {args.seed_start}–{args.seed_end} from '{args.data_dir}'...")
    count = process_batch(args.data_dir, args.summary, args.seed_start, args.seed_end, p)

    if count > 0:
        print(f"\nDone. Batch complete — {count} new records added.")
        print(f"You can now safely archive or delete the processed JSON files for seeds {args.seed_start}–{args.seed_end}.")
    else:
        print("\nDone. Nothing new was written.")


if __name__ == "__main__":
    main()