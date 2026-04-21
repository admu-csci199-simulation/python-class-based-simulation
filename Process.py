"""
extract_chunk.py
================
Processes a batch of simulation JSON files, extracts per-post virality
features, and appends them to a running summary CSV.

IMPORTANT — Cumulative series correction:
  The raw post_shared_data values are CUMULATIVE interaction counts, not
  per-slice counts. This script differentiates the active slice first
  (np.diff) to recover the actual interactions gained at each time step
  before computing growth rate and decay rate.

Usage:
    # Process 10 seeds at a time, appending to summary.csv each run
    python extract_chunk.py --files output/hyp9-config-seed001.json output/hyp9-config-seed002.json ...
    python extract_chunk.py --files output/hyp9-config-seed011.json ...

    # Or pass a glob pattern (quoted to prevent shell expansion)
    python extract_chunk.py --glob "output/hyp9-config-seed0[01]*.json"

    # Override defaults
    python extract_chunk.py --files ... --summary summary.csv --n_agents 300
"""

import gc
import csv
import json
import argparse
import warnings
import glob as glob_module
import numpy as np
from pathlib import Path
from scipy.stats import linregress

warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────────
N_AGENTS_DEFAULT = 300
SUMMARY_FILE     = Path("summary.csv")

# Columns written to summary CSV — order matters for append consistency
CSV_COLUMNS = [
    "source_file",
    "post_id",
    "interest_value",
    "post_time",
    "peak_interactions_norm",   # peak of differentiated series / n_agents
    "growth_rate",              # OLS slope of rise phase (differentiated)
    "decay_rate",               # OLS slope of fall phase (differentiated)
]


# ── FEATURE EXTRACTION ────────────────────────────────────────

def _extract_from_run(run: dict, source_file: str, n_agents: int) -> list[dict]:
    """
    Extract one feature row per post from a single run dict.

    The cumulative series is differentiated via np.diff so that growth_rate
    and decay_rate reflect actual per-slice interaction changes, not totals.
    Rows with no activity after post_time are recorded as zeros.
    """
    static = run.get("static_post_information", {})
    shared = run.get("post_shared_data", {})
    rows   = []

    for post_id, s_data in shared.items():
        if post_id not in static:
            continue

        iv        = float(static[post_id]["interest_value"])
        post_time = int(static[post_id]["post_time"])

        # Load as float32 to save memory, slice active window, then differentiate
        cumulative = np.asarray(s_data, dtype=np.float32)
        active_cum = cumulative[post_time:].astype(np.float64)
        del cumulative  # free immediately

        # Differentiate: recover per-slice interaction counts from cumulative
        # np.diff gives len(active_cum)-1 values; prepend 0 to keep alignment
        # so index 0 = interactions gained at post_time itself
        if active_cum.size < 2:
            rows.append(_zero_row(source_file, post_id, iv, post_time))
            continue

        active = np.diff(active_cum, prepend=active_cum[0])
        # Note: prepend active_cum[0] so first diff = active_cum[0] - active_cum[0] = 0,
        # which is correct since at post_time the post just appeared with that baseline.
        # Alternatively if the baseline is always 0 at post_time this is equivalent.
        del active_cum

        if active.sum() == 0:
            rows.append(_zero_row(source_file, post_id, iv, post_time))
            continue

        peak   = float(active.max())
        t_peak = int(np.argmax(active))

        # Growth rate: OLS slope from time 0 to peak on differentiated series
        growth_rate = 0.0
        if t_peak > 0:
            g_x = np.arange(t_peak + 1, dtype=np.float64)
            g_y = active[:t_peak + 1]
            if g_x.std() > 0:
                growth_rate = float(linregress(g_x, g_y).slope)

        # Decay rate: OLS slope from peak to last nonzero on differentiated series
        decay_rate = 0.0
        tail = active[t_peak:]
        nonzero_idx = np.where(tail > 0)[0]
        if nonzero_idx.size > 1:
            d_seg = tail[:nonzero_idx[-1] + 1]
            d_x   = np.arange(len(d_seg), dtype=np.float64)
            if d_x.std() > 0:
                decay_rate = float(linregress(d_x, d_seg).slope)

        rows.append({
            "source_file":            source_file,
            "post_id":                post_id,
            "interest_value":         iv,
            "post_time":              post_time,
            "peak_interactions_norm": peak / n_agents,
            "growth_rate":            growth_rate,
            "decay_rate":             decay_rate,
        })

    return rows


def _zero_row(source_file, post_id, iv, post_time) -> dict:
    return {
        "source_file":            source_file,
        "post_id":                post_id,
        "interest_value":         iv,
        "post_time":              post_time,
        "peak_interactions_norm": 0.0,
        "growth_rate":            0.0,
        "decay_rate":             0.0,
    }


# ── SUMMARY CSV APPEND ────────────────────────────────────────

def append_to_summary(rows: list[dict], summary_path: Path):
    """
    Append feature rows to the summary CSV.
    Creates the file with a header if it does not yet exist.
    """
    file_exists = summary_path.exists()
    with open(summary_path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        if not file_exists:
            writer.writeheader()
            print(f"  Created new summary file: {summary_path}")
        writer.writerows(rows)


# ── MAIN ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract virality features from a batch of JSON files and append to summary CSV."
    )
    parser.add_argument("--files",    nargs="*", default=[],
                        help="Explicit list of JSON file paths to process")
    parser.add_argument("--glob",     default=None,
                        help="Glob pattern for JSON files (e.g. 'output/hyp9-config-seed0*.json')")
    parser.add_argument("--summary",  default=str(SUMMARY_FILE),
                        help="Path to the running summary CSV (default: summary.csv)")
    parser.add_argument("--n_agents", type=int, default=N_AGENTS_DEFAULT,
                        help="Total number of agents in simulation (default: 300)")
    args = parser.parse_args()

    # Resolve file list
    file_paths = [Path(f) for f in args.files]
    if args.glob:
        file_paths += [Path(f) for f in sorted(glob_module.glob(args.glob))]
    if not file_paths:
        parser.error("No files specified. Use --files or --glob.")

    summary_path = Path(args.summary)

    print(f"\n{'='*60}")
    print(f"  extract_chunk.py  —  n_agents={args.n_agents}")
    print(f"  Files to process : {len(file_paths)}")
    print(f"  Summary CSV      : {summary_path}")
    print(f"{'='*60}\n")

    total_rows = 0
    for i, fpath in enumerate(file_paths, 1):
        print(f"  [{i:>3}/{len(file_paths)}] {fpath.name} ... ", end="", flush=True)

        with open(fpath, "r") as fh:
            run = json.load(fh)

        rows = _extract_from_run(run, fpath.name, args.n_agents)

        del run
        gc.collect()

        append_to_summary(rows, summary_path)
        total_rows += len(rows)
        print(f"{len(rows)} posts written  (CSV total so far: {total_rows})")

    print(f"\n  Done. {total_rows} rows appended to {summary_path}\n")


if __name__ == "__main__":
    main()