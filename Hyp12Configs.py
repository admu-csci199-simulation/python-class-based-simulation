"""
hyp12_config_generator.py
─────────────────────────────────────────────────────────────────────────────
Generates input configuration files for Hypothesis 12:
  "Two posts of different beliefs compete for interactions —
   one real news post vs. a three-wave misinformation campaign."

FACTORIAL DESIGN  (9 conditions × 25 seeds = 225 runs)
────────────────────────────────────────────────────────
Factor B — Real news interest level relative to misinfo spawn 1  (3 levels)
  weaker   : real news interest = min(base_misinfo_interest − 3, 21)
  equal    : real news interest = min(base_misinfo_interest + 3, 21)
  stronger : real news interest = min(base_misinfo_interest + 9, 21)

  Note on naming: even "equal" gives real news a +3 advantage over spawn 1,
  putting it level with spawn 2. "stronger" matches spawn 3's interest level.
  This tests whether real news needs to match the campaign's peak to compete.

Factor C — Real news posting time relative to misinfo spawn 1  (3 levels)
  real-news-first : real news posts 5–60 min before misinfo spawn 1 (control)
  simultaneous    : real news posts at the same time as misinfo spawn 1
  between-waves   : real news posts halfway between spawn 1 and spawn 2

MISINFORMATION STRUCTURE  (fixed design constraints)
──────────────────────────────────────────────────────
  Belief value  : fixed across all three spawns, drawn from {±2, ±3, ±4}.
                  Direction (red/blue) randomised per seed.
  Interest value: escalates by +3 per spawn, capped at 21.
    Spawn 1 (base): base_misinfo_interest
    Spawn 2       : base_misinfo_interest + 3
    Spawn 3       : base_misinfo_interest + 6
  Resurgence gap : 300–720 min between consecutive spawns (per seed).
  spawn field    : False for spawn 1 (base), True for spawns 2 and 3.

REAL NEWS STRUCTURE  (fixed design constraint)
───────────────────────────────────────────────
  Belief value  : equidistant mirror of misinfo belief relative to 0.
                  E.g. misinfo = −3 → real news = +3.
  Interest      : controlled by Factor B (see above).
  Posting time  : controlled by Factor C (see above).
  misinformation: False. spawn: False.

AGENT COMPOSITION  (fixed across all runs)
────────────────────────────────────────────
  1 002 agents, perfectly balanced: 334 red · 334 centrist · 334 blue.
  Cognitive types: 15% gullible, 65% normal, 20% stubborn.
  Share propensity: 90% lurker, 9% normal sharer, 1% active.
"""

import json
import os
import random

# ─────────────────────────────────────────────────────────────────────────────
#  GLOBAL CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_FOLDER  = "input"
SEEDS_PER_COND = 8           # runs per condition → 9 × 25 = 225 total files

AGENTS_PER_FILE = 1002        # must be divisible by 3
assert AGENTS_PER_FILE % 3 == 0

MAX_INTEREST_VALUE    = 21    # hard cap from manuscript calibration
MISINFO_INTEREST_STEP = 3     # interest increment per spawn wave

# Misinfo base interest range — chosen so all three spawns stay ≤ 21.
# Worst case: base + 6 ≤ 21  →  base ≤ 15. Use 6–15 for a comfortable range.
MISINFO_BASE_INTEREST_MIN = 6
MISINFO_BASE_INTEREST_MAX = 15

# Misinfo spawn 1 posting time — capped so all three spawns fit within 48 h.
# Worst case: t1 + 2 × 720 min gap = t1 + 1440 min ≤ 2880  →  t1 ≤ 1440.
MISINFO_BASE_TIME_MAX = 1440

# Resurgence gap between consecutive misinfo spawns (minutes)
RESURGENCE_MIN = 300          # 5 hours
RESURGENCE_MAX = 720          # 12 hours

# Allowed absolute misinfo belief values (direction is randomised per seed)
MISINFO_BELIEF_ABS_VALUES = [2, 3, 4]

# Real-news-first head-start range (minutes before misinfo spawn 1)
RN_FIRST_OFFSET_MIN = 5
RN_FIRST_OFFSET_MAX = 60


# ─────────────────────────────────────────────────────────────────────────────
#  FACTOR DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

# Factor B: delta applied to base_misinfo_interest to get real news interest.
# Result is capped at MAX_INTEREST_VALUE.
FACTOR_B = {
    "weaker":   -3,   # real news < spawn 1 interest
    "equal":    +3,   # real news == spawn 2 interest (matches second wave)
    "stronger": +9,   # real news == spawn 3 interest (matches third wave)
}

# Factor C: how real news posting time is computed relative to misinfo spawn 1.
FACTOR_C = ["real-news-first", "simultaneous", "between-waves"]


# ─────────────────────────────────────────────────────────────────────────────
#  AGENT HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def split_ratio_90_9_1(total):
    lurker        = round(total * 0.90)
    normal_sharer = round(total * 0.09)
    active        = total - lurker - normal_sharer
    return lurker, normal_sharer, active


def split_cognitive_types(total):
    gullible = round(total * 0.15)
    normal   = round(total * 0.65)
    stubborn = total - gullible - normal
    return gullible, normal, stubborn


def generate_agents():
    n     = AGENTS_PER_FILE
    third = n // 3

    gullible, normal_cog, stubborn = split_cognitive_types(n)
    lurker, normal_sharer, active  = split_ratio_90_9_1(n)

    return {
        "Agent Count":         n,
        "Gullible Count":      gullible,
        "Normal Count":        normal_cog,
        "Stubborn Count":      stubborn,
        "Lurker Count":        lurker,
        "Normal Sharer Count": normal_sharer,
        "Active Count":        active,
        "Red Count":           third,
        "Centrist Count":      third,
        "Blue Count":          third,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  POST GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def generate_posts(
    misinfo_sign,
    misinfo_belief_abs,
    base_misinfo_interest,
    interest_delta,
    timing_label,
    resurgence_gap,
    base_posting_time,
    rn_head_start,
):
    """
    Returns a list of four posts:
      postID 0 — misinfo spawn 1 (base)
      postID 1 — misinfo spawn 2 (resurgence 1)
      postID 2 — misinfo spawn 3 (resurgence 2)
      postID 3 — real news

    All posts share the same topic so topic preference is not a confound.
    """
    topic = random.randint(0, 4)

    # Misinfo belief is fixed across all three spawns
    misinfo_belief = misinfo_sign * misinfo_belief_abs

    # Misinfo posting times
    t1 = base_posting_time
    t2 = t1 + resurgence_gap
    t3 = t2 + resurgence_gap

    # Misinfo interest values — escalate by +3 each wave, capped at 21
    i1 = base_misinfo_interest
    i2 = min(base_misinfo_interest + MISINFO_INTEREST_STEP,     MAX_INTEREST_VALUE)
    i3 = min(base_misinfo_interest + 2 * MISINFO_INTEREST_STEP, MAX_INTEREST_VALUE)

    misinfo_posts = [
        {
            "postID":         0,
            "postTopic":      topic,
            "beliefValue":    misinfo_belief,
            "interestValue":  i1,
            "postingTime":    t1,
            "misinformation": True,
            "spawn":          False,   # base post — marks the start of the triad
        },
        {
            "postID":         1,
            "postTopic":      topic,
            "beliefValue":    misinfo_belief,
            "interestValue":  i2,
            "postingTime":    t2,
            "misinformation": True,
            "spawn":          True,    # resurgence 1
        },
        {
            "postID":         2,
            "postTopic":      topic,
            "beliefValue":    misinfo_belief,
            "interestValue":  i3,
            "postingTime":    t3,
            "misinformation": True,
            "spawn":          True,    # resurgence 2
        },
    ]

    # Real news belief: equidistant mirror of misinfo belief relative to 0
    rn_belief   = -misinfo_sign * misinfo_belief_abs

    # Real news interest: base misinfo interest + delta, capped at 21
    rn_interest = min(base_misinfo_interest + interest_delta, MAX_INTEREST_VALUE)

    # Real news posting time: resolved from Factor C label
    if timing_label == "real-news-first":
        rn_posting_time = max(0, t1 - rn_head_start)
    elif timing_label == "simultaneous":
        rn_posting_time = t1
    else:  # between-waves
        rn_posting_time = t1 + resurgence_gap // 2

    real_news_post = {
        "postID":         3,
        "postTopic":      topic,
        "beliefValue":    rn_belief,
        "interestValue":  rn_interest,
        "postingTime":    rn_posting_time,
        "misinformation": False,
        "spawn":          False,
    }

    return misinfo_posts + [real_news_post]


# ─────────────────────────────────────────────────────────────────────────────
#  FILE GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def generate_file(filename, factor_b_label, factor_c_label, seed):
    random.seed(seed)

    # Per-seed randomised values
    misinfo_sign          = random.choice([-1, +1])
    misinfo_belief_abs    = random.choice(MISINFO_BELIEF_ABS_VALUES)
    base_misinfo_interest = random.randint(
        MISINFO_BASE_INTEREST_MIN, MISINFO_BASE_INTEREST_MAX
    )
    base_posting_time = random.randint(0, MISINFO_BASE_TIME_MAX)
    resurgence_gap    = random.randint(RESURGENCE_MIN, RESURGENCE_MAX)
    rn_head_start     = random.randint(RN_FIRST_OFFSET_MIN, RN_FIRST_OFFSET_MAX)

    interest_delta = FACTOR_B[factor_b_label]

    posts = generate_posts(
        misinfo_sign          = misinfo_sign,
        misinfo_belief_abs    = misinfo_belief_abs,
        base_misinfo_interest = base_misinfo_interest,
        interest_delta        = interest_delta,
        timing_label          = factor_c_label,
        resurgence_gap        = resurgence_gap,
        base_posting_time     = base_posting_time,
        rn_head_start         = rn_head_start,
    )

    rn_post        = posts[3]
    misinfo_belief = misinfo_sign * misinfo_belief_abs

    data = {
        "Agents": generate_agents(),
        "Posts":  posts,
        "Metadata": {
            "hypothesis": "H12_competing_beliefs",
            "seed":       seed,

            # Factor labels
            "factor_B": factor_b_label,
            "factor_C": factor_c_label,

            # Misinfo details
            "misinfo_direction":   "red" if misinfo_sign == -1 else "blue",
            "misinfo_belief":      misinfo_belief,
            "misinfo_interest_s1": base_misinfo_interest,
            "misinfo_interest_s2": min(base_misinfo_interest + MISINFO_INTEREST_STEP,
                                       MAX_INTEREST_VALUE),
            "misinfo_interest_s3": min(base_misinfo_interest + 2 * MISINFO_INTEREST_STEP,
                                       MAX_INTEREST_VALUE),
            "misinfo_posting_s1":  posts[0]["postingTime"],
            "misinfo_posting_s2":  posts[1]["postingTime"],
            "misinfo_posting_s3":  posts[2]["postingTime"],
            "resurgence_gap_min":  resurgence_gap,

            # Real news details
            "real_news_belief":   rn_post["beliefValue"],
            "real_news_interest": rn_post["interestValue"],
            "real_news_posting":  rn_post["postingTime"],
            "interest_delta":     interest_delta,

            # Derived metrics useful for analysis grouping
            "rn_head_start_min":     (rn_head_start
                                      if factor_c_label == "real-news-first"
                                      else None),
            "interest_gap_rn_vs_s1": rn_post["interestValue"] - base_misinfo_interest,
            "interest_gap_rn_vs_s2": rn_post["interestValue"] - min(
                                         base_misinfo_interest + MISINFO_INTEREST_STEP,
                                         MAX_INTEREST_VALUE),
            "interest_gap_rn_vs_s3": rn_post["interestValue"] - min(
                                         base_misinfo_interest + 2 * MISINFO_INTEREST_STEP,
                                         MAX_INTEREST_VALUE),
        },
    }

    path = os.path.join(OUTPUT_FOLDER, filename)
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    file_index           = 0
    conditions_generated = 0

    for b_label in FACTOR_B:       # 3 levels
        for c_label in FACTOR_C:   # 3 levels

            condition_label = f"B-{b_label}_C-{c_label}"

            for seed_offset in range(SEEDS_PER_COND):   # 25 seeds
                seed     = conditions_generated * SEEDS_PER_COND + seed_offset
                filename = f"hyp12_{condition_label}_s{seed_offset:02d}.json"
                generate_file(
                    filename       = filename,
                    factor_b_label = b_label,
                    factor_c_label = c_label,
                    seed           = seed,
                )
                file_index += 1

            conditions_generated += 1

    print(f"Generated {file_index} config files across {conditions_generated} conditions.")
    print(f"Output folder: '{OUTPUT_FOLDER}/'")
    print()
    print("Condition breakdown:")
    print(f"  Factor B (real news interest level): {len(FACTOR_B)} levels → {list(FACTOR_B.keys())}")
    print(f"  Factor C (real news posting timing): {len(FACTOR_C)} levels → {FACTOR_C}")
    print(f"  Seeds per condition                : {SEEDS_PER_COND}")
    print(f"  Total files                        : {file_index}")


if __name__ == "__main__":
    main()