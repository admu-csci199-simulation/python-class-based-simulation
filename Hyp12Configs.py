import json
import os
import random

OUTPUT_FOLDER = "input"
NUM_FILES     = 100
AGENTS_PER_FILE = 1002

# ─────────────────────────────────────────────
#  AGENT HELPERS  (composition is FIXED)
# ─────────────────────────────────────────────

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
    """
    Balanced, fixed composition across ALL runs.
    1/3 red · 1/3 centrist · 1/3 blue so the network is politically
    neutral — any interaction gap between the two posts comes purely
    from the posts themselves, not from a majority camp.
    """
    n = AGENTS_PER_FILE
    assert n % 3 == 0, "AGENTS_PER_FILE must be divisible by 3 for a perfect balance."
    third = n // 3

    gullible, normal, stubborn        = split_cognitive_types(n)
    lurker, normal_sharer, active     = split_ratio_90_9_1(n)

    return {
        "Agent Count": n,

        "Gullible Count": gullible,
        "Normal Count":   normal,
        "Stubborn Count": stubborn,

        "Lurker Count":        lurker,
        "Normal Sharer Count": normal_sharer,
        "Active Count":        active,

        # Fixed: perfectly balanced so belief composition is not a factor
        "Red Count":      third,
        "Centrist Count": third,
        "Blue Count":     third,
    }


# ─────────────────────────────────────────────
#  POST VARIATION AXES
# ─────────────────────────────────────────────

# 1. BELIEF PAIRS  (red_belief, blue_belief)
#    Covers symmetric and asymmetric opposition at various distances.
#    Asymmetric pairs let us check whether the post closer to the
#    centrist band enjoys a structural reach advantage.
BELIEF_PAIRS = [
    # --- symmetric ---
    (-4,  4),   # maximum opposition, perfectly symmetric
    (-3,  3),   # strong opposition, symmetric
    (-2,  2),   # mild opposition, symmetric
    # --- asymmetric: red closer to centre ---
    (-2,  4),   # red has centrist reach, blue is extreme
    (-1,  3),   # red is nearly centrist
    # --- asymmetric: blue closer to centre ---
    (-4,  2),   # blue has centrist reach, red is extreme
    (-3,  1),   # blue is nearly centrist
]

# 2. INTEREST EDGE  (delta applied to blue post relative to red post)
#    Tests whether a small interest advantage decides the winner when
#    beliefs are otherwise symmetric.
#    0 → truly equal fight; +N → blue gets an edge; -N → red gets an edge.
INTEREST_DELTAS = [
    -3,   # red has slight interest edge
     0,   # perfectly equal
    +3,   # blue has slight interest edge
]

# 3. TIME OFFSET  (minutes between the two posts, red posts first)
#    Tests whether being first mover matters.
TIME_OFFSETS = [
    0,    # simultaneous — pure belief competition
    30,   # 30 min head-start for red
    120,  # 2 h head-start for red
]


# ─────────────────────────────────────────────
#  POST GENERATION
# ─────────────────────────────────────────────

def generate_competing_posts(red_belief, blue_belief,
                             base_interest, interest_delta,
                             base_posting_time, time_offset,
                             is_misinfo):
    """
    Two posts that compete for interactions.

    Controlled across the pair in every run:
      - postTopic      → same topic (agents are drawn to both equally)
      - misinformation → same flag (no trust asymmetry)

    Variable (the knobs being tested):
      - beliefValue    → the axis of opposition
      - interestValue  → base ± delta to test interest-edge effects
      - postingTime    → offset to test first-mover advantage
    """
    topic = random.randint(0, 4)   # Constants.N_TOPICS = 5

    post_red = {
        "postID":        0,
        "postTopic":     topic,
        "beliefValue":   red_belief,
        "interestValue": base_interest,                    # red gets base
        "postingTime":   base_posting_time,                # red posts first
        "misinformation": is_misinfo,
        "spawn":         False,
    }

    post_blue = {
        "postID":        1,
        "postTopic":     topic,
        "beliefValue":   blue_belief,
        "interestValue": base_interest + interest_delta,   # blue gets base ± delta
        "postingTime":   base_posting_time + time_offset,  # blue may post later
        "misinformation": is_misinfo,
        "spawn":         False,
    }

    return [post_red, post_blue]


# ─────────────────────────────────────────────
#  FILE GENERATION
# ─────────────────────────────────────────────

def generate_file(index):
    # Randomise all three competition axes per run so the 100 files
    # give broad, unbiased coverage of the parameter space.
    red_belief, blue_belief = random.choice(BELIEF_PAIRS)
    interest_delta          = random.choice(INTEREST_DELTAS)
    time_offset             = random.choice(TIME_OFFSETS)

    # Base interest value: mid-range so both posts are viable
    base_interest = random.randint(6, 14)

    # Anchor posting time in first 12 h so both posts have time to propagate
    base_posting_time = random.randint(0, 60 * 12)

    # Misinfo flag is the same for both posts in a given run
    is_misinfo = random.choice([True, False])

    data = {
        "Agents": generate_agents(),
        "Posts":  generate_competing_posts(
                      red_belief, blue_belief,
                      base_interest, interest_delta,
                      base_posting_time, time_offset,
                      is_misinfo,
                  ),
        # Not read by Main.py — attach to output rows during analysis
        # by joining on filename.
        "Metadata": {
            "hypothesis":      "competing_beliefs",

            # Post-level knobs
            "red_belief":      red_belief,
            "blue_belief":     blue_belief,
            "polarity_gap":    blue_belief - red_belief,
            "belief_symmetry": red_belief == -blue_belief,

            "red_interest":    base_interest,
            "blue_interest":   base_interest + interest_delta,
            "interest_delta":  interest_delta,          # +ve favours blue

            "red_posting_time":  base_posting_time,
            "blue_posting_time": base_posting_time + time_offset,
            "time_offset":       time_offset,           # +ve = red posts first

            "is_misinfo":      is_misinfo,
        },
    }

    filename = f"hyp12-config-{index}.json"
    path = os.path.join(OUTPUT_FOLDER, filename)
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    for i in range(NUM_FILES):
        generate_file(i)

    print(f"Generated {NUM_FILES} files in '{OUTPUT_FOLDER}' folder.")


if __name__ == "__main__":
    main()