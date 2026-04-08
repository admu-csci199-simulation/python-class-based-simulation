import json
import os
import random


OUTPUT_FOLDER = "input"
NUM_FILES_PER_GROUP = 20      # files generated per config group (5 groups = 100 total)
AGENTS_PER_FILE = 1000

# ── Config Groups ─────────────────────────────────────────────────────────────
# Each group isolates one variable to test competition under different conditions.
#
# Group A: Maximally opposed beliefs, balanced population, simultaneous posting
#           → Baseline competition case
# Group B: Moderately opposed beliefs, balanced population, simultaneous posting
#           → Does competition weaken when posts are less ideologically distant?
# Group C: Maximally opposed beliefs, skewed population, simultaneous posting
#           → Does majority camp always win?
# Group D: Maximally opposed beliefs, balanced population, staggered posting
#           → Does a head start break competitive symmetry?
# Group E: Maximally opposed beliefs (one misinfo), balanced population, simultaneous
#           → Does misinfo compete differently against real news?
# ─────────────────────────────────────────────────────────────────────────────

CONFIG_GROUPS = {
    "A": {
        "belief_pairs"   : [(-4, 4), (-3, 3), (-4, 3), (-3, 4)],  # sampled randomly per file
        "pop_skew"       : "balanced",
        "time_offset_range" : (0, 0),       # simultaneous
        "misinfo_pattern": "none",          # both real
    },
    "B": {
        "belief_pairs"   : [(-2, 2), (-1, 1), (-2, 1), (-1, 2)],  # moderate opposition
        "pop_skew"       : "balanced",
        "time_offset_range" : (0, 0),
        "misinfo_pattern": "none",
    },
    "C": {
        "belief_pairs"   : [(-4, 4), (-3, 3), (-4, 3), (-3, 4)],
        "pop_skew"       : "red_heavy",     # majority leans left → favors post 0
        "time_offset_range" : (0, 0),
        "misinfo_pattern": "none",
    },
    "D": {
        "belief_pairs"   : [(-4, 4), (-3, 3), (-4, 3), (-3, 4)],
        "pop_skew"       : "balanced",
        "time_offset_range" : (30, 90),     # post 1 is delayed 30–90 minutes
        "misinfo_pattern": "none",
    },
    "E": {
        "belief_pairs"   : [(-4, 4), (-3, 3), (-4, 3), (-3, 4)],
        "pop_skew"       : "balanced",
        "time_offset_range" : (0, 0),
        "misinfo_pattern": "first",         # post 0 is misinfo, post 1 is real
    },
}


# ── Agent generation ──────────────────────────────────────────────────────────

def split_into_three(total, min_value=1):
    a = random.randint(min_value, total - 2 * min_value)
    b = random.randint(min_value, total - a - min_value)
    c = total - a - b
    return a, b, c


def split_ratio_90_9_1(total):
    lurker        = round(total * 0.90)
    normal_sharer = round(total * 0.09)
    active        = total - lurker - normal_sharer
    return lurker, normal_sharer, active


def generate_agents(pop_skew: str) -> dict:
    agent_count = AGENTS_PER_FILE

    gullible, normal, stubborn = split_into_three(agent_count, agent_count // 4)
    lurker, normal_sharer, active = split_ratio_90_9_1(agent_count)

    if pop_skew == "balanced":
        red, centrist, blue = split_into_three(agent_count, agent_count // 4)

    elif pop_skew == "red_heavy":
        # Red gets 50–70 %, remainder split between centrist and blue
        red      = random.randint(int(agent_count * 0.50), int(agent_count * 0.70))
        leftover = agent_count - red
        centrist = random.randint(leftover // 4, leftover * 3 // 4)
        blue     = leftover - centrist

    elif pop_skew == "blue_heavy":
        blue     = random.randint(int(agent_count * 0.50), int(agent_count * 0.70))
        leftover = agent_count - blue
        centrist = random.randint(leftover // 4, leftover * 3 // 4)
        red      = leftover - centrist

    else:
        raise ValueError(f"Unknown pop_skew: {pop_skew}")

    return {
        "Agent Count"        : agent_count,
        "Gullible Count"     : gullible,
        "Normal Count"       : normal,
        "Stubborn Count"     : stubborn,
        "Lurker Count"       : lurker,
        "Normal Sharer Count": normal_sharer,
        "Active Count"       : active,
        "Red Count"          : red,
        "Centrist Count"     : centrist,
        "Blue Count"         : blue,
    }


# ── Post generation ───────────────────────────────────────────────────────────

def generate_posts(belief_pair: tuple, time_offset_range: tuple, misinfo_pattern: str) -> list:
    """
    Always generates exactly 2 competing posts.

    belief_pair        — (beliefValue_post0, beliefValue_post1)
    time_offset_range  — (min_offset, max_offset) in minutes; post 1 is delayed by this amount
    misinfo_pattern    — "none"  : both posts are real news
                         "first" : post 0 is misinformation, post 1 is real
                         "second": post 0 is real, post 1 is misinformation
                         "both"  : both posts are misinformation
    """
    belief0, belief1 = belief_pair

    # Base posting time for post 0 — keep it early enough that both posts fit in 48h window
    max_base_time   = 60 * 24 - time_offset_range[1] - 60   # leave 1h buffer
    base_post_time  = random.randint(0, max(0, max_base_time))
    time_offset     = random.randint(*time_offset_range)

    interest0 = random.randint(-10, 21)
    interest1 = random.randint(-10, 21)

    is_misinfo0 = misinfo_pattern in ("first",  "both")
    is_misinfo1 = misinfo_pattern in ("second", "both")

    post0 = {
        "postID"      : 0,
        "postTopic"   : 0,
        "beliefValue" : belief0,
        "interestValue": interest0,
        "postingTime" : base_post_time,
        "misinformation": is_misinfo0,
        "spawn"       : is_misinfo0,      # spawn = True marks the original misinfo seed post
    }

    post1 = {
        "postID"      : 1,
        "postTopic"   : 1,                # different topic so posts don't collapse into one chain
        "beliefValue" : belief1,
        "interestValue": interest1,
        "postingTime" : base_post_time + time_offset,
        "misinformation": is_misinfo1,
        "spawn"       : is_misinfo1,
    }

    return [post0, post1]


# ── File generation ───────────────────────────────────────────────────────────

def generate_file(group_name: str, group_cfg: dict, index: int) -> None:
    belief_pair = random.choice(group_cfg["belief_pairs"])

    data = {
        "Agents": generate_agents(group_cfg["pop_skew"]),
        "Posts" : generate_posts(
            belief_pair        = belief_pair,
            time_offset_range  = group_cfg["time_offset_range"],
            misinfo_pattern    = group_cfg["misinfo_pattern"],
        ),
        "_meta": {                         # bookkeeping — not read by Main.py
            "group"       : group_name,
            "belief_pair" : list(belief_pair),
            "pop_skew"    : group_cfg["pop_skew"],
            "misinfo_pattern": group_cfg["misinfo_pattern"],
        }
    }

    filename = f"hyp12-config-{group_name}-{index}.json"
    path     = os.path.join(OUTPUT_FOLDER, filename)
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    total = 0
    for group_name, group_cfg in CONFIG_GROUPS.items():
        for i in range(NUM_FILES_PER_GROUP):
            generate_file(group_name, group_cfg, i)
            total += 1

    print(f"Generated {total} files across {len(CONFIG_GROUPS)} groups in '{OUTPUT_FOLDER}/'.")
    print("Groups: " + ", ".join(
        f"{g} ({NUM_FILES_PER_GROUP} files)" for g in CONFIG_GROUPS
    ))


if __name__ == "__main__":
    main()