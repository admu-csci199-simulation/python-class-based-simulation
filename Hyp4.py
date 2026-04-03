import os
import random
import json

MAX_INTEREST_VAL = 21

def split_into_three(total, min_value=1):
    a = random.randint(min_value, total - 2*min_value)
    b = random.randint(min_value, total - a - min_value)
    c = total - a - b
    return a, b, c

def generate_belief_values(n, distribution):
    red_ratio, centrist_ratio, blue_ratio = distribution

    red_count = int(n * red_ratio)
    centrist_count = int(n * centrist_ratio)
    blue_count = n - red_count - centrist_count

    beliefs = []

    beliefs += [random.randint(-4, -2) for _ in range(red_count)]
    beliefs += [random.randint(-1, 1) for _ in range(centrist_count)]
    beliefs += [random.randint(2, 4) for _ in range(blue_count)]

    random.shuffle(beliefs)
    return beliefs

def generate_posts(n):
    posts = []
    beliefs = generate_belief_values(n, (1/3, 1/3, 1/3))

    i = 0
    while i < n:
        is_misinfo = random.choice([True, False])
        base_interest = random.randint(-10, 21)
        interest_val = min(base_interest + 3, MAX_INTEREST_VAL) if is_misinfo else min(base_interest, MAX_INTEREST_VAL)
        posting_time = (random.randint(0, 1440) if is_misinfo else random.randint(0, 2880))
        post_topic = random.randint(1, 5)
        belief_value = beliefs[i]

        post = {
            "postID": i,
            "postTopic": post_topic,
            "beliefValue": belief_value,
            "interestValue": interest_val,
            "postingTime": posting_time,
            "misinformation": is_misinfo,
            "spawn": True if is_misinfo else False
        }
        posts.append(post)
        i += 1

        if is_misinfo:
            previous_interest = interest_val
            previous_time = posting_time

            for j in range(2):
                new_interest = min(previous_interest + 3, MAX_INTEREST_VAL)
                delay = random.randint(300, 720)
                new_time = previous_time + delay

                spawned_post = {
                    "postID": i,
                    "postTopic": post_topic,
                    "beliefValue": belief_value,
                    "interestValue": new_interest,
                    "postingTime": new_time,
                    "misinformation": True,
                    "spawn": False
                }

                posts.append(spawned_post)
                i += 1
                
    return posts

def generate_agents(n, distribution):
    red_ratio, centrist_ratio, blue_ratio = distribution
    gullible, normal, stubborn = split_into_three(n, n//10)
    lurker = round(n * 0.90)
    normal_sharer = round(n* 0.09)
    active = n - lurker - normal_sharer

    return {
        "Agent Count": n,
        "Gullible Count": gullible,
        "Normal Count": normal,
        "Stubborn Count": stubborn,
        "Lurker Count": lurker,
        "Normal Sharer Count": normal_sharer,
        "Active Count": active,
        "Red Count": int(n * red_ratio),
        "Centrist Count": int(n * centrist_ratio),
        "Blue Count": int(n * blue_ratio)
    }

def save_json(filename, distribution):
    data = {
        "Agents": generate_agents(300, distribution),
        "Posts": generate_posts(300)
    }

    with open(filename, "w") as f:
        json.dump(data, f, indent=4)

def main():
    distributions = {
        "equal": (1/3, 1/3, 1/3),
        "blue": (0.1, 0.1, 0.8),
        "centrist": (0.1, 0.8, 0.1),
        "red": (0.8, 0.1, 0.1)
    }

    for name, dist in distributions.items():
        save_json(os.path.join("input", f"hyp4-config-{name}.json"), dist)

if __name__ == "__main__":
    main()