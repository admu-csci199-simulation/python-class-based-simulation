import os
import random
import json

MAX_INTEREST_VAL = 21

def generate_belief_values(n, distribution):
    neg_ratio, neu_ratio, pos_ratio = distribution

    red_count = int(n * neg_ratio)
    centrist_count = int(n * neu_ratio)
    blue_count = n - red_count - centrist_count

    beliefs = []

    beliefs += [random.randint(-4, -2) for _ in range(red_count)]
    beliefs += [random.randint(-1, 1) for _ in range(centrist_count)]
    beliefs += [random.randint(2, 4) for _ in range(blue_count)]

    random.shuffle(beliefs)
    return beliefs

def generate_posts(n, distribution):
    posts = []
    beliefs = generate_belief_values(n, distribution)

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

def generate_agents():
    return {
        "Agent Count": 60,
        "Gullible Count": 20,
        "Normal Count": 20,
        "Stubborn Count": 20,
        "Lurker Count": 54,
        "Normal Sharer Count": 5,
        "Active Count": 1,
        "Red Count": 20,
        "Centrist Count": 20,
        "Blue Count": 20
    }

def save_json(filename, distribution):
    data = {
        "Agents": generate_agents(),
        "Posts": generate_posts(3000, distribution)
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
        save_json(os.path.join("input", f"hyp3-config-{name}.json"), dist)

if __name__ == "__main__":
    main()