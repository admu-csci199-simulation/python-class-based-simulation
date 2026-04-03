import os
import random
import json

MAX_INTEREST_VAL = 21

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

    i = 0
    while i < n:
        is_misinfo = True if i % 2 == 0 else False
        base_interest = random.randint(-10, 21)
        interest_val = min(base_interest + 3, MAX_INTEREST_VAL) if is_misinfo else min(base_interest, MAX_INTEREST_VAL)
        posting_time = (random.randint(0, 1440) if is_misinfo else random.randint(0, 2880))
        post_topic = random.randint(1, 5)
        belief_value = random.randint(-4, 4)

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
    return {
        "Agent Count": n,
        "Gullible Count": 20,
        "Normal Count": 20,
        "Stubborn Count": 20,
        "Lurker Count": 54,
        "Normal Sharer Count": 5,
        "Active Count": 1,
        "Red Count": int(n * red_ratio),
        "Centrist Count": int(n * centrist_ratio),
        "Blue Count": int(n * blue_ratio)
    }

def save_json(filename, distribution):
    data = {
        "Agents": generate_agents(600, distribution),
        "Posts": generate_posts(2)
    }

    with open(filename, "w") as f:
        json.dump(data, f, indent=4)

def main():
    for i in range(3):
        save_json(os.path.join("input", f"hyp5-config-{i+1}.json"), (1/3, 1/3, 1/3))

if __name__ == "__main__":
    main()