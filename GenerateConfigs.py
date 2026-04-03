import random, json, os

def generate_random_posts(post_id, is_misinfo, interest_value):
    posts = []

    base_interest = interest_value
    interest_val = min(base_interest + 3 if is_misinfo else base_interest, 21)

    posting_time = (random.randint(0, 1440) if is_misinfo else random.randint(0, 2880))

    original_post = {
        "postID": post_id,
        "postTopic": random.randint(1, 5),
        "beliefValue": random.randint(-4, 4),
        "interestValue": interest_val,
        "postingTime": posting_time,
        "misinformation": is_misinfo,
        "spawn": is_misinfo
    }

    posts.append(original_post)

    # Spawn logic for OG misinformation
    if original_post["misinformation"] and original_post["spawn"]:

        previous_interest = interest_val
        previous_time = original_post["postingTime"]

        for _ in range(2):

            post_id += 1

            new_interest = min(21, previous_interest + 3)
            delay = random.randint(300, 720)
            new_time = previous_time + delay

            spawned_post = {
                "postID": post_id,
                "postTopic": original_post["postTopic"],
                "beliefValue": original_post["beliefValue"],
                "interestValue": new_interest,
                "postingTime": new_time,
                "misinformation": True,
                "spawn": False
            }

            posts.append(spawned_post)

            previous_interest = new_interest
            previous_time = new_time
    
    return posts

def split_into_three(total, min_value=1):
    a = random.randint(min_value, total - 2*min_value)
    b = random.randint(min_value, total - a - min_value)
    c = total - a - b
    return a, b, c

def split_ratio_90_9_1(total):
    lurker = round(total * 0.90)
    normal_sharer = round(total * 0.09)
    active = total - lurker - normal_sharer  # ensure exact sum
    return lurker, normal_sharer, active

def generate_agents(agent_count):
    gullible, normal, stubborn = split_into_three(agent_count, agent_count//3)
    lurker, normal_sharer, active = split_ratio_90_9_1(agent_count)
    red, centrist, blue = split_into_three(agent_count, agent_count//3)
    return {
        "Agent Count": agent_count,

        "Gullible Count": gullible,
        "Normal Count": normal,
        "Stubborn Count": stubborn,

        "Lurker Count": lurker,
        "Normal Sharer Count": normal_sharer,
        "Active Count": active,

        "Red Count": red,
        "Centrist Count": centrist,
        "Blue Count": blue
    }

test_cases = 100
num_agents = 300
for test_case in range(test_cases):
    conf_data = {}
    conf_data["Agents"] = generate_agents(num_agents)
    conf_data["Posts"] = []
    post_count = 300

    post_id = 0
    for post in range(post_count):
        generated = generate_random_posts(post_id, random.choice([True, False]), (post_id%32)-10)
        for x in generated:
            conf_data["Posts"].append(x)
            post_id += 1

    config_name = f"config_{test_case}.json"
    with open(os.path.join("input", config_name), 'w') as f:
        json.dump(conf_data, f, indent=4)