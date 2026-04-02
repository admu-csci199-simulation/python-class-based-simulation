import json
import os
import random


OUTPUT_FOLDER = "input"
NUM_FILES = 100               # how many json files to generate
POSTS_PER_FILE = 1            # number of posts in each file
AGENTS_PER_FILE = 10000

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

def generate_agents():
    agent_count = AGENTS_PER_FILE

    gullible, normal, stubborn = split_into_three(agent_count, agent_count//4)
    lurker, normal_sharer, active = split_ratio_90_9_1(agent_count)
    red, centrist, blue = split_into_three(agent_count, agent_count//4)

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

def generate_posts():
    posts = []

    id = 0
    topic = 0
    postID = id
    postTopic = topic
    beliefValue = random.choice([-3, 0, 3]) # center of each belief camp
    interestValue = random.randint(-10, 21)
    postingTime = 0
    misinformation = random.choice([True, False])
    spawn = True

    post = {
            "postID": postID,
            "postTopic": postTopic,
            "beliefValue": beliefValue,
            "interestValue": interestValue,
            "postingTime": postingTime,
            "misinformation": misinformation,
            "spawn": spawn
        }
    
    posts.append(post)

    if misinformation:
        post2 = {
                "postID": postID + 1,
                "postTopic": postTopic,
                "beliefValue": beliefValue,
                "interestValue": min(21, interestValue + 3),
                "postingTime": postingTime + random.randint(5, 12)*60,
                "misinformation": misinformation,
                "spawn": False
            }
        post3 = {
                "postID": postID + 2,
                "postTopic": postTopic,
                "beliefValue": beliefValue,
                "interestValue": min(21, interestValue + 3),
                "postingTime": post2["postingTime"] + random.randint(5, 12)*60,
                "misinformation": misinformation,
                "spawn": False
            }
        posts.append(post2)
        posts.append(post3)
    
    return posts

def generate_file(index):
    data = {
        "Agents": generate_agents(),
        "Posts": generate_posts()
    }

    filename = f"hyp6-config-{index}.json"
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