import json
import os
import random


OUTPUT_FOLDER = "input"
NUM_FILES = 100               # how many json files to generate
POSTS_PER_FILE = 10           # number of posts in each file
AGENTS_PER_FILE = 1000

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
    while id < POSTS_PER_FILE:
        postID = id
        postTopic = topic
        beliefValue = random.randint(-4, 4)
        interestValue = random.randint(-10, 21)
        postingTime = random.randint(0, 60*24)
        misinformation = (random.choice([True, False]) if id < POSTS_PER_FILE-3 else False)
        spawn = (True if misinformation else False)
        
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

        if not misinformation:
            id += 1
            topic += 1

        elif misinformation:
            assert(id < POSTS_PER_FILE-3)
            post2_postID = id+1
            post2_postTopic = topic
            post2_beliefValue = beliefValue
            post2_interestValue = min(21, interestValue+3)
            post2_postingTime = postingTime + random.randint(5, 12)*60
            post2_misinformation = True
            post2_spawn = False

            post3_postID = id+2
            post3_postTopic = topic
            post3_beliefValue = beliefValue
            post3_interestValue = min(21, post2_interestValue+3)
            post3_postingTime = post2_postingTime + random.randint(5, 12)*60
            post3_misinformation = True
            post3_spawn = False

            post2 = {
                    "postID": post2_postID,
                    "postTopic": post2_postTopic,
                    "beliefValue": post2_beliefValue,
                    "interestValue": post2_interestValue,
                    "postingTime": post2_postingTime,
                    "misinformation": post2_misinformation,
                    "spawn": post2_spawn
                }
            post3 = {
                    "postID": post3_postID,
                    "postTopic": post3_postTopic,
                    "beliefValue": post3_beliefValue,
                    "interestValue": post3_interestValue,
                    "postingTime": post3_postingTime,
                    "misinformation": post3_misinformation,
                    "spawn": post3_spawn
                }
            
            posts.append(post2)
            posts.append(post3)
            id += 3
            topic += 1
    
    return posts

def generate_file(index):
    data = {
        "Agents": generate_agents(),
        "Posts": generate_posts()
    }

    filename = f"hyp7-config-{index}.json"
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