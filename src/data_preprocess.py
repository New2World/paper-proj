import os, sys, json
import numpy as np

dataset = "avengers"
if len(sys.argv) > 1:
    dataset = sys.argv[1]

main_post = dataset+'/'+dataset+".json"
comms_path = dataset+'/'+dataset+'/'

def rename_comms(comms_path):
    files = list(os.walk(comms_path))[0][2]
    for each_file in files:
        name = each_file
        with open(comms_path+each_file, "r") as commfd:
            json_str = commfd.readline()
            if len(json_str) <= 0:
                continue
            name = json.loads(json_str)["link_id"]
        os.rename(comms_path+each_file, comms_path+name+".json")

def get_info_from_main_post(main_post):
    posts = dict()
    with open(main_post, "r") as mpfd:
        for each_item in mpfd:
            item = json.loads(each_item)
            posts[item["sub_id"]] = (item["numComms"], set([item["author"]]))
    return posts

def get_info_from_comms(comms_path, posts):
    for post_id in posts.keys():
        comms = comms_path+post_id+".json"
        if posts[post_id][0] == 0:
            continue
        with open(comms, "r") as commfd:
            for each_item in commfd:
                item = json.loads(each_item)
                posts[post_id][1].update([item["author"]])
    return posts

def list_all_users(posts):
    all_users = set()
    for post_info in posts.values():
        all_users.update(post_info[1])
    return list(all_users)

def generate_input_matrix(posts, all_users):
    mapping = dict((all_users[i], i) for i in range(len(all_users)))
    encoding = np.zeros((len(posts),len(all_users)))
    for i, post_info in enumerate(posts.values()):
        user_id = [mapping[user] for user in post_info[1]]
        encoding[[i]*len(user_id),user_id] = 1
    return encoding

# rename_comms(comms_path)

posts = get_info_from_main_post(main_post)
# with open("post_id.txt","w") as pfd:
#     pfd.write(' '.join(posts.keys()))
# exit(1)
print(f"all posts: {len(posts)}")
posts = get_info_from_comms(comms_path, posts)
all_users = list_all_users(posts)
print(f"all users: {len(all_users)}")
encoding = generate_input_matrix(posts, all_users)
np.save("one-hot-encoding.npy", encoding)

    