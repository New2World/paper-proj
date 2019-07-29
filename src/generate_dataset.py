import json
import numpy as np

thread_id = []
context = []
author = []
popularity = []
users = set()
reply = []
ll = 69
dictionary = dict(zip("abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:\"'/\\|_@#$%ˆ&*~`+-=<>()[]{}", range(ll)))
with open('../data/raw/avengers/avengers.json', 'r') as jf:
    for json_line in jf:
        json_obj = json.loads(json_line)
        if json_obj['numComms'] == 0:
            continue
        thread_id.append(json_obj['sub_id'])
        popularity.append(json_obj['numComms'])
        context.append(json_obj['title']+' '+json_obj['text'])
        author.append(json_obj['author'])
        users.add(json_obj['author'])
for i, thread in enumerate(thread_id):
    comms = []
    with open(f'../data/raw/avengers/avengers/{thread}.json', 'r') as jt:
        for json_line in jt:
            json_obj = json.loads(json_line)
            comms.append((json_obj['author'],json_obj['created_at']))
            users.add(json_obj['author'])
    comms = sorted(comms, key=lambda x: x[1])
    hot_reply = [comms[i][0] for i in range(min(len(comms),5))]
    if author[i] not in hot_reply:
        hot_reply = hot_reply[:-1]
        hot_reply.append(author[i])
    reply.append(hot_reply)

# with open('../data/thread_id.txt', 'w') as fd:
#     fd.write(','.join(thread_id))
# with open('../data/author_id.txt', 'w') as fd:
#     fd.write(','.join(author))
# with open('../data/user_id.txt', 'w') as fd:
#     fd.write(','.join(users))

user_dict = dict(zip(users, range(len(users))))

def char2onehot(cseq):
    seq = []
    for c in cseq:
        if c not in dictionary.keys():
            continue
        ss = [0]*ll
        ss[dictionary[c]] = 1
        seq.append(ss)
    return np.array(seq)

context_matrix = np.array([char2onehot(seq) for seq in context])
np.save('../data/context_matrix.npy', context_matrix)
# post_matrix = np.zeros((len(thread_id),len(users)))
# post_index = [[user_dict[usr] for usr in usrs] for usrs in reply]
# for i, idx in enumerate(post_index):
#     post_matrix[i,idx] = 1
# np.save('../data/first-5-reply-matrix.npy', post_matrix)
# np.save('../data/popularity.npy', np.array(popularity))