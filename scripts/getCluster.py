import numpy as np

a = np.zeros([6,6])

l = ['A', 'B', 'C', 'D','E', 'F']
indices_dict = {w:i for i, w in enumerate(l)}
cluster = [['A', 'B', 'C'], ['B', 'C', 'D', 'E']]
for clu in cluster:
    for w in clu:
        for z in clu:
            a[indices_dict[w], indices_dict[z]] = 1



def get_cluster(predict_indices, mention_label):
    for i in range(len(mention_label)):
        mention_label[i, i] = 1
        for j in range(len(mention_label[i])):
            if i <=  j:
                continue
            if mention_label[i, j] == 1:
                mention_label[j, i] = 1
    
    indices_dict = {w:i for i, w in enumerate(predict_indices)}
    init = run(predict_indices, indices_dict, mention_label)

    init = [sorted(w) for w in init]
    init = sorted(init, key=lambda x:len(x))[::-1]

    index = []
    while True:
        length = len(init)
        index = []
        for i,w in enumerate(init):
            if i in set(index):
                continue
            for k,p in enumerate(init):
                if k <= i:
                    continue
                tmp = set(w)
                all_in = True
                for ww in p:
                    if ww not in tmp:
                        all_in = False
                        break
                if all_in:
                    index.append(k)
        if len(index) == 0:
            break
        init = [w for i, w in enumerate(init) if i not in index]
    return init

    
def get(res, a, b):
        res = [[w for w in res if w != b], [w for w in res if w != a]]
        res = [w for w in res if len(w) > 0]
        return res

def split(res, mention_label, indices_dict):
    for w in res:
        for k in res:
            if w > k:
                continue
            if mention_label[indices_dict[k], indices_dict[w]] == 0:
                return get(res, k, w)
    return [res]


def run(predict_indices, indices_dict, mention_label):
    init = [predict_indices]
    while True:
        length = len(init)
        res = []
        for w in init:
            tmp = split(w, mention_label, indices_dict)
            res += tmp
        if len(res) == length:
            break
        init = res
    return init

