#!/usr/bin/env python
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


import os
import pickle as pkl
from torch.utils.data import Dataset
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import numpy as np

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)



class MyDataset(Dataset):
    def __init__(self, data):
        self.input_ids, self.input_lengths, self.input_labels, self.mention_set, _ = data

    def __getitem__(self, index):
        assert len(self.mention_set[index]) > 0
        res = {
            "input_ids": torch.tensor(self.input_ids[index]),
            "input_lengths": torch.tensor(self.input_lengths[index]),
            "input_labels": torch.tensor(self.input_labels[index]),
            "mention_sets": self.mention_set[index],
        }
        return res

    def __len__(self):
        return len(self.input_ids)


class MyDataLoader:
    def __init__(self, args, mode='train'):
        self.args = args

        path = os.path.join(args.data_dir, '{}.pkl'.format(mode))
        self.data = pkl.load(open(path, 'rb'))

    def collate_fn(self, batch_data):
        #input_ids = batch_data['input_ids']
        sentence_counts = torch.tensor([document['input_ids'].shape[0] for document in batch_data])
        input_ids = torch.stack([w for line in batch_data for w in line['input_ids']])
        input_lengths = torch.stack([w for line in batch_data for w in line['input_lengths']])
        input_labels = torch.stack([w for line in batch_data for w in line['input_labels']])
        mention_set = [w['mention_sets'] for w in batch_data]

        argsort = torch.argsort(input_lengths, descending=True)
        reverse_sort = argsort.argsort()
        input_ids = input_ids[argsort]
        input_lengths = input_lengths[argsort]
        res = {
            'input_ids': input_ids,
            'input_lengths': input_lengths,
            'input_labels': input_labels,
            'reverse_sort': reverse_sort,
            'mention_sets': mention_set,
            'sentence_counts': sentence_counts
        }
        return res


    def getdata(self):
        return DataLoader(MyDataset(self.data), shuffle=False, batch_size=self.args.batch_size, collate_fn=self.collate_fn)

def get_indices(bios):
    res = []
    start, end = -1, -1
    for i, word in enumerate(bios):
        if word == 'B':
            if start != -1:
                res.append((start, end))
            start, end = i, i
        elif word == 'O':
            if start != -1:
                res.append((start, end))
            start, end = -1, -1
        else:
            end = i
    if start != -1:
        res.append((start, end))
    return res

def get_f1_by_bio(predict_out, gold_bio, lengths):
    word_dict = {'B': 0, 'I': 1, 'O':2}
    word_dict = {v:k for k,v in word_dict.items()}
    predict_out = predict_out.argmax(-1)

    predict_res, gold_res = [], []
    for i, length in enumerate(lengths):
        predict_res += predict_out[i, :length].tolist()
        gold_res += gold_bio[i, :length].tolist()

    predict_bio = [word_dict[w] for w in predict_res]
    gold_bio = [word_dict[w] for w in gold_res]

    predict_bio = get_indices(predict_bio)
    gold_bio = get_indices(gold_bio)

    tp = 0
    for w in predict_bio:
        for z in gold_bio:
            if w[0] == z[0] and w[1] == z[1]:
                tp += 1

    p = tp / len(predict_bio) if len(predict_bio) > 0 else 0
    r = tp / len(gold_bio) if len(gold_bio) > 0 else 0
    f1 = 2 * p * r / (p + r) if p + r > 0 else 0
    return f1


def get_mention_f1(predict, gold):
    if len(predict) == 0:
        return 0, 0, 0
    predict = np.array(predict)
    gold = np.array(gold)
    predict = predict.argmax(-1)
    predict = predict.reshape([-1, ])
    gold = gold.reshape([-1, ])

    res_predict = []
    res_gold = []
    start, i = 0, 1

    f1 = f1_score(gold, predict)
    precision = precision_score(gold, predict)
    r = recall_score(gold, predict)
    return precision, r, f1

def get_f1_by_bio_nomask(predict_out, gold_bio, idx=None):
    #predict_out = np.array(predict_outs)
    #gold_bio = np.array(gold_bio)
    word_dict = {'B': 0, 'I': 1, 'O':2}
    word_dict = {v:k for k,v in word_dict.items()}
    predict_out = predict_out.argmax(-1)

    predict_bio = [word_dict[w.item()] for w in predict_out]
    gold_bio = [word_dict[w.item()] for w in gold_bio]

    predict_bio = get_indices(predict_bio)
    gold_bio = get_indices(gold_bio)
    fn = 0
    fp = 0
    tp = 0
    fn += len(gold_bio)
    fp += len(predict_bio)
    for w in predict_bio:
        for z in gold_bio:
            if w[0] == z[0] and w[1] == z[1]:
                tp += 1

    p = tp / fp if fp > 0 else 0
    r = tp / fn if fn > 0 else 0
    f1 = 2 * p * r / (p + r) if p + r > 0 else 0
    return (p, r, f1), (tp, fp, fn)
    p = tp / len(predict_bio) if len(predict_bio) > 0 else 0
    r = tp / len(gold_bio) if len(gold_bio) > 0 else 0
    f1 = 2 * p * r / (p + r) if p + r > 0 else 0
    return p,r , f1




import os
import numpy as np

def getPredictCluster(predict_indices, mention_interaction):
    """
    mention_interaction = [
        [[1.3,0.5], [0.1, 0.3], [1, 0.2],[0,0],[1,2]],
        [[0.3,0.5], [1.1, 0.3], [1, 0.2],[0,0],[1,2]],
        [[1.3,0.5], [0.1, 0.3], [1, 0.2],[0,0],[1,2]],
        [[0.3,0.5], [1, 0.3], [1.1, 0.2],[1,0],[1,2]],
        [[0.3,0.5], [1.1, 0.3], [0, 0.2],[0,1],[3,2]],
    ]
    predict_indices = [(1,2), (3, 4), (7, 8)]
    
    
    :param predict_indices:
    :param mention_interaction:
    :return:
    
    clusters:
    mention_to_predict:
    
    
    """
    mention_interaction = mention_interaction.cpu().tolist()
    exp = np.exp(mention_interaction)
    mention_interaction = exp / np.sum(exp, -1, keepdims=True)
    
    cluster = dict()
    
    cluster_id, idx = [0 for _ in mention_interaction], 1
    
    for i in range(len(mention_interaction)):
        indices = predict_indices[i]
        if i == 0:
            continue
        label = np.argmax(mention_interaction[i, :i], -1)
        if sum(label) == 0:
            cluster_id[i] = idx
            idx += 1
            continue
        ancestor = mention_interaction[i, :i, 1].argmax()
        cluster_id[i] = cluster_id[ancestor]
    
    cluster_id_dict = {}
    for i, index in enumerate(cluster_id):
        if index in cluster_id_dict:
            cluster_id_dict[index].append(i)
        else:
            cluster_id_dict[index] = [i]
    
    clusters = []
    mention_to_predict = {}
    for k, v in cluster_id_dict.items():
        cluster = [predict_indices[w] for w in v]
        cluster = tuple(tuple(w) for w in cluster)
        clusters.append(cluster)
        for w in cluster:
            mention_to_predict[w] = cluster
    
    return clusters, mention_to_predict


def evaluate_coref(predict_indices, mention_interaction, gold_mention_set, evaluator):
    
    gold_clusters = [tuple(tuple(m) for m in gc) for gc in gold_mention_set]
    mention_to_gold = {}
    for gc in gold_clusters:
        for mention in gc:
            mention_to_gold[mention] = gc
    predicted_clusters, mention_to_predicted = getPredictCluster(predict_indices, mention_interaction)
    
    evaluator.update(predicted_clusters, gold_clusters, mention_to_predicted, mention_to_gold)
    return predicted_clusters, gold_clusters


if __name__ == '__main__':
    mention_interaction = [
        [[0.3,0.5], [0.1, 0.3], [1, 0.2],[0,0],[1,2]],
        [[1.5,0.5], [0.1, 0.3], [1, 0.2],[0,0],[1,2]],
        [[0.3,0.5], [1.1, 0.3], [0, 0.2],[0,0],[1,2]],
        [[1.3,0.5], [0.1, 1.3], [0.1, 2.2],[-1,0],[1,2]],
        [[1.3,0.5], [1.1, 1.3], [1, 0.2],[2,1],[0,2]],
    ]
    predict_indices = [(1,2), (3, 4), (7, 8), (10,11), (12,13)]
