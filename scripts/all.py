#!/usr/bin/env python
import warnings
warnings.filterwarnings('ignore')
from preprocess import Preprocessor
from main import Pipeline
import argparse
import random
import numpy as np
import torch

seed = 42
random.seed(seed)
torch.random.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)


parser = argparse.ArgumentParser()
parser.add_argument('--test_mode', default='notest')
parser.add_argument('--cuda_index', default=1)
parser.add_argument('--doc', default='doc')
parser.add_argument('--use_pos', default='no_pos')
args = parser.parse_args()

num_iter = 1

def run(args):
    res = []
    for i in range(num_iter):
        pipeline = Pipeline(args)
        pipeline.main()
        res.append(pipeline.best_score_nine)
    return res
res = []
res.append([run(args), 'no_ner'])
setattr(args, 'use_pos', 'pos')
#res.append([run(args), 'ner'])
print(" p     r     f     "*3)
for r in res:
    r, z = r
    print(z, 'result')
    for w in r:
        w = [p*100  for p in w]
        print(("{:.2f},"*9).format(*w))

print("true doc, sememe, and ner, with max sememe")

