import json
import jieba.analyse
import numpy as np
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge import Rouge

def sim_enghlish(s1, s2):
    s1 = set(s1)
    s2 = set(s2)
    if len(s1) == 0:
        return 1
    return len(s1.intersection(s2)) / len(s1)

def sim_chinese(s1, s2):
    s1 = ''.join(s1)
    s1 = [x for x in s1]
    s2 = ''.join(s2)
    s2 = [x for x in s2]
    s1 = set(s1)
    s2 = set(s2)
    if len(s1) == 0:
        return 1
    return len(s1.intersection(s2)) / len(s1)

def load_data(filename):
    data = []
    with open(filename, encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

train_data = load_data('data/math23k/Math23K_keys_train.jsonl')
cached = set()
for d in train_data:
    cached.add(d['text'].lower())

data = {}
keys_cover = []
notrain = []
filename = 'result_math23k/results_test.jsonl'
for line in open(filename, 'r'):
    temp = json.loads(line)
    res = ''.join(temp['res'].split(' ')).lower()
    if res not in cached:
        notrain.append(1)
    else:
        notrain.append(0)
    hyp_keys = [x.lower() for x in temp['keys'].split(' ')]
    data[temp['id']] = temp
    keys_cover.append(sim_chinese(hyp_keys, res))
print('bleu:', np.mean([x[1]['bleu'] for x in data.items()]))
print('rouge-1:', np.mean([x[1]['rouge-1'] for x in data.items()]))
print('rouge-2:', np.mean([x[1]['rouge-2'] for x in data.items()]))
print('rouge-l:', np.mean([x[1]['rouge-l'] for x in data.items()]))
print('key_cover:', np.mean(keys_cover))
print('notrain:', np.mean(notrain))
