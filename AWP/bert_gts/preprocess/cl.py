import copy
import json
import numpy as np

def load_data(filename):
    data = []
    with open(filename, encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def sim(s1, s2):
    s1 = [x for x in s1]
    s2 = [x for x in s2]
    s1 = set(s1)
    s2 = set(s2)
    return len(s1.intersection(s2)) / len(s1.union(s2))

train_data = load_data('../data/Math23K_train_norm.jsonl')
treedis_data = load_data('../data/tree_dis.json')
cached = dict()
expr_id_dict = dict()
for d in train_data:
    if ' '.join(d['postfix']) not in expr_id_dict:
        expr_id_dict[' '.join(d['postfix'])] = [d['id']]
    else:
        expr_id_dict[' '.join(d['postfix'])].append(d['id'])
    cached[d['id']] = d['text']

treedis_matrix = np.zeros((len(expr_id_dict), len(expr_id_dict)))
expr_expr_dict = {x:i for i,x in enumerate(expr_id_dict.keys())}
expr_expr_reverse_dict = {x:i for i,x in expr_expr_dict.items()}
for d in treedis_data:
    expr1, expr2 = d[0].split(' ; ')
    treedis_matrix[expr_expr_dict[expr1], expr_expr_dict[expr2]] = d[1]

treedis_matrix1 = copy.deepcopy(treedis_matrix)
treedis_matrix2 = copy.deepcopy(treedis_matrix)
for i in range(len(treedis_matrix)):
    treedis_matrix1[i][i] = float('inf')
filter_ids = [expr_expr_dict[x[0]] for x in expr_id_dict.items() if len(x[1]) < 10]
for idx in filter_ids:
    treedis_matrix2[:, idx] = 0.0

res = []
for d in train_data:
    postfix_positive = ' '.join(d['postfix'])
    exprpos = expr_expr_dict[postfix_positive]
    src = d['text']
    positive_ids = expr_id_dict[postfix_positive]
    if len(positive_ids) == 1:
        postfix_positive = expr_expr_reverse_dict[np.argmin(treedis_matrix1[exprpos])]
        positive_ids = expr_id_dict[postfix_positive]
    minscore = float('inf')
    positive = src
    for idx in positive_ids:
        if idx != d['id'] and sim(src, cached[idx]) < minscore:
            minscore = sim(src, cached[idx])
            positive = idx
    
    postfix_negative = expr_expr_reverse_dict[np.argmax(treedis_matrix2[exprpos])]
    negative_ids = expr_id_dict[postfix_negative]
    maxscore = -float('inf')
    negative = src
    for idx in negative_ids:
        if idx != d['id'] and sim(src, cached[idx]) > maxscore:
            maxscore = sim(src, cached[idx])
            negative = idx
    d['positive'] = positive
    d['negative'] = negative
    res.append(d)
    if len(res) % 100 == 0:
        print(len(res)/len(train_data))

f = open('../data/Math23K_train_cl.jsonl', 'w')
for d in res:
    json.dump(d, f, ensure_ascii=False)
    f.write("\n")
f.close()