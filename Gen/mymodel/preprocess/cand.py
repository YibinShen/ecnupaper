import copy
import json
import numpy as np

def load_data(filename):
    data = []
    with open(filename, encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def sim_enghlish(s1, s2):
    s1 = set(s1)
    s2 = set(s2)
    if len(s1) == 0 and len(s2) == 0:
        return float('inf')
    elif len(s1) == 0 or len(s2) == 0:
        return 0
    return len(s1.intersection(s2)) / len(s1.union(s2))

def sim_chinese(s1, s2):
    s1 = ''.join(s1)
    s1 = [x for x in s1]
    s2 = ''.join(s2)
    s2 = [x for x in s2]
    s1 = set(s1)
    s2 = set(s2)
    if len(s1) == 0 and len(s2) == 0:
        return float('inf')
    elif len(s1) == 0 or len(s2) == 0:
        return 0
    return len(s1.intersection(s2)) / len(s1.union(s2))

for fold in range(5):
    train_data = load_data('../data/mawps_norm/MAWPS_'+str(fold)+'_keys_train.jsonl')
    test_data = load_data('../data/mawps_norm/MAWPS_'+str(fold)+'_keys_test.jsonl')
    treedis_data = load_data('../data/mawps_norm/tree_dis.json')
    cached = dict()
    expr_id_dict = dict()
    exprs = set()
    for d in train_data:
        if ' '.join(d['postfix_norm']) not in expr_id_dict:
            expr_id_dict[' '.join(d['postfix_norm'])] = [d['id']]
        else:
            expr_id_dict[' '.join(d['postfix_norm'])].append(d['id'])
        cached[d['id']] = d['keys']
        exprs.add(' '.join(d['postfix_norm']))
    for d in test_data:
        exprs.add(' '.join(d['postfix_norm']))

    treedis_matrix = np.zeros((len(exprs), len(exprs))) - float('inf')
    expr_expr_dict = {x:i for i,x in enumerate(expr_id_dict.keys())}
    expr_expr_reverse_dict = {x:i for i,x in expr_expr_dict.items()}
    for d in treedis_data:
        expr1, expr2 = d[0].split(' ; ')
        len1, len2 = len(expr1.split(' ')), len(expr2.split(' '))
        if expr1 in expr_expr_dict and expr2 in expr_expr_dict:
            # treedis_matrix[expr_expr_dict[expr1], expr_expr_dict[expr2]] = d[1]
            treedis_matrix[expr_expr_dict[expr1], expr_expr_dict[expr2]] = 1 - d[1] / (len1 + len2)

    for i in range(len(treedis_matrix)):
        treedis_matrix[i][i] = -float('inf')
    
    train_res = []
    for d in train_data:
        postfix_cand = ' '.join(d['postfix_norm'])
        exprpos = expr_expr_dict[postfix_cand]
        src = d['keys']
        cand_ids = expr_id_dict[postfix_cand]
        if len(cand_ids) == 1:
            postfix_cand = expr_expr_reverse_dict[np.argmax(treedis_matrix[exprpos])]
            cand_ids = expr_id_dict[postfix_cand]
        maxscore = -float('inf')
        cand = None
        for idx in cand_ids:
            if idx != d['id'] and sim_enghlish(src, cached[idx]) > maxscore:
                maxscore = sim_enghlish(src, cached[idx])
                cand = idx
        d['cand'] = cand
        train_res.append(d)
    
    test_res = []
    for d in test_data:
        postfix_cand = ' '.join(d['postfix_norm'])
        src = d['keys']
        if postfix_cand in expr_expr_dict:
            exprpos = expr_expr_dict[postfix_cand]
            cand_ids = expr_id_dict[postfix_cand]
        else:
            postfix_cand = expr_expr_reverse_dict[np.argmax(treedis_matrix[exprpos])]
            cand_ids = expr_id_dict[postfix_cand]
        maxscore = -float('inf')
        cand = None
        for idx in cand_ids:
            if idx != d['id'] and sim_enghlish(src, cached[idx]) > maxscore:
                maxscore = sim_enghlish(src, cached[idx])
                cand = idx
        d['cand'] = cand
        test_res.append(d)

    f = open('../data/mawps_norm/MAWPS_'+str(fold)+'_keys_cand_train.jsonl', 'w')
    for d in train_res:
        json.dump(d, f, ensure_ascii=False)
        f.write("\n")
    f.close()

    f = open('../data/mawps_norm/MAWPS_'+str(fold)+'_keys_cand_test.jsonl', 'w')
    for d in test_res:
        json.dump(d, f, ensure_ascii=False)
        f.write("\n")
    f.close()

train_data = load_data('../data/math23k_norm/Math23K_keys_train.jsonl')
dev_data = load_data('../data/math23k_norm/Math23K_keys_dev.jsonl')
test_data = load_data('../data/math23k_norm/Math23K_keys_test.jsonl')
treedis_data = load_data('../data/math23k_norm/tree_dis.json')
cached = dict()
expr_id_dict = dict()
exprs = set()
for d in train_data:
    if ' '.join(d['postfix_norm']) not in expr_id_dict:
        expr_id_dict[' '.join(d['postfix_norm'])] = [d['id']]
    else:
        expr_id_dict[' '.join(d['postfix_norm'])].append(d['id'])
    cached[d['id']] = d['keys']
    exprs.add(' '.join(d['postfix_norm']))
for d in dev_data:
    exprs.add(' '.join(d['postfix_norm']))
for d in test_data:
    exprs.add(' '.join(d['postfix_norm']))

treedis_matrix = np.zeros((len(exprs), len(exprs))) - float('inf')
expr_expr_dict = {x:i for i,x in enumerate(expr_id_dict.keys())}
expr_expr_reverse_dict = {x:i for i,x in expr_expr_dict.items()}
for d in treedis_data:
    expr1, expr2 = d[0].split(' ; ')
    len1, len2 = len(expr1.split(' ')), len(expr2.split(' '))
    if expr1 in expr_expr_dict and expr2 in expr_expr_dict:
        # treedis_matrix[expr_expr_dict[expr1], expr_expr_dict[expr2]] = d[1]
        treedis_matrix[expr_expr_dict[expr1], expr_expr_dict[expr2]] = 1 - d[1] / (len1 + len2)

for i in range(len(treedis_matrix)):
    treedis_matrix[i][i] = -float('inf')

train_res = []
for d in train_data:
    postfix_cand = ' '.join(d['postfix_norm'])
    exprpos = expr_expr_dict[postfix_cand]
    src = d['keys']
    cand_ids = expr_id_dict[postfix_cand]
    if len(cand_ids) == 1:
        postfix_cand = expr_expr_reverse_dict[np.argmax(treedis_matrix[exprpos])]
        cand_ids = expr_id_dict[postfix_cand]
    maxscore = -float('inf')
    cand = None
    for idx in cand_ids:
        if idx != d['id'] and sim_chinese(src, cached[idx]) > maxscore:
            maxscore = sim_chinese(src, cached[idx])
            cand = idx
    d['cand'] = cand
    train_res.append(d)

dev_res = []
for d in dev_data:
    postfix_cand = ' '.join(d['postfix_norm'])
    src = d['keys']
    if postfix_cand in expr_expr_dict:
        exprpos = expr_expr_dict[postfix_cand]
        cand_ids = expr_id_dict[postfix_cand]
    else:
        postfix_cand = expr_expr_reverse_dict[np.argmax(treedis_matrix[exprpos])]
        cand_ids = expr_id_dict[postfix_cand]
    maxscore = -float('inf')
    cand = None
    for idx in cand_ids:
        if idx != d['id'] and sim_chinese(src, cached[idx]) > maxscore:
            maxscore = sim_chinese(src, cached[idx])
            cand = idx
    d['cand'] = cand
    dev_res.append(d)

test_res = []
for d in test_data:
    postfix_cand = ' '.join(d['postfix_norm'])
    src = d['keys']
    if postfix_cand in expr_expr_dict:
        exprpos = expr_expr_dict[postfix_cand]
        cand_ids = expr_id_dict[postfix_cand]
    else:
        postfix_cand = expr_expr_reverse_dict[np.argmax(treedis_matrix[exprpos])]
        cand_ids = expr_id_dict[postfix_cand]
    maxscore = -float('inf')
    cand = None
    for idx in cand_ids:
        if idx != d['id'] and sim_chinese(src, cached[idx]) > maxscore:
            maxscore = sim_chinese(src, cached[idx])
            cand = idx
    d['cand'] = cand
    test_res.append(d)

f = open('../data/math23k_norm/Math23K_keys_cand_train.jsonl', 'w')
for d in train_res:
    json.dump(d, f, ensure_ascii=False)
    f.write("\n")
f.close()

f = open('../data/math23k_norm/Math23K_keys_cand_dev.jsonl', 'w')
for d in dev_res:
    json.dump(d, f, ensure_ascii=False)
    f.write("\n")
f.close()

f = open('../data/math23k_norm/Math23K_keys_cand_test.jsonl', 'w')
for d in test_res:
    json.dump(d, f, ensure_ascii=False)
    f.write("\n")
f.close()