import json
import torch
import random
import numpy as np
from multiprocessing import Pool
from zss import simple_distance, Node

def set_seed(seed=42): 
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    print('seed:', seed)
set_seed()

def load_data(filename):
    data = []
    with open(filename, encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

def from_postfix_to_tree(postfix):
    st = list()
    operators = ["+", "-", "^", "*", "/"]
    for p in postfix:
        if p not in operators:
            st.append(Node(p))
        elif p == "+" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(Node(p).addkid(b).addkid(a))
        elif p == "*" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(Node(p).addkid(b).addkid(a))
        elif p == "/" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(Node(p).addkid(b).addkid(a))
        elif p == "-" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(Node(p).addkid(b).addkid(a))
        elif p == "^" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(Node(p).addkid(b).addkid(a))
    return st.pop()

train_data = load_data('../data/Math23K_train.jsonl')
test_data = load_data('../data/Math23K_test.jsonl')
for d in train_data:
    d['tree'] = from_postfix_to_tree(d['postfix'])
for d in test_data:
    d['tree'] = from_postfix_to_tree(d['postfix'])

train_batches = []
test_batches = []
for i in range(len(train_data)):
    indexs = [random.randint(0, len(train_data)-1) for _ in range(10)]
    for j in indexs:
        d1, d2 = train_data[i], train_data[j]
        src1, src2 = d1['text'], d2['text']
        tree1, tree2 = d1['tree'], d2['tree']
        distance = simple_distance(tree1, tree2)
        temp = {}
        temp['src1'] = src1
        temp['src2'] = src2
        temp['prefix1'] = d1['prefix']
        temp['prefix2'] = d2['prefix']
        temp['distance'] = distance
        train_batches.append(temp)
    if i % 100 == 0:
        print(i)

for i in range(len(test_data)):
    indexs = [random.randint(0, len(test_data)-1) for _ in range(10)]
    for j in indexs:
        d1, d2 = test_data[i], test_data[j]
        src1, src2 = d1['text'], d2['text']
        tree1, tree2 = d1['tree'], d2['tree']
        distance = simple_distance(tree1, tree2)
        temp = {}
        temp['src1'] = src1
        temp['src2'] = src2
        temp['prefix1'] = d1['prefix']
        temp['prefix2'] = d2['prefix']
        temp['distance'] = distance
        test_batches.append(temp)
    if i % 100 == 0:
        print(i)

f = open('../data/Math23K_distance_train.jsonl', 'w')  
for d in train_batches:
    json.dump(d, f, ensure_ascii=False)
    f.write("\n")
f.close()
f = open('../data/Math23K_distance_test.jsonl', 'w')  
for d in test_batches:
    json.dump(d, f, ensure_ascii=False)
    f.write("\n")
f.close()