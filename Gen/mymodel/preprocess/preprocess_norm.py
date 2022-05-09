import re
import copy
import json
import sympy
from transformers import BertTokenizer

def from_infix_to_prefix(expression):
    st = list()
    res = list()
    priority = {"+": 0, "-": 0, "*": 1, "/": 1, "^": 2}
    expression = copy.deepcopy(expression)
    expression.reverse()
    for e in expression:
        if e in [")", "]"]:
            st.append(e)
        elif e == "(":
            c = st.pop()
            while c != ")":
                res.append(c)
                c = st.pop()
        elif e == "[":
            c = st.pop()
            while c != "]":
                res.append(c)
                c = st.pop()
        elif e in priority:
            while len(st) > 0 and st[-1] not in [")", "]"] and priority[e] < priority[st[-1]]:
                res.append(st.pop())
            st.append(e)
        else:
            res.append(e)
    while len(st) > 0:
        res.append(st.pop())
    res.reverse()
    return res

def from_infix_to_postfix(expression):
    st = list()
    res = list()
    priority = {"+": 0, "-": 0, "*": 1, "/": 1, "^": 2}
    for e in expression:
        if e in ["(", "["]:
            st.append(e)
        elif e == ")":
            c = st.pop()
            while c != "(":
                res.append(c)
                c = st.pop()
        elif e == "]":
            c = st.pop()
            while c != "[":
                res.append(c)
                c = st.pop()
        elif e in priority:
            while len(st) > 0 and st[-1] not in ["(", "["] and priority[e] <= priority[st[-1]]:
                res.append(st.pop())
            st.append(e)
        else:
            res.append(e)
    while len(st) > 0:
        res.append(st.pop())
    return res

def from_prefix_to_infix(prefix):
    st = list()
    operators = ["+", "-", "^", "*", "/"]
    prefix = copy.deepcopy(prefix)
    prefix.reverse()
    for p in prefix:
        if p not in operators:
            st.append(p)
        elif p == "+" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            operands = [a, b]
            operands.sort()
            a, b = operands
            st.append(" ".join([a, "+", b]))
        elif p == "*" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            operands = [a, b]
            operands.sort()
            a, b = operands
            st.append(" ".join(["(", a, ")", "*", "(", b, ")"]))
        elif p == "/" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(" ".join(["(", a, ")", "/", "(", b, ")"]))
        elif p == "-" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(" ".join(["(", a, ")", "-", "(", b, ")"]))
        elif p == "^" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(" ".join(["(", a, ")", "^", "(", b, ")"]))
        else:
            return None
    if len(st) == 1:
        return st.pop()
    return None

def from_postfix_to_infix(postfix):
    st = []
    operators = ["+", "-", "^", "*", "/"]
    for p in postfix:
        if p not in operators:
            st.append(p)
        elif p == "+" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            operands = [a, b]
            operands.sort()
            a, b = operands
            st.append(" ".join([a, "+", b]))
        elif p == "*" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            operands = [a, b]
            operands.sort()
            a, b = operands
            st.append(" ".join(["(", a, ")", "*", "(", b, ")"]))
        elif p == "/" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(" ".join(["(", b, ")", "/", "(", a, ")"]))
        elif p == "-" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(" ".join(["(", b, ")", "-", "(", a, ")"]))
        elif p == "^" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(" ".join(["(", b, ")", "^", "(", a, ")"]))
        else:
            return None
    if len(st) == 1:
        return st.pop()

def load_data(filename):
    data = []
    with open(filename, encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def norm(data):
    res = []
    for d in data:
        d['prefix_norm'] = from_infix_to_prefix(from_prefix_to_infix(d['prefix']).split(' '))
        d['postfix_norm'] = from_infix_to_postfix(from_postfix_to_infix(d['postfix']).split(' '))
        res.append(d)
    return res

train_data = load_data('../data/math23k/Math23K_train.jsonl')
dev_data = load_data('../data/math23k/Math23K_dev.jsonl')
test_data = load_data('../data/math23k/Math23K_test.jsonl')
train_data = norm(train_data)
dev_data = norm(dev_data)
test_data = norm(test_data)
f = open('../data/math23k_norm/Math23K_train_norm.jsonl', 'w')
for d in train_data:
    json.dump(d, f, ensure_ascii=False)
    f.write("\n")
f.close()
f = open('../data/math23k_norm/Math23K_dev_norm.jsonl', 'w')
for d in dev_data:
    json.dump(d, f, ensure_ascii=False)
    f.write("\n")
f.close()
f = open('../data/math23k_norm/Math23K_test_norm.jsonl', 'w')
for d in test_data:
    json.dump(d, f, ensure_ascii=False)
    f.write("\n")
f.close()

for fold in range(5):
    train_data = load_data('../data/mawps/MAWPS_'+str(fold)+'_train.jsonl')
    test_data = load_data('../data/mawps/MAWPS_'+str(fold)+'_test.jsonl')
    train_data = norm(train_data)
    test_data = norm(test_data)
    f = open('../data/mawps_norm/MAWPS_'+str(fold)+'_train_norm.jsonl', 'w')
    for d in train_data:
        json.dump(d, f, ensure_ascii=False)
        f.write("\n")
    f.close()
    f = open('../data/mawps_norm/MAWPS_'+str(fold)+'_test_norm.jsonl', 'w')
    for d in test_data:
        json.dump(d, f, ensure_ascii=False)
        f.write("\n")
    f.close()