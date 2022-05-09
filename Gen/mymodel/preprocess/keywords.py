import json
import jieba.analyse
topk= 1

def load_data(filename):
    data = []
    with open(filename, encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

for fold in range(5):
    train_data = load_data('../data/mawps_norm/MAWPS_'+str(fold)+'_train_norm.jsonl')
    test_data = load_data('../data/mawps_norm/MAWPS_'+str(fold)+'_test_norm.jsonl')
    train_keys_data = []
    test_keys_data = []
    for d in train_data:
        src = d['text']
        keys = jieba.analyse.extract_tags(sentence=src, topK=topk)
        d['keys'] = keys
        train_keys_data.append(d)
    for d in test_data:
        src = d['text']
        keys = jieba.analyse.extract_tags(sentence=src, topK=topk)
        d['keys'] = keys
        test_keys_data.append(d)
    f = open('../data/mawps_norm/MAWPS_'+str(fold)+'_keys_train.jsonl', 'w')
    for d in train_keys_data:
        json.dump(d, f, ensure_ascii=False)
        f.write("\n")
    f.close()
    f = open('../data/mawps_norm/MAWPS_'+str(fold)+'_keys_test.jsonl', 'w')
    for d in test_keys_data:
        json.dump(d, f, ensure_ascii=False)
        f.write("\n")
    f.close()

train_data = load_data('../data/math23k_norm/Math23K_train_norm.jsonl')
dev_data = load_data('../data/math23k_norm/Math23K_dev_norm.jsonl')
test_data = load_data('../data/math23k_norm/Math23K_test_norm.jsonl')
train_keys_data = []
dev_keys_data = []
test_keys_data = []
for d in train_data:
    src = d['text']
    keys = jieba.analyse.extract_tags(sentence=src, topK=topk)
    d['keys'] = keys
    train_keys_data.append(d)
for d in dev_data:
    src = d['text']
    keys = jieba.analyse.extract_tags(sentence=src, topK=topk)
    d['keys'] = keys
    dev_keys_data.append(d)
for d in test_data:
    src = d['text']
    keys = jieba.analyse.extract_tags(sentence=src, topK=topk)
    d['keys'] = keys
    test_keys_data.append(d)
f = open('../data/math23k_norm/Math23K_keys_train.jsonl', 'w')
for d in train_keys_data:
    json.dump(d, f, ensure_ascii=False)
    f.write("\n")
f.close()
f = open('../data/math23k_norm/Math23K_keys_dev.jsonl', 'w')
for d in dev_keys_data:
    json.dump(d, f, ensure_ascii=False)
    f.write("\n")
f.close()
f = open('../data/math23k_norm/Math23K_keys_test.jsonl', 'w')
for d in test_keys_data:
    json.dump(d, f, ensure_ascii=False)
    f.write("\n")
f.close()