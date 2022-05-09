import json

def load_data(filename):
    data = []
    with open(filename, encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def write_data(data, filename):
    f = open(filename, 'w')
    for d in data:
        json.dump(d, f, ensure_ascii=False)
        f.write("\n")
    f.close()

data_root_path = '../data/mawps/'
train_data = load_data(data_root_path + 'MAWPS_train.jsonl')
dev_data = load_data(data_root_path + 'MAWPS_dev.jsonl')
test_data = load_data(data_root_path + 'MAWPS_test.jsonl')
data = train_data + dev_data + test_data
# data = sorted(data, key=lambda x: int(x['id']))

fold_size = int(len(data) * 0.2)
fold_pairs = []
for split_fold in range(4):
    fold_start = fold_size * split_fold
    fold_end = fold_size * (split_fold + 1)
    fold_pairs.append(data[fold_start:fold_end])
fold_pairs.append(data[(fold_size * 4):])

for fold in range(5):
    pairs_tested = []
    pairs_trained = []
    for fold_t in range(5):
        if fold_t == fold:
            pairs_tested += fold_pairs[fold_t]
        else:
            pairs_trained += fold_pairs[fold_t]
    write_data(pairs_trained, "../data/mawps/MAWPS_"+str(4-fold)+"_train.jsonl")
    write_data(pairs_tested, "../data/mawps/MAWPS_"+str(4-fold)+"_test.jsonl")