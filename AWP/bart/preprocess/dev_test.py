import json

filename = open('../data/hmwp/hmwp.jsonl', 'r')
data = []
for i,line in enumerate(filename):
    temp = json.loads(line)
    temp['id'] = str(i)
    data.append(temp)

fold_size = int(len(data) * 0.2)
train_data = data[:int(len(data) * 0.6)]
dev_data = data[int(len(data) * 0.6):int(len(data) * 0.8)]
test_data = data[int(len(data) * 0.8):]
f = open('../data/hmwp/dev_test/HMWP_train.jsonl', 'w')
for d in train_data:
    json.dump(d, f, ensure_ascii=False)
    f.write('\n')
f.close()
f = open('../data/hmwp/dev_test/HMWP_dev.jsonl', 'w')
for d in dev_data:
    json.dump(d, f, ensure_ascii=False)
    f.write('\n')
f.close()
f = open('../data/hmwp/dev_test/HMWP_test.jsonl', 'w')
for d in test_data:
    json.dump(d, f, ensure_ascii=False)
    f.write('\n')
f.close()