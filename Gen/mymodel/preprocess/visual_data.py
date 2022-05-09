import json
from collections import Counter

# data = {}
# for line in open('../data/mawps/MAWPS_0_train.jsonl'):
#     temp = json.loads(line)
#     data[temp['id']] = {'expr': ' '.join(temp['prefix']), 'd':temp}
# for line in open('../data/mawps/MAWPS_0_test.jsonl'):
#     temp = json.loads(line)
#     data[temp['id']] = {'expr': ' '.join(temp['prefix']), 'd':temp}
# expr = {}
# for k,d in data.items():
#     if d['expr'] not in expr:
#         expr[d['expr']] = 1
#     else:
#         expr[d['expr']] += 1
# expr = sorted(expr.items(), key=lambda x: -x[-1])
# expr = [x[0] for x in expr[:5]]
# visual_data = []
# for k,d in data.items():
#     if d['expr'] in expr:
#         visual_data.append(d['d'])
# f = open('../data/mawps/MAWPS_Visual.jsonl', 'w')
# for d in visual_data:
#     json.dump(d, f, ensure_ascii=False)
#     f.write("\n")
# f.close()


data = {}
for line in open('../data/math23k/Math23K_train.jsonl'):
    temp = json.loads(line)
    data[temp['id']] = {'expr': ' '.join(temp['prefix']), 'd':temp}
for line in open('../data/math23k/Math23K_dev.jsonl'):
    temp = json.loads(line)
    data[temp['id']] = {'expr': ' '.join(temp['prefix']), 'd':temp}
for line in open('../data/math23k/Math23K_test.jsonl'):
    temp = json.loads(line)
    data[temp['id']] = {'expr': ' '.join(temp['prefix']), 'd':temp}
expr = {}
for k,d in data.items():
    if d['expr'] not in expr:
        expr[d['expr']] = 1
    else:
        expr[d['expr']] += 1
expr = sorted(expr.items(), key=lambda x: -x[-1])
expr = [x[0] for x in expr[:5]]
visual_data = []
for k,d in data.items():
    if d['expr'] in expr:
        visual_data.append(d['d'])
f = open('../data/math23k/Math23K_Visual.jsonl', 'w')
for d in visual_data:
    json.dump(d, f, ensure_ascii=False)
    f.write("\n")
f.close()

