import re
import copy
import json

def cut_sent(para):
    para = re.sub('([，．。！？\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
    para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
    para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
    para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    para = para.rstrip()  # 段尾如果有多余的\n就去掉它
    # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
    return para.split("\n")

def load_raw_data(filename):  # load the json data to list(dict()) for MATH 23K
    print("Reading lines...")
    f = open(filename, encoding="utf-8")
    js = ""
    data = []
    for i, s in enumerate(f):
        js += s
        i += 1
        if i % 7 == 0:  # every 7 line is a json
            data_d = json.loads(js)
            if "千米/小时" in data_d["equation"]:
                data_d["equation"] = data_d["equation"][:-5]
            data.append(data_d)
            js = ""

    return data

train_data = load_raw_data('../data/Math_23K_train.json')
train_data_question1 = []
for d in train_data:
    temp = copy.deepcopy(d)
    sentences = cut_sent(temp['segmented_text'])
    if len(sentences) > 1:
        for idx, sentence in enumerate(sentences):
            if '？' in sentences[idx]:
                break
        question = copy.deepcopy(sentence)
        if '？' not in sentence:
            print(question)
        question += ' 如果'
        res = [question] + sentences[:idx] + sentences[idx+1:]
        res = ' '.join(res)
        temp['segmented_text'] = res
        train_data_question1.append(temp)

f = open('../data/Math_23K_train_question1.json', 'w')
for d in train_data_question1:
    json.dump(d, f, ensure_ascii=False)
    f.write("\n")
f.close()

train_data = load_raw_data('../data/Math_23K_train.json')
train_data_question2 = []
for d in train_data:
    temp = copy.deepcopy(d)
    sentences = cut_sent(temp['segmented_text'])
    if len(sentences) > 1:
        sentences[0], sentences[1] = sentences[1], sentences[0]
        res = ' '.join(sentences)
        temp['segmented_text'] = res
        train_data_question2.append(temp)

f = open('../data/Math_23K_train_question2.json', 'w')
for d in train_data_question2:
    json.dump(d, f, ensure_ascii=False)
    f.write("\n")
f.close()
