import re
import copy
import json

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

def cut_sent(para):
    para = re.sub('([，．。！？\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
    para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
    para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
    para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    para = para.rstrip()  # 段尾如果有多余的\n就去掉它
    # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
    return para.split("\n")


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

def out_expression_list(test, nums):
    res = []
    for i in test:
        if i[0] == 'N':
            res.append(nums[int(i[2:])%len(nums)])
        elif i[0] == 'C':
            res.append(i[2:].replace('_', '.'))
        else:
            res.append(i)
    return res

def compute_prefix_expression(pre_fix):
    st = list()
    operators = ["+", "-", "^", "*", "/"]
    pre_fix = copy.deepcopy(pre_fix)
    pre_fix.reverse()
    for p in pre_fix:
        if p not in operators:
            st.append(eval(p))
        elif p == "+" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(a + b)
        elif p == "*" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(a * b)
        elif p == "/" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            if b == 0:
                return None
            st.append(a / b)
        elif p == "-" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(a - b)
        elif p == "^" and len(st) > 1:
            a = st.pop()
            b = st.pop()
#            if float(eval(b)) != 2.0 or float(eval(b)) != 3.0:
#                return None
            st.append(a ** b)
        else:
            return None
    if len(st) == 1:
        return st.pop()
    return None

def transfer_num(data):  # transfer num into "NUM"
    print("Transfer numbers...")
    pattern = re.compile("\d*\(\d+/\d+\)\d*|\d+\.\d+%?|\d+%?")
    pairs = []
    for d in data:
        nums = []
        input_seq = []
        seg = d["segmented_text"].strip().split(" ")
        equations = d["equation"][2:]

        for s in seg:
            pos = re.search(pattern, s)
            if pos and pos.start() == 0:
                nums.append(s[pos.start(): pos.end()])
                # input_seq.append('_'+s[pos.start():pos.end()]+'[N]')
                # input_seq.append("NUM")
                input_seq.append("N_" + str(len(nums)-1))
                if pos.end() < len(s):
                    input_seq.append(s[pos.end():])
            else:
                input_seq.append(s)

        nums_fraction = []
        for num in nums:
            if re.search("\d*\(\d+/\d+\)\d*", num):
                nums_fraction.append(num)
        nums_fraction = sorted(nums_fraction, key=lambda x: len(x), reverse=True)

        def seg_and_tag(st):  # seg the equation and tag the num
            res = []
            for n in nums_fraction:
                if n in st:
                    p_start = st.find(n)
                    p_end = p_start + len(n)
                    if p_start > 0:
                        res += seg_and_tag(st[:p_start])
                    if nums.count(n) >= 1:
                        res.append("N_"+str(nums.index(n)))
                    else:
                        n = "C_" + n
                        n = n.replace('.', '_')
                        res.append(n)
                    if p_end < len(st):
                        res += seg_and_tag(st[p_end:])
                    return res
            pos_st = re.search("\d+\.\d+%?|\d+%?", st)
            if pos_st:
                p_start = pos_st.start()
                p_end = pos_st.end()
                if p_start > 0:
                    res += seg_and_tag(st[:p_start])
                st_num = st[p_start:p_end]
                if nums.count(st_num) >= 1:
                    res.append("N_"+str(nums.index(st_num)))
                else:
                    st_num = "C_" + st_num
                    st_num = st_num.replace('.', '_')
                    res.append(st_num)
                if p_end < len(st):
                    res += seg_and_tag(st[p_end:])
                return res
            for ss in st:
                res.append(ss)
            return res

        out_seq = seg_and_tag(equations)
        out_seq = ' '.join(out_seq)
        out_seq = out_seq.replace('[', '(')
        out_seq = out_seq.replace(']', ')')
        out_seq = out_seq.split(' ')
        num_values = []
        for p in nums:
            pos1 = re.search("\d+\(", p)
            pos2 = re.search("\)\d+", p)
            if pos1:
                num_values.append(str(eval(p[pos1.start(): pos1.end() - 1] + "+" + p[pos1.end() - 1:])))
            elif pos2:
                num_values.append(str(eval(p[:pos2.start() + 1] + "+" + p[pos2.start() + 1: pos2.end()])))
            elif p[-1] == "%":
                num_values.append(str(float(p[:-1]) / 100))
            else:
                num_values.append(str(eval(p)))
        temp = {}
        temp['id'] = d['id']
        # temp['answer'] = d['ans']
        temp['text'] = ''.join(input_seq)
        temp['original_text'] = d['original_text']
        temp['infix'] = out_seq
        prefix = from_infix_to_prefix(out_seq)
        postfix = from_infix_to_postfix(out_seq)
        temp['prefix'] = prefix
        temp['postfix'] = postfix
        ans = compute_prefix_expression(out_expression_list(prefix, num_values))
        if ans is None:
            print(d)
            continue

        temp['nums'] = num_values
        temp['answer'] = ans
        pairs.append(temp)
    return pairs

test_data = load_raw_data('../data/math23k/Math_23K_test.json')
test_data_q1 = []
for d in test_data:
    temp = copy.deepcopy(d)
    sentences = cut_sent(temp['segmented_text'])
    if len(sentences) > 1:
        for idx, sentence in enumerate(sentences):
            if '？' in sentences[idx] or '=' in sentences[idx] or \
                '多少' in sentences[idx] or '问' in sentences[idx] or \
                '求' in sentences[idx] or '((())/(()))' in sentences[idx]:
                break
        question = copy.deepcopy(sentence)
        question += ' 如果'
        res = [question] + sentences[:idx] + sentences[idx+1:]
        res = ' '.join(res)
        temp['segmented_text'] = res
        test_data_q1.append(temp)
    else:
        test_data_q1.append(temp)

test_data_q1 = transfer_num(test_data_q1)
f = open('../data/math23k/Math23K_adversarial_q1_test.jsonl', 'w')
for d in test_data_q1:
    json.dump(d, f, ensure_ascii=False)
    f.write("\n")
f.close()

test_data = load_raw_data('../data/math23k/Math_23K_test.json')
test_data_q2 = []
for d in test_data:
    temp = copy.deepcopy(d)
    sentences = cut_sent(temp['segmented_text'])
    if len(sentences) > 1:
        sentences[0], sentences[1] = sentences[1], sentences[0]
        res = ' '.join(sentences)
        temp['segmented_text'] = res
        test_data_q2.append(temp)
    else:
        test_data_q2.append(temp)

test_data_q2 = transfer_num(test_data_q2)
f = open('../data/math23k/Math23K_adversarial_q2_test.jsonl', 'w')
for d in test_data_q2:
    json.dump(d, f, ensure_ascii=False)
    f.write("\n")
f.close()

test_data = load_raw_data('../data/math23k/Math_23K_test.json')
test_data_s = []
for d in test_data:
    temp = copy.deepcopy(d)
    sentences = cut_sent(temp['segmented_text'])
    if len(sentences) > 1:
        count = 0
        pos = []
        for i in range(len(sentences)):
            if count == 2:
                break
            pos_st = re.search("\d+\.\d+%?|\d+%?", sentences[i])
            if pos_st:
                count += 1
                pos.append(i)
        if len(pos) == 2:
            sentences[pos[0]], sentences[pos[1]] = sentences[pos[1]], sentences[pos[0]]
        res = ' '.join(sentences)
        temp['segmented_text'] = res
        test_data_s.append(temp)
    else:
        test_data_s.append(temp)

test_data_s = transfer_num(test_data_s)
f = open('../data/math23k/Math23K_adversarial_s_test.jsonl', 'w')
for d in test_data_s:
    json.dump(d, f, ensure_ascii=False)
    f.write("\n")
f.close()