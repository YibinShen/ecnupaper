from cProfile import label
from re import I
import copy
import json
import torch
import random
import numpy as np
from torch import nn
from tqdm import tqdm
from collections import Counter
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AdamW

from transformers import get_linear_schedule_with_warmup

from configuration.config import *
from bojone_snippets import DataGenerator, AutoRegressiveDecoder
from preprocess.metric import compute_postfix_tree_result

pretrain_model_path = 'facebook/bart-base'
max_text_len = 256
max_equ_len = 45
batch_size = 16
epochs = 50
lr = 5e-5
# max_grad_norm = 1.0

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

def load_data(filename):
    data = []
    with open(filename, encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def set_seed(seed=1): 
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    print('seed:', seed)

def generate_graph(max_num_len, nums):
    diag_ele = np.ones(max_num_len)
    graph1 = np.diag(diag_ele)
    for i in range(len(nums)):
        for j in range(len(nums)):
            if nums[i] <= nums[j]:
                graph1[i][j] = 1
            else:
                graph1[j][i] = 1
    graph2 = graph1.T
    return [graph1.tolist(), graph2.tolist()]

# def loss_function(predicted, target, PAD_ID):
#     predicted = predicted.flatten(0, -2)
#     target = target.flatten()
#     loss_fct = nn.CrossEntropyLoss(ignore_index=PAD_ID)
#     loss = loss_fct(predicted, target)
#     # loss /= bacth_size
#     return loss

def train():
    set_seed()
    for fold in range(5):
        # 加载数据集
        data_root_path = 'data/mawps/'
        train_data = load_data(data_root_path + 'MAWPS_' + str(fold) + '_train.jsonl')
        dev_data = load_data(data_root_path + 'MAWPS_' + str(fold) + '_test.jsonl')

        tokenizer = AutoTokenizer.from_pretrained(pretrain_model_path)
        tokens_count = Counter()
        max_nums_len = 0
        for d in train_data + dev_data:
            tokens_count += Counter(d['postfix'])
            max_nums_len = max(max_nums_len, len(d['nums']))
        tokens = list(tokens_count)
        op_tokens = [x for x in tokens if x[0].lower()!='c' and x[0].lower()!='n' and x[0].lower()!='x']
        constant_tokens = [x for x in tokens if x[0].lower()=='c' and tokens_count[x]>=5]
        number_tokens = ['N_' + str(x) for x in range(max_nums_len)]
        op_tokens.sort()
        constant_tokens = sorted(constant_tokens, key=lambda x: float(x[2:].replace('_', '.')))
        constant_tokens = constant_tokens + [x for x in tokens if x[0].lower()=='x']
        number_tokens = sorted(number_tokens, key=lambda x: int(x[2:]))
        # 字典
        tokens = op_tokens + constant_tokens + number_tokens

        tokenizer.add_special_tokens({'additional_special_tokens': tokens})
        train_batches = []
        for d in train_data:
            src = tokenizer.encode(d['text'], max_length=max_text_len)
            # src[-1] = tokenizer.convert_tokens_to_ids('en_XX')
            tgt = tokenizer.encode(''.join(d['postfix']))
            # tgt[-1] = tokenizer.convert_tokens_to_ids('en_XX')
            train_batches.append((src, tgt))

        dev_batches = []
        for d in dev_data:
            src = tokenizer.encode(d['text'], max_length=max_text_len)
            # src[-1] = tokenizer.convert_tokens_to_ids('en_XX')
            dev_batches.append((src, d))

        def data_generator(train_batches, batch_size):
            i = 0
            pairs = []
            while i + batch_size < len(train_batches):
                pair = train_batches[i: i+batch_size]
                pairs.append(pair)
                i += batch_size
            pairs.append(train_batches[i:])
            batches = []
            for pair in pairs:
                text_ids, equ_ids = [], []
                max_text = max([len(x[0]) for x in pair])
                max_equ = max([len(x[1]) for x in pair])
                for _, p in enumerate(pair):
                    text, equ = p
                    text_ids.append(text + [tokenizer.pad_token_id] * (max_text-len(text)))
                    equ_ids.append(equ + [tokenizer.pad_token_id] * (max_equ-len(equ)))
                text_ids = torch.tensor(text_ids, dtype=torch.long)
                equ_ids = torch.tensor(equ_ids, dtype=torch.long)
                text_pads = text_ids != tokenizer.pad_token_id
                text_pads = text_pads.float()
                equ_pads = equ_ids != tokenizer.pad_token_id
                equ_pads = equ_pads.float()
                batches.append((text_ids, text_pads, equ_ids, equ_pads))
            return batches
        
        class Solver(AutoRegressiveDecoder):
            @AutoRegressiveDecoder.wraps(default_rtype='probas')
            def predict(self, inputs, output_ids, states):  # output_ids: [1, s']
                encoded = inputs[0]
                with torch.no_grad():
                    decoder_hidden_state = pretrain_model.model.decoder(encoder_hidden_states=torch.tensor(encoded, dtype=torch.float, device=device),
                                                                        input_ids=torch.tensor(output_ids, dtype=torch.long, device=device))[0]
                    logits = pretrain_model.lm_head(decoder_hidden_state)
                pred = torch.softmax(logits[:, -1], dim=-1)
                pred = pred.cpu().detach().numpy()
                return pred

            def generate(self, text, topk=3):
                with torch.no_grad():
                    encoded = pretrain_model.model.encoder(input_ids=text)[0]  # [1,s,h]
                output_ids, output_score = self.beam_search(encoded.cpu().detach().numpy(), topk)  # 基于beam search
                # return tokenizer.convert_ids_to_tokens([int(i) for i in output_ids]), output_score
                return [tokenizer.decode(int(x)) for x in output_ids[1:-1]], output_score
     
        pretrain_model = AutoModelForSeq2SeqLM.from_pretrained(pretrain_model_path)
        pretrain_model.resize_token_embeddings(len(tokenizer))
        solver = Solver(start_id=tokenizer.bos_token_id, end_id=tokenizer.eos_token_id, maxlen=max_equ_len)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pretrain_model.to(device)
        
        train_generator = data_generator(train_batches, batch_size)
        optimizer = AdamW(pretrain_model.parameters(), lr=lr, weight_decay=0.01)
        global_steps = len(train_generator) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=global_steps*0.1, num_training_steps=global_steps)

        # train
        pretrain_model.zero_grad()
        log = open('result_mawps/log_' + str(fold) + '.txt', 'w')
        for e in range(epochs):
            print("epoch:", e)
            pretrain_model.train()
            loss_total = 0.0
            random.shuffle(train_batches)
            train_generator = data_generator(train_batches, batch_size)
            bar = tqdm(enumerate(train_generator), total=len(train_generator))
            for step, batch in bar:
                batch = [_.to(device) for _ in batch]
                text_ids, text_pads, equ_ids, equ_pads = batch
                encoded = pretrain_model.model.encoder(input_ids=text_ids, attention_mask=text_pads)[0]
                decoder_hidden_state = pretrain_model.model.decoder(encoder_hidden_states=encoded, input_ids=equ_ids, attention_mask=equ_pads)[0]
                logits = pretrain_model.lm_head(decoder_hidden_state)
                loss_fct = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, reduction="sum")
                shift_logits = logits[:, :-1].contiguous()
                shift_labels = equ_ids[:, 1:].contiguous()
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                loss /= shift_labels.size(0)
                # output = pretrain_model(input_ids=text_ids, attention_mask=text_pads, labels=equ_ids[:, 1:].contiguous())
                # loss = output.loss
                loss_total += loss.item()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(solver.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            loss_total /= len(train_generator)
            log.write("epoch:" + str(e)+ "\tloss:" + str(loss_total) + "\n")
            logger.info(f"epoch: {e} - loss: {loss_total}")
            
            if (e > 0 and e % 10 == 0) or epochs - e < 5:
                pretrain_model.eval()

                value_ac = 0
                equation_ac = 0
                eval_total = 0
                all_results = []
                correct_results = []
                wrong_results = []
                bar = tqdm(enumerate(dev_batches), total=len(dev_batches))
                for _,(text, d) in bar:
                    text_ids = torch.tensor([text], dtype=torch.long)
                    text_ids = text_ids.to(device)
                    res, score = solver.generate(text_ids)
                    val_ac, equ_ac, _, _ = compute_postfix_tree_result(res, d['postfix'], d['answer'], d['nums'])
                    value_ac += val_ac
                    equation_ac += equ_ac
                    eval_total += 1
                    temp = {}
                    temp['id'] = d['id']
                    temp['text'] = d['text']
                    temp['nums'] = d['nums']
                    temp['ans'] = d['answer']
                    temp['prefix'] = d['prefix']
                    temp['postfix'] = d['postfix']
                    temp['res'] = res
                    temp['score'] = score
                    if val_ac:
                        correct_results.append(temp)
                    else:
                        wrong_results.append(temp)
                    all_results.append(temp)
                
                f = open('result_mawps/correct_result_dev_' + str(fold) + '.jsonl', 'w')
                for d in correct_results:
                    json.dump(d, f, ensure_ascii=False)
                    f.write("\n")
                f.close()
                f = open('result_mawps/wrong_results_dev_' + str(fold) + '.jsonl', 'w')
                for d in wrong_results:
                    json.dump(d, f, ensure_ascii=False)
                    f.write("\n")
                f.close()
                f = open('result_mawps/all_results_dev_' + str(fold) + '.jsonl', 'w')
                for d in all_results:
                    json.dump(d, f, ensure_ascii=False)
                    f.write("\n")
                f.close()

                log.write("epoch:" + str(e)+ "\tequ_acc:" + str(float(equation_ac) / eval_total) + "\tval_acc:" + str(float(value_ac) / eval_total) + "\n")
                logger.info(f"epoch: {e} - equ_acc: {float(equation_ac) / eval_total} - val_acc: {float(value_ac) / eval_total}")

                pretrain_model.save_pretrained('result_mawps/fold_' + str(fold) +'/models')
                tokenizer.save_pretrained('result_mawps/fold_' + str(fold) +'/models')
        log.close()


def Visual(fold):
    print(fold)
    data_root_path = 'data/mawps/'
    test_data = load_data(data_root_path + 'MAWPS_Visual.jsonl')

    model_path = 'result_mawps/fold_'+str(fold)+'/models'
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    test_batches = []
    for d in test_data:
        src = tokenizer.encode(d['text'], max_length=max_text_len)
        test_batches.append((src, d))

    pretrain_model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrain_model.to(device)

    pretrain_model.eval()
    bar = tqdm(enumerate(test_batches), total=len(test_batches))
    embeddings = []
    for _,(text, d) in bar:
        text_ids = torch.tensor([text], dtype=torch.long)
        text_ids = text_ids.to(device)
        encoded = pretrain_model.model.encoder(input_ids=text_ids)[0]
        temp = {}
        temp['id'] = d['id']
        temp['prefix'] = ' '.join(d['prefix'])
        temp['embedding'] = list(encoded[:, 0].cpu().detach().numpy())
        embeddings.append(temp)
    
    f = open('result_mawps/embeddings_'+str(fold)+'.jsonl', 'w')
    for d in embeddings:
        json.dump(d, f, ensure_ascii=False, cls=MyEncoder)
        f.write("\n")
    f.close()

# train()
# Visual(0)
# Visual(1)
# Visual(2)
# Visual(3)
# Visual(4)