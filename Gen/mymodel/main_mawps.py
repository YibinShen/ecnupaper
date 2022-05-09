from cProfile import label
from re import I
import copy
import json
import torch
import random
import numpy as np
from torch import embedding, nn
from tqdm import tqdm
from collections import Counter
from transformers import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM, AdamW

from transformers import get_linear_schedule_with_warmup

from configuration.config import *
from bojone_snippets import DataGenerator, AutoRegressiveDecoder
from models.train_and_evaluate import Model

from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge import Rouge

pretrain_model_path = 'facebook/bart-base'
max_text_len = 256
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

def evaluate(data):
    rouge = Rouge()
    smooth = SmoothingFunction().method1
    total = 0
    rouge_1, rouge_2, rouge_l, bleu = 0, 0, 0, 0
    for title, pred_title in data:
        total += 1
        # title = ' '.join(title).lower()
        # pred_title = ' '.join(model.generate(content)).lower()
        if pred_title.strip():
            scores = rouge.get_scores(hyps=pred_title, refs=title)
            rouge_1 += scores[0]['rouge-1']['f']
            rouge_2 += scores[0]['rouge-2']['f']
            rouge_l += scores[0]['rouge-l']['f']
            bleu += sentence_bleu(
                references=[title.split(' ')],
                hypothesis=pred_title.split(' '),
                smoothing_function=smooth
            )
    rouge_1 /= total
    rouge_2 /= total
    rouge_l /= total
    bleu /= total
    return {
        'rouge-1': rouge_1,
        'rouge-2': rouge_2,
        'rouge-l': rouge_l,
        'bleu': bleu,
    }

def train():
    set_seed()
    for fold in range(5):
        # 加载数据集
        data_root_path = 'data/mawps_norm/'
        train_data = load_data(data_root_path + 'MAWPS_' + str(fold) + '_keys_cand_train.jsonl')
        dev_data = load_data(data_root_path + 'MAWPS_' + str(fold) + '_keys_cand_test.jsonl')
        # train_data = train_data[:100]
        # dev_data = dev_data[:10]

        config = AutoConfig.from_pretrained(pretrain_model_path)
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
        cached = {}
        for d in train_data:
            key = tokenizer.encode(' '.join(d['keys']))[1:-1]
            equ = tokenizer.encode(''.join(d['postfix']))
            tgt = tokenizer.encode(d['text'], max_length=max_text_len)
            cached[d['id']] = [key, equ, tgt]
        
        train_batches = []
        for d in train_data:
            temp = copy.deepcopy(cached[d['id']])
            temp.append(cached[d['cand']][-1])
            train_batches.append(temp)

        dev_batches = []
        for d in dev_data:
            key = tokenizer.encode(' '.join(d['keys']))[1:-1]
            equ = tokenizer.encode(''.join(d['postfix']))
            temp = [key, equ]
            temp.append(cached[d['cand']][-1])
            dev_batches.append((temp, d))

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
                key_ids, equ_ids, cand_ids, text_ids = [], [], [], []
                max_key = max([len(x[0]) for x in pair])
                max_equ = max([len(x[1]) for x in pair])
                max_text = max([len(x[2]) for x in pair])
                max_cand = max([len(x[-1]) for x in pair])
                for _, p in enumerate(pair):
                    key, equ, text, cand = p
                    key_ids.append(key + [tokenizer.pad_token_id] * (max_key-len(key)))
                    equ_ids.append(equ + [tokenizer.pad_token_id] * (max_equ-len(equ)))
                    text_ids.append(text + [tokenizer.pad_token_id] * (max_text-len(text)))
                    cand_ids.append(cand + [tokenizer.pad_token_id] * (max_cand-len(cand)))
                key_ids = torch.tensor(key_ids, dtype=torch.long)
                equ_ids = torch.tensor(equ_ids, dtype=torch.long)
                text_ids = torch.tensor(text_ids, dtype=torch.long)
                cand_ids = torch.tensor(cand_ids, dtype=torch.long)
                key_pads = key_ids != tokenizer.pad_token_id
                key_pads = key_pads.float()
                equ_pads = equ_ids != tokenizer.pad_token_id
                equ_pads = equ_pads.float()
                text_pads = text_ids != tokenizer.pad_token_id
                text_pads = text_pads.float()
                cand_pads = cand_ids != tokenizer.pad_token_id
                cand_pads = cand_pads.float()
                batches.append((key_ids, key_pads, equ_ids, equ_pads, cand_ids, cand_pads, text_ids, text_pads))
            return batches
        
        class Generator(AutoRegressiveDecoder):
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

            def generate(self, key, equ, cand, topk=3):
                with torch.no_grad():
                    encoded_key = pretrain_model.model.encoder.embed_tokens(key)
                    encoded_equ = pretrain_model.model.encoder(input_ids=equ)[0]  # [1,s,h]
                    encoded_cand = pretrain_model.model.encoder(input_ids=cand)[0]  # [1,s,h]
                    encoded = torch.cat((encoded_key, encoded_equ, encoded_cand[:, 0:1]), dim=1)
                output_ids, output_score = self.beam_search(encoded.cpu().detach().numpy(), topk)  # 基于beam search
                # return tokenizer.convert_ids_to_tokens([int(i) for i in output_ids]), output_score
                # return [tokenizer.decode(int(x)) for x in output_ids[1:-1]], output_score
                return tokenizer.decode(output_ids[1:-1]).lower(), output_score
     
        pretrain_model = AutoModelForSeq2SeqLM.from_pretrained(pretrain_model_path)
        pretrain_model.resize_token_embeddings(len(tokenizer))
        # pretrain_model.lm_head = nn.Linear(pretrain_model.lm_head.in_features*2, pretrain_model.lm_head.out_features)
        generator = Generator(start_id=tokenizer.bos_token_id, end_id=tokenizer.eos_token_id, maxlen=max_text_len)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pretrain_model.to(device)
        
        train_generator = data_generator(train_batches, batch_size)
        optimizer = AdamW(pretrain_model.parameters(), lr=lr, weight_decay=0.01)
        global_steps = len(train_generator) * epochs * 2
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
                key_ids, key_pads, equ_ids, equ_pads, cand_ids, cand_pads, text_ids, text_pads = batch
                encoded_key = pretrain_model.model.encoder.embed_tokens(key_ids)
                encoded_equ = pretrain_model.model.encoder(input_ids=equ_ids, attention_mask=equ_pads)[0]
                encoded_cand = pretrain_model.model.encoder(input_ids=cand_ids, attention_mask=cand_pads)[0]
                encoded1 = torch.cat((encoded_key, encoded_equ, encoded_cand[:, 0:1]), dim=1)
                decoder_hidden_state1 = pretrain_model.model.decoder(encoder_hidden_states=encoded1, input_ids=text_ids, attention_mask=text_pads)[0]
                logits1 = pretrain_model.lm_head(decoder_hidden_state1)
                loss_fct = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, reduction="sum")
                shift_logits1 = logits1[:, :-1].contiguous()
                shift_labels1 = text_ids[:, 1:].contiguous()
                loss1 = loss_fct(shift_logits1.view(-1, shift_logits1.size(-1)), shift_labels1.view(-1))
                loss1 /= shift_labels1.size(0)
                loss1.backward()
                # torch.nn.utils.clip_grad_norm_(solver.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()             

                encoded_text = pretrain_model.model.encoder(input_ids=text_ids, attention_mask=text_pads)[0]
                encoded2 = encoded_text
                decoder_hidden_state2 = pretrain_model.model.decoder(encoder_hidden_states=encoded2, input_ids=equ_ids, attention_mask=equ_pads)[0]
                logits2 = pretrain_model.lm_head(decoder_hidden_state2)
                loss_fct = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, reduction="sum")
                shift_logits2 = logits2[:, :-1].contiguous()
                shift_labels2 = equ_ids[:, 1:].contiguous()
                loss2 = loss_fct(shift_logits2.view(-1, shift_logits2.size(-1)), shift_labels2.view(-1))
                loss2 /= shift_labels2.size(0)
                loss2.backward()
                # torch.nn.utils.clip_grad_norm_(solver.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()                

                loss = loss1 + loss2
                loss_total += loss.item()

            loss_total /= len(train_generator)
            log.write("epoch:" + str(e)+ "\tloss:" + str(loss_total) + "\n")
            logger.info(f"epoch: {e} - loss: {loss_total}")
            
            if (e > 0 and e % 10 == 0) or epochs - e < 5:
                pretrain_model.eval()
                results = []
                hyp_refs = []

                bar = tqdm(enumerate(dev_batches), total=len(dev_batches))
                for _,([key, equ, cand], d) in bar:
                    key_ids = torch.tensor([key], dtype=torch.long)
                    key_ids = key_ids.to(device)
                    equ_ids = torch.tensor([equ], dtype=torch.long)
                    equ_ids = equ_ids.to(device)
                    cand_ids = torch.tensor([cand], dtype=torch.long)
                    cand_ids = cand_ids.to(device)
                    res, score = generator.generate(key_ids, equ_ids, cand_ids)
                    text = tokenizer.decode(tokenizer.encode(d['text'])[1:-1]).lower()
                    hyp_refs.append((text, res))
                    val = evaluate([(text, res)])
                    temp = copy.deepcopy(val)
                    temp['id'] = d['id']
                    temp['keys'] = ' '.join(d['keys'])
                    temp['postfix'] = ' '.join(d['postfix'])
                    temp['text'] = d['text']
                    temp['res'] = res
                    temp['score'] = score
                    results.append(temp)
                
                value = evaluate(hyp_refs)
                f = open('result_mawps/results_dev_' + str(fold) + '.jsonl', 'w')
                for d in results:
                    json.dump(d, f, ensure_ascii=False)
                    f.write("\n")
                f.close()
                
                log.write("epoch:" + str(e)+ "\tval:" + str(value) + "\n")
                print("epoch:" + str(e)+ "\tval:" + str(value) + "\n")

                pretrain_model.save_pretrained('result_mawps/fold_' + str(fold) +'/models')
                tokenizer.save_pretrained('result_mawps/fold_' + str(fold) +'/models')
        log.close()

# train()