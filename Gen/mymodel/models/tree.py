import copy
import torch
import torch.nn as nn

from .attention import MultiheadAttention, TransformerLayer
from transformers.activations import gelu_new as gelu_bert

class TreeNode:  # the class save the tree node
    def __init__(self, embedding, left_flag=False):
        self.embedding = embedding
        self.left_flag = left_flag

def copy_list(l):
    r = []
    if len(l) == 0:
        return r
    for i in l:
        if type(i) is list:
            r.append(copy_list(i))
        else:
            r.append(i)
    return r

class TreeBeam:  # the class save the beam node
    def __init__(self, score, node_stack, embedding_stack, left_childs, out):
        self.score = score
        self.embedding_stack = copy_list(embedding_stack)
        self.node_stack = copy_list(node_stack)
        self.left_childs = copy_list(left_childs)
        self.out = copy.deepcopy(out)

class TreeEmbedding:  # the class save the tree
    def __init__(self, embedding, terminal=False):
        self.embedding = embedding
        self.terminal = terminal

class Score(nn.Module):
    def __init__(self, config, input_size, hidden_size):
        super(Score, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # self.attn = nn.Linear(hidden_size + input_size, hidden_size)
        self.dropout_expand = nn.Dropout(config.hidden_dropout_prob)
        self.dropout_out = nn.Dropout(config.hidden_dropout_prob)
        self.norm_out = nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)
        self.lin_expand = nn.Linear(hidden_size + input_size, config.intermediate_size)
        self.lin_collapse = nn.Linear(config.intermediate_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, num_embeddings, num_mask=None):
        max_len = num_embeddings.size(1)
        # repeat_dims = [1] * hidden.dim()
        # repeat_dims[1] = max_len
        # hidden = hidden.repeat(*repeat_dims)  # B x O x H
        hidden = hidden.expand(-1, max_len, -1)
        # For each position of encoder outputs
        this_batch_size = num_embeddings.size(0)
        energy_in = torch.cat((hidden, num_embeddings), 2).view(-1, self.input_size + self.hidden_size)
        # score = self.score(torch.tanh(self.attn(energy_in)))  # (B x O) x 1
        score = self.lin_collapse(self.dropout_expand(gelu_bert(self.lin_expand(energy_in))))
        score = score + self.dropout_out(score)
        score = self.score(self.norm_out(score))
        score = score.squeeze(1)
        score = score.view(this_batch_size, -1)  # B x O
        if num_mask is not None:
            score = score.masked_fill_(num_mask.bool(), -float('inf'))
        return score

# class TreeAttn(nn.Module):
#     def __init__(self, input_size, hidden_size):
#         super(TreeAttn, self).__init__()
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.attn = nn.Linear(hidden_size + input_size, hidden_size)
#         self.score = nn.Linear(hidden_size, 1)

#     def forward(self, hidden, encoder_outputs, seq_mask=None):
#         max_len = encoder_outputs.size(0)

#         repeat_dims = [1] * hidden.dim()
#         repeat_dims[0] = max_len
#         hidden = hidden.repeat(*repeat_dims)  # S x B x H
#         this_batch_size = encoder_outputs.size(1)

#         energy_in = torch.cat((hidden, encoder_outputs), 2).view(-1, self.input_size + self.hidden_size)

#         score_feature = torch.tanh(self.attn(energy_in))
#         attn_energies = self.score(score_feature)  # (S x B) x 1
#         attn_energies = attn_energies.squeeze(1)
#         attn_energies = attn_energies.view(max_len, this_batch_size).transpose(0, 1)  # B x S
#         if seq_mask is not None:
#             attn_energies = attn_energies.masked_fill_(seq_mask.bool(), -float('inf'))
#         attn_energies = nn.functional.softmax(attn_energies, dim=1)  # B x S

#         return attn_energies.unsqueeze(1)

class Prediction(nn.Module):
    def __init__(self, config, hidden_size, op_nums, input_size):
        super().__init__()

        # Keep for reference
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.op_nums = op_nums

        # Define layers
        self.embedding_weight = nn.Parameter(torch.randn(1, input_size, hidden_size), requires_grad=True)

        # for Computational symbols and Generated numbers
        self.concat_l = nn.Linear(hidden_size, hidden_size)
        self.concat_r = nn.Linear(hidden_size * 2, hidden_size)
        self.concat_lg = nn.Linear(hidden_size, hidden_size)
        self.concat_rg = nn.Linear(hidden_size * 2, hidden_size)
        self.ops = nn.Linear(hidden_size, op_nums)

        # self.attn = TreeAttn(hidden_size, hidden_size)
        self.attn = MultiheadAttention(config)
        self.score = Score(config, hidden_size, hidden_size)
        self.dropout_node = nn.Dropout(config.hidden_dropout_prob)
        self.dropout_mem = nn.Dropout(config.hidden_dropout_prob)
        self.dropout_expand = nn.Dropout(config.hidden_dropout_prob)
        self.dropout_out = nn.Dropout(config.hidden_dropout_prob)
        self.dropout_num = nn.Dropout(config.hidden_dropout_prob)
        self.norm_mem = nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)
        self.norm_out = nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)
        self.lin_expand = nn.Linear(hidden_size, config.intermediate_size)
        self.lin_collapse = nn.Linear(config.intermediate_size, hidden_size)

    def forward(self, node_stacks, left_childs, encoder_outputs, num_pades, padding_hidden, seq_mask, mask_nums):
        current_embeddings = []

        for st in node_stacks:
            if len(st) == 0:
                current_embeddings.append(padding_hidden)
            else:
                current_node = st[-1]
                current_embeddings.append(current_node.embedding)

        current_node_temp = []
        for l, c in zip(left_childs, current_embeddings):
            if l is None:
                c = self.dropout_node(c)
                g = torch.tanh(self.concat_l(c))
                t = torch.sigmoid(self.concat_lg(c))
                current_node_temp.append(g * t)
            else:
                ld = self.dropout_node(l)
                c = self.dropout_node(c)
                g = torch.tanh(self.concat_r(torch.cat((ld, c), 1)))
                t = torch.sigmoid(self.concat_rg(torch.cat((ld, c), 1)))
                current_node_temp.append(g * t)

        current_node = torch.stack(current_node_temp)

        # current_embeddings = self.dropout(current_node)
        # current_attn = self.attn(current_embeddings.transpose(0, 1), encoder_outputs, seq_mask)
        # current_context = current_attn.bmm(encoder_outputs.transpose(0, 1))  # B x 1 x N
        current_context = self.attn(query=current_node, key_value=encoder_outputs.transpose(0, 1), key_ignorance_mask=seq_mask)
        current_node = current_node + self.dropout_mem(current_context)
        current_node = self.norm_mem(current_node)
        current_node = self.lin_collapse(self.dropout_expand(gelu_bert(self.lin_expand(current_node))))
        current_node = current_node + self.dropout_out(current_node)
        current_node = self.norm_out(current_node)

        # leaf_input = torch.cat((current_node, current_context), 2)
        # leaf_input = current_node.clone()
        # leaf_input = leaf_input.squeeze(1)
        # leaf_input = self.dropout(leaf_input)
        batch_size = current_node.size(0)
        # repeat_dims = [1] * self.embedding_weight.dim()
        # repeat_dims[0] = batch_size
        # embedding_weight = self.embedding_weight.repeat(*repeat_dims)  # B x input_size x N
        embedding_weight = self.embedding_weight.expand(batch_size, -1, -1)  # B x input_size x N
        embedding_weight = torch.cat((embedding_weight, num_pades), dim=1)  # B x O x N
        embedding_weight_ = self.dropout_num(embedding_weight)
        num_score = self.score(current_node, embedding_weight_, mask_nums)
        op_score = self.ops(current_node.squeeze(1))

        return num_score, op_score, current_node, embedding_weight
        # return num_score, op_score, current_node, current_context, embedding_weight


class GenerateNode(nn.Module):
    def __init__(self, config, hidden_size, embedding_size, op_nums):
        super(GenerateNode, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        self.embeddings = nn.Embedding(op_nums, embedding_size)
        self.dropout_node = nn.Dropout(config.hidden_dropout_prob)
        self.dropout_num = nn.Dropout(config.hidden_dropout_prob)
        self.generate_l = nn.Linear(hidden_size + embedding_size, hidden_size)
        self.generate_r = nn.Linear(hidden_size + embedding_size, hidden_size)
        self.generate_lg = nn.Linear(hidden_size + embedding_size, hidden_size)
        self.generate_rg = nn.Linear(hidden_size + embedding_size, hidden_size)

    def forward(self, node_embedding, node_label):
        node_label_ = self.embeddings(node_label)
        node_label = self.dropout_node(node_label_)
        node_embedding = node_embedding.squeeze(1)
        # current_context = current_context.squeeze(1)
        node_embedding = self.dropout_num(node_embedding)
        # current_context = self.em_dropout(current_context)

        l_child = torch.tanh(self.generate_l(torch.cat((node_embedding, node_label), 1)))
        l_child_g = torch.sigmoid(self.generate_lg(torch.cat((node_embedding, node_label), 1)))
        r_child = torch.tanh(self.generate_r(torch.cat((node_embedding, node_label), 1)))
        r_child_g = torch.sigmoid(self.generate_rg(torch.cat((node_embedding, node_label), 1)))
        l_child = l_child * l_child_g
        r_child = r_child * r_child_g
        return l_child, r_child, node_label_


class Merge(nn.Module):
    def __init__(self, config, hidden_size, embedding_size):
        super(Merge, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        self.dropout_tree1 = nn.Dropout(config.hidden_dropout_prob)
        self.dropout_tree2 = nn.Dropout(config.hidden_dropout_prob)
        self.dropout_node = nn.Dropout(config.hidden_dropout_prob)
        self.merge = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
        self.merge_g = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)

    def forward(self, node_embedding, sub_tree_1, sub_tree_2):
        sub_tree_1 = self.dropout_tree1(sub_tree_1)
        sub_tree_2 = self.dropout_tree2(sub_tree_2)
        node_embedding = self.dropout_node(node_embedding)

        sub_tree = torch.tanh(self.merge(torch.cat((node_embedding, sub_tree_1, sub_tree_2), 1)))
        sub_tree_g = torch.sigmoid(self.merge_g(torch.cat((node_embedding, sub_tree_1, sub_tree_2), 1)))
        sub_tree = sub_tree * sub_tree_g
        return sub_tree

class TreeDecoder(nn.Module):
    def __init__(self, config, op_size, constant_size):
        super().__init__()
        # self.dropout = config.hidden_dropout_prob
        self.predict = Prediction(config=config, hidden_size=config.hidden_size, op_nums=op_size, input_size=constant_size)
        self.generate = GenerateNode(config=config, hidden_size=config.hidden_size, embedding_size=config.hidden_size, op_nums=op_size)
        self.merge = Merge(config=config, hidden_size=config.hidden_size, embedding_size=config.hidden_size)