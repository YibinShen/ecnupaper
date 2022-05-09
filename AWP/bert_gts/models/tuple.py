import torch
from torch import nn

from .attention import TransformerLayer, MultiheadAttentionWeights, MultiheadAttention, Squeeze, AveragePooling
from .util import PositionalEncoding, get_embedding_without_pad, apply_module_dict, apply_across_dim


def mask_forward(sz, diagonal=1):
    return torch.ones(sz, sz, dtype=torch.bool, requires_grad=False).triu(diagonal=diagonal).contiguous()

def masked_cross_entropy(logits, target, mask):
    target[~mask] = 0
    logits_flat = logits.reshape(-1, logits.size(-1))
    target_flat = target.reshape(-1, 1)
    losses_flat = torch.gather(logits_flat, index=target_flat, dim=1)
    losses = losses_flat.reshape(*target.size())
    losses = losses * mask.float()
    loss = losses.sum() / logits.size(0)
    return loss

class BaseDecoder(nn.Module):
    def __init__(self, config, operator_size, constant_size):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_size
        self.embedding_dim = config.hidden_size
        self.num_hidden_layers = 1
        self.layernorm_eps = config.layer_norm_eps
        self.num_heads = config.num_attention_heads
        self.num_pointer_heads = config.num_attention_heads
        self.operator_size = operator_size
        self.constant_size = constant_size

        """ Embedding layers """
        self.operator_word_embedding = nn.Embedding(self.operator_size, self.hidden_dim)
        self.operator_pos_embedding = PositionalEncoding(self.hidden_dim)
        self.dropout_operator = nn.Dropout(config.hidden_dropout_prob)
        self.dropout_operand = nn.Dropout(config.hidden_dropout_prob)
        # self.operand_source_embedding = nn.Embedding(3, self.hidden_dim)

        """ Scalar parameters """
        # degrade_factor = self.embedding_dim ** 0.5
        # self.operator_pos_factor = nn.Parameter(torch.tensor(degrade_factor), requires_grad=True)
        # self.operand_source_factor = nn.Parameter(torch.tensor(degrade_factor), requires_grad=True)

        """ Layer Normalizations """
        # self.operator_norm = nn.LayerNorm(self.hidden_dim, eps=self.layernorm_eps)
        # self.operand_norm = nn.LayerNorm(self.hidden_dim, eps=self.layernorm_eps)

        """ Linear Transformation """
        self.embed_to_hidden = nn.Linear(self.hidden_dim * 3, self.hidden_dim)

        """ Transformer layer """
        self.shared_decoder_layer = TransformerLayer(config)
        # self.attn = MultiheadAttention(config)

        """ Output layer """
        self.operator_out = nn.Linear(self.hidden_dim, self.operator_size)
        # # Softmax layers, which can handle infinity values properly (used in Equation 6, 10)
        # self.softmax = LogSoftmax(dim=-1)

    def _build_operand_embed(self, ids, mem_pos, nums):
        raise NotImplementedError()

    def _build_decoder_input(self, ids, nums):
        operator = get_embedding_without_pad(self.operator_word_embedding, ids.select(dim=-1, index=0))
        operator = self.dropout_operator(operator)
        operator_pos = self.operator_pos_embedding(ids.shape[1])
        # operator = self.operator_norm(operator * self.operator_pos_factor + operator_pos.unsqueeze(0)).unsqueeze(2)
        operator = operator.unsqueeze(2)

        # operand = get_embedding_without_pad(self.operand_source_embedding, ids[:, :, 1::2]) * self.operand_source_factor
        # operand += self._build_operand_embed(ids, operator_pos, nums)
        # operand = self.operand_norm(operand)
        operand = self._build_operand_embed(ids, operator_pos, nums)
        operand = self.dropout_operator(operand)

        operator_operands = torch.cat([operator, operand], dim=2).contiguous().flatten(start_dim=2)
        return self.embed_to_hidden(operator_operands)

    def _build_decoder_context(self, embedding, embedding_pad=None, text=None, text_pad=None):
        mask = mask_forward(embedding.shape[1]).to(embedding.device)
        output = embedding
        for _ in range(self.num_hidden_layers):
            output = self.shared_decoder_layer(target=output, memory=text, target_attention_mask=mask,
                                               target_ignorance_mask=embedding_pad, memory_ignorance_mask=text_pad)

        return output

    def _forward_single(self, text=None, text_pad=None, text_num=None, equation=None):
        operator_ids = equation.select(dim=2, index=0)
        output = self._build_decoder_input(ids=equation, nums=text_num)
        output_pad = operator_ids == -1

        # # Ignore the result of equality at the function output
        output_not_usable = output_pad.clone()
        # output_not_usable[:, :-1].masked_fill_(operator_ids[:, 1:] == FUN_EQ_SGN_ID, True)
        # # We need offset '1' because 'function_word' is input and output_not_usable is 1-step shifted output.

        output = self._build_decoder_context(embedding=output, embedding_pad=output_pad, text=text, text_pad=text_pad)
        operator_out = self.operator_out(output)

        if not self.training:
            operator_out[:, :, self.operator_size-2] = -1e12
            # Can end after equation formed, i.e. END_EQN is available when the input is EQ_SGN.
            # operator_out[:, :, self.operator_size-1].masked_fill_(operator_ids != FUN_EQ_SGN_ID, NEG_INF)

        result = {'operator': torch.log_softmax(operator_out, dim=-1), '_out': output, '_not_usable': output_not_usable}
        return result


class TupleDecoder(BaseDecoder):
    def __init__(self, config, operator_size, constant_size):
        super().__init__(config, operator_size, constant_size)

        """ Operand embedding """
        self.constant_word_embedding = nn.Embedding(self.constant_size, self.hidden_dim)

        """ Output layer """
        self.operand_out = nn.ModuleList([
            nn.ModuleDict({
                '0_attn': MultiheadAttentionWeights(config),
                '1_mean': Squeeze(dim=-1) if self.num_pointer_heads == 1 else AveragePooling(dim=-1)
            }) for _ in range(2)
        ])

    def _build_operand_embed(self, ids, mem_pos, nums):
        operand_source = ids[:, :, 1::2]
        operand_value = ids[:, :, 2::2]

        number_operand = operand_value.masked_fill(operand_source != 1, -1)
        operand = torch.stack([get_embedding_without_pad(nums[b], number_operand[b]) for b in range(ids.shape[0])], dim=0).contiguous()
        operand += get_embedding_without_pad(self.constant_word_embedding, operand_value.masked_fill(operand_source != 0, -1))
        prior_result_operand = operand_value.masked_fill(operand_source != 2, -1)
        operand += get_embedding_without_pad(mem_pos, prior_result_operand)
        return operand

    def _build_attention_keys(self, num, mem, num_pad=None, mem_pad=None):
        batch_sz = num.shape[0]
        const_sz = self.constant_size
        const_num_sz = const_sz + num.shape[1]
        const_key = self.constant_word_embedding.weight.unsqueeze(0).expand(batch_sz, const_sz, self.hidden_dim)
        key = torch.cat([const_key, num, mem], dim=1).contiguous()
        key_ignorance_mask = torch.zeros(key.shape[:2], dtype=torch.bool, device=key.device)
        if num_pad is not None:
            key_ignorance_mask[:, const_sz:const_num_sz] = num_pad
        if mem_pad is not None:
            key_ignorance_mask[:, const_num_sz:] = mem_pad

        attention_mask = torch.zeros(mem.shape[1], key.shape[1], dtype=torch.bool, device=key.device)
        attention_mask[:, const_num_sz:] = mask_forward(mem.shape[1], diagonal=0).to(key_ignorance_mask.device)

        return key, key_ignorance_mask, attention_mask

    def _forward_single(self, text=None, text_pad=None, text_num=None, text_numpad=None, equation=None):
        result = super()._forward_single(text, text_pad, text_num, equation)
        output = result.pop('_out')
        output_not_usable = result.pop('_not_usable')
        key, key_ign_msk, attn_msk = self._build_attention_keys(num=text_num, mem=output, num_pad=text_numpad, mem_pad=output_not_usable)

        for j, layer in enumerate(self.operand_out):
            score = apply_module_dict(layer, encoded=output, key=key, key_ignorance_mask=key_ign_msk, attention_mask=attn_msk)
            # result['operand_%s' % j] = self.softmax(score)
            result['operand_%s' % j] = torch.log_softmax(score, dim=-1)

        return result

    def _build_target_dict(self, text_num=None, equation=None):
        num_offset = self.constant_size
        mem_offset = num_offset + text_num.shape[1]
        targets = {'operator': equation.select(dim=-1, index=0)}
        for i in range(2):
            operand_source = equation[:, :, (i * 2 + 1)]
            operand_value = equation[:, :, (i * 2 + 2)].clamp_min(0)
            operand_value += operand_source.masked_fill(operand_source == 1, num_offset).masked_fill_(operand_source == 2, mem_offset)
            targets['operand_%s' % i] = operand_value

        return targets
    
    def forward(self, text, text_pad, text_num, text_numpad, equation=None):
        if self.training:
            outputs = self._forward_single(text, text_pad, text_num, text_numpad, equation)
            with torch.no_grad():
                targets = self._build_target_dict(text_num, equation)
                return outputs, targets
        else:
            with torch.no_grad():
                tensor = {}
                tensor['text'] = text
                tensor['text_pad'] = text_pad
                tensor['text_num'] = text_num
                tensor['text_numpad'] = text_numpad
                tensor['equation'] = equation
                outputs = apply_across_dim(self._forward_single, dim=1, shared_keys={'text', 'text_pad', 'text_num', 'text_numpad'}, **tensor)
        return outputs