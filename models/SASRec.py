import torch
from torch import nn
import copy
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, hidden_size, dropout_prob):
        super(MultiHeadAttention, self).__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, n_heads))

        self.n_heads = n_heads
        self.hidden_size = hidden_size
        self.attention_head_size = hidden_size // n_heads

        self.query_linear = nn.Linear(hidden_size, hidden_size)
        self.key_linear = nn.Linear(hidden_size, hidden_size)
        self.value_linear = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout_prob)
        self.dense = torch.nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.n_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        input_v = hidden_states
        q = k = v = hidden_states
        mixed_query_layer = self.query_linear(q)
        mixed_key_layer = self.key_linear(k)
        mixed_value_layer = self.value_linear(v)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores = attention_scores + attention_mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        context_layer = self.dense(context_layer)
        context_layer = self.dropout(context_layer)
        hidden_states = self.LayerNorm(context_layer + input_v)

        return hidden_states


class FeedForward(nn.Module):
    def __init__(self, hidden_size, dropout_prob):
        super(FeedForward, self).__init__()
        self.dense_1 = nn.Linear(hidden_size, hidden_size)
        self.dense_2 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)

    def forward(self, input_tensor):
        hidden_states = self.dropout(torch.relu(self.dense_1(input_tensor)))
        hidden_states = self.dropout(self.dense_2(hidden_states))
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class TransformerLayer(nn.Module):
    def __init__(self, n_heads, hidden_size, dropout_prob):
        super(TransformerLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(n_heads, hidden_size, dropout_prob)
        self.feed_forward = FeedForward(hidden_size, dropout_prob)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.multi_head_attention(hidden_states, attention_mask)
        feedforward_output = self.feed_forward(attention_output)
        return feedforward_output


class TransformerEncoder(nn.Module):
    def __init__(self, n_layers=2, n_heads=1, hidden_size=100, dropout_prob=0):
        super(TransformerEncoder, self).__init__()
        layer = TransformerLayer(n_heads, hidden_size, dropout_prob)
        self.layer = nn.ModuleList([copy.deepcopy(layer)
                                    for _ in range(n_layers)])

    def forward(self, hidden_states, attention_mask):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class SASRec(torch.nn.Module):
    def __init__(self, config):
        super(SASRec, self).__init__()

        self.n_items = config["item_num"]
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.hidden_size = config['hidden_size']
        self.dropout_prob = config['dropout_prob']
        self.max_seq_length = config['seq_length']

        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length + 1, self.hidden_size)
        self.trm_encoder = TransformerEncoder(n_layers=self.n_layers,
                                              n_heads=self.n_heads,
                                              hidden_size=self.hidden_size,
                                              dropout_prob=self.dropout_prob)

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(self.dropout_prob)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def get_attention_mask(self, item_seq):
        """Generate left-to-right uni-directional attention mask for multi-head attention."""
        attention_mask = (item_seq > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        # mask for left-to-right unidirectional
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).to(item_seq.device)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long()

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def forward(self, data):
        item_seq = data['seq']

        position_ids = torch.arange(0, item_seq.shape[1]).unsqueeze(0).expand_as(item_seq).to(item_seq.device)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_embedding(item_seq)
        item_emb = self.dropout(item_emb)
        item_emb = self.LayerNorm(item_emb + position_embedding)

        extended_attention_mask = self.get_attention_mask(item_seq)

        trm_output = self.trm_encoder(hidden_states=item_emb,
                                      attention_mask=extended_attention_mask)
        output = trm_output[-1][:, -1, :]

        scores = output @ self.item_embedding.weight.T
        return scores, 0
