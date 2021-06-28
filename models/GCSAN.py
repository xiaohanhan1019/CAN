import math
import numpy as np

import torch
from torch import nn
from torch.nn import Parameter
from torch.nn import functional as F
import copy


class TransformerEncoder(torch.nn.Module):
    def __init__(self, n_layers, n_heads, hidden_size, dropout_prob):
        super(TransformerEncoder, self).__init__()
        layer = TransformerLayer(n_heads, hidden_size, dropout_prob)
        self.layer = torch.nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])

    def forward(self, hidden_states, attention_mask):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class TransformerLayer(torch.nn.Module):
    def __init__(self, n_heads, hidden_size, dropout_prob):
        super(TransformerLayer, self).__init__()
        self.multi_head_attention = MultiHeadSelfAttention(n_heads, hidden_size, dropout_prob)
        self.feed_forward = FeedForward(hidden_size, dropout_prob)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.multi_head_attention(hidden_states, attention_mask)
        feedforward_output = self.feed_forward(attention_output)
        return feedforward_output


class FeedForward(torch.nn.Module):
    def __init__(self, hidden_size, dropout_prob):
        super(FeedForward, self).__init__()
        self.dense_1 = torch.nn.Linear(hidden_size, hidden_size)
        self.dense_2 = torch.nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = torch.nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = torch.nn.Dropout(dropout_prob)

    def forward(self, input_tensor):
        output = torch.relu(self.dense_1(input_tensor))
        output = self.dropout(self.dense_2(output))
        output = self.LayerNorm(output + input_tensor)
        return output


class MultiHeadSelfAttention(torch.nn.Module):
    def __init__(self, n_heads, hidden_size, dropout_prob):
        super(MultiHeadSelfAttention, self).__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, n_heads))

        self.num_attention_heads = n_heads
        self.attention_head_size = hidden_size // n_heads
        self.hidden_size = hidden_size

        self.query_linear = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.key_linear = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.value_linear = torch.nn.Linear(hidden_size, hidden_size, bias=False)

        self.dropout = torch.nn.Dropout(dropout_prob)

        self.dense = torch.nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = torch.nn.LayerNorm(hidden_size, eps=1e-12)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)  # (b, seq, head, head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)  # (b, head, seq, head_size)

    def forward(self, hidden_states, attention_mask):
        input_v = hidden_states
        q = k = v = hidden_states
        q = self.query_linear(q)  # (b, seq, dim)
        k = self.key_linear(k)  # (b, seq, dim)
        v = self.value_linear(v)  # (b, seq, dim)

        q = self.transpose_for_scores(q)  # (b, head, seq, head_size)
        k = self.transpose_for_scores(k)  # (b, head, seq, head_size)
        v = self.transpose_for_scores(v)  # (b, head, seq, head_size)

        attention_scores = q @ k.transpose(-1, -2) / math.sqrt(self.attention_head_size)  # (b, head, seq, seq)
        attention_scores = attention_scores + attention_mask

        attention_weights = self.dropout(torch.softmax(attention_scores, dim=-1))  # (b, head, seq, seq)
        v = attention_weights @ v  # (b, head, seq, head_size)
        v = v.permute(0, 2, 1, 3).contiguous()
        v_shape = v.size()[:-2] + (self.hidden_size,)
        v = v.view(*v_shape)  # (b, seq, dim)
        v = self.dense(v)
        v = self.dropout(v)
        v = self.LayerNorm(v + input_v)

        return v


class GNN(nn.Module):
    r"""Graph neural networks are well-suited for session-based recommendation,
    because it can automatically extract features of session graphs with considerations of rich node connections.
    """

    def __init__(self, embedding_size, step=1):
        super(GNN, self).__init__()
        self.step = step
        self.embedding_size = embedding_size
        self.input_size = embedding_size * 2
        self.gate_size = embedding_size * 3
        self.w_ih = Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh = Parameter(torch.Tensor(self.gate_size, self.embedding_size))
        self.b_ih = Parameter(torch.Tensor(self.gate_size))
        self.b_hh = Parameter(torch.Tensor(self.gate_size))

        self.linear_edge_in = nn.Linear(self.embedding_size, self.embedding_size, bias=True)
        self.linear_edge_out = nn.Linear(self.embedding_size, self.embedding_size, bias=True)

        # parameters initialization
        self._reset_parameters()

    def _reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.embedding_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def GNNCell(self, A, hidden):
        r"""Obtain latent vectors of nodes via gated graph neural network.

        Args:
            A (torch.FloatTensor): The connection matrix,shape of [batch_size, max_session_len, 2 * max_session_len]

            hidden (torch.FloatTensor): The item node embedding matrix, shape of
                [batch_size, max_session_len, embedding_size]

        Returns:
            torch.FloatTensor: Latent vectors of nodes,shape of [batch_size, max_session_len, embedding_size]

        """

        input_in = torch.matmul(A[:, :, :A.size(1)], self.linear_edge_in(hidden))
        input_out = torch.matmul(A[:, :, A.size(1): 2 * A.size(1)], self.linear_edge_out(hidden))
        # [batch_size, max_session_len, embedding_size * 2]
        inputs = torch.cat([input_in, input_out], 2)

        # gi.size equals to gh.size, shape of [batch_size, max_session_len, embdding_size * 3]
        gi = F.linear(inputs, self.w_ih, self.b_ih)
        gh = F.linear(hidden, self.w_hh, self.b_hh)
        # (batch_size, max_session_len, embedding_size)
        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)
        hy = (1 - inputgate) * hidden + inputgate * newgate
        return hy

    def forward(self, A, hidden):
        for i in range(self.step):
            hidden = self.GNNCell(A, hidden)
        return hidden


class GCSAN(torch.nn.Module):

    def __init__(self, config):
        super(GCSAN, self).__init__()

        # load parameters info
        self.n_items = config["item_num"]
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.hidden_size = config['hidden_size']
        self.dropout_prob = config['dropout_prob']
        self.max_seq_length = config['seq_length']

        self.step = config['step']
        self.device = config['device']
        self.weight = config['weight']

        # define layers and loss
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.gnn = GNN(self.hidden_size, self.step)
        self.self_attention = TransformerEncoder(n_layers=self.n_layers,
                                                 n_heads=self.n_heads,
                                                 hidden_size=self.hidden_size,
                                                 dropout_prob=self.dropout_prob)

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
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
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long().to(item_seq.device)

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def _get_slice(self, item_seq):
        items, n_node, A, alias_inputs = [], [], [], []
        max_n_node = item_seq.size(1)
        item_seq = item_seq.cpu().numpy()

        for u_input in item_seq:
            node = np.unique(u_input)
            items.append(node.tolist() + (max_n_node - len(node)) * [0])
            u_A = np.zeros((max_n_node, max_n_node))
            for i in np.arange(len(u_input) - 1):
                if u_input[i + 1] == 0:
                    break
                u = np.where(node == u_input[i])[0][0]
                v = np.where(node == u_input[i + 1])[0][0]
                u_A[u][v] = 1
            u_sum_in = np.sum(u_A, 0)
            u_sum_in[np.where(u_sum_in == 0)] = 1
            u_A_in = np.divide(u_A, u_sum_in)
            u_sum_out = np.sum(u_A, 1)
            u_sum_out[np.where(u_sum_out == 0)] = 1
            u_A_out = np.divide(u_A.transpose(), u_sum_out)
            u_A = np.concatenate([u_A_in, u_A_out]).transpose()
            A.append(u_A)

            alias_inputs.append([np.where(node == i)[0][0] for i in u_input])
        # The relative coordinates of the item node, shape of [batch_size, max_session_len]
        alias_inputs = torch.LongTensor(alias_inputs).to(self.device)
        # The connecting matrix, shape of [batch_size, max_session_len, 2 * max_session_len]
        A = torch.FloatTensor(A).to(self.device)
        # The unique item nodes, shape of [batch_size, max_session_len]
        items = torch.LongTensor(items).to(self.device)

        return alias_inputs, A, items

    def gather_indexes(self, output, gather_index):
        """Gathers the vectors at the specific positions over a mini-batch"""
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)

    def forward(self, data):
        item_seq = data['seq']
        item_seq_len = data['seq_length']

        assert 0 <= self.weight <= 1
        alias_inputs, A, items = self._get_slice(item_seq)
        hidden = self.item_embedding(items)
        hidden = self.gnn(A, hidden)
        alias_inputs = alias_inputs.view(-1, alias_inputs.size(1), 1).expand(-1, -1, self.hidden_size)
        seq_hidden = torch.gather(hidden, dim=1, index=alias_inputs)
        # fetch the last hidden state of last timestamp
        ht = self.gather_indexes(seq_hidden, item_seq_len - 1)
        a = seq_hidden
        attention_mask = self.get_attention_mask(item_seq)

        outputs = self.self_attention(hidden_states=a, attention_mask=attention_mask)
        output = outputs[-1]
        at = self.gather_indexes(output, item_seq_len - 1)
        seq_output = self.weight * at + (1 - self.weight) * ht
        scores = seq_output @ self.item_embedding.weight.T
        return scores, 0
