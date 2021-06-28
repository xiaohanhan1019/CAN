import torch
from torch import nn

torch.set_printoptions(profile="full")
from torch.nn.init import xavier_normal_, constant_
import torch.nn.functional as F

import torch
import math
import copy


class TransformerEncoder(torch.nn.Module):
    def __init__(self, n_layers, n_heads, hidden_size, dropout_prob):
        super(TransformerEncoder, self).__init__()
        layer = TransformerLayer(n_heads, hidden_size, dropout_prob)
        self.layer = torch.nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])

    def forward(self, hidden_states, attention_mask):
        all_encoder_layers = []
        all_attention_weights = []
        for layer_module in self.layer:
            hidden_states, attention_weights = layer_module(hidden_states, attention_mask)
            all_encoder_layers.append(hidden_states)
            all_attention_weights.append(attention_weights)
        return all_encoder_layers, all_attention_weights


class TransformerLayer(torch.nn.Module):
    def __init__(self, n_heads, hidden_size, dropout_prob):
        super(TransformerLayer, self).__init__()
        self.multi_head_attention = MultiHeadSelfAttention(n_heads, hidden_size, dropout_prob)
        self.feed_forward = FeedForward(hidden_size, dropout_prob)

    def forward(self, hidden_states, attention_mask):
        attention_output, attention_weights = self.multi_head_attention(hidden_states, attention_mask)
        feedforward_output = self.feed_forward(attention_output)
        return feedforward_output, attention_weights


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
        q = torch.selu(self.query_linear(q))  # (b, seq, dim)
        k = torch.selu(self.key_linear(k))  # (b, seq, dim)

        q = self.transpose_for_scores(q)  # (b, head, seq, head_size)
        k = self.transpose_for_scores(k)  # (b, head, seq, head_size)
        v = self.transpose_for_scores(v)  # (b, head, seq, head_size)

        # attention_scores = inner_cosine_product(q, k.transpose(-1, -2)) / math.sqrt(
        #     self.attention_head_size)  # (b, head, seq, seq)
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

        return v, attention_scores


class CapsuleNetwork(torch.nn.Module):
    def __init__(self, hidden_size, num_interest, iter_times=3):
        super(CapsuleNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.num_interest = num_interest
        self.iter_times = iter_times

        self.S_matrix = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=False)

    @staticmethod
    def squash(s):
        s_norm = torch.norm(s, p=2, dim=-1, keepdim=True)
        s_norm2 = torch.pow(s_norm, 2)
        v = (s_norm2 / (1.0 + s_norm2)) * (s / (s_norm + 1e-12))
        return v

    def forward(self, item_embedding, mask):
        """
        :param item_embedding: (b, seq, dim)
        :param mask: (b, seq)
        """
        batch_size = item_embedding.size(0)
        seq_length = item_embedding.size(1)

        item_embedding = self.S_matrix(item_embedding)  # (b, seq, dim)
        item_embedding = item_embedding.unsqueeze(2).expand(batch_size, seq_length, self.num_interest, self.hidden_size)
        item_embedding = item_embedding.transpose(1, 2)
        item_embedding = item_embedding.detach()  # (b, interest, seq, dim)

        capsule_b = torch.normal(0, 1.0, (batch_size, self.num_interest, seq_length),
                                 device=mask.device, requires_grad=False)  # (b, interest, seq)

        # iteration
        interest_capsule = None
        mask = mask.unsqueeze(1).expand(batch_size, self.num_interest, seq_length)
        for i in range(self.iter_times):
            capsule_b = capsule_b.masked_fill(mask == 0, -10000)  # (b, interest, seq)
            capsule_c = torch.softmax(capsule_b, dim=1)  # (b, interest, seq)

            interest_capsule = capsule_c.unsqueeze(2) @ item_embedding
            interest_capsule = interest_capsule.squeeze(2)  # (b, interest, dim)
            interest_capsule = self.squash(interest_capsule)  # (b, interest, dim)

            temp = (inner_cosine_product(item_embedding, interest_capsule.unsqueeze(2), w=1)).squeeze(3)
            capsule_b = capsule_b + temp

        return interest_capsule


def inner_cosine_product(x, y, w=1):
    """
    :param x: (b, seq, dim)
    :param y: (b, seq, dim)
    :param w: scalar
    :return (b, seq, seq)
    """
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return x @ y.transpose(-1, -2) * w


class CAN(torch.nn.Module):

    def __init__(self, config):
        super(CAN, self).__init__()

        self.n_items = config["item_num"]
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.hidden_size = config['hidden_size']
        self.dropout_prob = config['dropout_prob']
        self.max_seq_length = config['seq_length']

        self.n = config['n']  # memory item num(exclude the last n item)
        self.m = config['k'][1]  # m interests
        self.k = config['k'][0]  # choose k interests
        self.l = config['l']  # now item num
        self.threshold = config['threshold']  # threshold

        # embedding layer
        self.item_embedding = torch.nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.position_embedding = torch.nn.Embedding(self.l + 1, self.hidden_size, padding_idx=0)

        # capsule network
        self.capsule_network = CapsuleNetwork(hidden_size=self.hidden_size, num_interest=self.m, iter_times=3)
        self.capsule_linear = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        # transformer
        self.transformer = TransformerEncoder(n_layers=self.n_layers,
                                              n_heads=self.n_heads,
                                              hidden_size=self.hidden_size,
                                              dropout_prob=self.dropout_prob)

        # dropout
        self.dropout = torch.nn.Dropout(p=self.dropout_prob)

        # layer norm
        self.layer_norm = torch.nn.LayerNorm(self.hidden_size, eps=1e-12)

        # predict
        self.w0 = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=False)
        self.w1 = nn.Linear(self.hidden_size, 1, bias=False)
        self.predict_linear = torch.nn.Linear(self.hidden_size * 2, self.hidden_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, data):
        item_seq = data['seq']
        batch_size = item_seq.size(0)

        # memory_item
        memory_item = item_seq[:, :self.max_seq_length - self.n]  # (b, n)
        memory_item_mask = (memory_item != 0).float()  # (b, n)
        memory_item_embedding = self.item_embedding(memory_item)  # (b, n, dim)

        # now_item
        now_item = item_seq[:, -self.l:]  # (b, l)
        now_item_mask = (now_item != 0).float()  # (b, l)
        now_item_embedding = self.item_embedding(now_item)  # (b, l, dim)

        # capsule network
        interest_capsule_original = self.capsule_network(memory_item_embedding, memory_item_mask)  # (b, k, dim)
        interest_capsule_original, capsule_mask = self.choose_capsule(interest_capsule_original)  # (b, k)
        interest_capsule = self.capsule_linear(interest_capsule_original)

        # embedding concatenate
        position_ids = torch.arange(now_item.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = now_item.size(1) - position_ids
        position_ids = position_ids.unsqueeze(0).expand(batch_size, now_item.size(1))  # (b, l)
        position_embedding = self.position_embedding(position_ids)

        now_embedding = self.layer_norm(now_item_embedding + position_embedding)
        all_embedding = torch.cat((now_embedding, interest_capsule), dim=1)  # (b, k+l, dim)

        # transformer
        all_seq = torch.cat((now_item, capsule_mask), dim=-1)
        attention_mask = self.get_attention_mask(all_seq)  # (b, head, k+l, k+l)
        sa_embedding, sa_weight = self.transformer(hidden_states=self.dropout(all_embedding),
                                                   attention_mask=attention_mask)  # (b, k+l, dim)

        # predict
        # final_embedding = sa_embedding[-1][:, -self.k - 1, :]  # (b, dim)
        # final_embedding = self.predict_linear(final_embedding)

        # predict use all sa_embedding
        sa_embedding = sa_embedding[-1]  # (b, k+l, dim)
        last_item = sa_embedding[:, -self.k - 1, :]  # (b, dim)
        predict_mask = torch.cat((now_item_mask, capsule_mask), dim=-1)
        alpha = torch.cat((sa_embedding, last_item.unsqueeze(1).expand_as(sa_embedding)), dim=-1)  # (b, k+l, dim*2)
        alpha = self.w1(torch.relu(self.w0(alpha))).squeeze(2)  # (b, k+l)
        alpha = alpha + (1 - predict_mask) * -10000
        alpha = torch.softmax(alpha, dim=-1)  # (b, k+l)
        global_embedding = (alpha.unsqueeze(1) @ sa_embedding).squeeze(1)
        final_embedding = torch.cat((last_item, global_embedding), dim=-1)
        final_embedding = self.predict_linear(final_embedding)

        scores = final_embedding @ self.item_embedding.weight.T

        # return memory_item_embedding, capsule_mask, interest_capsule_original, sa_weight, alpha, scores, 0
        return scores, 0

    def choose_capsule(self, capsule):
        """
        :param capsule: (b, m, dim)
        """
        batch_size = capsule.size(0)
        prob = torch.sqrt(torch.sum(capsule ** 2, dim=-1))  # (b, m)

        rank = torch.argsort(-prob, dim=1)[:, :self.k]  # (b, k)
        rank_idx = rank.unsqueeze(2).expand(batch_size, self.k, self.hidden_size)  # (b, k, dim)
        chosen_capsule = capsule.gather(dim=1, index=rank_idx)  # (b, k, dim)
        chosen_prob = prob.gather(dim=1, index=rank)  # (b, k)

        mask = (chosen_prob > self.threshold).float()  # (b, k)
        return chosen_capsule, mask

    def get_attention_mask(self, seq):
        """
        get attention mask
        [0,0,0,1,2,3] -> [[0,0,0,1,1,1],
                          [0,0,0,1,1,1],
                          [0,0,0,1,1,1],
                          [0,0,0,0,1,1],
                          [0,0,0,1,0,1],
                          [0,0,0,1,1,0],
        """
        mask = (seq != 0).float()  # (b, seq)
        seq_len = mask.size(-1)
        mask = mask.unsqueeze(1).unsqueeze(2).expand(-1, self.n_heads, seq_len, -1)  # (b, head, seq, seq)

        diag_mask = (~torch.eye(seq_len).bool()).to(seq.device).float().unsqueeze(0).unsqueeze(0)  # (b, head, seq, seq)

        mask = mask * diag_mask
        mask = (1.0 - mask) * -10000.0  # (b, head, seq, seq)
        return mask
