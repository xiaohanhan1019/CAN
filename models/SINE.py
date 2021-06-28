import torch
import torch.nn as nn


class SINE(torch.nn.Module):
    def __init__(self, config):
        super(SINE, self).__init__()
        self.item_num = config["item_num"]
        self.hidden_size = config['hidden_size']
        self.l = config['l']
        self.k = config['k']
        self.lambdaa = config['lambda']
        self.max_seq_length = config['seq_length']
        self.tao = 0.1

        self.item_embedding = torch.nn.Embedding(self.item_num, self.hidden_size, padding_idx=0)
        self.position_embedding = torch.nn.Embedding(self.max_seq_length + 1, self.hidden_size, padding_idx=0)
        self.prototype_embedding = torch.nn.Embedding(self.l, self.hidden_size)

        # self.attentive
        self.w1 = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.w2 = torch.nn.Linear(self.hidden_size, 1, bias=False)

        # intention assignment
        self.w3 = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.layer_norm1 = torch.nn.LayerNorm(self.hidden_size)
        self.layer_norm2 = torch.nn.LayerNorm(self.hidden_size)

        # attention weighting
        self.w4 = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.w5 = torch.nn.Linear(self.hidden_size, self.k, bias=False)

        # interest embedding generation
        self.layer_norm3 = torch.nn.LayerNorm(self.hidden_size)

        # interest aggregation module
        self.w6 = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.w7 = torch.nn.Linear(self.hidden_size, 1, bias=False)
        self.layer_norm4 = torch.nn.LayerNorm(self.hidden_size)

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
        seq = data['seq']
        batch_size = seq.size(0)
        mask = (seq != 0).float()  # (b, seq)

        item_embedding = self.item_embedding(seq)  # (b, seq, dim)
        self_attentive_score = self.w2(torch.tanh(self.w1(item_embedding))).squeeze(2)  # (b, seq)
        self_attentive_score = self_attentive_score - (1 - mask) * 10000
        self_attentive_alpha = torch.softmax(self_attentive_score, dim=-1)  # (b, seq)
        virtual_concept_vector = (self_attentive_alpha.unsqueeze(1) @ item_embedding).squeeze(1)  # (b, dim)

        # prototype
        prototype = self.prototype_embedding.weight.unsqueeze(0)  # (1, l, dim)
        s = (virtual_concept_vector.unsqueeze(1) @ prototype.transpose(1, 2)).squeeze(1)  # (b, l)
        rank = torch.argsort(-s, dim=1)[:, :self.k]  # (b, k)
        rank_idx = rank.unsqueeze(2).expand(batch_size, self.k, self.hidden_size)  # (b, k, dim)
        chosen_prototype = prototype.expand(batch_size, -1, -1).gather(dim=1, index=rank_idx)  # (b, k, dim)
        chosen_s = s.gather(dim=1, index=rank)  # (b, k)
        chosen_prototype = (torch.sigmoid(chosen_s).unsqueeze(2)) * chosen_prototype  # (b, k, dim)

        # intention assignment
        p_kt_score = self.layer_norm1(self.w3(item_embedding)) @ self.layer_norm2(chosen_prototype).transpose(1, 2)
        p_kt = torch.softmax(p_kt_score, dim=-1)  # (b, seq, k)
        p_kt = p_kt * mask.unsqueeze(2).expand(-1, -1, self.k)

        # attention weighting
        position = self.max_seq_length - torch.arange(0, self.max_seq_length).unsqueeze(0).expand_as(seq).to(
            seq.device)
        pos_embedding = self.position_embedding(position)
        embedding = pos_embedding + item_embedding
        a_k_score = self.w5(torch.tanh(self.w4(embedding))).transpose(1, 2)  # (b, k, seq)
        a_k = torch.softmax(a_k_score, dim=-1)  # (b, k, seq)
        a_k = a_k * mask.unsqueeze(1).expand(-1, self.k, -1)
        a_k = a_k.transpose(1, 2)  # (b, seq, k)

        # interest embedding generation
        p = p_kt * a_k  # (b, seq, k)
        interesting_embedding = self.layer_norm3(p.transpose(1, 2) @ embedding)  # (b, k, dim)

        # interest aggregation module
        xu = p_kt @ chosen_prototype  # (b, seq, dim)
        c_apt_score = self.w7(torch.tanh(self.w6(xu))).squeeze(2)  # (b, seq)
        c_apt_score = c_apt_score - (1 - mask) * 10000
        c_apt = torch.softmax(c_apt_score, dim=-1).unsqueeze(1)
        c_apt = self.layer_norm4((c_apt @ xu).squeeze(1))  # (b, dim)
        e_score = (c_apt.unsqueeze(1) @ interesting_embedding.transpose(1, 2)).squeeze(1)  # (b, k)
        e = torch.softmax(e_score / self.tao, dim=-1)  # (b, k)
        final_embedding = (e.unsqueeze(1) @ interesting_embedding).squeeze(1)  # (b, dim)

        scores = final_embedding @ self.item_embedding.weight.T

        return scores, self.get_loss()

    def get_loss(self):
        prototype_embeddings = self.prototype_embedding.weight  # (l, dim)
        mean_prototype = torch.mean(self.prototype_embedding.weight, dim=0).unsqueeze(0)
        M = (prototype_embeddings - mean_prototype) @ (prototype_embeddings - mean_prototype).T / self.hidden_size
        Lc = (M.norm() - M.diag().norm()) * 0.5
        return self.lambdaa * Lc
