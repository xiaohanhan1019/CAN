import torch
from torch import nn


class CapsuleNetwork(torch.nn.Module):
    def __init__(self, hidden_size, num_interest, seq_length, iter_times=3):
        super(CapsuleNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.num_interest = num_interest
        self.iter_times = iter_times

        self.s = torch.nn.Linear(self.hidden_size, self.hidden_size * self.num_interest)

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
        item_embedding = self.s(item_embedding)
        item_embedding = item_embedding.reshape(batch_size, seq_length, self.num_interest, self.hidden_size)
        item_embedding = item_embedding.transpose(1, 2)
        item_embedding = item_embedding.detach()  # (b, interest, seq, dim)

        capsule_b = torch.zeros((batch_size, self.num_interest, seq_length),
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

            capsule_b = capsule_b + (item_embedding @ interest_capsule.unsqueeze(3)).squeeze(3)

        return interest_capsule


class ComiRec_DR(torch.nn.Module):
    def __init__(self, config):
        super(ComiRec_DR, self).__init__()

        self.n_items = config["item_num"]
        self.hidden_size = config['hidden_size']
        self.max_seq_length = config['seq_length']
        self.k = config['k']

        # embedding layer
        self.item_embedding = torch.nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        # capsule network
        self.capsule_network = CapsuleNetwork(hidden_size=self.hidden_size, seq_length=self.max_seq_length,
                                              num_interest=self.k, iter_times=3)

    def forward(self, data):
        item_seq = data['seq']
        batch_size = item_seq.size(0)
        mask = (item_seq != 0).float()

        item_embedding = self.item_embedding(item_seq)  # (b, seq, dim)
        interest = self.capsule_network(item_embedding, mask)  # (b, k, dim)

        last_item = self.item_embedding(item_seq[:, -1]).squeeze(1)  # (b, dim)
        attention_score = (last_item.unsqueeze(1) @ interest.transpose(1, 2)).squeeze(1)  # (b, k)
        attention_score = torch.pow(attention_score, self.pow)
        attention_weight = torch.softmax(attention_score, dim=-1)  # (b, k)
        final_embedding = (attention_weight.unsqueeze(1) @ interest).squeeze(1)  # (b, dim)

        score = final_embedding @ self.item_embedding.weight.T
        return score, 0


class ComiRec_SA(torch.nn.Module):
    def __init__(self, config):
        super(ComiRec_SA, self).__init__()

        self.n_items = config["item_num"]
        self.hidden_size = config['hidden_size']
        self.max_seq_length = config['seq_length']
        self.k = config['k']

        # embedding layer
        self.item_embedding = torch.nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.position_embedding = torch.nn.Embedding(self.max_seq_length + 1, self.hidden_size, padding_idx=0)

        # self-attentive
        self.linear = torch.nn.Linear(self.hidden_size, self.hidden_size * 4)
        self.interest_linear = torch.nn.Linear(self.hidden_size * 4, self.k)

    def forward(self, data):
        item_seq = data['seq']
        batch_size = item_seq.size(0)
        seq_length = item_seq.size(1)
        mask = (item_seq != 0).float()

        item_embedding = self.item_embedding(item_seq)  # (b, seq, dim)
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device) + 1
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)
        embedding = item_embedding + position_embedding

        att_score = torch.tanh(self.linear(embedding))  # (b, seq, dim*4)
        att_score = self.interest_linear(att_score)  # (b, seq, k)
        att_score = att_score.transpose(1, 2)  # (b, k, seq)

        mask = mask.unsqueeze(1).expand(batch_size, self.k, seq_length)
        att_score = att_score.masked_fill(mask == 0, -10000)
        att_weight = torch.softmax(att_score, dim=-1)  # (b, k, seq)
        interest = att_weight @ item_embedding  # (b, k, dim)

        last_item = self.item_embedding(item_seq[:, -1]).squeeze(1)  # (b, dim)
        predict_att_score = (last_item.unsqueeze(1) @ interest.transpose(1, 2)).squeeze(1)  # (b, k)
        predict_att_weight = torch.softmax(predict_att_score, dim=-1)  # (b, k)

        rank = torch.argsort(-predict_att_weight, dim=1)[:, :1]  # (b, 1)
        rank_idx = rank.unsqueeze(2).expand(batch_size, 1, self.hidden_size)  # (b, 1, dim)
        final_embedding = interest.gather(dim=1, index=rank_idx).squeeze(1)  # (b, dim)

        score = final_embedding @ self.item_embedding.weight.T

        return score, 0
