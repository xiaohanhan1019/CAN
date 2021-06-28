import torch
from torch import nn


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

            capsule_b = capsule_b + (item_embedding @ interest_capsule.unsqueeze(3)).squeeze(3)

        return interest_capsule


class MIND(torch.nn.Module):
    def __init__(self, config):
        super(MIND, self).__init__()

        self.n_items = config["item_num"]
        self.hidden_size = config['hidden_size']
        self.max_seq_length = config['seq_length']
        self.k = config['k']
        self.pow = config['pow']

        # embedding layer
        self.item_embedding = torch.nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        # capsule network
        self.capsule_network = CapsuleNetwork(hidden_size=self.hidden_size, num_interest=self.k, iter_times=3)
        # capsule linear
        self.capsule_linear = torch.nn.Linear(self.hidden_size, self.hidden_size)

    # def forward(self, data):
    #     item_seq = data['seq']
    #     batch_size = item_seq.size(0)
    #     mask = (item_seq != 0).float()
    #
    #     item_embedding = self.item_embedding(item_seq)  # (b, seq, dim)
    #     interest = self.capsule_network(item_embedding, mask)  # (b, k, dim)
    #     all_item = self.item_embedding.weight.unsqueeze(0).expand(batch_size, -1, -1)  # (b, item, dim)
    #
    #     attention_score = all_item @ interest.transpose(1, 2)  # (b, item, k)
    #     attention_score = torch.pow(attention_score, self.pow)
    #     attention_weight = torch.softmax(attention_score, dim=-1)  # (b, item, k)
    #     final_embedding = attention_weight @ interest  # (b, item, dim)
    #     score = torch.sum(final_embedding * all_item, dim=-1)
    #     return score, 0

    def forward(self, data):
        item_seq = data['seq']
        mask = (item_seq != 0).float()

        item_embedding = self.item_embedding(item_seq)  # (b, seq, dim)
        interest = self.capsule_network(item_embedding, mask)  # (b, k, dim)
        interest = torch.relu(self.capsule_linear(interest))
        last_item = self.item_embedding(item_seq[:, -1]).squeeze(1)  # (b, dim)

        attention_score = (last_item.unsqueeze(1) @ interest.transpose(1, 2)).squeeze(1)  # (b, k)
        attention_score = torch.pow(attention_score, self.pow)
        attention_weight = torch.softmax(attention_score, dim=-1)  # (b, k)
        final_embedding = (attention_weight.unsqueeze(1) @ interest).squeeze(1)  # (b, dim)
        score = final_embedding @ self.item_embedding.weight.T
        return score, 0
