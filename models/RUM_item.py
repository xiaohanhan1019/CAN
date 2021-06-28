import torch
import torch.nn as nn
import math


class RUM_item(torch.nn.Module):
    def __init__(self, config):
        super(RUM_item, self).__init__()
        self.config = config
        self.alpha = self.config["alpha"]
        self.k = self.config["k"]  # memory slots
        self.user_num = self.config["user_num"]
        self.item_num = self.config["item_num"]
        self.embedding_dim = self.config["embedding_dim"]

        self.item_embedding = torch.nn.Embedding(self.item_num, self.embedding_dim, padding_idx=0)
        self.user_embedding = torch.nn.Embedding(self.user_num, self.embedding_dim, padding_idx=0)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    # def forward(self, data):
    #     users = data['user_id']
    #     seq = data['seq']
    #     batch_size = seq.size(0)
    #     user_embedding = self.user_embedding(users)  # (b, dim)
    #     item_embedding = self.item_embedding.weight.unsqueeze(0).expand(batch_size, self.item_num, self.embedding_dim)  # (b, item_num, dim)
    #     memory = self.item_embedding(seq[:, -self.k:])  # (b, k, dim)
    
    #     mask = (seq[:, -self.k:] != 0).float().unsqueeze(1).expand(batch_size, self.item_num, self.k)  # (b, item_num, k)
    
    #     w = item_embedding @ memory.transpose(1, 2)  # (b, item_num, k)
    #     w = w.masked_fill(mask == 0, -10000)
    #     z = torch.softmax(w, dim=2)  # (b, m, k)
    #     pmu = z @ memory  # (b, item_num, dim)
    #     pu = user_embedding.unsqueeze(1) + self.alpha * pmu  # (b, item_num, dim)
    #     y = pu * item_embedding  # (b, item_num, dim)
    
    #     scores = y.sum(dim=2)
    
    #     return scores, 0

    def forward(self, data):
        users = data['user_id']
        seq = data['seq']
        batch_size = seq.size(0)
        user_embedding = self.user_embedding(users)  # (b, dim)
        item_embedding = self.item_embedding(seq[:, -1]).squeeze(1)  # (b, dim)
        memory = self.item_embedding(seq[:, -self.k:])  # (b, k, dim)

        mask = (seq[:, -self.k:] != 0).float().expand(batch_size, self.k)  # (b, k)

        w = (item_embedding.unsqueeze(1) @ memory.transpose(1, 2)).squeeze(1)  # (b, k)
        w = w.masked_fill(mask == 0, -10000)
        z = torch.softmax(w, dim=-1)  # (b, k)
        p = (z.unsqueeze(1) @ memory).squeeze(1)  # (b, dim)
        pu = user_embedding + self.alpha * p  # (b, dim)
        scores = pu @ self.item_embedding.weight.T

        return scores, 0
