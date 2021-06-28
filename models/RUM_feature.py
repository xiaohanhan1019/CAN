import torch
import torch.nn as nn
import math


class RUM_feature(torch.nn.Module):
    def __init__(self, config):
        super(RUM_feature, self).__init__()
        self.config = config
        self.alpha = self.config["alpha"]
        self.k = self.config["k"]  # memory slots
        self.user_num = self.config["user_num"]
        self.item_num = self.config["item_num"]
        self.embedding_dim = self.config["embedding_dim"]
        self.device = self.config['device']

        self.item_embedding = torch.nn.Embedding(self.item_num, self.embedding_dim, padding_idx=0)
        self.user_embedding = torch.nn.Embedding(self.user_num, self.embedding_dim, padding_idx=0)

        self.erase_linear = torch.nn.Linear(self.embedding_dim, self.embedding_dim)
        self.add_linear = torch.nn.Linear(self.embedding_dim, self.embedding_dim)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def read(self, items, memory):
        """
        read memory
        :param items: (b, dim)
        :param memory: (b, k, dim)
        """
        w = (items.unsqueeze(1) @ memory.transpose(1, 2)).squeeze(1)  # (b, k)
        z = torch.softmax(w, dim=1)  # (b, k)

        return z  # (b, k)

    def write(self, items, z, memory):
        """
        write memory
        :param items: (b, dim)
        :param z: (b, k) attention weights
        :param memory: (b, k, dim)
        """
        erase = torch.sigmoid(items)  # (b, dim)
        add = torch.tanh(items)  # (b, dim)
        new_memory = memory * (1 - z.unsqueeze(2) @ erase.unsqueeze(1)) + z.unsqueeze(2) @ add.unsqueeze(1)
        return new_memory

    # def forward(self, data):
    #     users = data['user_id']
    #     seq = data['seq']
    #     length = seq.shape[1]
    #     item_seq_len = data['seq_length']
    #     batch_size = seq.size(0)

    #     user_embedding = self.user_embedding(users)  # (b, dim)
    #     item_embedding = self.item_embedding(seq)  # (b, seq, dim)

    #     memories = [torch.nn.init.kaiming_uniform_(
    #         torch.randn((batch_size, self.k, self.embedding_dim), requires_grad=False)).to(self.device)
    #                 for _ in range(length + 1)]  # memory (b, k, dim) * (length + 1)

    #     for i in range(length):
    #         # read
    #         cur_memory = memories[i]  # (b, k, dim)
    #         z = self.read(item_embedding[:, i, :], cur_memory)
    #         # write
    #         memories[i + 1] = self.write(item_embedding[:, i, :], z, cur_memory)

    #     memory_embedding = torch.zeros((batch_size, self.k, self.embedding_dim), requires_grad=False).to(self.device)
    #     for i, l in enumerate(item_seq_len):
    #         memory_embedding[i] = memories[l][i, :, :]

    #     item_embedding = self.item_embedding.weight.unsqueeze(0).expand(batch_size, self.item_num, self.embedding_dim)  # (b, item_num, dim)
    #     w = item_embedding @ memory_embedding.transpose(1, 2)  # (b, item_num, k)
    #     z = torch.softmax(w, dim=-1)  # (b, k)
    #     pmu = z @ memory_embedding  # (b, item_num, dim)
    #     pu = user_embedding.unsqueeze(1) + self.alpha * pmu  # (b, item_num, dim)
    #     y = pu * item_embedding  # (b, item_num, dim)

    #     scores = y.sum(dim=2)

    #     return scores, 0

    def forward(self, data):
        users = data['user_id']
        seq = data['seq']
        length = seq.shape[1]
        item_seq_len = data['seq_length']
        batch_size = seq.size(0)

        user_embedding = self.user_embedding(users)  # (b, dim)
        item_embedding = self.item_embedding(seq)  # (b, seq, dim)

        memories = [torch.nn.init.kaiming_uniform_(
            torch.randn((batch_size, self.k, self.embedding_dim), requires_grad=False)).to(self.device)
                    for _ in range(length + 1)]  # memory (b, k, dim) * (length + 1)

        for i in range(length):
            # read
            cur_memory = memories[i]  # (b, k, dim)
            z = self.read(item_embedding[:, i, :], cur_memory)
            # write
            memories[i + 1] = self.write(item_embedding[:, i, :], z, cur_memory)

        memory_embedding = torch.zeros((batch_size, self.k, self.embedding_dim), requires_grad=False).to(self.device)
        for i, l in enumerate(item_seq_len):
            memory_embedding[i] = memories[l][i, :, :]

        item_embedding = self.item_embedding(seq[:, -1]).squeeze(1)  # (b, dim)
        w = (item_embedding.unsqueeze(1) @ memory_embedding.transpose(1, 2)).squeeze(1)  # (b, k)
        z = torch.softmax(w, dim=-1)  # (b, k)
        p = (z.unsqueeze(1) @ memory_embedding).squeeze(1)  # (b, dim)
        pu = user_embedding + self.alpha * p  # (b, dim)
        scores = pu @ self.item_embedding.weight.T

        return scores, 0
