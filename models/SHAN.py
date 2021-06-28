import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import normal_, uniform_


class SHAN(torch.nn.Module):

    def __init__(self, config):
        super(SHAN, self).__init__()

        # load the dataset information
        self.n_items = config["item_num"]
        self.n_users = config["user_num"]

        # load the parameter information
        self.embedding_size = config["hidden_size"]
        self.short_item_length = config["short_item_length"]  # the length of the short session items

        # define layers and loss
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size, padding_idx=0)
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)

        self.long_w = nn.Linear(self.embedding_size, self.embedding_size)
        self.long_b = nn.Parameter(
            uniform_(
                tensor=torch.zeros(self.embedding_size),
                a=-np.sqrt(3 / self.embedding_size),
                b=np.sqrt(3 / self.embedding_size)
            ),
            requires_grad=True
        )
        self.long_short_w = nn.Linear(self.embedding_size, self.embedding_size)
        self.long_short_b = nn.Parameter(
            uniform_(
                tensor=torch.zeros(self.embedding_size),
                a=-np.sqrt(3 / self.embedding_size),
                b=np.sqrt(3 / self.embedding_size)
            ),
            requires_grad=True
        )

        self.relu = nn.ReLU()

        # init the parameter of the model
        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, nn.Embedding):
            normal_(module.weight.data, 0., 0.01)
        elif isinstance(module, nn.Linear):
            uniform_(module.weight.data, -np.sqrt(3 / self.embedding_size), np.sqrt(3 / self.embedding_size))
        elif isinstance(module, nn.Parameter):
            uniform_(module.data, -np.sqrt(3 / self.embedding_size), np.sqrt(3 / self.embedding_size))
            print(module.data)

    def forward(self, data):

        item_seq = data['seq']
        user = data['user_id']

        seq_item_embedding = self.item_embedding(item_seq)
        user_embedding = self.user_embedding(user)

        # get the mask
        mask = item_seq.data.eq(0)
        long_term_attention_based_pooling_layer = self.long_term_attention_based_pooling_layer(
            seq_item_embedding, user_embedding, mask
        )
        # batch_size * 1 * embedding_size

        short_item_embedding = seq_item_embedding[:, -self.short_item_length:, :]
        mask_long_short = mask[:, -self.short_item_length:]
        batch_size = mask_long_short.size(0)
        x = torch.zeros(size=(batch_size, 1)).eq(1).to(item_seq.device)
        mask_long_short = torch.cat([x, mask_long_short], dim=1)
        # batch_size * short_item_length * embedding_size
        long_short_item_embedding = torch.cat([long_term_attention_based_pooling_layer, short_item_embedding], dim=1)
        # batch_size * 1_plus_short_item_length * embedding_size

        long_short_item_embedding = self.long_and_short_term_attention_based_pooling_layer(
            long_short_item_embedding, user_embedding, mask_long_short
        )
        # batch_size * embedding_size

        scores = long_short_item_embedding @ self.item_embedding.weight.T

        return scores, 0

    def long_and_short_term_attention_based_pooling_layer(self, long_short_item_embedding, user_embedding, mask=None):
        """
        fusing the long term purpose with the short-term preference
        """
        long_short_item_embedding_value = long_short_item_embedding

        long_short_item_embedding = self.relu(self.long_short_w(long_short_item_embedding) + self.long_short_b)
        long_short_item_embedding = torch.matmul(long_short_item_embedding, user_embedding.unsqueeze(2)).squeeze(-1)
        # batch_size * seq_len
        if mask is not None:
            long_short_item_embedding.masked_fill_(mask, -1e9)
        long_short_item_embedding = nn.Softmax(dim=-1)(long_short_item_embedding)
        long_short_item_embedding = torch.mul(long_short_item_embedding_value,
                                              long_short_item_embedding.unsqueeze(2)).sum(dim=1)

        return long_short_item_embedding

    def long_term_attention_based_pooling_layer(self, seq_item_embedding, user_embedding, mask=None):
        """
        get the long term purpose of user
        """
        seq_item_embedding_value = seq_item_embedding

        seq_item_embedding = self.relu(self.long_w(seq_item_embedding) + self.long_b)
        user_item_embedding = torch.matmul(seq_item_embedding, user_embedding.unsqueeze(2)).squeeze(-1)
        # batch_size * seq_len
        if mask is not None:
            user_item_embedding.masked_fill_(mask, -1e9)
        user_item_embedding = nn.Softmax(dim=1)(user_item_embedding)
        user_item_embedding = torch.mul(seq_item_embedding_value,
                                        user_item_embedding.unsqueeze(2)).sum(dim=1, keepdim=True)
        # batch_size * 1 * embedding_size

        return user_item_embedding
