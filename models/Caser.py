import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.init import normal_, xavier_normal_, constant_


class Caser(torch.nn.Module):
    def __init__(self, config):
        super(Caser, self).__init__()

        # load parameters info
        self.n_users = config['user_num']
        self.n_items = config['item_num']
        self.embedding_size = config['embedding_size']
        self.n_h = config['n_h']
        self.n_v = config['n_v']
        self.dropout_prob = config['dropout_prob']
        self.max_seq_length = config['seq_length']

        # define layers and loss
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size, padding_idx=0)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size, padding_idx=0)

        # vertical conv layer
        self.conv_v = nn.Conv2d(in_channels=1, out_channels=self.n_v, kernel_size=(self.max_seq_length, 1))

        # horizontal conv layer
        lengths = [i + 1 for i in range(self.max_seq_length)]
        self.conv_h = nn.ModuleList(
            [nn.Conv2d(in_channels=1, out_channels=self.n_h, kernel_size=(i, self.embedding_size)) for i in lengths])

        # fully-connected layer
        self.fc1_dim_v = self.n_v * self.embedding_size
        self.fc1_dim_h = self.n_h * len(lengths)
        fc1_dim_in = self.fc1_dim_v + self.fc1_dim_h
        self.fc1 = nn.Linear(fc1_dim_in, self.embedding_size)
        self.fc2 = nn.Linear(self.embedding_size + self.embedding_size, self.embedding_size)

        self.dropout = nn.Dropout(self.dropout_prob)
        self.ac_conv = nn.ReLU()
        self.ac_fc = nn.ReLU()

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            normal_(module.weight.data, 0, 1.0 / module.embedding_dim)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def forward(self, data):
        item_seq = data['seq']
        user = data['user_id']

        # Embedding Look-up
        # use unsqueeze() to get a 4-D input for convolution layers. (batchsize * 1 * max_length * embedding_size)
        item_seq_emb = self.item_embedding(item_seq).unsqueeze(1)
        user_emb = self.user_embedding(user).squeeze(1)

        # Convolutional Layers
        out, out_h, out_v = None, None, None
        # vertical conv layer
        if self.n_v:
            out_v = self.conv_v(item_seq_emb)
            out_v = out_v.view(-1, self.fc1_dim_v)  # prepare for fully connect

        # horizontal conv layer
        out_hs = list()
        if self.n_h:
            for conv in self.conv_h:
                conv_out = self.ac_conv(conv(item_seq_emb).squeeze(3))
                pool_out = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
                out_hs.append(pool_out)
            out_h = torch.cat(out_hs, 1)  # prepare for fully connect

        # Fully-connected Layers
        out = torch.cat([out_v, out_h], 1)
        # apply dropout
        out = self.dropout(out)
        # fully-connected layer
        z = self.ac_fc(self.fc1(out))
        x = torch.cat([z, user_emb], 1)
        seq_output = self.ac_fc(self.fc2(x))
        # the hidden_state of the predicted item, size:(batch_size * hidden_size)
        scores = seq_output @ self.item_embedding.weight.T
        return scores, 0
