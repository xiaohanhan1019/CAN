import torch
from torch import nn
from torch.nn.init import xavier_uniform_, xavier_normal_


class GRU4Rec(torch.nn.Module):

    def __init__(self, config):
        super(GRU4Rec, self).__init__()

        self.n_items = config["item_num"]
        self.hidden_size = config['hidden_size']
        self.num_layers = config['num_layers']
        self.dropout_prob = config['dropout_prob']

        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.emb_dropout = nn.Dropout(self.dropout_prob)
        self.gru_layers = nn.GRU(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
        )
        self.dense = nn.Linear(self.hidden_size, self.hidden_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)

    def gather_indexes(self, output, gather_index):
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)

    def forward(self, data):
        item_seq = data['seq']
        item_seq_len = data['seq_length']

        item_seq_emb = self.item_embedding(item_seq)
        item_seq_emb_dropout = self.emb_dropout(item_seq_emb)
        gru_output, _ = self.gru_layers(item_seq_emb_dropout)
        seq_output = self.gather_indexes(gru_output, item_seq_len - 1)
        seq_output = torch.relu(self.dense(seq_output))

        scores = seq_output @ self.item_embedding.weight.T
        return scores, 0
