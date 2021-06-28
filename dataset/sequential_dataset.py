# -*- coding: UTF-8 -*-
import torch
import random
from torch.utils.data import DataLoader, Dataset
import pickle
from utils.path import ROOT_DIR


def get_data(dataset, seq_length=50, front_padding=True):
    seqs = pickle.load(open(f"{ROOT_DIR}/data/{dataset}/seq.txt", "rb"))

    train_user_id = []
    train_seq = []
    train_label = []

    valid_user_id = []
    valid_seq = []
    valid_label = []

    test_user_id = []
    test_seq = []
    test_label = []
    for user_id in seqs:
        seq = seqs[user_id]
        for i in range(1, len(seq)):
            if i == len(seq) - 2:
                valid_user_id += [user_id]
                valid_seq += [seq[:i]]
                valid_label += [seq[i]]
            elif i == len(seq) - 1:
                test_user_id += [user_id]
                test_seq += [seq[:i]]
                test_label += [seq[i]]
            else:
                train_user_id += [user_id]
                train_seq += [seq[:i]]
                train_label += [seq[i]]

    train_dataset = SequentialDataset(user=train_user_id,
                                      seq=train_seq,
                                      label=train_label,
                                      seq_length=seq_length,
                                      front_padding=front_padding)
    valid_dataset = SequentialDataset(user=valid_user_id,
                                      seq=valid_seq,
                                      label=valid_label,
                                      seq_length=seq_length,
                                      front_padding=front_padding)
    test_dataset = SequentialDataset(user=test_user_id,
                                     seq=test_seq,
                                     label=test_label,
                                     seq_length=seq_length,
                                     front_padding=front_padding)
    return train_dataset, valid_dataset, test_dataset


class SequentialDataset(Dataset):
    def __init__(self, user, seq, label, seq_length=50, front_padding=True):
        self.length = []
        # padding
        self.seq = torch.zeros((len(user), seq_length), dtype=torch.long)
        for i, s in enumerate(seq):
            l = min(len(s), seq_length)
            if front_padding:
                self.seq[i][-l:] = torch.tensor(s[-l:], dtype=torch.long)
                self.length += [l]
            else:
                self.seq[i][:l] = torch.tensor(s[-l:], dtype=torch.long)
                self.length += [l]
        # change to tensor
        self.user = torch.tensor(user, dtype=torch.long)
        self.labels = torch.tensor(label, dtype=torch.long)
        self.length = torch.tensor(self.length, dtype=torch.long)

    def __getitem__(self, idx):
        return self.user[idx], self.seq[idx], self.length[idx], self.labels[idx]

    def __len__(self):
        return len(self.user)

    def get_data_loader(self, device, batch_size=128, shuffle=True):
        pin_memory = device not in ["cpu", "CPU"]
        data_loader = DataLoader(self, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory)
        return data_loader


if __name__ == '__main__':
    train_dataset, valid_dataset, test_dataset = get_data('tmall', seq_length=50, front_padding=True)
    data_loader = train_dataset.get_data_loader(device="cpu", batch_size=16, shuffle=False)
    for i, d in enumerate(data_loader):
        print(d)
        if i == 15:
            break
