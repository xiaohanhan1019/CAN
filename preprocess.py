# -*- coding: utf-8 -*-
import pandas as pd
from utils.path import ROOT_DIR
from datetime import datetime
import pickle
import random
import csv


def reindex(seqs):
    item_cnt = 1
    item2index = {}
    index2item = {}

    user_cnt = 1
    user2index = {}
    index2user = {}

    seq_greater_than_n_cnt = 0

    seq_length = []
    clicks = 0
    new_seqs = {}
    for user_id in seqs:
        seq = seqs[user_id]

        if user_id in user2index:
            user_id = user2index[user_id]
        else:
            user2index[user_id] = user_cnt
            index2user[user_cnt] = user_id
            user_cnt += 1
            user_id = user2index[user_id]

        new_seq = []
        for i in seq:
            if i in item2index:
                new_seq += [item2index[i]]
            else:
                item2index[i] = item_cnt
                index2item[item_cnt] = i
                item_cnt += 1
                new_seq += [item2index[i]]
        new_seqs[user_id] = new_seq
        clicks += len(new_seq)
        seq_length += [len(new_seq)]
        if len(new_seq) > 20:
            seq_greater_than_n_cnt += 1

    print(f"users: {len(new_seqs)}")
    print(f"items: {len(item2index)}")
    print(f"clicks: {clicks}")
    print(f"max seq length: {max(seq_length)}")
    print(f"min seq length: {min(seq_length)}")
    print(f"avg seq length: {sum(seq_length) / len(seq_length)}")
    print(f"avg item click: {clicks / len(item2index)}")
    print(f"sequence length greater than 20: {seq_greater_than_n_cnt}")

    pickle.dump(item2index, open(f"{ROOT_DIR}/data/{dataset}/item2index.txt", "wb"))
    pickle.dump(index2item, open(f"{ROOT_DIR}/data/{dataset}/index2item.txt", "wb"))
    pickle.dump(user2index, open(f"{ROOT_DIR}/data/{dataset}/user2index.txt", "wb"))
    pickle.dump(index2user, open(f"{ROOT_DIR}/data/{dataset}/index2user.txt", "wb"))
    pickle.dump(new_seqs, open(f"{ROOT_DIR}/data/{dataset}/seq.txt", "wb"))


def filter(all_seq, item_min_occurrence=5, seq_min_length=5, seq_max_length=1000):
    item_cnt = {}
    for user_id in all_seq:
        seq = all_seq[user_id]
        for i in seq:
            if i in item_cnt:
                item_cnt[i] += 1
            else:
                item_cnt[i] = 1

    all_seq_after_filter_item = {}
    for user_id in all_seq:
        seq = all_seq[user_id]
        new_seq = []
        for i in seq:
            if item_cnt[i] >= item_min_occurrence:
                new_seq += [i]
        all_seq_after_filter_item[user_id] = new_seq

    all_seq_after_filter_seq = {}
    for user_id in all_seq_after_filter_item:
        if seq_min_length <= len(all_seq_after_filter_item[user_id]):
            if len(all_seq_after_filter_item[user_id]) >= seq_max_length:
                all_seq_after_filter_seq[user_id] = all_seq_after_filter_item[user_id][-seq_max_length:]
            else:
                all_seq_after_filter_seq[user_id] = all_seq_after_filter_item[user_id]
    return all_seq_after_filter_seq


def preprocess_data(file, dataset, item_min_occurrence, seq_min_length, seq_max_length):
    all_seq = {}
    with open(file, "r") as f:
        next(f)
        for line in f.readlines():
            temp = line.split("\t")
            if dataset == 'ml-1m' or dataset == 'Amazon_Books' or dataset == 'ml-10m' or dataset == "Amazon_Electronics":
                timestamp = int(temp[3])
                user = temp[0]
                item = temp[1]
            elif dataset == 'tmall':
                timestamp = int(temp[4])
                user = temp[0]
                item = temp[2]
            elif dataset == 'steam':
                timestamp = int(temp[5])
                user = temp[0]
                item = temp[3]
            elif dataset == 'ta-feng':
                timestamp = int(temp[0])
                user = temp[1]
                item = temp[2]
            elif dataset == 'gowalla':
                timestamp = float(temp[2])
                user = temp[0]
                item = temp[1]
            else:
                raise NotImplementedError("we don't have that dataset")
            if user not in all_seq:
                all_seq[user] = [(timestamp, item)]
            else:
                all_seq[user] += [(timestamp, item)]
    for u in all_seq:
        temp = sorted(all_seq[u], key=lambda x: x[0])
        sorted_seq = []
        for t, i in temp:
            sorted_seq += [i]
        all_seq[u] = sorted_seq

    all_seq = filter(all_seq,
                     item_min_occurrence=item_min_occurrence,
                     seq_min_length=seq_min_length,
                     seq_max_length=seq_max_length)

    reindex(all_seq)


if __name__ == '__main__':
    dataset = 'ml-1m'
    dataset_to_file = {
        'ml-1m': f"{ROOT_DIR}/raw_data/ml-1m.inter",
        'ml-10m': f"{ROOT_DIR}/raw_data/ml-10m.inter",
        'Amazon_Electronics': f"{ROOT_DIR}/raw_data/Amazon_Electronics.inter",
        'steam': f"{ROOT_DIR}/raw_data/steam.inter",
        'Amazon_Books': f"{ROOT_DIR}/raw_data/Amazon_Books.inter",
        'tmall': f"{ROOT_DIR}/raw_data/tmall.inter",
        'ta-feng': f"{ROOT_DIR}/raw_data/ta-feng.inter",
        'gowalla': f"{ROOT_DIR}/raw_data/gowalla.inter"
    }
    dataset_config = {
        'ml-1m': {
            'item_min_occurrence': 5,
            'seq_min_length': 5,
            'seq_max_length': 9999,
        },
        'ml-10m': {
            'item_min_occurrence': 20,
            'seq_min_length': 50,
            'seq_max_length': 1000,
        },
        'steam': {
            'item_min_occurrence': 5,
            'seq_min_length': 20,
            'seq_max_length': 1000,
        },
        'Amazon_Books': {
            'item_min_occurrence': 20,
            'seq_min_length': 50,
            'seq_max_length': 1000,
        },
        'tmall': {
            'item_min_occurrence': 50,
            'seq_min_length': 100,
            'seq_max_length': 1000,
        },
        'Amazon_Electronics': {
            'item_min_occurrence': 5,
            'seq_min_length': 5,
            'seq_max_length': 1000,
        },
        'ta-feng': {
            'item_min_occurrence': 5,
            'seq_min_length': 5,
            'seq_max_length': 1000,
        },
        'gowalla': {
            'item_min_occurrence': 5,
            'seq_min_length': 5,
            'seq_max_length': 1000,
        }
    }
    preprocess_data(dataset_to_file[dataset], dataset,
                    item_min_occurrence=dataset_config[dataset]['item_min_occurrence'],
                    seq_min_length=dataset_config[dataset]['seq_min_length'],
                    seq_max_length=dataset_config[dataset]['seq_max_length'])

# ml-1m
# users: 6040
# items: 3416
# clicks: 999611
# max seq length: 2277
# min seq length: 18
# avg seq length: 165.49850993377484
# avg item click: 292.6261709601874

# steam
# users: 39751
# items: 12898
# clicks: 1790990
# max seq length: 1000
# min seq length: 20
# avg seq length: 45.05521873663556
# avg item click: 138.8579624748023

# Amazons Books
# users: 18319
# items: 168724
# clicks: 2171282
# max seq length: 1000
# min seq length: 50
# avg seq length: 118.52622959768546
# avg item click: 12.86883905075745

# tmall
# users: 50419
# items: 124059
# clicks: 10473273
# max seq length: 1000
# min seq length: 100
# avg seq length: 207.72472678950396
# avg item click: 84.42171063768046

# ml-10m
# users: 43600
# items: 8940
# clicks: 8797450
# max seq length: 1000
# min seq length: 50
# avg seq length: 201.776376146789
# avg item click: 984.0548098434004

# Amazon Electronics
# users: 226753
# items: 143316
# clicks: 2008150
# max seq length: 468
# min seq length: 5
# avg seq length: 8.85611215728127
# avg item click: 14.012043316866226
# sequence length greater than 20: 10149

# gowalla
# users: 77124
# items: 308903
# clicks: 4562972
# max seq length: 1000
# min seq length: 5
# avg seq length: 59.16409937243919
# avg item click: 14.771536695985471
# sequence length greater than 20: 44411
