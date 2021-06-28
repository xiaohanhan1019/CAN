import torch

dataset_configs = {
    "ml-1m": {
        "dataset": "ml-1m",
        "user_num": 6040 + 1,
        "item_num": 3416 + 1,
        "seq_length": 50,
        "metric": "hr20",
        "epoch": 35,
        "grace_period": 25,
        "metrics": [5, 10, 20]
    },
    "Amazon_Books": {
        "dataset": "Amazon_Books",
        "user_num": 18319 + 1,
        "item_num": 168724 + 1,
        "seq_length": 50,
        "metric": "hr50",
        "epoch": 15,
        "grace_period": 10,
        "metrics": [10, 20, 50, 100]
    },
    "ml-10m": {
        "dataset": "ml-10m",
        "user_num": 43600 + 1,
        "item_num": 8940 + 1,
        "seq_length": 50,
        "metric": "hr20",
        "epoch": 30,
        "grace_period": 20,
        "metrics": [5, 10, 20, 50]
    }
}

model_configs = {
    "CAN": {
        "model": "CAN",
        "front_padding": True,
        "hidden_size": 100,
        "lr": 1e-3,
        "reg": 0,
        "n_layers": 2,
        "n_heads": 1,
        "dropout_prob": 0.25,  # 0.25 for ml-1m and Amazon_Books, 0 for ml-10m
        "n": 0,
        "k": [5, 10],  # [5,10] for ml-1m and ml-10m, [8,10] for Amazon_Books
        "l": 20,
        'threshold': 1e-3,
        "batch_size": 256,
        "device": torch.device("cuda:3"),
        "verbose": True,
        "report": False
    },
    "SASRec": {
        "model": "SASRec",
        "front_padding": True,
        "hidden_size": 100,
        "lr": 1e-3,
        "reg": 0,
        "n_layers": 2,
        "n_heads": 1,
        "dropout_prob": 0.25,
        "batch_size": 256,
        "seq_length": 50,
        "device": torch.device("cuda:1"),
        "verbose": True,
        "report": False
    },
    "GRU4Rec": {
        "model": "GRU4Rec",
        "front_padding": False,
        "hidden_size": 100,
        "lr": 1e-3,
        "reg": 0,
        "num_layers": 1,
        "dropout_prob": 0.25,
        "batch_size": 256,
        "device": torch.device("cuda:0"),
        "verbose": True,
        "report": False
    },
    "Caser": {
        "model": "Caser",
        "front_padding": False,
        "embedding_size": 100,
        "n_h": 4,
        "n_v": 2,
        "lr": 1e-3,
        "reg": 1e-4,
        "dropout_prob": 0.25,
        "batch_size": 512,
        "epoch": 20,
        "device": torch.device("cuda:0"),
        "verbose": True,
        "report": False
    },
    "GCSAN": {
        "model": "GCSAN",
        "front_padding": False,
        "hidden_size": 100,
        "lr": 1e-3,
        "reg": 1e-6,
        "n_layers": 1,
        "n_heads": 1,
        "weight": 0.8,
        "step": 1,
        "dropout_prob": 0,
        "batch_size": 512,
        "epoch": 12,
        "device": torch.device("cuda:1"),
        "verbose": True,
        "report": False
    },
    "RUM_item": {
        "model": "RUM_item",
        "front_padding": True,
        "embedding_dim": 100,
        "k": 5,
        "alpha": 0.6,
        "lr": 1e-3,
        "reg": 1e-5,
        "batch_size": 256,
        "device": torch.device("cuda:3"),
        "verbose": True,
        "report": False
    },
    "RUM_feature": {
        "model": "RUM_feature",
        "front_padding": False,
        "embedding_dim": 100,
        "k": 5,
        "alpha": 0.2,
        "lr": 1e-3,
        "reg": 1e-5,
        "batch_size": 256,
        "device": torch.device("cuda:1"),
        "verbose": True,
        "report": False
    },
    "SINE": {
        "model": "SINE",
        "front_padding": True,
        "hidden_size": 100,
        "lr": 1e-3,
        "reg": 0,
        "l": 50,
        "k": 4,
        "lambda": 0.5,
        "batch_size": 256,
        "epoch": 50,
        "device": torch.device("cuda:2"),
        "verbose": True,
        "report": False
    },
    "MIND": {
        "model": "MIND",
        "front_padding": True,
        "hidden_size": 100,
        "lr": 1e-3,
        "reg": 0,
        "k": 4,
        "pow": 2,
        "batch_size": 256,
        "device": torch.device("cuda:2"),
        "verbose": True,
        "report": False
    },
    "ComiRec_DR": {
        "model": "ComiRec_DR",
        "front_padding": True,
        "hidden_size": 100,
        "lr": 1e-3,
        "reg": 0,
        "k": 4,
        "batch_size": 256,
        "device": torch.device("cuda:0"),
        "verbose": True,
        "report": False
    },
    "ComiRec_SA": {
        "model": "ComiRec_SA",
        "front_padding": True,
        "hidden_size": 100,
        "lr": 1e-3,
        "reg": 0,
        "k": 4,
        "batch_size": 256,
        "device": torch.device("cuda:0"),
        "verbose": True,
        "report": False
    },
    "SHAN": {
        "model": "SHAN",
        "front_padding": True,
        "hidden_size": 100,
        "lr": 1e-3,
        "reg": 1e-5,
        "batch_size": 256,
        "seq_length": 50,
        "short_item_length": 5,
        "device": torch.device("cuda:1"),
        "verbose": True,
        "report": False
    },
}


def get_config(model='Test', dataset='ml-1m'):
    # model config can replace the dataset configs
    return dict(dataset_configs[dataset], **model_configs[model])


if __name__ == '__main__':
    print(get_config(model="SASRec", dataset="taobao"))
