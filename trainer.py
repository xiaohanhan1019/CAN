# -*- coding: UTF-8 -*-
import torch
import os

from utils.metric import cal_hr, cal_mrr, cal_ndcg
from dataset.sequential_dataset import get_data
from models.CAN import CAN
from models.SASRec import SASRec
from models.GRU4Rec import GRU4Rec
from models.Caser import Caser
from models.GCSAN import GCSAN
from models.RUM_item import RUM_item
from models.RUM_feature import RUM_feature
from models.SINE import SINE
from models.MIND import MIND
from models.ComiRec import ComiRec_DR
from models.ComiRec import ComiRec_SA
from models.SHAN import SHAN
from utils.path import ROOT_DIR
from utils.setup_seed import setup_seed
import datetime
from utils.configs import get_config

from tqdm import tqdm
from ray import tune


def train(model, optimizer, criterion, data_loader, device, verbose=False):
    model.train()
    loss_sum = 0

    data_iter = enumerate(data_loader)
    if verbose:
        data_iter = tqdm(enumerate(data_loader), total=len(data_loader))

    for i, (user_id, seq, seq_length, labels) in data_iter:
        user_id, seq, seq_length, labels = user_id.to(device), seq.to(device), seq_length.to(device), labels.to(device)
        data = {
            'user_id': user_id,
            'seq': seq,
            'seq_length': seq_length
        }

        optimizer.zero_grad()

        # forward & backward
        outputs, loss = model(data)

        loss = loss + criterion(outputs.to(device), labels.to(device))
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * seq.shape[0]
    if verbose:
        print(f"loss: {loss_sum}")


def eval(model, data_loader, device, metrics=None, verbose=False, valid=True):
    if metrics is None:
        metrics = [5, 10, 20]

    model.eval()
    with torch.no_grad():
        hrs = [0 for _ in range(len(metrics))]
        mrrs = [0 for _ in range(len(metrics))]
        ndcgs = [0 for _ in range(len(metrics))]
        for _, (user_id, seq, seq_length, labels) in enumerate(data_loader):
            user_id, seq, seq_length, labels = user_id.to(device), seq.to(device), seq_length.to(device), labels.to(
                device)
            data = {
                'user_id': user_id,
                'seq': seq,
                'seq_length': seq_length
            }

            # forward & backward
            outputs, _ = model(data)

            # metric
            result = torch.topk(outputs, k=metrics[-1], dim=1)[1]
            for i, k in enumerate(metrics):
                hrs[i] += cal_hr(result[:, :k].cpu().numpy(), labels.cpu().numpy())
                mrrs[i] += cal_mrr(result[:, :k].cpu().numpy(), labels.cpu().numpy())
                ndcgs[i] += cal_ndcg(result[:, :k].cpu().numpy(), labels.cpu().numpy())

        for i, k in enumerate(metrics):
            hrs[i] = hrs[i] / len(data_loader.dataset)
            mrrs[i] = mrrs[i] / len(data_loader.dataset)
            ndcgs[i] = ndcgs[i] / len(data_loader.dataset)
            if verbose:
                if valid:
                    print(f'valid, HR@{k}: {hrs[i]:.4f} MRR@{k}: {mrrs[i]:.4f} NDCG@{k}: {ndcgs[i]:.4f}')
                else:
                    print(f'test, HR@{k}: {hrs[i]:.4f} MRR@{k}: {mrrs[i]:.4f} NDCG@{k}: {ndcgs[i]:.4f}')
    return hrs, mrrs, ndcgs


def trainer(config, checkpoint_dir=None):
    setup_seed(2021)
    device = config["device"]
    verbose = config["verbose"]
    metrics = config["metrics"]
    train_dataset, valid_dataset, test_dataset = get_data(dataset=config["dataset"],
                                                          seq_length=config["seq_length"],
                                                          front_padding=config["front_padding"])

    train_loader = train_dataset.get_data_loader(device=device, batch_size=config["batch_size"], shuffle=True)
    valid_loader = valid_dataset.get_data_loader(device=device, batch_size=config["batch_size"], shuffle=False)
    test_loader = test_dataset.get_data_loader(device=device, batch_size=config["batch_size"], shuffle=False)

    if config['model'] == 'CAN':
        model = CAN(config).to(device)
    elif config['model'] == 'Test_':
        model = Test_(config).to(device)
    elif config['model'] == 'SASRec':
        model = SASRec(config).to(device)
    elif config['model'] == 'GRU4Rec':
        model = GRU4Rec(config).to(device)
    elif config['model'] == 'NARM':
        model = NARM(config).to(device)
    elif config['model'] == 'Test_GRU':
        model = Test_GRU(config).to(device)
    elif config['model'] == 'Test_GRU2':
        model = Test_GRU2(config).to(device)
    elif config['model'] == 'Caser':
        model = Caser(config).to(device)
    elif config['model'] == 'GCSAN':
        model = GCSAN(config).to(device)
    elif config['model'] == 'RUM_item':
        model = RUM_item(config).to(device)
    elif config['model'] == 'RUM_feature':
        model = RUM_feature(config).to(device)
    elif config['model'] == 'SINE':
        model = SINE(config).to(device)
    elif config['model'] == 'MIND':
        model = MIND(config).to(device)
    elif config['model'] == 'ComiRec_DR':
        model = ComiRec_DR(config).to(device)
    elif config['model'] == 'ComiRec_SA':
        model = ComiRec_SA(config).to(device)
    elif config['model'] == 'OnlySA':
        model = OnlySA(config).to(device)
    elif config['model'] == 'SHAN':
        model = SHAN(config).to(device)
    else:
        raise NotImplementedError("we don't have that model")

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["reg"], amsgrad=True)
    criterion = torch.nn.CrossEntropyLoss()

    best_hr = 0
    best_test_hrs = []
    best_test_mrrs = []
    best_test_ndcgs = []
    for epoch in range(config["epoch"]):
        if config["verbose"]:
            print(f"epoch: {epoch}/{config['epoch']}")
            print("Start Training:" + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        train(model, optimizer, criterion, train_loader, device, verbose=verbose)

        if config["verbose"]:
            print("Start Validating:" + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        valid_hrs, valid_mrrs, valid_ndcgs = eval(model, valid_loader, device, metrics=metrics, verbose=verbose,
                                                  valid=True)

        if config["verbose"]:
            print("Start Testing:" + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        test_hrs, test_mrrs, test_ndcgs = eval(model, test_loader, device, metrics=metrics, verbose=verbose,
                                               valid=False)

        if valid_hrs[2] > best_hr:
            best_hr = max(best_hr, valid_hrs[2])

            # torch.save(model, f"{ROOT_DIR}/saved_models/ml-10m_can.pth")

            best_test_hrs = test_hrs
            best_test_mrrs = test_mrrs
            best_test_ndcgs = test_ndcgs

            if config["report"]:
                if config['dataset'] == 'ml-1m':
                    tune.report(hr5=test_hrs[0], mrr5=test_mrrs[0], ndcg5=test_ndcgs[0],
                                hr10=test_hrs[1], mrr10=test_mrrs[1], ndcg10=test_ndcgs[1],
                                hr20=test_hrs[2], mrr20=test_mrrs[2], ndcg20=test_ndcgs[2])
                elif config['dataset'] == 'ml-10m':
                    tune.report(hr5=test_hrs[0], mrr5=test_mrrs[0], ndcg5=test_ndcgs[0],
                                hr10=test_hrs[1], mrr10=test_mrrs[1], ndcg10=test_ndcgs[1],
                                hr20=test_hrs[2], mrr20=test_mrrs[2], ndcg20=test_ndcgs[2],
                                hr50=test_hrs[3], mrr50=test_mrrs[3], ndcg50=test_ndcgs[3])
                elif config['dataset'] == 'steam':
                    tune.report(hr5=test_hrs[0], mrr5=test_mrrs[0], ndcg5=test_ndcgs[0],
                                hr10=test_hrs[1], mrr10=test_mrrs[1], ndcg10=test_ndcgs[1],
                                hr20=test_hrs[2], mrr20=test_mrrs[2], ndcg20=test_ndcgs[2],
                                hr50=test_hrs[3], mrr50=test_mrrs[3], ndcg50=test_ndcgs[3])
                elif config['dataset'] == 'Amazon_Books':
                    tune.report(hr10=test_hrs[0], mrr10=test_mrrs[0], ndcg10=test_ndcgs[0],
                                hr20=test_hrs[1], mrr20=test_mrrs[1], ndcg20=test_ndcgs[1],
                                hr50=test_hrs[2], mrr50=test_mrrs[2], ndcg50=test_ndcgs[2],
                                hr100=test_hrs[3], mrr100=test_mrrs[3], ndcg100=test_ndcgs[3])
                elif config['dataset'] == 'tmall':
                    tune.report(hr10=test_hrs[0], mrr10=test_mrrs[0], ndcg10=test_ndcgs[0],
                                hr20=test_hrs[1], mrr20=test_mrrs[1], ndcg20=test_ndcgs[1],
                                hr50=test_hrs[2], mrr50=test_mrrs[2], ndcg50=test_ndcgs[2],
                                hr100=test_hrs[3], mrr100=test_mrrs[3], ndcg100=test_ndcgs[3])
                elif config['dataset'] == 'gowalla':
                    tune.report(hr5=test_hrs[0], mrr5=test_mrrs[0], ndcg5=test_ndcgs[0],
                                hr10=test_hrs[1], mrr10=test_mrrs[1], ndcg10=test_ndcgs[1],
                                hr20=test_hrs[2], mrr20=test_mrrs[2], ndcg20=test_ndcgs[2],
                                hr50=test_hrs[3], mrr50=test_mrrs[3], ndcg50=test_ndcgs[3])

    if config["verbose"]:
        print(f'results:\n')
        for i, k in enumerate(metrics):
            print(f'test, HR@{k}: {best_test_hrs[i]:.4f} '
                  f'MRR@{k}: {best_test_mrrs[i]:.4f} '
                  f'NDCG@{k}: {best_test_ndcgs[i]:.4f}')


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

    model = "CAN"
    dataset = "ml-1m"
    config = get_config(model, dataset)
    print(config)
    trainer(config)
