from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune import CLIReporter
import ray

import torch

from utils.path import ROOT_DIR
from utils.tune_configs import get_tune_config
from trainer import trainer

import os


def get_metric_columns(metrics):
    metric_columns = []
    for k in metrics:
        metric_columns += [f"hr{k}", f"mrr{k}", f"ndcg{k}"]
    metric_columns += ["training_iteration"]
    return metric_columns


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

    dataset = "ml-1m"
    model = "CAN"
    config = get_tune_config(model, dataset)

    ray.init()

    reporter = CLIReporter(metric_columns=get_metric_columns(config["metrics"]))

    asha_scheduler = ASHAScheduler(
        time_attr='training_iteration',
        metric=config["metric"],
        mode='max',
        max_t=config['epoch'],
        grace_period=config['grace_period'])

    analysis = tune.run(
        trainer.trainer,
        name=model,
        resources_per_trial={"cpu": 2, "gpu": 1},
        scheduler=asha_scheduler,
        config=config,
        num_samples=1,
        progress_reporter=reporter,
        resume=False,
        local_dir=f"{ROOT_DIR}/tune",
        verbose=1)

    best_trial = analysis.get_best_trial(metric=config["metric"], mode="max")
    best_config = best_trial.config

    print(f"best config: {best_config}\n")
