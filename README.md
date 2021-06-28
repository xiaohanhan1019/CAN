# Capturing Multi-granularity Interests with Capsule Attentive Network for Sequential Recommendation

This code is used to reproduce the main experiment of our paper.

## Requirements

- Python 3.7.10
- Pytorch 1.8.0

## Datasets

The original dataset can be downloaded from:

- ml-1m: https://grouplens.org/datasets/movielens/1m/
- Amazon Books: http://jmcauley.ucsd.edu/data/amazon/
- ml-10m: https://grouplens.org/datasets/movielens/10m/

The format of the preprocessed dataset is the same as [RecBole](https://github.com/RUCAIBox/RecBole) which can be downloaded from:

- ml-1m: https://drive.google.com/drive/folders/1OkDVEqetvOrtbuWebxl4y1JlZ_YjjfWj
- Amazon Books: https://drive.google.com/drive/folders/1FUAScpZtCmArqQS0xdrIqpFx5F7u3TaY
- ml-10m: https://drive.google.com/drive/folders/1OkDVEqetvOrtbuWebxl4y1JlZ_YjjfWj

The `*.inter` file should be stored in the `./raw_data` folder

## Code

-  `./models` folder contains all the baselines and our model `CAN`
- `./data` folder is used to store the processed data

## Training

- First you need to process the data by running`./raw_data/preprocess.py`, make sure the raw dataset is downloaded and stored in the `./raw_data` folder
- After processing the data, you can train a model by running `./trainer.py`
  - The model's configuration can be modified in `./utils/configs.py`
- `./ray_tune.py` for tuning the hyper-parameters
  - The model's configuration can be modified in `./utils/tune_configs.py`

