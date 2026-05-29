"""
DEPRECATED: Competition prediction has been removed.
The model now uses TFTEncoder -> PortfolioPolicy end-to-end with GRPO.
Use predict_rl.py for RL-based predictions.
"""
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from model import TFTEncoder, PortfolioPolicy, DiffusionDenoiser
from dataset import StockDataset
from data_loader import build_merged_dataset
from feature_engine import build_features, DYNAMIC_FEATURES, STATIC_FEATURES
from feature_engine import STATIC_CATEGORICAL, STATIC_CONTINUOUS
import config


def main():
    print("Competition prediction is deprecated.")
    print("Use 'python predict_rl.py --date YYYYMMDD' instead.")
    print("Or run: python predict_rl.py --date 20260501")


if __name__ == "__main__":
    main()
