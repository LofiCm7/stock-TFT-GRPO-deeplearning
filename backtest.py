"""
DEPRECATED: Competition prediction backtest has been removed.
The model now uses TFTEncoder -> PortfolioPolicy end-to-end with GRPO.
Use backtest_rl.py for RL-based backtesting.
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
from plot import plot_backtest_nav
import config


def main():
    print("Competition backtest is deprecated.")
    print("Use 'python backtest_rl.py' instead.")


if __name__ == "__main__":
    main()
