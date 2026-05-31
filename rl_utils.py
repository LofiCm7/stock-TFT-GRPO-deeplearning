"""Shared utilities for RL pipeline (train_rl, backtest_rl, predict_rl)."""
import random
import numpy as np
import torch

import config
from feature_engine import STATIC_CATEGORICAL


class ObsCache:
    """预计算所有股票的特征数据和日期索引，批量标准化避免逐股票循环。"""

    def __init__(self, grouped, avail_features, env, seq_len):
        self.seq_len = seq_len
        self.avail_features = avail_features
        self.n_feat = len(avail_features)
        self.n_stocks = len(env.codes)
        self.n_dates = len(env.dates)

        cat_features = list(STATIC_CATEGORICAL.keys())

        self.dynamic_matrix = np.full(
            (self.n_stocks, self.n_dates, self.n_feat), np.nan,
            dtype=np.float32)
        self.static_cat_arr = np.zeros(
            (self.n_stocks, len(cat_features)), dtype=np.float32)
        self.stock_age_matrix = np.zeros(
            (self.n_stocks, self.n_dates), dtype=np.float32)
        self.valid_start = np.full(self.n_stocks, self.n_dates, dtype=np.int32)

        date_to_idx = env.date_to_idx
        for i, code in enumerate(env.codes):
            if code not in grouped.groups:
                continue
            stock_df = grouped.get_group(code)
            self.static_cat_arr[i] = stock_df[cat_features].iloc[0].values
            dates = stock_df['trade_date'].values
            data = stock_df[avail_features].values.astype(np.float32)
            ages = stock_df['stock_age'].values.astype(np.float32) \
                if 'stock_age' in stock_df.columns else np.zeros(len(stock_df))
            for row_idx, d in enumerate(dates):
                if d in date_to_idx:
                    di = date_to_idx[d]
                    self.dynamic_matrix[i, di] = data[row_idx]
                    self.stock_age_matrix[i, di] = ages[row_idx]
            first_valid = np.where(
                ~np.isnan(self.dynamic_matrix[i, :, 0]))[0]
            if len(first_valid) >= seq_len:
                self.valid_start[i] = first_valid[seq_len - 1]

    def get_obs(self, date_idx, env, device):
        seq_len = self.seq_len

        # phase-aware observation window (anti look-ahead):
        #   open  -> decision at T's OPEN: may only use info up to T-1 close,
        #            so the window must END at T-1 (T's row is unknown yet).
        #   close -> decision at T's CLOSE: T's row is already realized and we
        #            also execute at T close, so ending at T is legitimate.
        end = date_idx if env.phase == "close" else date_idx - 1
        start = end - seq_len + 1

        if start < 0:
            # insufficient history (only at the very earliest dates; all three
            # callers guard date_idx >= SEQ_LEN so this is defensive).
            normalized = np.zeros(
                (self.n_stocks, seq_len, self.n_feat), dtype=np.float32)
            valid = np.zeros(self.n_stocks, dtype=bool)
        else:
            valid = (end >= self.valid_start) & (date_idx < self.n_dates)
            windows = self.dynamic_matrix[:, start:end + 1, :]
            mean = np.nanmean(windows, axis=1, keepdims=True)
            std = np.nanstd(windows, axis=1, keepdims=True, ddof=0) + 1e-8
            normalized = (windows - mean) / std
            normalized = np.nan_to_num(normalized, 0.0)
            normalized[~valid] = 0.0

        mask_arr = valid & ~env.suspended[date_idx]

        # stock_age is calendar-deterministic (known at open), not look-ahead.
        stock_ages = self.stock_age_matrix[:, date_idx:date_idx + 1]
        stat_full = np.concatenate(
            [self.static_cat_arr, stock_ages], axis=-1)

        dyn_t = torch.tensor(normalized, device=device)
        stat_t = torch.tensor(stat_full, device=device)
        mask_t = torch.tensor(mask_arr, device=device, dtype=torch.bool)
        return dyn_t, stat_t, mask_t


def build_port_state(env, device):
    """Build portfolio state tensor [N_stocks, 6]."""
    prices = env.get_valuation_prices()
    nav = env._compute_nav(prices)
    n = env.n_stocks

    cash_frac = np.full(n, env.cash / (nav + 1e-8), dtype=np.float32)
    hold_val = env.holdings.astype(np.float64) * np.nan_to_num(prices, 0)
    hold_frac = (hold_val / (nav + 1e-8)).astype(np.float32)
    lock_val = env.locked.astype(np.float64) * np.nan_to_num(prices, 0)
    lock_frac = (lock_val / (nav + 1e-8)).astype(np.float32)
    prev_w = env.prev_weights.astype(np.float32)

    ep_len = getattr(env, 'episode_len', config.EPISODE_LEN)
    ep_day = getattr(env, 'episode_day', 0)
    ep_progress = np.full(n, ep_day / max(ep_len, 1), dtype=np.float32)
    is_last = np.full(n, float(ep_day >= ep_len - 1), dtype=np.float32)

    state = np.stack([cash_frac, hold_frac, lock_frac, prev_w,
                      ep_progress, is_last], axis=-1)
    return torch.tensor(state, device=device)


def randomize_portfolio_state(env, date_idx):
    """Randomly initialize portfolio state so policy trains on diverse conditions."""
    prices = env.close_prices[date_idx]
    valid = ~np.isnan(prices) & (prices > 0)
    valid_indices = np.where(valid)[0]

    if len(valid_indices) == 0 or random.random() < 0.2:
        return

    n_held = random.randint(1, min(config.N_HOLD, len(valid_indices)))
    held_indices = np.random.choice(valid_indices, size=n_held, replace=False)

    cash_frac = random.uniform(0.1, 0.6)
    invest_capital = env.init_capital * (1 - cash_frac)
    env.cash = env.init_capital * cash_frac

    weights = np.random.dirichlet(np.ones(n_held))
    for j, idx in enumerate(held_indices):
        raw_shares = weights[j] * invest_capital / prices[idx]
        shares = int(raw_shares / env.lots[idx]) * env.lots[idx]
        if random.random() < 0.3:
            env.locked[idx] = shares
        else:
            env.holdings[idx] = shares

    nav = env._compute_nav(prices)
    if nav > 0:
        total_held = (env.holdings + env.locked).astype(np.float64)
        env.prev_weights = (total_held * prices) / nav
        env.prev_weights = np.nan_to_num(env.prev_weights, 0.0)
