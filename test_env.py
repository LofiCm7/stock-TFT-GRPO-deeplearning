"""Unit tests for the three critical correctness risks in env.py."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest
from env import AShareTradingEnv


def make_simple_panel(n_dates=5, codes=None, base_price=10.0):
    """Create a minimal panel DataFrame for testing."""
    if codes is None:
        codes = ["A.SZ", "B.SZ"]
    rows = []
    for di in range(n_dates):
        for code in codes:
            rows.append({
                "ts_code": code,
                "trade_date": f"2025010{di+1}",
                "open": base_price + di * 0.1,
                "high": base_price + di * 0.1 + 0.5,
                "low": base_price + di * 0.1 - 0.3,
                "close": base_price + di * 0.1 + 0.2,
                "pre_close": base_price + (di - 1) * 0.1 + 0.2 if di > 0
                    else base_price,
            })
    return pd.DataFrame(rows)


class TestT1Lock:
    """Risk 1: T+1 unlock timing."""

    def test_cannot_sell_same_day_purchase(self):
        panel = make_simple_panel(n_dates=3)
        env = AShareTradingEnv(panel, init_capital=100000)
        env.reset(start_date_idx=1)
        env.phase = "open"

        target_w = np.array([0.5, 0.0])
        env.step(target_w)

        assert env.locked[0] > 0, "Should have locked shares after buy"
        cash_before_close = env.cash

        sell_w = np.array([0.0, 0.0])
        env.step(sell_w)

        assert cash_before_close == env.cash or env.cash <= cash_before_close, \
            "Should not gain cash from selling locked shares"
        assert env.phase == "open", "Should transition to next day open"

    def test_unlock_next_day_open(self):
        panel = make_simple_panel(n_dates=4)
        env = AShareTradingEnv(panel, init_capital=100000)
        env.reset(start_date_idx=1)
        env.phase = "open"

        target_w = np.array([0.5, 0.0])
        env.step(target_w)
        bought_locked = env.locked[0]
        assert bought_locked > 0

        env.step(np.array([0.5, 0.0]))

        assert env.locked[0] == 0, "Locked should be cleared after day transition"
        assert env.holdings[0] >= bought_locked, \
            "Holdings should include yesterday's purchase"


class TestCashInvariant:
    """Risk 2: Floor rounding must never cause negative cash."""

    @pytest.mark.parametrize("seed", range(100))
    def test_cash_non_negative(self, seed):
        rng = np.random.default_rng(seed)
        n_stocks = 5
        codes = [f"S{i}.SZ" for i in range(n_stocks)]
        panel = make_simple_panel(n_dates=3, codes=codes,
                                  base_price=rng.uniform(5, 500))
        env = AShareTradingEnv(panel, init_capital=1_000_000)
        env.reset(start_date_idx=1)
        env.phase = "open"

        weights = rng.dirichlet(np.ones(n_stocks))
        env.step(weights)

        assert env.cash >= 0, f"Cash went negative: {env.cash}"


class TestLimitRejection:
    """Risk 3: Limit-up rejection is per-stock, not whole order."""

    def test_limit_up_blocks_single_stock(self):
        rows = []
        pre_close_a = 10.00
        limit_up_a = round(pre_close_a * 1.10, 2)

        for di, date in enumerate(["20250101", "20250102"]):
            rows.append({
                "ts_code": "A.SZ", "trade_date": date,
                "open": limit_up_a if di == 1 else pre_close_a,
                "high": limit_up_a, "low": pre_close_a - 0.1,
                "close": limit_up_a if di == 1 else pre_close_a,
                "pre_close": pre_close_a,
            })
            rows.append({
                "ts_code": "B.SZ", "trade_date": date,
                "open": 20.0, "high": 21.0, "low": 19.5,
                "close": 20.5,
                "pre_close": 19.8 if di == 1 else 20.0,
            })

        panel = pd.DataFrame(rows)
        env = AShareTradingEnv(panel, init_capital=100000)
        env.reset(start_date_idx=1)
        env.phase = "open"

        target_w = np.array([0.5, 0.5])
        env.step(target_w)

        assert env.locked[0] == 0, "A hit limit-up, should not buy"
        assert env.locked[1] > 0, "B should be bought normally"

    def test_pre_close_boundary_rounding(self):
        pre_close = 10.005
        limit_up = round(pre_close * 1.10, 2)
        assert limit_up == 11.01, f"Expected 11.01, got {limit_up}"
