import numpy as np
import pandas as pd
import config


def _is_star_code(code):
    """判断股票代码是否为科创板（688xxx）。"""
    return code.startswith('688')


def _is_gem_code(code):
    """判断股票代码是否为创业板（300xxx / 301xxx）。"""
    return code.startswith('300') or code.startswith('301')


def _round_to_lot(shares, lot):
    """将股数向下取整到最小交易单位的整数倍。
    
    科创板（lot=200）：超过 200 后以 1 股递增，即只保证 >=200 且为整数即可。
    主板/创业板（lot=100）：以 100 股递增。
    """
    if lot == 200:
        # 科创板：200 股起，之后 1 股递增
        if shares < 200:
            return 0
        return int(shares)  # 直接取整，>=200 即可
    else:
        # 主板/创业板：100 股递增
        return int(shares / lot) * lot


class AShareTradingEnv:
    def __init__(self, panel_df, init_capital=config.INIT_CAPITAL,
                 lot=config.LOT, commission=config.COMMISSION,
                 stamp=config.STAMP, min_commission=config.MIN_COMMISSION,
                 limit_pct=config.LIMIT_PCT):
        self.init_capital = init_capital
        self.lot = lot
        self.commission = commission
        self.stamp = stamp
        self.min_commission = min_commission
        self.limit_pct = limit_pct

        self._prepare_panel(panel_df)
        self._build_lots()
        self._build_limit_pcts()
        self._load_benchmark_returns()

    def _build_lots(self):
        """根据股票代码构建 per-stock 最小交易单位数组。

        科创板（688xxx）：200 股
        其他（主板/创业板）：100 股
        """
        self.lots = np.array([
            config.STAR_LOT if _is_star_code(code) else config.LOT
            for code in self.codes
        ], dtype=np.int64)

    def _build_limit_pcts(self):
        """根据股票代码构建 per-stock 涨跌停幅度数组。

        科创板（688xxx）：±20%
        创业板（300xxx / 301xxx）：±20%
        主板（其余）：±10%
        """
        self.limit_pcts = np.array([
            config.GEM_STAR_LIMIT_PCT
            if _is_star_code(code) or _is_gem_code(code)
            else config.LIMIT_PCT
            for code in self.codes
        ], dtype=np.float64)

    def _prepare_panel(self, panel_df):
        self.codes = sorted(panel_df['ts_code'].unique())
        self.code_to_idx = {c: i for i, c in enumerate(self.codes)}
        self.n_stocks = len(self.codes)

        self.dates = sorted(panel_df['trade_date'].unique())
        self.date_to_idx = {d: i for i, d in enumerate(self.dates)}

        n_dates = len(self.dates)
        self.open_prices = np.full((n_dates, self.n_stocks), np.nan)
        self.close_prices = np.full((n_dates, self.n_stocks), np.nan)
        self.pre_close_prices = np.full((n_dates, self.n_stocks), np.nan)
        self.suspended = np.ones((n_dates, self.n_stocks), dtype=bool)

        pdf = panel_df.copy()
        pdf['_di'] = pdf['trade_date'].map(self.date_to_idx)
        pdf['_si'] = pdf['ts_code'].map(self.code_to_idx)
        valid = pdf.dropna(subset=['_di', '_si'])
        di = valid['_di'].astype(int).values
        si = valid['_si'].astype(int).values

        self.open_prices[di, si] = valid['open'].values
        self.close_prices[di, si] = valid['close'].values
        self.suspended[di, si] = False

        if 'pre_close' in valid.columns:
            pre_vals = valid['pre_close'].values.astype(float)
            has_pre = ~np.isnan(pre_vals)
            self.pre_close_prices[di[has_pre], si[has_pre]] = pre_vals[has_pre]

        mask = np.isnan(self.pre_close_prices)
        mask[0, :] = False
        shifted_close = np.empty_like(self.pre_close_prices)
        shifted_close[0, :] = np.nan
        shifted_close[1:, :] = self.close_prices[:-1, :]
        self.pre_close_prices = np.where(mask, shifted_close, self.pre_close_prices)

    # === PLACEHOLDER_ENV_METHODS ===

    def _load_benchmark_returns(self):
        import os
        path = os.path.join(config.MARKET_DIR, "000300.SH.csv")
        self.bench_returns = np.zeros(len(self.dates))
        if not os.path.exists(path):
            return
        bench = pd.read_csv(path)
        bench['trade_date'] = bench['trade_date'].astype(str)
        bench['ret'] = bench['close'].pct_change().fillna(0)
        ret_map = dict(zip(bench['trade_date'], bench['ret']))
        for i, d in enumerate(self.dates):
            self.bench_returns[i] = ret_map.get(d, 0.0)

    def reset(self, start_date_idx=0, episode_len=None):
        self.cash = self.init_capital
        self.holdings = np.zeros(self.n_stocks, dtype=np.int64)
        self.locked = np.zeros(self.n_stocks, dtype=np.int64)
        self.prev_weights = np.zeros(self.n_stocks, dtype=np.float64)
        self.current_idx = start_date_idx
        self.phase = "open"
        self.episode_start_idx = start_date_idx
        self.episode_len = episode_len if episode_len is not None else config.EPISODE_LEN
        self.episode_day = 0
        return self._get_state()

    def _get_state(self):
        nav = self._compute_nav()
        return {
            "date_idx": self.current_idx,
            "phase": self.phase,
            "cash": self.cash,
            "holdings": self.holdings.copy(),
            "locked": self.locked.copy(),
            "prev_weights": self.prev_weights.copy(),
            "nav": nav,
            "episode_day": self.episode_day,
            "is_last_day": self.episode_day >= self.episode_len - 1,
        }

    def _compute_nav(self, prices=None):
        if prices is None:
            prices = self.close_prices[self.current_idx]
        stock_value = np.nansum(
            (self.holdings + self.locked).astype(np.float64) * prices)
        return self.cash + stock_value

    def get_valuation_prices(self):
        """Return the appropriate prices for current phase valuation.

        open phase: use previous day's close (last known price)
        close phase: use current day's open (known after morning auction)
        """
        if self.phase == "open":
            if self.current_idx > 0:
                return self.close_prices[self.current_idx - 1]
            return self.open_prices[self.current_idx]
        return self.open_prices[self.current_idx]

    def get_execution_prices(self):
        """Return the execution prices for the current phase.

        These are the prices used for order settlement. We use the last
        known price at decision time (no look-ahead):
          open phase  → T-1 close (decision made before T opens)
          close phase → T open    (decision made after morning auction)
        """
        if self.phase == "open":
            if self.current_idx > 0:
                return self.close_prices[self.current_idx - 1]
            return self.open_prices[self.current_idx]
        return self.open_prices[self.current_idx]

    def _limit_prices(self, date_idx):
        pre = self.pre_close_prices[date_idx]
        limit_up = np.round(pre * (1 + self.limit_pcts), 2)
        limit_down = np.round(pre * (1 - self.limit_pcts), 2)
        return limit_up, limit_down

    def _can_buy(self, date_idx):
        """Check which stocks can be bought (not suspended, not limit-up).

        Limit-up check uses the actual market price (open for open phase,
        close for close phase) — this reflects real market constraints,
        independent of what execution price we use for settlement.
        """
        limit_up, _ = self._limit_prices(date_idx)
        if self.phase == "open":
            market_prices = self.open_prices[date_idx]
        else:
            market_prices = self.close_prices[date_idx]
        valid_price = ~np.isnan(market_prices)
        hit_limit_up = np.nan_to_num(market_prices, nan=np.inf) >= limit_up
        mask = ~self.suspended[date_idx] & ~hit_limit_up & valid_price
        return mask

    def _can_sell(self, date_idx):
        """Check which stocks can be sold (not suspended, not limit-down).

        Limit-down check uses the actual market price (open for open phase,
        close for close phase) — this reflects real market constraints.
        """
        _, limit_down = self._limit_prices(date_idx)
        if self.phase == "open":
            market_prices = self.open_prices[date_idx]
        else:
            market_prices = self.close_prices[date_idx]
        valid_price = ~np.isnan(market_prices)
        hit_limit_down = np.nan_to_num(market_prices, nan=-np.inf) <= limit_down
        mask = ~self.suspended[date_idx] & ~hit_limit_down & valid_price
        return mask

    def step(self, target_weights, force_liquidate=False):
        """Execute one decision step.

        Args:
            target_weights: np.ndarray [n_stocks], target portfolio weight.
            force_liquidate: if True, sell all holdings ignoring target_weights.

        Returns:
            (next_state, reward, done, info)
        """
        date_idx = self.current_idx
        nav_before = self._compute_nav(self.get_valuation_prices())

        is_last_day = self.episode_day >= self.episode_len - 1
        if force_liquidate or (is_last_day and self.phase == "close"):
            target_weights = np.zeros(self.n_stocks)

        prices = self.get_execution_prices()

        can_buy = self._can_buy(date_idx)
        can_sell = self._can_sell(date_idx)

        target_values = target_weights * nav_before
        target_shares = np.zeros(self.n_stocks, dtype=np.int64)
        for i in range(self.n_stocks):
            if np.isnan(prices[i]) or prices[i] <= 0:
                continue
            raw_shares = target_values[i] / prices[i]
            target_shares[i] = _round_to_lot(raw_shares, self.lots[i])

        available = self.holdings.copy()
        sell_shares = np.maximum(available - target_shares, 0)
        sell_shares = np.where(can_sell, sell_shares, 0)

        sell_revenue = 0.0
        for i in range(self.n_stocks):
            if sell_shares[i] > 0:
                revenue = sell_shares[i] * prices[i]
                comm = max(revenue * self.commission, self.min_commission)
                stamp_tax = revenue * self.stamp
                net = revenue - comm - stamp_tax
                sell_revenue += net
                self.holdings[i] -= sell_shares[i]

        self.cash += sell_revenue

        buy_order = []
        for i in range(self.n_stocks):
            want = target_shares[i] - self.holdings[i]
            if want > 0 and can_buy[i]:
                buy_order.append((target_weights[i], i, want))
        buy_order.sort(key=lambda x: -x[0])

        for _, i, shares in buy_order:
            cost = shares * prices[i]
            comm = max(cost * self.commission, self.min_commission)
            total = cost + comm
            if total > self.cash:
                raw_shares = self.cash / (prices[i] * (1 + self.commission))
                shares = _round_to_lot(raw_shares, self.lots[i])
                if shares <= 0:
                    continue
                cost = shares * prices[i]
                comm = max(cost * self.commission, self.min_commission)
                total = cost + comm
                if total > self.cash:
                    continue
            self.cash -= total
            self.locked[i] += shares

        cash_penalty = 0.0
        if self.cash > config.MAX_CASH:
            cash_penalty = config.LAMBDA_CASH_PENALTY * (
                (self.cash - config.MAX_CASH) / self.init_capital)
            self._force_reduce_cash(prices, can_buy)

        turnover = (np.abs(target_weights - self.prev_weights)).sum()

        if self.phase == "open":
            # per_stock_returns: open[T] / close[T-1] - 1 (overnight gap)
            future_prices = self.open_prices[date_idx]
            per_stock_returns = np.where(
                ~np.isnan(prices) & ~np.isnan(future_prices) & (prices > 0),
                future_prices / prices - 1.0, 0.0).astype(np.float32)
            nav_after = self._compute_nav(self.open_prices[date_idx])
            self.phase = "close"
        else:
            # per_stock_returns: close[T] / open[T] - 1 (intraday)
            future_prices = self.close_prices[date_idx]
            per_stock_returns = np.where(
                ~np.isnan(prices) & ~np.isnan(future_prices) & (prices > 0),
                future_prices / prices - 1.0, 0.0).astype(np.float32)
            nav_after = self._compute_nav(self.close_prices[date_idx])
            self.current_idx += 1
            self.phase = "open"
            self.holdings += self.locked
            self.locked[:] = 0
            self.episode_day += 1

        episode_done = self.episode_day >= self.episode_len
        data_done = self.current_idx >= len(self.dates)
        done = episode_done or data_done

        self.prev_weights = target_weights.copy()
        portfolio_ret = np.log(nav_after / nav_before + 1e-10)
        bench_ret = self.bench_returns[date_idx] if date_idx < len(self.bench_returns) else 0.0
        excess = portfolio_ret - bench_ret
        benchmark_bonus = config.LAMBDA_BENCHMARK * excess
        reward = portfolio_ret - config.LAMBDA_TURNOVER * turnover \
            - cash_penalty + benchmark_bonus

        info = {"nav": nav_after, "turnover": turnover,
                "cash_penalty": cash_penalty,
                "per_stock_returns": per_stock_returns}
        return self._get_state() if not done else None, reward, done, info

    def _force_reduce_cash(self, prices, can_buy):
        """Buy stocks to bring cash below MAX_CASH."""
        buyable = np.where(
            can_buy & ~np.isnan(prices) & (prices > 0))[0]
        if len(buyable) == 0:
            return
        current_val = (self.holdings + self.locked).astype(float) * \
            np.nan_to_num(prices, 0)
        weights = current_val[buyable]
        w_sum = weights.sum()
        if w_sum < 1e-8:
            weights = np.ones(len(buyable)) / len(buyable)
        else:
            weights = weights / w_sum

        excess = self.cash - config.MAX_CASH
        for j, idx in enumerate(buyable):
            alloc = excess * weights[j]
            raw_shares = alloc / (prices[idx] * (1 + self.commission))
            shares = _round_to_lot(raw_shares, self.lots[idx])
            if shares <= 0:
                continue
            cost = shares * prices[idx]
            comm = max(cost * self.commission, self.min_commission)
            total = cost + comm
            if total > self.cash:
                continue
            self.cash -= total
            self.locked[idx] += shares
            if self.cash <= config.MAX_CASH:
                break

    def clone(self):
        """Create a lightweight copy for GRPO group sampling."""
        import copy
        new_env = copy.copy(self)
        new_env.holdings = self.holdings.copy()
        new_env.locked = self.locked.copy()
        new_env.prev_weights = self.prev_weights.copy()
        return new_env