import os
import argparse
import numpy as np
import pandas as pd
import torch

import config
from data_loader import build_merged_dataset, universe_tag, load_daily_open_for_date
from feature_engine import build_features, DYNAMIC_FEATURES, STATIC_FEATURES
from feature_engine import STATIC_CATEGORICAL, STATIC_CONTINUOUS
from model import TFTEncoder, PortfolioPolicy, DiffusionDenoiser
from env import AShareTradingEnv
from rl_utils import build_port_state, ObsCache


def get_next_trade_date(last_date):
    cal_path = os.path.join(config.DATA_ROOT, "trade_cal.csv")
    cal = pd.read_csv(cal_path, dtype={"cal_date": str})
    cal = cal[cal["is_open"] == 1].sort_values("cal_date")
    future = cal[cal["cal_date"] > last_date]
    if future.empty:
        return None
    return future.iloc[0]["cal_date"]


def load_portfolio(path, env):
    port_df = pd.read_csv(path, dtype={"ts_code": str})
    holdings = np.zeros(env.n_stocks, dtype=np.int64)
    unknown = []
    for _, row in port_df.iterrows():
        code = row["ts_code"]
        shares = int(row["shares"])
        if code in env.code_to_idx:
            holdings[env.code_to_idx[code]] = shares
        else:
            unknown.append(code)
    if unknown:
        print(f"WARNING: {len(unknown)} codes not in universe: {unknown[:5]}")
    return holdings


def inject_placeholder_row(df, last_real_date, target_date):
    placeholder = df[df["trade_date"] == last_real_date].copy()
    placeholder["trade_date"] = target_date
    return pd.concat([df, placeholder], ignore_index=True)


def inject_daily_open(env, target_date, date_idx):
    """Load T-day open prices from daily_open and inject into env."""
    open_df = load_daily_open_for_date(target_date)
    if open_df is None:
        print(f"WARNING: no daily_open data for {target_date}, "
              f"close phase will use placeholder open prices")
        return False
    injected = 0
    for _, row in open_df.iterrows():
        code = row["ts_code"]
        if code in env.code_to_idx:
            si = env.code_to_idx[code]
            env.open_prices[date_idx, si] = row["open"]
            if "pre_close" in row and not pd.isna(row["pre_close"]):
                env.pre_close_prices[date_idx, si] = row["pre_close"]
            injected += 1
    print(f"Injected {injected} open prices for {target_date}")
    return True


def compute_open_price_ret(env, date_idx):
    """Compute T-day open return: open[T] / close[T-1] - 1."""
    if date_idx <= 0:
        return np.zeros(env.n_stocks, dtype=np.float32)
    prev_close = env.close_prices[date_idx - 1]
    cur_open = env.open_prices[date_idx]
    opr = np.where(
        ~np.isnan(cur_open) & ~np.isnan(prev_close) & (prev_close > 0),
        cur_open / prev_close - 1.0, 0.0).astype(np.float32)
    return opr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=str, required=True,
                        choices=["open", "close"],
                        help="Which phase to predict: open or close")
    parser.add_argument("--date", type=str, required=True,
                        help="Target trading date YYYYMMDD")
    parser.add_argument("--portfolio", type=str, default=None,
                        help="CSV with current holdings (ts_code, shares)")
    parser.add_argument("--cash", type=float, default=None,
                        help="Current cash (default: INIT_CAPITAL if no portfolio)")
    parser.add_argument("--episode-day", type=int, default=0,
                        help="Current day within the episode (0-9)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = os.path.join(config.CACHE_DIR, "best_rl_policy.pt")
    if not os.path.exists(ckpt_path):
        print(f"No RL model found at {ckpt_path}. Run train_rl.py first.")
        return

    # --- Build dataset (up to T-1 for features) ---
    end_date = config.TEST_END
    if args.date:
        end_date = max(end_date, args.date)
    df = build_merged_dataset(config.TRAIN_START, end_date)
    df, avail_features = build_features(df)

    # For close phase, inject a placeholder row if T-day data not in daily
    target_date = args.date
    last_real_date = df["trade_date"].max()
    if target_date > last_real_date:
        df = inject_placeholder_row(df, last_real_date, target_date)
        print(f"[info] target {target_date} > last data {last_real_date}, "
              f"using placeholder row")

    env = AShareTradingEnv(df)

    if target_date not in env.date_to_idx:
        print(f"Date {target_date} not in data range "
              f"[{env.dates[0]}, {env.dates[-1]}].")
        return
    date_idx = env.date_to_idx[target_date]

    if date_idx < config.SEQ_LEN:
        print(f"ERROR: date_idx={date_idx} < SEQ_LEN={config.SEQ_LEN}.")
        return

    # For close phase, inject daily_open data
    if args.phase == "close":
        inject_daily_open(env, target_date, date_idx)

    # --- Load model ---
    dynamic_dim = len(avail_features)
    static_dim = len(STATIC_FEATURES)

    encoder = TFTEncoder(dynamic_dim, static_dim,
                         hidden_dim=config.HIDDEN_DIM,
                         seq_len=config.SEQ_LEN,
                         num_heads=config.NUM_HEADS,
                         dropout=config.DROPOUT,
                         static_categorical=STATIC_CATEGORICAL,
                         static_n_continuous=len(STATIC_CONTINUOUS)).to(device)
    policy = PortfolioPolicy(config.HIDDEN_DIM, n_bins=config.N_BINS,
                             n_extra_state=config.N_EXTRA_STATE,
                             dropout=config.DROPOUT).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    if 'config' in ckpt:
        saved_cfg = ckpt['config']
        saved_univ = saved_cfg.get('UNIVERSE')
        cur_univ = universe_tag(getattr(config, 'UNIVERSE', None))
        if saved_univ is not None and saved_univ != cur_univ:
            print(f"WARNING: universe mismatch — "
                  f"trained={saved_univ}, current={cur_univ}")
        for key, cur in {'HIDDEN_DIM': config.HIDDEN_DIM,
                         'SEQ_LEN': config.SEQ_LEN,
                         'NUM_HEADS': config.NUM_HEADS,
                         'N_BINS': config.N_BINS,
                         'N_HOLD': config.N_HOLD}.items():
            saved = saved_cfg.get(key)
            if saved is not None and saved != cur:
                print(f"ERROR: config mismatch — {key}: "
                      f"trained={saved}, current={cur}")
                return

        saved_bins = saved_cfg.get('BINS')
        if saved_bins is not None and saved_bins != config.BINS:
            print(f"ERROR: BINS mismatch — "
                  f"trained={saved_bins}, current={config.BINS}")
            return

        saved_features = saved_cfg.get('DYNAMIC_FEATURES')
        if saved_features is not None and saved_features != avail_features:
            missing = set(saved_features) - set(avail_features)
            extra = set(avail_features) - set(saved_features)
            print(f"ERROR: feature mismatch.\n"
                  f"  Missing from current: {missing}\n"
                  f"  Extra in current: {extra}")
            return

    if any(k.startswith('denoiser.') for k in ckpt["encoder"].keys()):
        denoiser = DiffusionDenoiser(
            feature_dim=dynamic_dim,
            seq_len=config.SEQ_LEN,
            hidden_dim=config.DIFFUSION_HIDDEN_DIM,
            time_dim=config.DIFFUSION_TIME_DIM,
            n_timesteps=config.DIFFUSION_T,
            beta_start=config.DIFFUSION_BETA_START,
            beta_end=config.DIFFUSION_BETA_END,
        ).to(device)
        encoder.denoiser = denoiser

    encoder.load_state_dict(ckpt["encoder"])
    policy.load_state_dict(ckpt["policy"])
    encoder.eval()
    policy.eval()

    # --- Setup env state ---
    grouped = df.sort_values('trade_date').groupby('ts_code')
    obs_cache = ObsCache(grouped, avail_features, env, config.SEQ_LEN)

    env.reset(start_date_idx=date_idx, episode_len=config.EPISODE_LEN)
    env.episode_day = args.episode_day
    env.phase = args.phase

    # Inject real portfolio if provided
    if args.portfolio:
        holdings = load_portfolio(args.portfolio, env)
        env.holdings = holdings
        if args.cash is not None:
            env.cash = args.cash
        else:
            val_prices = env.get_valuation_prices()
            stock_val = np.nansum(holdings.astype(np.float64) * val_prices)
            env.cash = max(0, config.INIT_CAPITAL - stock_val)
        val_prices = env.get_valuation_prices()
        nav = env._compute_nav(val_prices)
        if nav > 0:
            hold_val = holdings.astype(np.float64) * np.nan_to_num(val_prices, 0)
            env.prev_weights = (hold_val / nav).astype(np.float64)
    elif args.cash is not None:
        env.cash = args.cash

    # --- Inference ---
    bins = torch.tensor(config.BINS, device=device, dtype=torch.float32)
    is_last_day = args.episode_day >= config.EPISODE_LEN - 1

    # Force liquidate on last day close phase
    if is_last_day and args.phase == "close":
        orders = []
        prices_now = env.get_execution_prices()
        for i in range(env.n_stocks):
            if env.holdings[i] > 0:
                if not np.isnan(prices_now[i]) and prices_now[i] > 0:
                    orders.append({
                        "date": target_date, "phase": args.phase,
                        "code": env.codes[i], "side": "sell",
                        "target_weight": 0.0,
                        "est_shares": 0,
                        "delta_shares": int(env.holdings[i]),
                        "ref_price": round(float(prices_now[i]), 3),
                        "current_held": int(env.holdings[i]),
                    })
        _output_results(orders, args, env, target_date)
        return

    # Compute open_price_ret for close phase
    open_price_ret = None
    if args.phase == "close":
        open_price_ret = compute_open_price_ret(env, date_idx)

    dyn_t, stat_t, mask_t = obs_cache.get_obs(date_idx, env, device)
    port_state = build_port_state(env, device, open_price_ret)

    with torch.no_grad():
        enc_bs = getattr(config, 'ENCODER_BATCH_SIZE', 0)
        n_stocks = dyn_t.shape[0]
        if enc_bs > 0 and n_stocks > enc_bs:
            enc_parts = []
            for _i in range(0, n_stocks, enc_bs):
                _end = min(_i + enc_bs, n_stocks)
                enc_parts.append(encoder(dyn_t[_i:_end], stat_t[_i:_end]))
            enc = torch.cat(enc_parts, dim=0)
        else:
            enc = encoder(dyn_t, stat_t)
        dist = policy(enc, port_state, mask_t, args.phase)
        action = dist.probs.argmax(dim=-1)

    weights = bins[action].cpu().numpy()
    top_k_idx = np.argsort(weights)[-config.N_HOLD:]

    target_w = np.zeros(env.n_stocks)
    for idx in top_k_idx:
        target_w[idx] = weights[idx]
    w_sum = target_w.sum()
    if w_sum > 0:
        target_w /= w_sum

    # --- Generate orders ---
    exec_prices = env.get_execution_prices()
    nav = env._compute_nav(env.get_valuation_prices())
    sellable_holdings = env.holdings.copy()
    current_holdings = env.holdings + env.locked
    orders = []

    # Stocks to buy/hold
    for i in top_k_idx:
        if target_w[i] <= 0:
            continue
        ref_price = exec_prices[i]
        if np.isnan(ref_price) or ref_price <= 0:
            continue
        from env import _round_to_lot
        est_shares = _round_to_lot(
            target_w[i] * nav / ref_price, env.lots[i])
        current_held = int(current_holdings[i])
        sellable = int(sellable_holdings[i])
        delta = est_shares - current_held
        if delta > 0:
            side = "buy"
        elif delta < 0:
            side = "sell"
            delta = min(-delta, sellable)
        else:
            side = "hold"
            delta = 0
        orders.append({
            "date": target_date, "phase": args.phase,
            "code": env.codes[i], "side": side,
            "target_weight": round(target_w[i], 6),
            "est_shares": est_shares,
            "delta_shares": delta if side != "hold" else 0,
            "ref_price": round(float(ref_price), 3),
            "current_held": current_held,
            "sellable": sellable,
        })

    # Stocks to sell (held but not in target)
    for i in range(env.n_stocks):
        held = int(current_holdings[i])
        sellable = int(sellable_holdings[i])
        if held > 0 and target_w[i] <= 0:
            ref_price = exec_prices[i]
            orders.append({
                "date": target_date, "phase": args.phase,
                "code": env.codes[i], "side": "sell",
                "target_weight": 0.0,
                "est_shares": 0,
                "delta_shares": sellable,
                "ref_price": round(float(ref_price), 3),
                "current_held": held,
                "sellable": sellable,
            })

    _output_results(orders, args, env, target_date)


def _output_results(orders, args, env, target_date):
    """Format and save prediction results."""
    result_df = pd.DataFrame(orders)
    out_path = os.path.join(config.CACHE_DIR, "latest_rl_signals.csv")
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    result_df.to_csv(out_path, index=False)

    print(f"\n{'='*60}")
    print(f"  {args.phase.upper()} PHASE prediction for {target_date}")
    print(f"  Episode day: {args.episode_day}/{config.EPISODE_LEN}")
    print(f"  Portfolio: {'custom' if args.portfolio else 'empty (100% cash)'}")
    print(f"{'='*60}")

    if result_df.empty:
        print("\n  No orders generated.")
        print(f"\nSaved to {out_path}")
        return

    buys = result_df[result_df["side"] == "buy"].sort_values(
        "target_weight", ascending=False)
    sells = result_df[result_df["side"] == "sell"]
    holds = result_df[result_df["side"] == "hold"]

    if not buys.empty:
        print(f"\n  BUY ({len(buys)} stocks):")
        print(buys[["code", "target_weight", "delta_shares",
                    "ref_price"]].to_string(index=False))
    if not sells.empty:
        print(f"\n  SELL ({len(sells)} stocks):")
        print(sells[["code", "target_weight", "delta_shares",
                     "ref_price", "current_held"]].to_string(index=False))
    if not holds.empty:
        print(f"\n  HOLD ({len(holds)} stocks):")
        print(holds[["code", "target_weight", "est_shares",
                     "ref_price"]].to_string(index=False))

    price_source = "T-1 close" if args.phase == "open" else "T-day open"
    print(f"\n  NOTE: est_shares based on {price_source} as ref_price.")
    print(f"  target_weight is the DEFINITIVE signal; shares are estimates.")

    # Generate next-call portfolio file
    next_holdings = []
    for _, row in result_df.iterrows():
        est = int(row.get("est_shares", 0))
        if est > 0:
            next_holdings.append({"ts_code": row["code"], "shares": est})
    if next_holdings:
        next_port_df = pd.DataFrame(next_holdings)
        next_port_path = os.path.join(config.CACHE_DIR, "next_holdings.csv")
        next_port_df.to_csv(next_port_path, index=False)

        spent = sum(r["delta_shares"] * r["ref_price"]
                    for _, r in result_df[result_df["side"] == "buy"].iterrows())
        est_cash = env.cash - spent * (1 + config.COMMISSION)
        for _, r in result_df[result_df["side"] == "sell"].iterrows():
            est_cash += r["delta_shares"] * r["ref_price"] * (
                1 - config.COMMISSION - config.STAMP)

        next_phase = "close" if args.phase == "open" else "open"
        next_day = args.episode_day if args.phase == "open" \
            else args.episode_day + 1
        next_date = target_date if args.phase == "open" \
            else "(next trading day)"

        print(f"\n  Next-call portfolio saved: {next_port_path}")
        print(f"  Estimated cash: {est_cash:,.0f}")
        print(f"  Next command:")
        print(f"    python predict_rl.py --phase {next_phase} "
              f"--date {next_date} "
              f"--episode-day {next_day} "
              f"--portfolio {next_port_path} "
              f"--cash {est_cash:.2f}")

    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
