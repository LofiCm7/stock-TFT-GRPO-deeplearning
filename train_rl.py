import os
import random
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import config
from data_loader import build_merged_dataset, universe_tag
from feature_engine import build_features, DYNAMIC_FEATURES, STATIC_FEATURES
from feature_engine import STATIC_CATEGORICAL, STATIC_CONTINUOUS
from model import TFTEncoder, PortfolioPolicy, DiffusionDenoiser, ReturnPredictor
from dataset import StockDataset
from env import AShareTradingEnv
from grpo_trainer import GRPOTrainer
from rl_utils import build_port_state, ObsCache, randomize_portfolio_state
from plot import plot_rl_reward_curve, plot_greedy_eval_curve
from greedy_eval import build_eval_windows, evaluate_greedy
from backtest_rl import load_benchmark_close_map


SEED = 37


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_data():
    print("Loading and preparing data...")
    df = build_merged_dataset(config.TRAIN_START, config.TEST_END)
    df, avail_features = build_features(df)
    return df, avail_features


def build_env(df):
    env = AShareTradingEnv(df)
    return env


def load_denoiser(device, dynamic_dim):
    """Load pretrained denoiser and freeze it."""
    if not config.USE_DIFFUSION_DENOISER:
        return None
    ckpt_path = os.path.join(config.CACHE_DIR, "denoiser_pretrained.pt")
    if not os.path.exists(ckpt_path):
        print(f"WARNING: {ckpt_path} not found. Running without denoiser.")
        return None
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    denoiser = DiffusionDenoiser(
        feature_dim=dynamic_dim,
        seq_len=config.SEQ_LEN,
        hidden_dim=config.DIFFUSION_HIDDEN_DIM,
        time_dim=config.DIFFUSION_TIME_DIM,
        n_timesteps=config.DIFFUSION_T,
        beta_start=config.DIFFUSION_BETA_START,
        beta_end=config.DIFFUSION_BETA_END,
    ).to(device)
    denoiser.load_state_dict(state['denoiser_state'])
    denoiser.eval()
    for p in denoiser.parameters():
        p.requires_grad = False
    print("Loaded and froze pretrained denoiser.")
    return denoiser


def warmup_encoder(encoder, return_predictor, df, avail_features, device):
    """用 close 前向收益率预热 encoder，与 aux 头目标统一。

    不再依赖 StockDataset 的 y（所有特征变化/std），而是直接从原始数据
    计算 close 的前向收益率作为回归目标，与 GRPO trainer 内部的 aux 头
    （return_predictor）目标一致。
    """
    print("Warming up encoder with close forward return...")
    close_idx = avail_features.index('close')
    seq_len = config.SEQ_LEN
    pred_horizon = config.PRED_HORIZON

    train_df = df[(df['trade_date'] >= config.TRAIN_START) &
                  (df['trade_date'] <= config.TRAIN_END)].copy()
    train_df = train_df.sort_values(['ts_code', 'trade_date'])

    # 直接计算 close 前向收益率：fwd_ret_t = close_{t+pred_horizon} / close_t - 1
    train_df['_close_fwd'] = train_df.groupby('ts_code')['close'].shift(-pred_horizon)
    train_df['_fwd_ret'] = train_df['_close_fwd'] / train_df['close'] - 1.0
    train_df = train_df.dropna(subset=['_fwd_ret'])
    train_df = train_df.reset_index(drop=True)

    # 构建样本：取 seq_len 天窗口，预测窗口结束日的 close 前向收益
    samples_dyn = []
    samples_stat = []
    samples_target = []
    for code, group in train_df.groupby('ts_code'):
        group = group.reset_index(drop=True)
        n = len(group)
        if n < seq_len + pred_horizon:
            continue
        dyn_data = group[avail_features].values.astype(np.float32)
        stat_data = group[STATIC_FEATURES].values.astype(np.float32)
        targets = group['_fwd_ret'].values.astype(np.float32)
        for i in range(seq_len - 1, n - pred_horizon):
            window = dyn_data[i - seq_len + 1:i + 1]
            if np.all(np.isnan(window)):
                continue
            # 窗口内标准化
            mean = window.mean(axis=0)
            std = window.std(axis=0, ddof=0) + 1e-8
            normed = (window - mean) / std
            normed = np.nan_to_num(normed, 0.0).astype(np.float32)
            samples_dyn.append(normed)
            samples_stat.append(stat_data[i])
            samples_target.append(targets[i])

    if not samples_dyn:
        print("WARNING: no warmup samples generated. Skipping warmup.")
        return

    dyn_arr = np.stack(samples_dyn)
    stat_arr = np.stack(samples_stat)
    target_arr = np.array(samples_target, dtype=np.float32)
    print(f"  Warmup samples: {len(samples_dyn)}")

    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(dyn_arr),
        torch.from_numpy(stat_arr),
        torch.from_numpy(target_arr))
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE,
                        shuffle=True, num_workers=0)

    encoder.train()
    return_predictor.train()
    opt = torch.optim.Adam(
        list(encoder.parameters()) + list(return_predictor.parameters()),
        lr=config.WARMUP_LR)
    for epoch in range(config.WARMUP_EPOCHS):
        total_loss = 0.0
        n_batch = 0
        for x_dyn, x_stat, y in loader:
            x_dyn = x_dyn.to(device)
            x_stat = x_stat.to(device)
            y = y.to(device).float()
            enc_out = encoder(x_dyn, x_stat)
            pred = return_predictor(enc_out)
            loss = F.mse_loss(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batch += 1
        print(f"  Warmup epoch {epoch+1}/{config.WARMUP_EPOCHS} "
              f"loss={total_loss/max(n_batch,1):.6f}")
    print("Encoder warmup complete.")


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df, avail_features = prepare_data()
    env = build_env(df)

    dynamic_dim = len(avail_features)
    static_dim = len(STATIC_FEATURES)

    encoder = TFTEncoder(dynamic_dim, static_dim,
                         hidden_dim=config.HIDDEN_DIM,
                         seq_len=config.SEQ_LEN,
                         num_heads=config.NUM_HEADS,
                         dropout=config.DROPOUT,
                         static_categorical=STATIC_CATEGORICAL,
                         static_n_continuous=len(STATIC_CONTINUOUS)).to(device)

    denoiser = load_denoiser(device, dynamic_dim)
    if denoiser is not None:
        encoder.denoiser = denoiser

    return_predictor = ReturnPredictor(config.HIDDEN_DIM).to(device)

    warmup_encoder(encoder, return_predictor, df, avail_features, device)

    policy = PortfolioPolicy(config.HIDDEN_DIM, n_bins=config.N_BINS,
                             n_extra_state=config.N_EXTRA_STATE,
                             dropout=config.DROPOUT).to(device)

    trainer = GRPOTrainer(encoder, policy, env,
                          return_predictor=return_predictor, device=device)

    train_dates = [i for i, d in enumerate(env.dates)
                   if config.TRAIN_START <= d <= config.TRAIN_END]
    seq_len = config.SEQ_LEN

    valid_starts = [d for d in train_dates[seq_len:]
                    if d + config.EPISODE_LEN - 1 < len(env.dates)]

    grouped = df.sort_values('trade_date').groupby('ts_code')
    obs_cache = ObsCache(grouped, avail_features, env, seq_len)

    # 固定窗口贪心评估：在 held-out(VAL) 区间取一组固定窗口，构成可跨步比较的学习曲线
    eval_starts = build_eval_windows(
        env, config.VAL_START, config.VAL_END, seq_len,
        config.EPISODE_LEN, config.EVAL_MAX_WINDOWS)
    eval_close_map = load_benchmark_close_map()
    if config.EVAL_INTERVAL > 0:
        print(f"Greedy eval: {len(eval_starts)} fixed VAL windows "
              f"every {config.EVAL_INTERVAL} steps.")

    print(f"RL training for {config.RL_STEPS} episodes "
          f"({len(valid_starts)} valid starts)...")
    print("Training encoder + policy end-to-end with GRPO from scratch.")
    best_ret = -np.inf
    reward_history = []
    eval_history = []  # (step, excess_mean, mean_ret, episode_ir, win_rate)
    t_start = time.time()

    # 用启动时间戳生成本次训练的唯一文件名，避免覆盖上一次的最佳模型
    run_tag = time.strftime("%Y%m%d_%H%M%S", time.localtime(t_start))
    best_model_name = f"best_rl_policy_{run_tag}.pt"

    def save_best(step):
        save_path = os.path.join(config.CACHE_DIR, best_model_name)
        os.makedirs(config.CACHE_DIR, exist_ok=True)
        torch.save({
            "encoder": encoder.state_dict(),
            "policy": policy.state_dict(),
            "step": step,
            "config": {
                'UNIVERSE': universe_tag(getattr(config, 'UNIVERSE', None)),
                'HIDDEN_DIM': config.HIDDEN_DIM,
                'NUM_HEADS': config.NUM_HEADS,
                'SEQ_LEN': config.SEQ_LEN,
                'DROPOUT': config.DROPOUT,
                'N_BINS': config.N_BINS,
                'BINS': config.BINS,
                'N_HOLD': config.N_HOLD,
                'N_EXTRA_STATE': config.N_EXTRA_STATE,
                'EPISODE_LEN': config.EPISODE_LEN,
                'MAX_CASH': config.MAX_CASH,
                'USE_DIFFUSION_DENOISER': config.USE_DIFFUSION_DENOISER,
                'DYNAMIC_FEATURES': avail_features,
            },
        }, save_path)

    for step in range(config.RL_STEPS):
        start_idx = random.choice(valid_starts)

        env.reset(start_date_idx=start_idx, episode_len=config.EPISODE_LEN)

        metrics = trainer.collect_trajectory_and_update(
            env, obs_cache, start_idx, device)

        reward_history.append(metrics['mean_reward'])
        if step % 10 == 0:
            elapsed = time.time() - t_start
            speed = (step + 1) / elapsed if elapsed > 0 else 0
            eta = (config.RL_STEPS - step - 1) / speed if speed > 0 else 0
            print(f"Episode {step}/{config.RL_STEPS} | "
                  f"loss={metrics['loss']:.4f} | "
                  f"mean_reward={metrics['mean_reward']:.6f} | "
                  f"best_reward={metrics['best_reward']:.6f} | "
                  f"kl={metrics['kl']:.4f} | "
                  f"{speed:.2f} ep/s | ETA {eta/60:.1f}min")

        # 固定窗口贪心评估：跨步可比的学习曲线，并据 mean_ret(绝对收益) 选最优
        if (config.EVAL_INTERVAL > 0 and eval_starts and
                step % config.EVAL_INTERVAL == 0):
            em = evaluate_greedy(encoder, policy, env, obs_cache, device,
                                 eval_starts, eval_close_map)
            if em is not None:
                alpha = em['excess_mean']
                mean_ret = em['mean_ret']
                eval_history.append((step, alpha, mean_ret,
                                     em['episode_ir'], em['win_rate']))
                tag = ""
                if mean_ret > best_ret:
                    best_ret = mean_ret
                    save_best(step)
                    tag = "  <- best"
                print(f"  [EVAL step {step}] ret={mean_ret*100:+.3f}%  "
                      f"alpha(excess)={alpha*100:+.3f}%  "
                      f"IR={em['episode_ir']:.3f}  "
                      f"beat={em['beat_bench_rate']*100:.0f}%  "
                      f"win={em['win_rate']*100:.0f}%{tag}")

    print(f"Training done. Best greedy-eval ret: {best_ret*100:+.3f}%")
    total_time = time.time() - t_start
    print(f"Total training time: {total_time/60:.1f} min "
          f"({total_time/config.RL_STEPS:.2f} s/episode)")
    print(f"Model saved to {os.path.join(config.CACHE_DIR, best_model_name)}")
    plot_rl_reward_curve(reward_history)
    plot_greedy_eval_curve(eval_history)


if __name__ == "__main__":
    main()
