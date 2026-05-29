"""
Diffusion Denoiser Pre-training

Pre-trains the DiffusionDenoiser on dynamic features from the merged dataset.
The trained denoiser can then be loaded and frozen in train_rl.py to help
stabilize the denoising process (TFTEncoder.forward).

This script uses the same data pipeline as train_rl.py:
  build_merged_dataset() -> build_features() -> extract dynamic arrays

Usage:
    python train.py
"""
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import config
from data_loader import build_merged_dataset
from feature_engine import build_features, DYNAMIC_FEATURES, STATIC_FEATURES
from model import DiffusionDenoiser


SEED = 42


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_denoiser_dataset(df, avail_features, seq_len):
    """Extract sliding windows of dynamic features from the merged DataFrame.

    For each stock, extract all valid [seq_len] windows and normalize
    using the window's own mean/std (same logic as ObsCache / dataset.py).

    Returns:
        windows: np.ndarray of shape (N, seq_len, n_features)
    """
    print("Building denoiser training samples...")
    windows = []
    grouped = df.sort_values('trade_date').groupby('ts_code')

    for code, group in grouped:
        data = group[avail_features].values.astype(np.float32)
        n = len(data)
        if n < seq_len + 1:
            continue

        for i in range(seq_len, n):
            window = data[i - seq_len:i]
            if np.any(np.isnan(window)):
                continue
            mean = window.mean(axis=0, keepdims=True)
            std = window.std(axis=0, keepdims=True, ddof=0) + 1e-8
            normed = (window - mean) / std
            windows.append(np.nan_to_num(normed, 0.0))

    windows = np.stack(windows, axis=0).astype(np.float32)
    print(f"Total samples: {len(windows)}, shape: {windows.shape}")
    return windows


def train_denoiser(denoiser, loader, optimizer, device, scaler=None):
    """Train the diffusion denoiser for one epoch."""
    denoiser.train()
    total_loss = 0.0
    n = 0
    for x_dyn, in loader:
        x_dyn = x_dyn.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=scaler is not None):
            loss = denoiser.compute_loss(x_dyn)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(denoiser.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(denoiser.parameters(), 1.0)
            optimizer.step()
        total_loss += loss.item() * len(x_dyn)
        n += len(x_dyn)
    return total_loss / n


def main():
    if not config.USE_DIFFUSION_DENOISER:
        print("USE_DIFFUSION_DENOISER=False, skipping denoiser pretraining.")
        return

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Data pipeline (same as train_rl.py) ----
    print("Loading data...")
    df = build_merged_dataset(start_date=config.TRAIN_START,
                              end_date=config.VAL_END)
    print("Building features...")
    df, avail_features = build_features(df)

    # Use only training period
    train_df = df[df['trade_date'] <= config.TRAIN_END]
    print(f"Train rows: {len(train_df)}")

    # Build sliding-window samples
    windows = build_denoiser_dataset(train_df, avail_features, config.SEQ_LEN)

    # ---- DataLoader ----
    dataset = TensorDataset(torch.from_numpy(windows))
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE,
                        shuffle=True, num_workers=4, pin_memory=True,
                        worker_init_fn=lambda wid: np.random.seed(SEED + wid))

    # ---- Model ----
    denoiser = DiffusionDenoiser(
        feature_dim=len(avail_features),
        seq_len=config.SEQ_LEN,
        hidden_dim=config.DIFFUSION_HIDDEN_DIM,
        time_dim=config.DIFFUSION_TIME_DIM,
        n_timesteps=config.DIFFUSION_T,
        beta_start=config.DIFFUSION_BETA_START,
        beta_end=config.DIFFUSION_BETA_END,
    ).to(device)

    optimizer = torch.optim.Adam(denoiser.parameters(), lr=config.LR)

    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    # ---- Training ----
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    denoiser_path = os.path.join(config.CACHE_DIR, "denoiser_pretrained.pt")

    print(f"Training denoiser for {config.EPOCHS} epochs...")
    for epoch in range(config.EPOCHS):
        train_loss = train_denoiser(denoiser, loader, optimizer, device, scaler)
        print(f"Epoch {epoch + 1:02d} | Denoiser Loss: {train_loss:.6f}")

    torch.save(denoiser.state_dict(), denoiser_path)
    print(f"Denoiser saved to: {denoiser_path}")


if __name__ == "__main__":
    main()
