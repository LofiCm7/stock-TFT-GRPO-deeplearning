# Render Mermaid diagrams from ARCHITECTURE.md to PNG images.
# Uses mermaid.ink API (free, no auth needed) to convert Mermaid code to PNG.
# Saves to: C:/Users/Lenovo/Desktop/python_project/stock4/

import base64
import requests
import os
from pathlib import Path

OUTPUT_DIR = Path("C:/Users/Lenovo/Desktop/python_project/stock4")

# Mermaid diagrams - simplified to avoid encoding issues with special chars
DIAGRAMS = {
    "01_overall_dataflow": """
flowchart TB
    subgraph Config["config.py"]
        direction LR
        C1["DATA_ROOT / CACHE_DIR"]
        C2["TRAIN_START/END, VAL_START/END, TEST_START/END"]
        C3["SEQ_LEN=30, PRED_HORIZON=1"]
        C4["HIDDEN_DIM=128, N_BINS=6, N_EXTRA_STATE=6"]
        C5["EPISODE_LEN=10, N_HOLD=20"]
        C6["DIFFUSION_T=200, DENOISE_STEPS=3"]
    end

    subgraph DataPipeline["Data Loading & Feature Engineering"]
        direction TB
        DP1["data_loader.py<br/>build_merged_dataset()"]
        DP2["feature_engine.py<br/>build_features()"]
        DP3["dataset.py<br/>StockDataset / rolling_normalize_window"]
    end

    subgraph Model["Model (model.py)"]
        direction TB
        M1["DiffusionDenoiser"]
        M2["TFTEncoder"]
        M3["PortfolioPolicy"]
        M4["StaticEmbedding"]
        M5["VariableSelectionNetwork"]
    end

    subgraph Training["Training"]
        direction TB
        T1["train.py<br/>Denoiser Pretraining"]
        T2["train_rl.py<br/>GRPO End-to-End"]
        T3["grpo_trainer.py<br/>GRPOTrainer"]
    end

    subgraph Inference["Inference & Backtest"]
        direction TB
        I1["predict_rl.py<br/>Generate Signals"]
        I2["backtest_rl.py<br/>Backtest"]
        I3["env.py<br/>AShareTradingEnv"]
    end

    subgraph Utils["Utilities"]
        direction TB
        U1["rl_utils.py<br/>ObsCache / build_port_state"]
        U2["plot.py<br/>Visualization"]
    end

    C1 --> DP1
    C2 --> DP1
    C3 --> DP1
    DP1 --> DP2
    DP2 --> DP3
    DP2 --> T1
    DP2 --> T2
    DP2 --> I1
    DP2 --> I2

    C4 --> M3
    C6 --> M1
    M1 --> M2
    M2 --> M3

    T1 -->|"denoiser_pretrained.pt"| T2
    T2 -->|"best_rl_policy.pt"| I1
    T2 -->|"best_rl_policy.pt"| I2
    T3 --> T2

    I1 --> I3
    I2 --> I3
    U1 --> T2
    U1 --> I1
    U1 --> I2
    U2 --> T2
    U2 --> I2
""",

    "02_training_dataflow": """
flowchart LR
    subgraph Raw["Raw Data"]
        R1["daily/*.csv<br/>Daily Quotes"]
        R2["metric/*.csv<br/>Valuation Metrics"]
        R3["moneyflow/*.csv<br/>Money Flow"]
        R4["market/*.csv<br/>Index Data"]
        R5["basic.csv<br/>Stock Info"]
        R6["stock_st/*.csv<br/>ST Marks"]
    end

    subgraph Merge["data_loader.py"]
        M1["load_daily_data()"]
        M2["load_metric_data()"]
        M3["load_moneyflow_data()"]
        M4["load_market_features()"]
        M5["load_basic_info()"]
        M6["load_st_codes()"]
        M7["filter_stocks()"]
        M8["build_merged_dataset()"]
    end

    subgraph Feature["feature_engine.py"]
        F1["add_technical_features()<br/>MA/RSI/MACD/ATR/Bollinger"]
        F2["add_moneyflow_features()<br/>Net Inflow Ratio"]
        F3["add_fundamental_features()<br/>logMktCap/PE/PB"]
        F4["build_features()"]
    end

    subgraph Output["Output"]
        O1["merged DataFrame<br/>(ts_code, trade_date, features...)"]
        O2["DYNAMIC_FEATURES<br/>33 Dynamic Features"]
        O3["STATIC_FEATURES<br/>5 Static Features"]
    end

    R1 --> M1
    R2 --> M2
    R3 --> M3
    R4 --> M4
    R5 --> M5
    R6 --> M6
    M1 --> M8
    M2 --> M8
    M3 --> M8
    M4 --> M8
    M5 --> M8
    M6 --> M8
    M7 --> M8
    M8 --> F4
    F1 --> F4
    F2 --> F4
    F3 --> F4
    F4 --> O1
    F4 --> O2
    F4 --> O3
""",

    "03_model_architecture": """
flowchart TB
    subgraph Input["Input"]
        I1["dynamic_x<br/>(B, SEQ_LEN, 33)"]
        I2["static_x<br/>(B, 5)"]
    end

    subgraph Denoiser["DiffusionDenoiser (Optional)"]
        D1["Noise: x_t = sqrt(ac)*x0 + sqrt(1-ac)*eps"]
        D2["U-Net Style Denoising"]
        D3["DDIM Sampling"]
    end

    subgraph Encoder["TFTEncoder"]
        E1["dynamic_embedding<br/>Linear(1 -> hidden_dim)"]
        E2["VariableSelectionNetwork<br/>BatchedVariableGRN x 33"]
        E3["pos_embedding<br/>Position Encoding"]
        E4["static_embedding<br/>StaticEmbedding"]
        E5["static_encoder<br/>GRN"]
        E6["LSTM x 2 layers"]
        E7["enrichment_grn<br/>GRN"]
        E8["MultiheadAttention<br/>4 heads"]
        E9["attention_gate<br/>Sigmoid Gate"]
        E10["post_attn_grn<br/>GRN"]
    end

    subgraph Policy["PortfolioPolicy"]
        P1["Concat(enc_features, port_state)"]
        P2["head_open<br/>MLP -> N_BINS"]
        P3["head_close<br/>MLP -> N_BINS"]
        P4["Categorical Distribution"]
    end

    subgraph PortState["Portfolio State (6-dim)"]
        PS1["cash_frac"]
        PS2["hold_frac"]
        PS3["lock_frac"]
        PS4["prev_w"]
        PS5["ep_progress"]
        PS6["is_last"]
    end

    I1 --> D1
    D1 --> D2
    D2 --> D3
    D3 --> E1
    I1 --> E1
    E1 --> E2
    E2 --> E3
    I2 --> E4
    E4 --> E5
    E5 --> E7
    E3 --> E6
    E6 --> E7
    E7 --> E8
    E8 --> E9
    E9 --> E10
    E10 --> P1
    PS1 --> P1
    PS2 --> P1
    PS3 --> P1
    PS4 --> P1
    PS5 --> P1
    PS6 --> P1
    P1 --> P2
    P1 --> P3
    P2 --> P4
    P3 --> P4
""",

    "04_grpo_training": """
sequenceDiagram
    participant Main as train_rl.py
    participant Trainer as GRPOTrainer
    participant Env as AShareTradingEnv
    participant Model as TFTEncoder + Policy
    participant Ref as Reference Policy

    Main->>Trainer: collect_trajectory_and_update()
    loop G=4 trajectories
        Trainer->>Env: clone() + reset()
        loop EPISODE_LEN=10 days
            loop open/close 2 phases
                Trainer->>Env: get_obs() + build_port_state()
                Trainer->>Model: encoder(dyn, stat) -> policy(enc, port, mask, phase)
                Model-->>Trainer: action distribution
                Trainer->>Trainer: sample action -> target_weights
                Trainer->>Env: step(target_weights)
                Env-->>Trainer: reward
                Trainer->>Ref: KL(current || reference)
            end
        end
        Trainer->>Trainer: record total_reward, total_log_prob, avg_kl
    end
    Trainer->>Trainer: advantages = (rewards - mean) / std
    Trainer->>Trainer: policy_loss = -mean(advantages * log_probs)
    Trainer->>Trainer: loss = policy_loss + beta * kl_avg
    Trainer->>Model: backward() + optimizer.step()
    Trainer-->>Main: metrics {loss, mean_reward, best_reward, kl}
""",

    "05_file_dependencies": """
flowchart LR
    subgraph Core["Core Modules"]
        config["config.py"]
        model["model.py"]
        env["env.py"]
    end

    subgraph Data["Data Processing"]
        dl["data_loader.py"]
        fe["feature_engine.py"]
        ds["dataset.py"]
    end

    subgraph RL["Reinforcement Learning"]
        rl["rl_utils.py"]
        grpo["grpo_trainer.py"]
    end

    subgraph Scripts["Execution Scripts"]
        train["train.py"]
        train_rl["train_rl.py"]
        predict["predict.py (DEPRECATED)"]
        backtest["backtest.py (DEPRECATED)"]
        predict_rl["predict_rl.py"]
        backtest_rl["backtest_rl.py"]
    end

    subgraph Viz["Visualization"]
        plot["plot.py"]
    end

    subgraph Test["Testing"]
        test_env["tests/test_env.py"]
    end

    dl --> config
    fe --> config
    ds --> config
    ds --> fe
    model --> config
    env --> config
    rl --> config
    rl --> ds
    rl --> fe
    grpo --> config
    grpo --> rl

    train --> config
    train --> dl
    train --> fe
    train --> model

    train_rl --> config
    train_rl --> dl
    train_rl --> fe
    train_rl --> model
    train_rl --> env
    train_rl --> grpo
    train_rl --> rl
    train_rl --> plot

    predict_rl --> config
    predict_rl --> dl
    predict_rl --> fe
    predict_rl --> model
    predict_rl --> env
    predict_rl --> rl

    backtest_rl --> config
    backtest_rl --> dl
    backtest_rl --> fe
    backtest_rl --> model
    backtest_rl --> env
    backtest_rl --> rl
    backtest_rl --> plot

    test_env --> env
""",
}


def mermaid_to_png(mermaid_code: str, output_path: str):
    """Convert Mermaid code to PNG using mermaid.ink API with base64 encoding."""
    # Use plain base64 encoding (no compression) for better compatibility
    encoded = base64.b64encode(mermaid_code.encode('utf-8')).decode('ascii')
    url = f"https://mermaid.ink/img/{encoded}"

    print(f"  Fetching from mermaid.ink API...")
    response = requests.get(url, timeout=60)
    response.raise_for_status()

    with open(output_path, 'wb') as f:
        f.write(response.content)
    print(f"  Saved to {output_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Rendering {len(DIAGRAMS)} Mermaid diagrams to PNG...")
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    for name, mermaid_code in DIAGRAMS.items():
        output_path = OUTPUT_DIR / f"{name}.png"
        print(f"[{name}]")
        try:
            mermaid_to_png(mermaid_code.strip(), str(output_path))
        except Exception as e:
            print(f"  ERROR: {e}")
        print()

    print("Done! All diagrams rendered.")


if __name__ == "__main__":
    main()
