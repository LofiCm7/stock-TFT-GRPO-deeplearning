# 项目架构与数据流图

## 整体数据流

```mermaid
flowchart TB
    subgraph Config["⚙️ config.py"]
        direction LR
        C1["DATA_ROOT / CACHE_DIR"]
        C2["TRAIN_START/END, VAL_START/END, TEST_START/END"]
        C3["SEQ_LEN=30, PRED_HORIZON=1"]
        C4["HIDDEN_DIM=128, N_BINS=6, N_EXTRA_STATE=6"]
        C5["EPISODE_LEN=10, N_HOLD=20"]
        C6["DIFFUSION_T=200, DENOISE_STEPS=3"]
    end

    subgraph DataPipeline["📦 数据加载与特征工程"]
        direction TB
        DP1["data_loader.py<br/>build_merged_dataset()"]
        DP2["feature_engine.py<br/>build_features()"]
        DP3["dataset.py<br/>StockDataset / rolling_normalize_window"]
    end

    subgraph Model["🧠 模型定义 (model.py)"]
        direction TB
        M1["DiffusionDenoiser<br/>扩散去噪器"]
        M2["TFTEncoder<br/>Temporal Fusion Transformer"]
        M3["PortfolioPolicy<br/>组合策略头"]
        M4["StaticEmbedding<br/>静态特征嵌入"]
        M5["VariableSelectionNetwork<br/>变量选择网络"]
    end

    subgraph Training["🎯 训练流程"]
        direction TB
        T1["train.py<br/>Denoiser 预训练"]
        T2["train_rl.py<br/>GRPO 端到端训练"]
        T3["grpo_trainer.py<br/>GRPOTrainer"]
    end

    subgraph Inference["🔮 推理与回测"]
        direction TB
        I1["predict_rl.py<br/>生成交易信号"]
        I2["backtest_rl.py<br/>回测评估"]
        I3["env.py<br/>AShareTradingEnv<br/>模拟交易环境"]
    end

    subgraph Utils["🛠️ 工具模块"]
        direction TB
        U1["rl_utils.py<br/>ObsCache / build_port_state"]
        U2["plot.py<br/>可视化"]
    end

    %% 数据流连接
    C1 --> DP1
    C2 --> DP1
    C3 --> DP1
    DP1 --> DP2
    DP2 --> DP3
    DP2 --> T1
    DP2 --> T2
    DP2 --> I1
    DP2 --> I2

    %% 模型连接
    C4 --> M3
    C6 --> M1
    M1 --> M2
    M2 --> M3

    %% 训练流程
    T1 -->|"denoiser_pretrained.pt"| T2
    T2 -->|"best_rl_policy.pt"| I1
    T2 -->|"best_rl_policy.pt"| I2
    T3 --> T2

    %% 推理流程
    I1 --> I3
    I2 --> I3
    U1 --> T2
    U1 --> I1
    U1 --> I2
    U2 --> T2
    U2 --> I2
```

## 训练数据流（详细）

```mermaid
flowchart LR
    subgraph Raw["原始数据"]
        R1["daily/*.csv<br/>日线行情"]
        R2["metric/*.csv<br/>估值指标"]
        R3["moneyflow/*.csv<br/>资金流"]
        R4["market/*.csv<br/>指数行情"]
        R5["basic.csv<br/>股票基本信息"]
        R6["stock_st/*.csv<br/>ST 标记"]
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
        F1["add_technical_features()<br/>MA偏差/RSI/MACD/ATR/布林带"]
        F2["add_moneyflow_features()<br/>主力净流入比/净流入MA5"]
        F3["add_fundamental_features()<br/>log市值/PE/PB截断"]
        F4["build_features()"]
    end

    subgraph Output["输出"]
        O1["merged DataFrame<br/>(ts_code, trade_date, 特征...)"]
        O2["DYNAMIC_FEATURES<br/>33维动态特征列表"]
        O3["STATIC_FEATURES<br/>5维静态特征"]
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
```

## 模型架构

```mermaid
flowchart TB
    subgraph Input["输入"]
        I1["dynamic_x<br/>(B, SEQ_LEN, 33)"]
        I2["static_x<br/>(B, 5)"]
    end

    subgraph Denoiser["DiffusionDenoiser (可选)"]
        D1["加噪: x_t = √ᾱ·x₀ + √(1-ᾱ)·ε"]
        D2["U-Net 风格去噪网络"]
        D3["DDIM 采样去噪"]
    end

    subgraph Encoder["TFTEncoder"]
        E1["dynamic_embedding<br/>Linear(1 → hidden_dim)"]
        E2["VariableSelectionNetwork<br/>BatchedVariableGRN × 33"]
        E3["pos_embedding<br/>位置编码"]
        E4["static_embedding<br/>StaticEmbedding<br/>(类别嵌入 + 连续投影)"]
        E5["static_encoder<br/>GRN"]
        E6["LSTM × 2层"]
        E7["enrichment_grn<br/>GRN"]
        E8["MultiheadAttention<br/>4头"]
        E9["attention_gate<br/>Sigmoid门控"]
        E10["post_attn_grn<br/>GRN"]
    end

    subgraph Policy["PortfolioPolicy"]
        P1["Concat(enc_features, port_state)"]
        P2["head_open<br/>MLP → N_BINS"]
        P3["head_close<br/>MLP → N_BINS"]
        P4["Categorical 分布"]
    end

    subgraph PortState["组合状态 (6维)"]
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
```

## GRPO 训练流程

```mermaid
sequenceDiagram
    participant Main as train_rl.py
    participant Trainer as GRPOTrainer
    participant Env as AShareTradingEnv
    participant Model as TFTEncoder + Policy
    participant Ref as Reference Policy

    Main->>Trainer: collect_trajectory_and_update()
    loop G=4 条轨迹
        Trainer->>Env: clone() + reset()
        loop EPISODE_LEN=10 天
            loop open/close 两个 phase
                Trainer->>Env: get_obs() + build_port_state()
                Trainer->>Model: encoder(dyn, stat) → policy(enc, port, mask, phase)
                Model-->>Trainer: action 分布
                Trainer->>Trainer: sample action → target_weights
                Trainer->>Env: step(target_weights)
                Env-->>Trainer: reward
                Trainer->>Ref: KL(当前策略 || 参考策略)
            end
        end
        Trainer->>Trainer: 记录 total_reward, total_log_prob, avg_kl
    end
    Trainer->>Trainer: 计算 advantages = (rewards - mean) / std
    Trainer->>Trainer: policy_loss = -mean(advantages × log_probs)
    Trainer->>Trainer: loss = policy_loss + β × kl_avg
    Trainer->>Model: backward() + optimizer.step()
    Trainer-->>Main: metrics {loss, mean_reward, best_reward, kl}
```

## 文件依赖关系

```mermaid
flowchart LR
    subgraph Core["核心模块"]
        config["config.py"]
        model["model.py"]
        env["env.py"]
    end

    subgraph Data["数据处理"]
        dl["data_loader.py"]
        fe["feature_engine.py"]
        ds["dataset.py"]
    end

    subgraph RL["强化学习"]
        rl["rl_utils.py"]
        grpo["grpo_trainer.py"]
    end

    subgraph Scripts["执行脚本"]
        train["train.py"]
        train_rl["train_rl.py"]
        predict["predict.py (DEPRECATED)"]
        backtest["backtest.py (DEPRECATED)"]
        predict_rl["predict_rl.py"]
        backtest_rl["backtest_rl.py"]
    end

    subgraph Viz["可视化"]
        plot["plot.py"]
    end

    subgraph Test["测试"]
        test_env["tests/test_env.py"]
    end

    %% 依赖关系
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
