# TFT + 扩散去噪 + GRPO 的 A 股强化学习选股系统 — 代码说明文档

> 本文档对应当前仓库的**强化学习（RL）版本**代码。
> 旧文档 [`PROJECT_DOC.md`](PROJECT_DOC.md) 描述的是早期「TFT + MSE 回归 + Top-K 轮动」方案，
> 其配置值（`SEQ_LEN=20 / HIDDEN_DIM=32 / EPOCHS=2`）与回归式架构均已**过时**，
> 请以本文件为准。文末「第 6 节」集中列出了需要优化的问题。

---

## 1. 系统概述

本项目把 A 股选股建模成一个**序列决策（强化学习）问题**，而非单纯的收益率回归：

- **特征编码器**：`TFTEncoder`（Temporal Fusion Transformer 变体），把每只股票过去 `SEQ_LEN=30`
  天的量价 / 资金流 / 基本面 / 指数特征，编码成一个隐向量。
- **扩散去噪器**：`DiffusionDenoiser`，先在训练集上独立预训练，再以**冻结**方式挂到编码器前端，
  对输入特征做去噪 / 数据增强。
- **组合策略**：`PortfolioPolicy`，基于编码向量 + 组合状态，为每只股票输出一个「仓位档位」分布。
- **交易环境**：`AShareTradingEnv`，模拟 A 股的 T+1、涨跌停、佣金 / 印花税 / 最低佣金、
  现金上限等真实约束，按 `open / close` 两个阶段撮合。
- **训练算法**：`GRPO`（Group Relative Policy Optimization），编码器与策略**端到端联合训练**，
  用「同一起点跑 G 条轨迹、组内相对优势」替代价值网络，并对参考策略加 KL 正则。

评估口径：样本外区间的净值曲线、年化收益、夏普、最大回撤（见 `backtest_rl.py`）。

---

## 2. 整体架构与数据流

```
                  ┌─────────────── 数据层 ───────────────┐
 daily/*.csv      │                                      │
 metric/*.csv ───►│ data_loader.build_merged_dataset()   │  按 CSI300 过滤、去 ST/北交所、
 moneyflow/*.csv  │   → 合并面板 + 静态分类编码 + IPO 过滤 │  剔除上市初期 IPO_SKIP_DAYS 天
 market/*.csv     │                                      │
 index_weight/    └──────────────────┬───────────────────┘
                                     ▼
                  feature_engine.build_features()
                   技术指标 / 资金流 / 基本面 / 指数特征 → avail_features
                                     ▼
        ┌────────────────────────────┴────────────────────────────┐
        ▼                                                          ▼
 train.py（阶段一）                                     train_rl.py（阶段二）
 DiffusionDenoiser 预训练                               TFTEncoder + PortfolioPolicy
   → denoiser_pretrained.pt                               端到端 GRPO 训练
                                                          ├─ env.py        (撮合与奖励)
                                                          ├─ rl_utils.py   (ObsCache/组合状态)
                                                          └─ grpo_trainer.py (GRPO 更新)
                                                            → best_rl_policy.pt
                                     ▼
              backtest_rl.py（样本外回测） / predict_rl.py（生成最新下单信号）
                                     ▼
                          plot.py（净值、回撤、奖励曲线等）
```

数据划分（`config.py`）：

| 用途 | 区间 | 配置项 |
|------|------|--------|
| 训练 | 2022-07-01 ~ 2024-07-01 | `TRAIN_START` / `TRAIN_END` |
| 验证 | 2024-07-02 ~ 2025-04-30 | `VAL_START` / `VAL_END` |
| 测试（样本外回测） | 2025-05-01 ~ 2026-05-01 | `TEST_START` / `TEST_END` |

> 注意：`VAL_START/VAL_END` 目前**只用于去噪器数据加载范围**，RL 训练（`train_rl.py`）
> 并没有用验证集做模型选择 / early stopping，详见第 6 节。

---

## 3. 文件 / 模块逐一说明

| 文件 | 角色 | 关键产物 |
|------|------|----------|
| `config.py` | 全局配置：路径、模型超参、回测约束、RL/扩散超参 | — |
| `data_loader.py` | 读取并合并 daily/metric/moneyflow/market CSV，过滤股票池，生成静态分类编码 | `merged_*.pkl` |
| `feature_engine.py` | 特征工程，定义 `DYNAMIC_FEATURES` / `STATIC_FEATURES` | `avail_features` |
| `dataset.py` | 滚动窗口标准化 + 样本缓存（旧回归流程用，RL 不直接用，仅复用 `rolling_normalize_window`） | `train_*.pkl` 等 |
| `model.py` | 模型定义：`DiffusionDenoiser` / `TFTEncoder` / `PortfolioPolicy` 等 | — |
| `train.py` | **阶段一**：扩散去噪器预训练 | `denoiser_pretrained.pt` |
| `env.py` | A 股交易环境（撮合 / 约束 / 奖励） | — |
| `rl_utils.py` | `ObsCache`（批量观测）、`build_port_state`（组合状态） | — |
| `grpo_trainer.py` | **阶段二**核心：GRPO 轨迹采样与参数更新 | — |
| `train_rl.py` | 阶段二入口：组装 env/encoder/policy，跑 GRPO | `best_rl_policy.pt` |
| `backtest_rl.py` | 样本外回测，贪心动作，输出净值与指标 | `backtest_rl_nav.csv` |
| `predict_rl.py` | 对指定日期生成可执行下单清单 | `latest_rl_signals.csv` |
| `plot.py` | 训练曲线 / 净值 / 回撤 / 奖励曲线 / 特征重要性 | `figures/*.png` |
| `backtest.py` / `predict.py` | **已废弃**，仅打印提示，转向 `*_rl.py` | — |
| `tests/test_env.py` | env 的 T+1、现金非负、涨停拒单单元测试（104 条） | — |

### 3.1 `data_loader.py`
- `build_merged_dataset(start, end)`：按日期范围读取各类 CSV → 合并 → 过滤（北交所 / ST / IPO 前 `IPO_SKIP_DAYS=40` 天）→ 生成 `industry_code/area_code/market_code/ent_type_code/stock_age` 静态特征 → pickle 缓存。
- 缓存键含 `data_loader.py + feature_engine.py` 的 MD5（`_data_loader_version`），源码改动自动失效。
- `load_market_features()`：构造 3 个指数（上证 / 沪深300 / 创业板）的涨跌幅、已实现波动、量比，以及跨指数的 `mkt_mean_ret_5d / large_small_spread / mkt_overnight_gap`。

### 3.2 `feature_engine.py`
- 36 个动态特征（量价 + 技术指标 + 资金流 + 基本面 + 指数），4 个静态分类 + 1 个连续（`stock_age`）。
- `build_features` 逐股 `groupby` 计算技术指标后 `concat`，返回 `(df, avail_features)`，`avail_features` 是实际存在的动态列。
- `TRIVIAL_FEATURES` 已定义但当前未被使用。

### 3.3 `dataset.py`
- `rolling_normalize_window(data, seq_len, end_idx)`：窗口内自标准化的**公共工具**，被 `rl_utils.get_obs_for_date` 复用。
- `StockDataset` 属于旧回归流程，目标 `y` 是「未来一日特征增量 / std」。RL 主链路**不经过它**，但缓存逻辑和标准化口径是参考实现。

---

## 4. 模型详解（`model.py`）

### 4.1 DiffusionDenoiser（扩散去噪器）
对「展平后的窗口特征」`(B, seq_len×feature_dim)` 做 DDPM 式建模：

- **前向加噪**：线性 beta 调度（`DIFFUSION_T=200`，beta 从 1e-4 到 0.02），`x_t = √ᾱ·x₀ + √(1-ᾱ)·ε`。
- **噪声预测网络**：`Linear→LayerNorm→SiLU` 三段 MLP，时间步用正弦嵌入经 `time_mlp` 注入（相加）。
- `compute_loss(x_0)`：随机 t 加噪后预测噪声，`MSE(pred_noise, noise)`（标准 DDPM 训练目标）。
- `denoise(x_raw, t_start, n_steps)`：从 `t_start` 出发，用 DDIM 式确定性反演逐步去噪 `DENOISE_STEPS=3` 步，返回去噪后的窗口。

挂载方式（`TFTEncoder.forward`）：训练阶段先对干净输入**重新加噪到 `DENOISE_T_START=30`** 再去噪，相当于一种「扩散式数据增强」。去噪器在 RL 阶段**全程冻结**（`requires_grad=False`，`eval()`）。

### 4.2 StaticEmbedding（静态特征嵌入）
- 每个分类特征独立 `nn.Embedding(cardinality, STATIC_EMBED_DIM=16)`；连续特征（`stock_age`）直接拼接。
- 拼接后经两层 MLP 投影到 `hidden_dim`。
- 基数定义在 `STATIC_CATEGORICAL`（industry=120, area=40, market=5, ent_type=12）。

### 4.3 GatedResidualNetwork（GRN）
TFT 的基础构件：`ELU → fc1 → dropout → fc2 → sigmoid 门控 → 残差 → LayerNorm`，输入输出维度不一致时用投影对齐。

### 4.4 BatchedVariableGRN + VariableSelectionNetwork（VSN）
- `BatchedVariableGRN`：用 `einsum('bsvh,vhd->bsvd')` 一次性对所有变量并行做 GRN，每个变量一套独立权重，等价于「逐变量 GRN」但向量化。
- `VSN`：把 `(B,seq,num_vars,hidden)` 展平后过 `joint_grn` 得到每个变量的 softmax 权重，再对 `batched_grn` 的逐变量输出加权求和 → `(B,seq,hidden)`。

### 4.5 TFTEncoder（主编码器）
前向流程（`forward(dynamic_x, static_x)`）：
1. （可选）扩散去噪 `dynamic_x`；
2. `dynamic_embedding`：标量特征 `Linear(1, hidden)` 升维 → `(B,seq,num_vars,hidden)`；
3. `VSN` 做变量选择 → `(B,seq,hidden)`；
4. 加 learnable 位置编码；
5. 静态分支：`StaticEmbedding → static_encoder(GRN)` 得 `static_context`；
6. `LSTM(2 层)` 处理时序；
7. `enrichment_grn(lstm_out + static_context)` 静态富化；
8. `MultiheadAttention(NUM_HEADS=4)` 自注意力；
9. 静态门控融合 `attn_gate * attn + (1-gate) * enrichment`；
10. `post_attn_grn` → 取**最后一个时间步** `[:, -1, :]` 作为编码向量 `(B, hidden)`。

### 4.6 PortfolioPolicy（策略头）
- 输入 `[enc_features(hidden) ‖ port_state(N_EXTRA_STATE=6)]`。
- 两个独立 MLP 头 `head_open` / `head_close`，各输出 `N_BINS=6` 个 logits。
- `mask` 把停牌 / 无效股票的 logits 置 `-1e9`。
- 返回 `Categorical(logits)` 分布；动作是「仓位档位」索引，`BINS=[0,0.025,0.05,0.10,0.15,0.20]` 给出对应目标权重。

> 组合层面的 Top-K 选择与归一化（取 `N_HOLD=20`、权重归一）**在网络之外**，由训练 / 回测循环手工完成（见 `grpo_trainer.py` 与 `backtest_rl.py`）。

---

## 5. 交易环境与 GRPO 训练

### 5.1 AShareTradingEnv（`env.py`）
- **面板预处理**：把长表转成 `(n_dates, n_stocks)` 的 `open/close/pre_close` 矩阵与 `suspended` 掩码；`pre_close` 缺失时用前一日 close 回填。
- **两阶段撮合**：每个交易日分 `open`（按开盘价）和 `close`（按收盘价）两次决策；`close` 结束后 `current_idx += 1`，并把当日买入的 `locked` 转入 `holdings`（**T+1**：当日买入不可卖）。
- **约束**：涨停不可买（`_can_buy`）、跌停不可卖（`_can_sell`）、佣金 `COMMISSION=3e-4`（最低 `MIN_COMMISSION=5`）、印花税 `STAMP=1e-3`（卖出）、整手 `LOT=100`。
- **现金上限**：现金超过 `MAX_CASH=150000` 触发 `cash_penalty` 并 `_force_reduce_cash` 强制按市值比例买入压低现金。
- **奖励**：`log(nav_after/nav_before) − LAMBDA_TURNOVER·turnover − cash_penalty`（对数收益 − 换手惩罚 − 现金惩罚）。
- **episode**：`EPISODE_LEN=10` 天为一段，最后一天 close 强制清仓。
- `clone()`：浅拷贝 + 关键数组深拷贝，供 GRPO 同起点采样 G 条轨迹。

### 5.2 ObsCache / build_port_state（`rl_utils.py`）
- `ObsCache` 预先把所有股票特征铺成 `(n_stocks, n_dates, n_feat)` 矩阵；`get_obs(date_idx)` 用 `nanmean/nanstd` 对窗口批量标准化，一次返回全市场的 `(dyn, stat, mask)`，避免逐股循环。
- `build_port_state` 输出每只股票 6 维状态：`[cash_frac, hold_frac, lock_frac, prev_w, ep_progress, is_last]`。

### 5.3 GRPOTrainer（`grpo_trainer.py`）
`collect_trajectory_and_update`：
1. 从同一 `start_idx` 用 `env.clone()` 跑 `G=4` 条完整 episode；
2. 每个 phase：编码 → 策略采样动作 → 映射成目标权重（Top-K 归一）→ `env.step`，累计 `log_prob`、奖励、与参考策略的逐步 KL；
3. 组内优势 `A = (R - mean(R)) / std(R)`（**无价值网络**，组内相对）；
4. 损失 `−mean(A·logπ) + beta·KL`（`GRPO_BETA=0.04`）；
5. 编码器（`LR_ENCODER=1e-5`）与策略（`LR_POLICY=3e-4`）分别用 Adam，梯度裁剪 1.0，**端到端**更新；
6. 参考策略每 `GRPO_REF_REFRESH=150` 步同步一次。

### 5.4 训练 / 回测 / 推理入口
- `train.py`：仅当 `USE_DIFFUSION_DENOISER=True` 时预训练去噪器，`EPOCHS=12`、`BATCH_SIZE=2048`。
- `train_rl.py`：`RL_STEPS=1500` 步，每步随机起点 + `randomize_portfolio_state` 随机初始组合；以**近 100 步平均奖励**作为最优判据保存 `best_rl_policy.pt`（含 config 快照）。
- `backtest_rl.py`：测试区间贪心 `argmax` 动作，逐 episode 滚动，算年化 / 夏普 / 回撤 / 超额。
- `predict_rl.py`：对单一日期输出 buy/sell 清单（含限价信息），处理最后一日清仓与现金上限补单。

---

## 6. 需要优化的问题清单

> 按「影响结果可信度 → 性能 → 架构 → 工程」分级。标注 🔴 严重 / 🟡 中等 / 🟢 轻微。

### 6.1 正确性 / 数据泄漏（最高优先级）

**🔴 P1 — 去噪器在推理时注入随机噪声，导致回测不可复现、GRPO 基线失真**
`model.py:241-253`，`TFTEncoder.forward` 无论训练还是 eval，每次前向都 `torch.randn_like(dynamic_x)` 重新加噪后去噪。后果有二：
- 回测 `backtest_rl.py` 里 `dist.probs.argmax` 的输入是随机的，**同一份权重跑两次结果不同**，评估指标不可复现。
- GRPO 同起点采样 G 条轨迹时，同一交易日每条轨迹看到的输入特征都不同；而组内相对优势 `A=(R-mean)/std` 的前提是「同一状态下比较不同动作」，输入都不一致会让 baseline 失真、增大梯度方差。
- **建议**：eval 走确定性路径（直接喂干净特征，或固定随机种子 / 去掉重加噪）；训练若要保留扩散增强，应与「同起点同输入」的 GRPO 假设解耦（例如对一个起点只采样一次噪声并在 G 条轨迹间共享）。

**🔴 P2 — open 阶段疑似使用当日收盘信息（未来函数）**
`rl_utils.py:56` 的窗口 `dynamic_matrix[:, date_idx-seq_len+1 : date_idx+1]` **包含 `date_idx` 当日**，而当日动态特征含 `close / pct_chg / ma*_dev / rsi_14 / macd_hist / vwap` 等收盘后才能知道的量。但 `env.py` 的 `open` 阶段是按 `open_prices[date_idx]` 撮合的。也就是**用当日开盘价下单时，模型已经看到了当日收盘价及其衍生指标** → 系统性高估回测收益。
- **建议**：确认是否设计意图。若非，`open` 阶段的观测窗口应截止到 `date_idx-1`（或对当日只暴露开盘可得的字段）。`close` 阶段使用当日全特征是合理的。

**🟡 P3 — RL 训练没有验证集模型选择**
`train_rl.py:173-197` 以「**训练区间内**近 100 步平均奖励」作为 `best_rl_policy.pt` 的保存判据，等于用训练集表现选模型，易过拟合。`VAL_START/VAL_END` 实际只用于去噪器数据范围。
- **建议**：周期性在验证区间跑一次贪心评估（年化 / 夏普 / Rank IC），用验证指标做 checkpoint 选择与早停。

**🟡 P4 — 标准化口径不一致**
`dataset.rolling_normalize_window` 用 `window.mean()/std()`（无 NaN 处理），而 `ObsCache.get_obs` 用 `nanmean/nanstd`。`get_obs_for_date`（走前者）与 `ObsCache`（走后者）对含 NaN 的窗口会给出不同结果。RL 主链路用 `ObsCache`，但两条路径并存容易埋坑。
- **建议**：统一为 NaN-safe 口径，废弃或修正 `get_obs_for_date`。

### 6.2 性能

**🟡 P5 — 编码器每步重复计算约 8×**
`grpo_trainer.py:69-74`：每个训练步执行 `G(4) × EPISODE_LEN(10) × 2 phase ≈ 80` 次全市场 TFT 前向。但 `enc = encoder(dyn,stat)` 只取决于 `date_idx`，与组合状态、phase、轨迹无关（去掉 P1 随机性后更是如此）。
- **建议**：对一个 episode 的每个唯一 `date_idx` 只编码一次，在 G 条轨迹与 open/close 间复用。预计可省到约 1/8 前向算力。注意编码器要参与反向传播，需保留计算图（不能 `no_grad` 缓存）。

**🟡 P6 — `env.step` 的逐股 Python 循环**
`env.py:155-196` 对 `n_stocks≈300` 逐只算 `target_shares`、卖出、买入。叠加 `80 step × 1500 episode`，是 CPU 端主要瓶颈。
- **建议**：用 numpy 向量化目标股数与卖出，买入排序后批量处理；`_compute_nav` 等已是向量化，可一并复用。

**🟢 P7 — 去噪器对 batch 内每个样本重复跑 3 步反演**
全市场每个 phase 都对 ~300 股做 `DENOISE_STEPS=3` 的逐步反演。若 P5 落地（按日期缓存编码），去噪也随之只跑一次/日，问题缓解。

### 6.3 模型 / 架构

**🟡 P8 — VSN 未使用静态上下文**
`model.py:197-203` 的变量选择只吃动态特征。标准 TFT 用 static context 调制变量选择权重；这里静态信息只在 LSTM 之后才以 `enrichment + gate` 注入。
- **建议**：把 `static_context` 投影后并入 `joint_grn` 输入，让变量重要性随个股属性自适应。

**🟡 P9 — 选股是逐股独立决策，缺少截面建模**
策略对每只股票独立输出档位，组合的 Top-K 与归一化在网络外完成（`grpo_trainer.py:90-97`）。网络从未显式学习**截面排序**，而选股本质是截面相对强弱比较。
- **建议**：在编码向量上加截面归一化（cross-sectional normalize / ranking）或一层跨股票注意力，让模型感知「相对其他股票」的强弱。

**🟢 P10 — 动作空间表达力有限**
`BINS=[0,0.025,0.05,0.10,0.15,0.20]` 6 档 + 外部 Top-20 归一，权重粒度粗，且「先独立选档位再归一」与真实目标权重存在偏差。可考虑直接输出连续权重（再投影到单纯形）或更细档位。

**🟢 P11 — `head_open` / `head_close` 完全独立**
两个 phase 用两套独立 MLP，参数翻倍且不共享表示。可改为共享主干 + phase 嵌入，减小参数、共享统计强度。

### 6.4 配置 / 工程

**🟢 P12 — 数据路径硬编码为 Linux 绝对路径**
`config.py:4-7` 默认 `/data/sczli/Adata`，本机（Windows）不存在，导致除 `tests/` 外的脚本无法直接跑。建议用相对路径或在 README 注明必须设 `STOCK_DATA_ROOT` 环境变量。

**🟢 P13 — 旧文档与废弃文件**
`PROJECT_DOC.md` 描述的是已废弃的回归流程，配置值与现状不符；`backtest.py` / `predict.py` 已是空壳。建议归档或删除，避免误导。

**🟢 P14 — `randomize_portfolio_state` 与回测起始状态不一致**
训练时随机初始化组合（`train_rl.py:78`），但回测每个 episode 都从空仓 + 满现金开始（`backtest_rl.py:129`）。训练分布与评估分布存在 gap，可能影响策略在「持仓延续」场景下的表现。

### 6.5 建议的落地顺序
1. 先修 **P1 / P2**（不修，后续所有指标都不可信）；
2. 加 **P3** 验证集评估（建立可靠的改进度量）；
3. 做 **P5 / P6** 提速（让实验迭代更快）；
4. 再尝试 **P8 / P9** 架构增强（用 P3 的指标验证收益）。

> 说明：本机数据根目录 `/data/sczli/Adata` 不存在、torch 为 CPU 版，因此上述涉及训练 / 回测的改动**无法在本机端到端验证**；`tests/test_env.py`（104 条，当前全部通过）可覆盖 env 撮合逻辑，模型前向可用随机张量做 smoke test。
