import os

# ====== 修改这里为你本地的数据路径 ======
DATA_ROOT = os.environ.get("STOCK_DATA_ROOT",
                           "/data/sczli/Adata")
CACHE_DIR = os.environ.get("STOCK_CACHE_DIR",
                           "/data/sczli/Adata/cache")
# ==========================================

DAILY_DIR = os.path.join(DATA_ROOT, "daily")
METRIC_DIR = os.path.join(DATA_ROOT, "metric")
MONEYFLOW_DIR = os.path.join(DATA_ROOT, "moneyflow")
ST_DIR = os.path.join(DATA_ROOT, "stock_st")
BASIC_CSV = os.path.join(DATA_ROOT, "basic.csv")
MARKET_DIR = os.path.join(DATA_ROOT, "market")
DAILY_OPEN_DIR = os.path.join(DATA_ROOT, "daily_open")
INDEX_WEIGHT_DIR = os.path.join(DATA_ROOT, "index_weight")

# ====== 股票池选择 ======
# 旧开关，向后兼容：仅当 UNIVERSE 为 None 时生效
USE_CSI300 = True

# 新的可组合选股规格。非 None 时优先于 USE_CSI300 生效。
# 语义：include 各选择器结果取并集得到候选池，再减去 exclude 的并集。
# 北交所 / ST 始终被强制剔除（见 data_loader.filter_stocks），与此处无关。
#
# 选择器类型：
#   {"type": "index",    "code": "000300.SH"}         指数成分（可用 000300.SH / 399006.SZ）
#   {"type": "market",   "value": "主板"}             按板块（主板/创业板/科创板/北交所），value 可为 str 或 list
#   {"type": "industry", "value": ["银行", "半导体"]}  按行业，value 可为 str 或 list
#   {"type": "area",     "value": "深圳"}             按地域，value 可为 str 或 list
#   {"type": "codes",    "value": ["600519.SH"]}      按显式代码列表
#   {"type": "all"}                                    全市场（include 为空亦视为全市场）
#
# 示例与各设置下大致股票数见 docs/UNIVERSE.md。设为 None 则回退到 USE_CSI300。
UNIVERSE = {
    "name": "main_gem",
    "include": [
        {"type": "market", "value": ["主板", "创业板"]},
    ],
    "exclude": [],
}
UNIVERSE_TOP_N = 1500

IPO_SKIP_DAYS = 40
STATIC_EMBED_DIM = 16

TRAIN_START = "20210701"
TRAIN_END = "20240701"
VAL_START = "20240702"
VAL_END = "20250430"
TEST_START = "20210701"
TEST_END = "20260608"

SEQ_LEN = 30
PRED_HORIZON = 1

HIDDEN_DIM = 128
NUM_HEADS = 4
DROPOUT = 0.15
LR = 5e-4
EPOCHS = 150
BATCH_SIZE = 2048
PATIENCE = 25

N_HOLD = 30
K_GRAD = 60
K_GRAD_MARGIN_WEIGHT = 0.3
K_SWAP = 3
INIT_CAPITAL = 1_000_000

# ====== RL (GRPO) 超参 ======
N_BINS = 11
BINS = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20]
GRPO_G = 16
GRPO_BETA = 0.04
GRPO_REF_REFRESH = 150
GRPO_GAMMA = 0.95        # #3: MC return-to-go 的折扣系数。1.0=纯蒙特卡洛
                         # （总收益折合到每一步）；<1 降低逐步优势方差。
                         # 借鉴#5：由 1.0 下调至 0.95，给 t=0 的 return-to-go
                         # 减负（否则它背负整段 episode 的噪声）。
GFPO_KEEP_RATIO = 0.5    # 借鉴#3 (GFPO, 2508.09726)：每个 update 采 G 条轨迹，
                         # 按过滤指标只保留 top-(ratio*G) 条算梯度，其余优势置零。
                         # 用筛选做隐式 reward shaping、降低梯度方差。
                         # =1.0 时完全退回过滤前行为（n_kept==G）。
GFPO_TURNOVER_KAPPA = 0.08  # 借鉴#4 (GFPO token-efficiency 的避坑版)：过滤指标为
                         # filter_score = total_return - KAPPA * total_turnover。
                         # 用【减法】而非 reward/turnover 比值，避免除零（不交易轨迹
                         # turnover≈0 比值爆炸）与符号反转（亏损轨迹下比值排序反向）。
                         # KAPPA=0 时 filter_score==total_return，等价于纯收益排序。
                         # 仅用于轨迹排序，不进梯度/奖励，与 LAMBDA_TURNOVER 解耦。
LAMBDA_TURNOVER = 5e-5
COMMISSION = 3e-4
STAMP = 1e-3
MIN_COMMISSION = 5.0
# 涨跌停板限制：主板 ±10%，创业板/科创板 ±20%。
# 运行时 env 根据股票代码动态决定 per-stock limit_pct。
LIMIT_PCT = 0.10
GEM_STAR_LIMIT_PCT = 0.20  # 创业板（300/301）和科创板（688）
# A股最小交易单位：主板/创业板 100股，科创板 200股。
# 运行时 env 会根据股票代码动态决定 per-stock lot，此处仅作默认值。
LOT = 100
STAR_LOT = 200  # 科创板（688xxx）最小交易单位
RL_STEPS = 3000
LR_POLICY = 3e-4
LR_ENCODER = 3e-4
N_EXTRA_STATE = 7

# ====== Episode & Competition Constraints ======
EPISODE_LEN = 10
MAX_CASH = 150_000
LAMBDA_CASH_PENALTY = 0.01
ENCODER_BATCH_SIZE = 0

# ====== 改进方案超参 ======
WARMUP_EPOCHS = 5
WARMUP_LR = 1e-3
LAMBDA_AUX = 0.3
LAMBDA_BENCHMARK = 0.5
ALPHA_TEMPORAL = 0.3   # 时间维度优势的混合权重（截面优势权重 = 1 - ALPHA_TEMPORAL）

# ====== 诊断开关（默认不影响原算法）======
DIAG_INTERVAL = 0       # 每多少个 update 打印一次诊断指标；设 0 关闭
DIAG_SMOKE_TEST = False  # True 时用合成奖励测试 RL 管线（需手动开启）

# ====== 固定窗口贪心评估（衡量 RL 真实学习效果）======
# 每隔 EVAL_INTERVAL 个 RL step，在一组固定的 held-out(VAL) 窗口上以贪心
# (argmax) 方式跑策略，得到跨步可比的学习曲线，并据此(alpha)选最优模型。
EVAL_INTERVAL = 30       # 每多少个 RL step 评估一次；设 0 关闭
EVAL_MAX_WINDOWS = 40    # 评估用的固定窗口数（非重叠、均匀采样自 VAL 区间）

# ====== Diffusion Denoiser 超参 ======
USE_DIFFUSION_DENOISER = True
DIFFUSION_T = 200
DIFFUSION_BETA_START = 1e-4
DIFFUSION_BETA_END = 0.02
DIFFUSION_HIDDEN_DIM = 256
DIFFUSION_TIME_DIM = 128
DENOISE_T_START = 30
DENOISE_STEPS = 3
LAMBDA_DENOISE = 0.1
