"""电力变压器时序异构图知识图谱数据集。

从 ETTh1 / ETTh2 等 CSV 格式的小时级变压器运行数据出发：

1. 解析日期时间、计算健康状态标签（基于行业阈值）
2. 对特征列执行 z-score 标准化，保持原始值用于展示和反标准化
3. 构建 PyG `HeteroData` 异构图，包含
   - `transformer` 节点：设备元数据（id / 类别 / 额定电压）
   - `time_slice` 节点：时序运行切片（特征、健康标签、油温标签、时间戳）
   - `health_state` 节点：4 种健康状态的嵌入
   - `feature_indicator` 节点：特征指标的嵌入
   - 以及连接这些节点的 5 种边（含 2 种带时间戳的时序边）

调用示例::

    from src.dataset.transformer_temporal_kg import TransformerTemporalKG
    dataset = TransformerTemporalKG(csv_path="/path/to/ETTh2.csv")
    kg_graph = dataset.data          # HeteroData，可直接送入 RGCN / TGN
    df_raw = dataset.df_raw           # 原始值 DataFrame，方便可视化
    df_train = dataset.df_train       # 标准化后的 DataFrame
    print(dataset.feature_list)       # ['HUFL', ..., 'OT']
    print(dataset.health_mapping)     # {0: "正常运行", ..., 3: "过载故障"}
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Tuple

# 允许在 `python src/dataset/temporal_dataset.py` 这样的直接执行
# 场景下找到顶层的 numpy / pandas / torch 等依赖；正常 import 时无副作用
_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from torch_geometric.data import HeteroData  # noqa: E402
from typing import Optional  # noqa: E402

from src.core.config import load_health_mapping, logger  # noqa: E402

# ---------------------------------------------------------------------------
# 健康状态标签映射（从 config/config.yaml 读取）
# ---------------------------------------------------------------------------
# 懒加载：首次调用 get_health_mapping() 时才读取配置并打日志，
# 避免本模块被 import 时（如 src/tlp 仅需要 src.dataset.tlp_dataset）
# 就触发整条 src 包导入链并输出无关 INFO 日志。

def get_health_mapping() -> Dict[int, str]:
    """返回健康状态标签映射（dict[int, str]），首次调用时从 config.yaml 加载并缓存。"""
    if get_health_mapping._cache is None:
        get_health_mapping._cache = load_health_mapping()
    return get_health_mapping._cache
get_health_mapping._cache = None


class KGTemporalDataset:
    """知识图谱时序异构图数据集。

    所有列名与阈值均通过构造参数配置或从 CSV 自动推断，无任何模块级硬编码。

    Parameters
    ----------
    csv_path : str
        时序 CSV 数据文件路径。
    transformer_id : int, default 0
        transformer 节点的元数据 id，用于多设备场景区分。
    feature_list : list[str] | None, default None
        指定参与训练的特征列；不传时自动从 CSV 提取所有数值型列。
    date_col : str | None, default None
        时间戳列名（用于时序边顺序）；不传时自动从 CSV 中查找含
        "date"/"time"/"timestamp" 关键词的列。
    target_col : str | None, default None
        目标列名（用于预测 & 过热诊断）；不传时默认取 CSV 最后一个数值列。
    overload_cols : list[str] | None, default None
        过载判据列名；不传则无过载列（健康状态 3 不会被触发）。
    overload_thresholds : list[float] | None, default None
        与 ``overload_cols`` 一一对应的过载阈值（真实物理单位）；
        不传且指定了 overload_cols 时按"均值 + 2σ"自动估计。
    target_mild_threshold : float | None, default None
        目标列的轻微过热阈值（真实单位）；不传时取 80 百分位。
    target_severe_threshold : float | None, default None
        目标列的严重过热阈值（真实单位）；不传时取 95 百分位。
    future_steps : int, default 3
        目标列的超前预测步数。
    chunk_size : int, default 0
        分块流式读取的块大小（行数）。>0 时启用**流式读取**，避免一次性
        把超大 CSV 全量载入内存；峰值内存仅与 chunk_size 成正比。
    reservoir_size : int, default 200000
        流式模式下用于估计百分位阈值（80/95 百分位过热阈值）的蓄水池采样上限。
    device : torch.device | str | None, default None
        HeteroData 最终落盘的设备；默认使用 CUDA 如可用否则 CPU。
    """

    # ------------------------------------------------------------------
    # 构造
    # ------------------------------------------------------------------
    def __init__(
            self,
            csv_path: str,
            transformer_id: int = 0,
            feature_list: Optional[List[str]] = None,
            date_col: Optional[str] = None,
            target_col: Optional[str] = None,
            overload_cols: Optional[List[str]] = None,
            overload_thresholds: Optional[List[float]] = None,
            target_mild_threshold: Optional[float] = None,
            target_severe_threshold: Optional[float] = None,
            future_steps: int = 3,
            chunk_size: int = 0,
            reservoir_size: int = 200000,
            device=None,
    ) -> None:
        self.csv_path = csv_path
        self.transformer_id = transformer_id
        # 流式读取开关：chunk_size>0 即启用分批读取
        self.chunk_size = int(chunk_size)
        self.streaming = self.chunk_size > 0
        self.reservoir_size = int(reservoir_size)

        # ---------- 轻量 CSV 预读：用于自动检测列名、推断统计阈值 ----------
        # 只读前 200 行足以做 dtype 检测与简单统计，避免加载大文件
        sample = pd.read_csv(csv_path, nrows=200)

        # ---------- 列名自动检测 ----------
        if date_col is None:
            date_col = KGTemporalDataset._detect_date_col(sample)
        self.date_col = date_col

        if target_col is None:
            target_col = KGTemporalDataset._detect_target_col(
                sample, self.date_col
            )
        self.target_col = target_col

        if overload_cols is None:
            overload_cols = []
        self.overload_cols: List[str] = list(overload_cols)

        # ---------- 阈值自动推断 ----------
        # 流式模式下，阈值推迟到流式扫描时基于「全量数据」统计（更准确），
        # 这里仅处理「非流式」或「用户显式给定」的情况。
        if overload_thresholds is None:
            if not self.streaming:
                # 非流式：按"均值 + 2σ"从预读样本估计
                overload_thresholds = (
                    KGTemporalDataset._infer_thresholds(sample, self.overload_cols)
                    if self.overload_cols else []
                )
            else:
                # 流式：留空，由 _load_and_preprocess_streaming 基于全量数据估算
                overload_thresholds = None
        # 仅当显式给定阈值时才做长度校验（流式估算的结果天然一致）
        if overload_thresholds is not None and len(self.overload_cols) != len(overload_thresholds):
            raise ValueError(
                "overload_cols 与 overload_thresholds 长度必须一致："
                f"{len(self.overload_cols)} vs {len(overload_thresholds)}"
            )
        self.overload_thresholds: List[float] = (
            list(overload_thresholds) if overload_thresholds is not None else []
        )

        # 目标列过热阈值：未显式指定时，非流式从预读样本估计；
        # 流式模式留 None，由流式扫描基于全量数据估 80/95 百分位。
        if target_mild_threshold is None or target_severe_threshold is None:
            if not self.streaming:
                mild, severe = KGTemporalDataset._infer_target_thresholds(
                    sample, self.target_col
                )
                if target_mild_threshold is None:
                    target_mild_threshold = mild
                if target_severe_threshold is None:
                    target_severe_threshold = severe
        self.target_mild_threshold = (
            float(target_mild_threshold) if target_mild_threshold is not None else None
        )
        self.target_severe_threshold = (
            float(target_severe_threshold) if target_severe_threshold is not None else None
        )

        self.future_steps = future_steps
        self.device = self._resolve_device(device)

        # 健康状态映射（与列配置无关的行业语义标签）
        self.health_mapping: Dict[int, str] = dict(get_health_mapping())
        self.health_num = len(get_health_mapping())

        # 由配置推导的必需列（不再写死在模块顶层）
        self._required_cols = (
            {self.date_col, self.target_col}
            | set(self.overload_cols)
        )

        # ---- 0. 从 CSV 自动推断特征列（feature_list 未显式传入时）----
        if feature_list is None:
            feature_list = self._infer_feature_columns(
                csv_path,
                date_col=self.date_col,
                required_cols=self._required_cols,
            )
        self.feature_list: List[str] = list(feature_list)
        self.feature_num = len(self.feature_list)

        # ---- 1. 加载 CSV + 生成健康标签 + 标准化 ----
        self.df_train, self.df_raw, self.feat_mean, self.feat_std = (
            self._load_and_preprocess()
        )
        self.slice_num = len(self.df_train)

        # ---- 2. 构建异构图知识图谱 ----
        self.data: HeteroData = self._build_hetero_graph()

        logger.info(
            "变压器时序异构图构建完成 | time_slice=%d | device=%s",
            self.slice_num,
            self.device,
        )

    # ------------------------------------------------------------------
    # 工具函数
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_device(device):
        if device is None:
            return torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        if isinstance(device, torch.device):
            return device
        return torch.device(device)

    @staticmethod
    def _detect_date_col(sample: pd.DataFrame) -> str:
        """从 CSV 样例中自动检测时间戳列。

        策略：
        1. 优先匹配列名包含 "date"/"time"/"timestamp"/"datetime"（大小写不敏感）
        2. 兜底：尝试按顺序解析各列为 datetime，首个可解析者即被选中
        """
        cols = list(sample.columns)

        # 1) 关键词匹配（保留原始列名大小写）
        for col in cols:
            col_lower = str(col).lower()
            if any(
                kw in col_lower
                for kw in ("date", "time", "timestamp", "datetime")
            ):
                return col

        # 2) 兜底：尝试解析为 datetime
        for col in cols:
            try:
                pd.to_datetime(sample[col])
                return col
            except (ValueError, TypeError):
                continue

        raise ValueError(
            "无法自动检测时间戳列，请显式指定 date_col 参数；"
            f"CSV 实际列: {cols}"
        )

    @staticmethod
    def _detect_target_col(sample: pd.DataFrame, date_col: str) -> str:
        """自动检测目标列：默认取最后一个数值列（时序 CSV 常见约定）。"""
        numeric_cols = sample.select_dtypes(include="number").columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != date_col]
        if not numeric_cols:
            raise ValueError(
                "无法自动检测目标列（CSV 中无数值列），请显式指定 target_col 参数；"
                f"CSV 实际列: {list(sample.columns)}"
            )
        return numeric_cols[-1]

    @staticmethod
    def _infer_thresholds(
            sample: pd.DataFrame,
            cols: List[str],
    ) -> List[float]:
        """按"均值 + 2 倍标准差"为每列估计过载阈值。

        对每列额外做 NaN 过滤，避免样本缺失值污染统计。
        """
        thresholds: List[float] = []
        for col in cols:
            if col not in sample.columns:
                raise ValueError(
                    f"列 '{col}' 不在 CSV 中；实际列: {list(sample.columns)}"
                )
            values = pd.to_numeric(sample[col], errors="coerce").dropna().values
            if len(values) == 0:
                raise ValueError(
                    f"列 '{col}' 全为非数值或缺失，无法估计阈值"
                )
            mu = float(np.mean(values))
            sigma = float(np.std(values))
            thresholds.append(mu + 2.0 * sigma)
        return thresholds

    @staticmethod
    def _infer_target_thresholds(
            sample: pd.DataFrame,
            target_col: str,
    ) -> Tuple[float, float]:
        """按 80 / 95 百分位估计目标列的轻微过热 / 严重过热阈值。"""
        if target_col not in sample.columns:
            raise ValueError(
                f"target_col '{target_col}' 不在 CSV 中；"
                f"实际列: {list(sample.columns)}"
            )
        values = pd.to_numeric(sample[target_col], errors="coerce").dropna().values
        if len(values) == 0:
            raise ValueError(
                f"target_col '{target_col}' 全为非数值或缺失，无法估计阈值"
            )
        mild = float(np.percentile(values, 80))
        severe = float(np.percentile(values, 95))
        return mild, severe

    @staticmethod
    def _infer_feature_columns(
            csv_path: str,
            date_col: Optional[str] = None,
            required_cols: Optional[set] = None,
    ) -> List[str]:
        """从 CSV 表头自动推断特征列。

        规则：
        1. 检查 ``required_cols`` 中指定的必需列是否齐全
        2. 读取少量样本推断 dtype，保留所有数值型列
        3. 排除 ``date_col``，将其余数值列按 CSV 原始顺序作为特征列

        Parameters
        ----------
        csv_path : str
            CSV 文件路径。
        date_col : str | None
            时间戳列名；为 None 时不做额外列名过滤。
        required_cols : set[str] | None
            必需存在的列集合；不传则跳过必需列检查。

        Returns
        -------
        List[str]
            推断出的特征列名列表。
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV 文件不存在: {csv_path}")

        # 仅读取前 5 行，足以推断各列的 dtype
        sample = pd.read_csv(csv_path, nrows=5)

        if required_cols:
            missing = required_cols - set(sample.columns)
            if missing:
                raise ValueError(
                    f"CSV 缺少必需列: {sorted(missing)}；"
                    f"实际列: {list(sample.columns)}"
                )

        numeric_cols = sample.select_dtypes(include="number").columns.tolist()
        # 排除 date_col（虽然它通常已不是数值）作为防御性处理
        feature_cols = [
            c for c in sample.columns
            if c in numeric_cols and c != date_col
        ]

        if not feature_cols:
            raise ValueError(
                f"未能从 CSV 推断出任何数值型特征列；"
                f"实际列: {list(sample.columns)}"
            )

        return feature_cols

    # ------------------------------------------------------------------
    # 加载与预处理
    # ------------------------------------------------------------------
    def _load_and_preprocess(self):
        """加载 CSV、生成健康状态标签、标准化特征。

        当 ``chunk_size > 0``（流式模式）时，改为分块读取以控制峰值内存。

        Returns
        -------
        df_train : pd.DataFrame
            特征列已 z-score 标准化的 DataFrame。
        df_raw : pd.DataFrame
            与 ``df_train`` 行对齐的原始值 DataFrame。
        feat_mean : np.ndarray shape (feature_num,)
        feat_std  : np.ndarray shape (feature_num,)
        """
        if self.streaming:
            return self._load_and_preprocess_streaming()

        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV 文件不存在: {self.csv_path}")

        df_raw = pd.read_csv(self.csv_path)
        # 验证：构造参数派生的必需列 + 已推断/指定的 feature_list 列都存在
        required_cols = self._required_cols | set(self.feature_list)
        missing = required_cols - set(df_raw.columns)
        if missing:
            raise ValueError(
                f"CSV 缺少必需列: {sorted(missing)}；实际列: {list(df_raw.columns)}"
            )

        df_raw[self.date_col] = pd.to_datetime(df_raw[self.date_col])

        # 基于真实值生成健康状态标签（按行，矢量化）
        # - 状态 1/2：由 target_col 的 mild / severe 阈值
        # - 状态 3：任一 overload_col 的对应阈值
        target_vals = df_raw[self.target_col].to_numpy()
        labels = np.zeros(len(df_raw), dtype=np.int64)
        labels[
            (target_vals >= self.target_mild_threshold)
            & (target_vals < self.target_severe_threshold)
        ] = 1
        labels[target_vals >= self.target_severe_threshold] = 2

        # 任意一个 overload 条件命中即为过载故障（状态 3）
        if self.overload_cols:
            overload_mask = np.zeros(len(df_raw), dtype=bool)
            for col, thr in zip(self.overload_cols, self.overload_thresholds):
                overload_mask = overload_mask | (df_raw[col].to_numpy() > thr)
            labels[overload_mask] = 3
        df_raw["health_label"] = labels

        # 目标列超前预测值（保持真实物理单位，不标准化）
        future_col = f"future_{self.target_col}"
        df_raw[future_col] = df_raw[self.target_col].shift(-self.future_steps)

        # 仅对特征列执行 z-score 标准化，其余列保持原始值
        feat_mean = (
            df_raw[self.feature_list].mean().values.astype(np.float32)
        )
        feat_std = df_raw[self.feature_list].std().values.astype(np.float32)
        # 避免 std=0 导致 NaN（常数列，如某设备的无用指标）
        feat_std = np.where(feat_std < 1e-6, 1.0, feat_std)

        df_train = df_raw.copy()
        df_train[self.feature_list] = (
                                              df_train[self.feature_list] - feat_mean
                                      ) / feat_std

        # shift(-N) 会在最后 N 行产生 NaN 的 future_OT，统一 dropna
        df_train = df_train.dropna().reset_index(drop=True)
        df_raw = df_raw.dropna().reset_index(drop=True)

        if len(df_train) < 2:
            raise ValueError(
                "有效时序切片不足 2，无法构建相邻时序边："
                f"原始 {len(df_raw) + self.future_steps} 行，dropna 后剩余 {len(df_train)}"
            )

        logger.info(
            "CSV 加载完成: %s | 有效切片=%d | 特征数=%d",
            self.csv_path,
            len(df_train),
            self.feature_num,
        )
        return df_train, df_raw, feat_mean, feat_std

    # ------------------------------------------------------------------
    # 流式读取辅助：Welford 在线统计 + 蓄水池采样
    # ------------------------------------------------------------------
    @staticmethod
    def _welford_merge(state: List[float], vals: np.ndarray) -> None:
        """把一批数值（允许含 NaN）合并进 Welford 状态 [count, mean, M2]。

        使用并行合并公式，全程向量化，避免逐元素 Python 循环。
        """
        vals = vals[~np.isnan(vals)]
        if vals.size == 0:
            return
        n = vals.size
        mean = float(vals.mean())
        m2 = float(((vals - mean) ** 2).sum())
        count, mean_old, m2_old = state
        if count == 0:
            state[0], state[1], state[2] = n, mean, m2
            return
        count_new = count + n
        delta = mean - mean_old
        mean_new = (count * mean_old + n * mean) / count_new
        m2_new = m2_old + m2 + delta * delta * count * n / count_new
        state[0], state[1], state[2] = count_new, mean_new, m2_new

    @staticmethod
    def _welford_mean(state: List[float]) -> float:
        return state[1]

    @staticmethod
    def _welford_std(state: List[float]) -> float:
        # 使用样本标准差(ddof=1)，与 pandas .std() 默认行为保持一致
        if state[0] < 2:
            return 0.0
        return float(np.sqrt(state[2] / (state[0] - 1)))

    @staticmethod
    def _reservoir_update(reservoir: List[float], vals: np.ndarray,
                          cap: int, rng: np.random.Generator) -> None:
        """有界蓄水池采样：满 cap 后以随机替换方式维持统计代表性。

        用于在不持有全量数据的前提下估计 target_col 的百分位阈值。
        每批最多取 5000 个样本参与，限制超大文件下的采样开销。
        """
        vals = vals[~np.isnan(vals)]
        if vals.size == 0:
            return
        if vals.size > 5000:
            vals = vals[rng.integers(0, vals.size, size=5000)]
        for v in vals.tolist():
            if len(reservoir) < cap:
                reservoir.append(float(v))
            else:
                reservoir[int(rng.integers(0, len(reservoir)))] = float(v)

    @staticmethod
    def _reservoir_percentiles(reservoir: List[float], qs) -> List[float]:
        arr = np.asarray(reservoir, dtype=np.float64)
        return [float(np.percentile(arr, q)) for q in qs]

    # ------------------------------------------------------------------
    # 流式读取主流程
    # ------------------------------------------------------------------
    def _load_and_preprocess_streaming(self):
        """分块流式读取超大 CSV，峰值内存仅与 chunk_size 成正比。

        采用两遍流式扫描（文件被顺序读两遍，避免同时持有全量数据）：

        * 第 1 遍：用 Welford 在线算法累计各特征的均值/标准差；用蓄水池采样
          保留 target_col 的一个有界样本，用于估计 80/95 百分位过热阈值；
          过载阈值（若未显式给定）按"全量均值 + 2σ"估算。
        * 第 2 遍：逐块做健康标签、未来值平移、z-score 标准化，并追加到列式
          数组；最后拼接成与全量读取结构完全一致的 DataFrame，交给
          ``_build_hetero_graph``（无需改动建图逻辑）。

        说明：模型输入（time_slice 节点特征）本身占 O(N·F) 内存，这是推理所需、
        无法避免的；流式读取的意义在于避免「原始 DataFrame + 副本 + 标准化副本」
        同时驻留，把峰值内存从约数倍数据降到约 1 倍，从而支持任意大的 CSV
        （只要单块能放进内存）。若需进一步突破内存上限，应配合磁盘缓存 +
        邻居采样的小批量训练（见 train_joint_rgcn_tgn 的 batch_size 参数说明）。
        """
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV 文件不存在: {self.csv_path}")

        feature_cols = self.feature_list
        overload_cols = self.overload_cols
        target_col = self.target_col
        date_col = self.date_col
        rng = np.random.default_rng(42)

        # ---------- 第 1 遍：在线统计 + 蓄水池采样 ----------
        welford = {c: [0, 0.0, 0.0]
                   for c in feature_cols + overload_cols + [target_col]}
        reservoir: List[float] = []

        reader = pd.read_csv(self.csv_path, chunksize=self.chunk_size)
        for chunk in reader:
            for c in welford:
                self._welford_merge(
                    welford[c],
                    pd.to_numeric(chunk[c], errors="coerce").to_numpy(dtype=np.float64),
                )
            self._reservoir_update(
                reservoir,
                pd.to_numeric(chunk[target_col], errors="coerce").to_numpy(dtype=np.float64),
                self.reservoir_size, rng,
            )

        # 最终化阈值
        feat_mean = np.array(
            [self._welford_mean(welford[c]) for c in feature_cols], dtype=np.float32
        )
        feat_std = np.array(
            [self._welford_std(welford[c]) for c in feature_cols], dtype=np.float32
        )
        feat_std = np.where(feat_std < 1e-6, 1.0, feat_std)  # 常数列保护

        if self.target_mild_threshold is None or self.target_severe_threshold is None:
            mild, severe = self._reservoir_percentiles(reservoir, [80, 95])
            if self.target_mild_threshold is None:
                self.target_mild_threshold = float(mild)
            if self.target_severe_threshold is None:
                self.target_severe_threshold = float(severe)

        if overload_cols and not self.overload_thresholds:
            self.overload_thresholds = [
                float(self._welford_mean(welford[c]) + 2.0 * self._welford_std(welford[c]))
                for c in overload_cols
            ]

        # ---------- 第 2 遍：逐块处理并追加 ----------
        feat_raw_chunks: List[np.ndarray] = []   # 各块原始特征矩阵
        target_chunks: List[np.ndarray] = []     # 各块 target 原始值
        overload_chunks: List[np.ndarray] = []   # 各块 overload 原始值
        health_chunks: List[np.ndarray] = []     # 各块健康标签
        date_list: List[str] = []                # 各块日期字符串

        reader = pd.read_csv(self.csv_path, chunksize=self.chunk_size)
        for chunk in reader:
            chunk = chunk.copy()
            chunk[date_col] = pd.to_datetime(chunk[date_col])
            tvals = pd.to_numeric(
                chunk[target_col], errors="coerce"
            ).to_numpy(dtype=np.float32)

            # 健康标签（矢量，按行）
            labels = np.zeros(len(chunk), dtype=np.int64)
            labels[
                (tvals >= self.target_mild_threshold)
                & (tvals < self.target_severe_threshold)
            ] = 1
            labels[tvals >= self.target_severe_threshold] = 2
            if overload_cols:
                ov_mask = np.zeros(len(chunk), dtype=bool)
                for c, thr in zip(overload_cols, self.overload_thresholds):
                    ov = pd.to_numeric(
                        chunk[c], errors="coerce"
                    ).to_numpy(dtype=np.float32)
                    ov_mask |= (ov > thr)
                labels[ov_mask] = 3

            fraw = chunk[feature_cols].to_numpy(dtype=np.float32)

            feat_raw_chunks.append(fraw)
            target_chunks.append(tvals)
            if overload_cols:
                overload_chunks.append(np.stack(
                    [pd.to_numeric(chunk[c], errors="coerce").to_numpy(dtype=np.float32)
                     for c in overload_cols], axis=1))
            health_chunks.append(labels)
            date_list.extend(
                chunk[date_col].dt.strftime("%Y-%m-%d %H:%M:%S").tolist()
            )

        # ---------- 拼接 + 平移 + 标准化 ----------
        feat_raw_all = np.concatenate(feat_raw_chunks, axis=0)        # (N, F)
        feat_std_all = (feat_raw_all - feat_mean) / feat_std         # 标准化
        target_all = np.concatenate(target_chunks, axis=0)           # (N,)
        health_all = np.concatenate(health_chunks, axis=0)           # (N,)
        if overload_cols:
            overload_all = np.concatenate(overload_chunks, axis=0)
        date_all = date_list

        # 未来值平移：跨块边界由全量拼接自然处理
        future_col = f"future_{target_col}"
        future_all = np.concatenate([
            target_all[self.future_steps:],
            np.full(self.future_steps, np.nan, dtype=np.float32),
        ])

        # 丢弃尾部 future_steps 行（无未来标签）
        keep = ~np.isnan(future_all)
        feat_raw_all = feat_raw_all[keep]
        feat_std_all = feat_std_all[keep]
        target_all = target_all[keep]
        health_all = health_all[keep]
        future_all = future_all[keep]
        date_all = [d for d, k in zip(date_all, keep) if k]
        if overload_cols:
            overload_all = overload_all[keep]

        slice_num = feat_raw_all.shape[0]
        if slice_num < 2:
            raise ValueError(
                "有效时序切片不足 2，无法构建相邻时序边："
                f"流式读取后剩余 {slice_num} 行"
            )

        # 组装成与全量读取结构一致的 DataFrame，使 _build_hetero_graph 无需改动
        df_raw = pd.DataFrame(feat_raw_all, columns=feature_cols)
        df_raw[target_col] = target_all
        df_raw[date_col] = pd.to_datetime(date_all)
        df_raw["health_label"] = health_all
        df_raw[future_col] = future_all
        if overload_cols:
            for c, col_arr in zip(overload_cols, overload_all.T):
                df_raw[c] = col_arr

        df_train = df_raw.copy()
        df_train[feature_cols] = feat_std_all

        self.df_train = df_train
        self.df_raw = df_raw
        self.feat_mean = feat_mean
        self.feat_std = feat_std
        self.slice_num = slice_num

        logger.info(
            "CSV 流式加载完成: %s | 有效切片=%d | 特征数=%d | chunk_size=%d",
            self.csv_path, slice_num, self.feature_num, self.chunk_size,
        )
        return df_train, df_raw, feat_mean, feat_std

    # ------------------------------------------------------------------
    # 异构图构建
    # ------------------------------------------------------------------
    def _build_hetero_graph(self) -> HeteroData:
        """构建时序异构图知识图谱：transformer / time_slice /
        health_state / feature_indicator 四种节点 + 五种边。"""
        data = HeteroData()
        slice_num = self.slice_num
        feature_num = self.feature_num
        health_num = self.health_num

        # --- 1. transformer 节点：单台设备元数据 ---
        data["transformer"].x = torch.tensor(
            [[self.transformer_id, 2.0, 110.0]], dtype=torch.float32
        )

        # --- 2. time_slice 节点：时序运行切片 ---
        data["time_slice"].x = torch.tensor(
            self.df_train[self.feature_list].values, dtype=torch.float32
        )
        data["time_slice"].x_raw = torch.tensor(
            self.df_raw[self.feature_list].values, dtype=torch.float32
        )
        data["time_slice"].y_health = torch.tensor(
            self.df_raw["health_label"].values, dtype=torch.long
        )
        future_col = f"future_{self.target_col}"
        data["time_slice"].y_future_ot = torch.tensor(
            self.df_raw[future_col].values, dtype=torch.float32
        )

        # 对目标列预测值做 z-score 标准化（让 loss 量级与其他任务可比）
        target_raw = self.df_raw[self.target_col].values.astype(np.float32)
        target_mean = np.float32(target_raw.mean())
        target_std = np.float32(max(target_raw.std(), 1e-6))
        data["time_slice"].y_future_ot_norm = torch.tensor(
            (self.df_raw[future_col].values.astype(np.float32) - target_mean)
            / target_std,
            dtype=torch.float32,
        )
        data["time_slice"].ot_mean = torch.tensor(target_mean, dtype=torch.float32)
        data["time_slice"].ot_std = torch.tensor(target_std, dtype=torch.float32)

        # 顺序序号用于 TGN 时序编码，同时保存真实日期字符串
        data["time_slice"].time = torch.tensor(
            np.arange(slice_num), dtype=torch.float32
        ).unsqueeze(1)
        data["time_slice"].date_str = (
            self.df_raw[self.date_col]
            .dt.strftime("%Y-%m-%d %H:%M:%S")
            .tolist()
        )
        data["time_slice"].feat_mean = torch.tensor(
            self.feat_mean, dtype=torch.float32
        )
        data["time_slice"].feat_std = torch.tensor(
            self.feat_std, dtype=torch.float32
        )

        # --- 3. health_state 节点：随机初始化的可学习嵌入载体 ---
        data["health_state"].x = torch.randn(health_num, 4)

        # --- 4. feature_indicator 节点：特征指标的可学习嵌入载体 ---
        data["feature_indicator"].x = torch.randn(feature_num, 2)

        # --- 边 1: transformer → has_time_slice → time_slice ---
        trans2slice = torch.tensor(
            [[0, sid] for sid in range(slice_num)], dtype=torch.long
        ).t().contiguous()
        data["transformer", "has_time_slice", "time_slice"].edge_index = (
            trans2slice
        )
        data["transformer", "has_time_slice", "time_slice"].edge_time = (
            data["time_slice"].time.squeeze()
        )

        # --- 边 2: time_slice → next → time_slice（相邻时序边）---
        next_src = torch.arange(slice_num - 1, dtype=torch.long)
        next_dst = torch.arange(1, slice_num, dtype=torch.long)
        data["time_slice", "next", "time_slice"].edge_index = torch.stack(
            [next_src, next_dst], dim=0
        )
        data["time_slice", "next", "time_slice"].edge_time = (
            data["time_slice"].time.squeeze()[1:]
        )

        # --- 边 3: time_slice → has_health_state → health_state ---
        slice2health = torch.tensor(
            [
                [sid, hid]
                for sid, hid in enumerate(
                self.df_raw["health_label"].values.tolist()
            )
            ],
            dtype=torch.long,
        ).t().contiguous()
        data["time_slice", "has_health_state", "health_state"].edge_index = (
            slice2health
        )

        # --- 边 4: time_slice → has_feature → feature_indicator ---
        slice2feature_src = np.repeat(np.arange(slice_num), feature_num)
        slice2feature_dst = np.tile(np.arange(feature_num), slice_num)
        data["time_slice", "has_feature", "feature_indicator"].edge_index = (
            torch.tensor(
                np.stack([slice2feature_src, slice2feature_dst], axis=0),
                dtype=torch.long,
            )
        )

        # --- 边 5: health_state → state_has_symbol → feature_indicator ---
        #     行业先验规则：
        #       - 轻微过热/严重过热 (state 1, 2) → target_col
        #       - 过载故障 (state 3) → overload_cols 中的前若干列
        #     若用户未将这些列包含进 feature_list，则使用 feature_list[0] 作为兜底。
        symbol_edges: list[list[int]] = []

        # --- 规则: 状态 1/2 → target_col ---
        if self.target_col in self.feature_list:
            idx_target = self.feature_list.index(self.target_col)
            symbol_edges.append([1, idx_target])
            symbol_edges.append([2, idx_target])
        else:
            logger.warning(
                "target_col=%s 不在 feature_list=%s 中，state_has_symbol 规则边"
                "将不包含 target 相关连接",
                self.target_col,
                self.feature_list,
            )

        # --- 规则: 状态 3 → overload_cols ---
        any_overload_added = False
        for col in self.overload_cols:
            if col in self.feature_list:
                idx_overload = self.feature_list.index(col)
                symbol_edges.append([3, idx_overload])
                any_overload_added = True
        if not any_overload_added and self.overload_cols:
            logger.warning(
                "overload_cols=%s 中没有任何列出现在 feature_list=%s 中，"
                "state_has_symbol 规则边将不包含过载相关连接",
                self.overload_cols,
                self.feature_list,
            )

        if not symbol_edges:
            raise ValueError(
                "无法构建 state_has_symbol 规则边：target_col 与 overload_cols 均"
                "未出现在 feature_list 中；请调整 feature_list 或列配置。"
                f" feature_list={self.feature_list}"
            )

        state_feature_rule = torch.tensor(
            symbol_edges, dtype=torch.long
        ).t().contiguous()
        data[
            "health_state", "state_has_symbol", "feature_indicator"
        ].edge_index = state_feature_rule

        return data.to(self.device)

    # ------------------------------------------------------------------
    # 便捷访问 API
    # ------------------------------------------------------------------
    def health_label_counts(self) -> Dict[str, int]:
        """返回各健康状态标签的样本计数。"""
        counts = self.df_raw["health_label"].value_counts().to_dict()
        return {
            self.health_mapping.get(int(k), str(k)): int(v)
            for k, v in sorted(counts.items())
        }

    def describe(self) -> Dict[str, object]:
        """返回数据集概况的可读字典，用于日志 / 调试。"""
        return {
            "csv_path": self.csv_path,
            "slice_num": self.slice_num,
            "feature_num": self.feature_num,
            "feature_list": list(self.feature_list),
            "date_col": self.date_col,
            "target_col": self.target_col,
            "overload_cols": list(self.overload_cols),
            "target_mild_threshold": self.target_mild_threshold,
            "target_severe_threshold": self.target_severe_threshold,
            "future_steps": self.future_steps,
            "health_mapping": dict(self.health_mapping),
            "label_counts": self.health_label_counts(),
            "device": str(self.device),
        }

    def summary(self) -> str:
        """返回数据集概况的多行字符串摘要。"""
        info = self.describe()
        lines = [
            "===== Transformer Temporal KG =====",
            f"csv_path        : {info['csv_path']}",
            f"time_slice num  : {info['slice_num']}",
            f"feature columns : {info['feature_list']}",
            f"future OT steps : {info['future_ot_steps']}",
            f"health mapping  : {info['health_mapping']}",
            f"label counts    : {info['label_counts']}",
            f"device          : {info['device']}",
            "====================================",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # 数据切分
    # ------------------------------------------------------------------
    def split_hold_out_indices(
            self,
            hold_out_n: int = 5,
            test_ratio: float = 0.2,
            random_state: int = 42,
            device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """从时序样本尾部预留未知样本，剩余部分按比例划分 train/test。

        Parameters
        ----------
        hold_out_n : int
            从尾部预留的"未知样本"数，这些样本既不参与训练也不参与测试。
        test_ratio : float
            剩余样本中划分给测试集的比例。
        random_state : int
            随机切分种子。
        device : Optional[torch.device]
            结果张量放置的设备；默认使用数据集本身的 device。

        Returns
        -------
        (train_idx, test_idx, hold_out_idx)
            三组 long 型张量，形状为 [n]。
        """
        if device is None:
            device = self.device
        total_num = self.slice_num
        available = np.arange(total_num - hold_out_n)
        hold_out_idx = np.arange(total_num - hold_out_n, total_num)
        train_idx, test_idx = train_test_split(
            available, test_size=test_ratio, random_state=random_state,
        )

        def _to(x):
            t = torch.tensor(x, dtype=torch.long)
            return t if device is None else t.to(device)

        return _to(train_idx), _to(test_idx), _to(hold_out_idx)


# ---------------------------------------------------------------------------
# 便捷工厂函数（与原有 demo 中的函数签名保持一致，便于平滑迁移）
# ---------------------------------------------------------------------------

def load_and_preprocess_dataset(csv_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """兼容 demo: 直接返回 (df_train, df_raw, feat_mean, feat_std)。"""
    dataset = KGTemporalDataset(csv_path=csv_path)
    return dataset.df_train, dataset.df_raw, dataset.feat_mean, dataset.feat_std


def build_transformer_kg(
        df_train: pd.DataFrame,
        df_raw: pd.DataFrame,
        feat_mean: np.ndarray,
        feat_std: np.ndarray,
        transformer_id: int = 0,
        date_col: Optional[str] = None,
        target_col: Optional[str] = None,
        overload_cols: Optional[List[str]] = None,
        overload_thresholds: Optional[List[float]] = None,
        target_mild_threshold: Optional[float] = None,
        target_severe_threshold: Optional[float] = None,
        future_steps: int = 3,
        device=None,
) -> HeteroData:
    """兼容 demo: 从外部准备好的 DataFrame 直接构建 HeteroData。

    所有列名与阈值均为可选——未显式指定时从 ``df_raw`` 自动检测或
    按统计量推断，无任何与特定数据集绑定的硬编码。

    对于新代码，推荐直接使用 ``KGTemporalDataset`` 类。
    """
    # --- 列名自动检测 ---
    if date_col is None:
        date_col = KGTemporalDataset._detect_date_col(df_raw)
    if target_col is None:
        target_col = KGTemporalDataset._detect_target_col(df_raw, date_col)
    if overload_cols is None:
        overload_cols = []

    # --- 阈值自动推断（从 df_raw 真实值估计）---
    if overload_cols and overload_thresholds is None:
        overload_thresholds = KGTemporalDataset._infer_thresholds(
            df_raw, overload_cols
        )
    elif overload_thresholds is None:
        overload_thresholds = []

    if target_mild_threshold is None or target_severe_threshold is None:
        mild, severe = KGTemporalDataset._infer_target_thresholds(
            df_raw, target_col
        )
        if target_mild_threshold is None:
            target_mild_threshold = mild
        if target_severe_threshold is None:
            target_severe_threshold = severe

    dataset = KGTemporalDataset.__new__(KGTemporalDataset)
    dataset.transformer_id = transformer_id
    dataset.date_col = date_col
    dataset.target_col = target_col
    dataset.overload_cols = list(overload_cols)
    dataset.overload_thresholds = list(overload_thresholds)
    dataset.target_mild_threshold = float(target_mild_threshold)
    dataset.target_severe_threshold = float(target_severe_threshold)
    dataset.future_steps = future_steps
    dataset.device = KGTemporalDataset._resolve_device(device)
    dataset.health_mapping = dict(get_health_mapping())
    dataset.health_num = len(get_health_mapping())
    dataset._required_cols = (
        {date_col, target_col} | set(overload_cols)
    )

    # 从 df_train 推断特征列：所有数值列，排除非特征元数据列
    _exclude_cols = {date_col, "health_label", f"future_{target_col}"}
    numeric_cols = df_train.select_dtypes(include="number").columns.tolist()
    inferred_features = [
        c for c in df_train.columns if c in numeric_cols and c not in _exclude_cols
    ]
    if not inferred_features:
        raise ValueError(
            f"未能从 df_train 推断出特征列；实际列: {list(df_train.columns)}"
        )
    dataset.feature_list = inferred_features
    dataset.feature_num = len(inferred_features)

    dataset.df_train = df_train
    dataset.df_raw = df_raw
    dataset.feat_mean = feat_mean
    dataset.feat_std = feat_std
    dataset.slice_num = len(df_train)
    dataset.csv_path = "<external>"
    dataset.data = dataset._build_hetero_graph()
    return dataset.data