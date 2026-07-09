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

from src.core.config import logger  # noqa: E402

# ---------------------------------------------------------------------------
# 行业先验阈值（与 demo/fault_prediction.py 保持一致）
# ---------------------------------------------------------------------------

HEALTH_MAPPING: Dict[int, str] = {
    0: "正常运行",
    1: "轻微过热",
    2: "严重过热",
    3: "过载故障",
}
HEALTH_NUM = len(HEALTH_MAPPING)


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
            device=None,
    ) -> None:
        self.csv_path = csv_path
        self.transformer_id = transformer_id

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
        # 过载阈值：若用户未显式指定，按"均值 + 2σ"从预读样本估计
        if self.overload_cols and overload_thresholds is None:
            overload_thresholds = KGTemporalDataset._infer_thresholds(
                sample, self.overload_cols
            )
        elif overload_thresholds is None:
            overload_thresholds = []
        if len(self.overload_cols) != len(overload_thresholds):
            raise ValueError(
                "overload_cols 与 overload_thresholds 长度必须一致："
                f"{len(self.overload_cols)} vs {len(overload_thresholds)}"
            )
        self.overload_thresholds: List[float] = list(overload_thresholds)

        # 目标列过热阈值：未显式指定时按 80/95 百分位从预读样本估计
        if target_mild_threshold is None or target_severe_threshold is None:
            mild, severe = KGTemporalDataset._infer_target_thresholds(
                sample, self.target_col
            )
            if target_mild_threshold is None:
                target_mild_threshold = mild
            if target_severe_threshold is None:
                target_severe_threshold = severe
        self.target_mild_threshold = float(target_mild_threshold)
        self.target_severe_threshold = float(target_severe_threshold)

        self.future_steps = future_steps
        self.device = self._resolve_device(device)

        # 健康状态映射（与列配置无关的行业语义标签）
        self.health_mapping: Dict[int, str] = dict(HEALTH_MAPPING)
        self.health_num = HEALTH_NUM

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

        Returns
        -------
        df_train : pd.DataFrame
            特征列已 z-score 标准化的 DataFrame。
        df_raw : pd.DataFrame
            与 ``df_train`` 行对齐的原始值 DataFrame。
        feat_mean : np.ndarray shape (feature_num,)
        feat_std  : np.ndarray shape (feature_num,)
        """
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
    dataset.health_mapping = dict(HEALTH_MAPPING)
    dataset.health_num = HEALTH_NUM
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