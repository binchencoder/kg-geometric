"""TGN 模型 —— Temporal Graph Network 用于时序异构图趋势预测。

结合 PyG 的 TGNMemory（节点记忆）、TimeEncoder（余弦时序编码）
与 HeteroConv（异构图卷积），实现在时序知识图谱上的因果消息传递，
可用于变压器油温预测、故障风险评分等时序任务。

参考文献:
    Temporal Graph Networks for Deep Learning on Dynamic Graphs
    (Rossi et al., ICML 2020 Workshop)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import Linear, HeteroConv, SAGEConv
from torch_geometric.nn.models.tgn import (
    TGNMemory,
    IdentityMessage,
    LastAggregator,
    TimeEncoder,
)


class TGN(nn.Module):
    """Temporal Graph Network: 节点记忆 + 时序编码 + 多层异构图卷积。

    关键特性:
    - 余弦时序编码器 (TimeEncoder) 捕捉周期性模式
    - TGNMemory 为每个节点维护可更新的记忆向量
    - HeteroConv 支持多关系/多节点类型的消息传递
    - Float 时间戳与 msg_store 统一处理，避免 dtype 冲突

    Parameters
    ----------
    in_channels : Dict[str, int]
        每种节点类型的输入特征维度。
        例如 ``{"transformer": 3, "time_slice": 7, "health_state": 4}``。
    hidden_channels : int
        隐藏层维度。
    out_channels : int
        输出嵌入维度。
    edge_types : List[Tuple[str, str, str]]
        需要参与异构图卷积的边类型列表。
    temporal_edge_types : Optional[List[Tuple[str, str, str]]]
        需要参与 TGN 记忆更新的时序边类型。
        若为 ``None``，则使用所有 edge_types。
    num_layers : int
        HeteroConv 层数，默认 2。
    num_nodes_for_memory : int
        TGNMemory 中保留的记忆槽位数。
        需至少等于图中参与记忆更新的节点总数（例如 transformer + time_slice）。
    """

    def __init__(
        self,
        in_channels: Dict[str, int],
        hidden_channels: int,
        out_channels: int,
        edge_types: List[Tuple[str, str, str]],
        temporal_edge_types: Optional[List[Tuple[str, str, str]]],
        num_layers: int,
        num_nodes_for_memory: int,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.edge_types = list(edge_types)
        self.temporal_edge_types = (
            list(temporal_edge_types)
            if temporal_edge_types is not None
            else list(edge_types)
        )

        # 余弦时序编码器
        self.time_enc = TimeEncoder(hidden_channels)

        # TGN 节点记忆模块
        self.memory = TGNMemory(
            num_nodes=num_nodes_for_memory,
            raw_msg_dim=hidden_channels,
            memory_dim=hidden_channels,
            time_dim=hidden_channels,
            message_module=IdentityMessage(
                hidden_channels, hidden_channels, hidden_channels
            ),
            aggregator_module=LastAggregator(),
        )
        # 将 last_update 与 msg_store 中的空时间戳统一为 Float，
        # 避免 eval() / 二次 forward 时 Long→Float 冲突。
        self.memory.register_buffer(
            "last_update", self.memory.last_update.float()
        )
        for _store in (self.memory.msg_s_store, self.memory.msg_d_store):
            for _j in range(self.memory.num_nodes):
                _s, _d, _t, _m = _store[_j]
                _store[_j] = (_s, _d, _t.float(), _m)

        # 节点特征投影层
        self.node_proj = nn.ModuleDict(
            {
                node_type: Linear(in_dim, hidden_channels)
                for node_type, in_dim in in_channels.items()
            }
        )

        # 多层异构图卷积
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv(
                {
                    edge_type: SAGEConv((-1, -1), hidden_channels)
                    for edge_type in self.edge_types
                },
                aggr="mean",
            )
            self.convs.append(conv)

        # 输出投影层
        self.out_proj = nn.ModuleDict(
            {
                node_type: Linear(hidden_channels, out_channels)
                for node_type in in_channels.keys()
            }
        )

    # ------------------------------------------------------------------
    # 辅助：将原始节点 id 映射到 TGNMemory 中的索引
    # ------------------------------------------------------------------
    @staticmethod
    def _map_node_indices(
        node_type: str,
        indices: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """将原始节点 id 映射到 TGNMemory slot。

        约定: transformer 节点始终占用 slot 0；time_slice i 占用 slot i+1。
        对于其他节点类型，返回一个全零占位（这些节点不参与记忆更新）。
        """
        if node_type == "transformer":
            return torch.zeros(len(indices), dtype=torch.long, device=device)
        if node_type == "time_slice":
            return indices + 1
        return torch.zeros(len(indices), dtype=torch.long, device=device)

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
        edge_time_dict: Optional[
            Dict[Tuple[str, str, str], torch.Tensor]
        ] = None,
    ) -> Dict[str, torch.Tensor]:
        """前向传播，返回每种节点类型的嵌入字典。

        Parameters
        ----------
        x_dict:
            节点类型 -> 特征张量。
        edge_index_dict:
            边类型三元组 -> ``[2, num_edges]`` 索引张量。
        edge_time_dict:
            （可选）需要参与 TGN 记忆更新的边 -> 时间戳张量。

        Returns
        -------
        Dict[str, torch.Tensor]
            节点类型 -> 输出嵌入 ``[num_nodes, out_channels]``。
        """
        device = x_dict["time_slice"].device

        # 1) 将各类型节点特征投影到统一的隐空间
        x_dict = {k: self.node_proj[k](v) for k, v in x_dict.items()}

        # 2) TGN 时序记忆更新（因果性：每次 forward 重置记忆）
        if edge_time_dict is not None:
            # detach 记忆状态，避免 "backward through the graph a second time"
            if hasattr(self.memory, "detach"):
                self.memory.detach()
            if hasattr(self.memory, "memory") and self.memory.memory is not None:
                self.memory.memory = self.memory.memory.detach()
            if (
                hasattr(self.memory, "last_update")
                and self.memory.last_update is not None
            ):
                self.memory.last_update = self.memory.last_update.detach()

            # 重置记忆与消息存储
            self.memory.reset_state()
            # reset_state 会将 msg_store 空时间戳初始化为 Long，
            # 与我们传入的 Float 时间戳冲突，这里统一转成 Float。
            for store in (self.memory.msg_s_store, self.memory.msg_d_store):
                for j in range(self.memory.num_nodes):
                    s, d, t, m = store[j]
                    store[j] = (s, d, t.float(), m)

            for edge_type, edge_time in edge_time_dict.items():
                src_type, _rel, dst_type = edge_type
                if edge_type not in edge_index_dict:
                    continue
                edge_index = edge_index_dict[edge_type]
                src, dst = edge_index[0], edge_index[1]

                # 节点 id -> TGNMemory slot
                src_mapped = self._map_node_indices(src_type, src, device)
                dst_mapped = self._map_node_indices(dst_type, dst, device)

                # 按时序排序，保证因果性
                time_order = edge_time.argsort()
                src_mapped = src_mapped[time_order]
                dst_mapped = dst_mapped[time_order]
                t_sorted = edge_time[time_order].float()

                # 构造时序消息：源节点特征 + 时间编码
                time_emb = self.time_enc(t_sorted.unsqueeze(-1))
                raw_msg = x_dict[src_type][src[time_order]] + time_emb

                # 更新 TGN 节点记忆
                self.memory.update_state(
                    src_mapped, dst_mapped, t_sorted, raw_msg
                )

            # 将 time_slice 节点的最新记忆注入特征
            num_slices = x_dict["time_slice"].shape[0]
            time_indices = torch.arange(1, num_slices + 1, device=device)
            mem_z, _ = self.memory(time_indices)
            x_dict["time_slice"] = x_dict["time_slice"] + mem_z

        # 3) 逐层异构图卷积，注意保留仅作为源节点的类型
        for conv in self.convs:
            prev = x_dict
            x_dict = conv(x_dict, edge_index_dict)
            for k, v in prev.items():
                if k not in x_dict:
                    x_dict[k] = v
            x_dict = {k: F.relu(v) for k, v in x_dict.items()}

        # 4) 输出投影
        out_dict = {k: self.out_proj[k](v) for k, v in x_dict.items()}
        return out_dict


class TGNOilTemperaturePredict(nn.Module):
    """基于 TGN 的变压器油温预测与故障风险评分模型。

    包含两个预测头：
    - ``ot_head``:  未来油温回归（与当前 OT 做 skip-connection，保证至少达到基线水平）
    - ``risk_head``: 故障风险评分（Sigmoid 输出 [0, 1]）

    Parameters
    ----------
    in_channels : Dict[str, int]
        同 :class:`TGN`。
    edge_types : List[Tuple[str, str, str]]
        同 :class:`TGN`。
    temporal_edge_types : Optional[List[Tuple[str, str, str]]]
        同 :class:`TGN`。
    hidden_dim : int
        隐藏层维度。
    num_time_slices : int
        图中 time_slice 节点总数，用于分配 TGNMemory 槽位。
    num_layers : int
        TGN HeteroConv 层数，默认 2。
    ot_feature_index : int
        x["time_slice"] 中 OT 特征所在列（用于 skip-connection），默认 -1。
    """

    def __init__(
        self,
        in_channels: Dict[str, int],
        edge_types: List[Tuple[str, str, str]],
        temporal_edge_types: Optional[List[Tuple[str, str, str]]],
        hidden_dim: int,
        num_time_slices: int,
        num_layers: int = 2,
        ot_feature_index: int = -1,
    ) -> None:
        super().__init__()
        self.ot_feature_index = ot_feature_index

        self.gnn = TGN(
            in_channels=in_channels,
            hidden_channels=hidden_dim,
            out_channels=hidden_dim,
            edge_types=edge_types,
            temporal_edge_types=temporal_edge_types,
            num_layers=num_layers,
            num_nodes_for_memory=num_time_slices + 1,
        )

        # 油温预测头：GNN embedding + 当前 OT 特征（skip-connection）
        self.ot_head = nn.Sequential(
            Linear(hidden_dim + 1, 16),
            nn.ReLU(),
            Linear(16, 1),
        )

        # 故障风险预测头
        self.risk_head = nn.Sequential(
            Linear(hidden_dim, 16),
            nn.ReLU(),
            Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        hetero_data,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """对异构图数据执行 TGN 前向传播。

        Parameters
        ----------
        hetero_data : HeteroData
            需包含:
            - ``x_dict`` / ``edge_index_dict``
            - 可选 ``edge_time`` 用于时序边的记忆更新

        Returns
        -------
        (future_ot, fault_risk, slice_emb)
            - future_ot: ``[num_time_slices, 1]``，未来油温（z-score 空间）
            - fault_risk: ``[num_time_slices, 1]``，故障风险 [0, 1]
            - slice_emb: ``[num_time_slices, hidden_dim]``，time_slice 节点嵌入
        """
        # 构造时序边的时间戳字典
        edge_time_dict = {}
        for et in self.gnn.temporal_edge_types:
            if (
                et in hetero_data.edge_index_dict
                and hasattr(hetero_data[et], "edge_time")
                and hetero_data[et].edge_time is not None
            ):
                edge_time_dict[et] = hetero_data[et].edge_time

        emb_dict = self.gnn(
            x_dict=hetero_data.x_dict,
            edge_index_dict=hetero_data.edge_index_dict,
            edge_time_dict=edge_time_dict if edge_time_dict else None,
        )
        slice_emb = emb_dict["time_slice"]

        # skip-connection: 拼接当前 OT 特征
        ot_feature = hetero_data["time_slice"].x[:, self.ot_feature_index].unsqueeze(-1)
        ot_input = torch.cat([slice_emb, ot_feature], dim=-1)

        future_ot = self.ot_head(ot_input)
        fault_risk = self.risk_head(slice_emb)
        return future_ot, fault_risk, slice_emb