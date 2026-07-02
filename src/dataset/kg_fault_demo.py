"""内置示例知识图谱数据集 —— 车辆故障知识图谱。

基于 CVFFAD 中文车辆故障关联数据集结构构建，
包含故障现象、故障原因、维修措施、所需工具等多关系类型。

支持 R-GCN 所需的关系类型索引和图遍历查询。
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

import torch

from src.core.types import Triple


class KGFaultDataset:
    """车辆故障知识图谱数据集。

    关系类型:
    - "表现为"    : Fault → Symptom（故障的表现症状，正向）
    - "由...引起" : Fault → Cause（故障的根本原因，正向）
    - "维修措施"   : Fault → Action（修复措施，正向）
    - "需要工具"   : Fault → Tool（维修所需工具，正向）
    - "属于系统"   : Fault → System（所属车辆系统，正向）
    - "属于类别"   : Fault → Category（故障分类，正向）

    图遍历 API:
    - get_forward(head, relation) → List[tail] : 获取与 head 通过 relation 相连的所有 tail
    - get_backward(tail, relation) → List[head] : 获取通过 relation 指向 tail 的所有 head
    - get_symptom_nodes() → List[str] : 获取所有症状描述节点
    - get_fault_category_nodes() → List[str] : 获取所有故障类别节点（head 节点）
    """

    def __init__(self) -> None:
        self.triples: List[Triple] = [
            # ================================================================
            # 发动机启动困难
            # ================================================================
            Triple("发动机启动困难", "属于系统", "动力系统"),
            Triple("发动机启动困难", "属于类别", "启动系统"),
            Triple("发动机启动困难", "表现为", "早上启动困难，需要多次尝试"),
            Triple("发动机启动困难", "表现为", "启动时仪表盘闪烁然后熄灭"),
            Triple("发动机启动困难", "表现为", "启动时有吱吱异响"),
            Triple("发动机启动困难", "表现为", "冷车启动困难，热车正常"),
            Triple("发动机启动困难", "由...引起", "蓄电池电量不足"),
            Triple("发动机启动困难", "由...引起", "启动机故障"),
            Triple("发动机启动困难", "由...引起", "点火系统故障"),
            Triple("发动机启动困难", "由...引起", "燃油系统供油不畅"),
            Triple("发动机启动困难", "维修措施", "更换蓄电池"),
            Triple("发动机启动困难", "维修措施", "检修启动机"),
            Triple("发动机启动困难", "维修措施", "更换火花塞"),
            Triple("发动机启动困难", "维修措施", "清洗喷油嘴"),
            Triple("发动机启动困难", "需要工具", "万用表"),
            Triple("发动机启动困难", "需要工具", "蓄电池检测仪"),
            # ================================================================
            # 发动机怠速不稳
            # ================================================================
            Triple("发动机怠速不稳", "属于系统", "动力系统"),
            Triple("发动机怠速不稳", "属于类别", "怠速系统"),
            Triple("发动机怠速不稳", "表现为", "发动机怠速时抖动明显"),
            Triple("发动机怠速不稳", "表现为", "怠速转速忽高忽低不稳定"),
            Triple("发动机怠速不稳", "表现为", "等红灯时感觉车身不停颤抖"),
            Triple("发动机怠速不稳", "表现为", "怠速时发动机转速波动大"),
            Triple("发动机怠速不稳", "由...引起", "节气门积碳"),
            Triple("发动机怠速不稳", "由...引起", "怠速马达故障"),
            Triple("发动机怠速不稳", "由...引起", "进气系统漏气"),
            Triple("发动机怠速不稳", "由...引起", "燃油压力不稳"),
            Triple("发动机怠速不稳", "维修措施", "清洗节气门体"),
            Triple("发动机怠速不稳", "维修措施", "更换怠速马达"),
            Triple("发动机怠速不稳", "维修措施", "检修进气系统"),
            Triple("发动机怠速不稳", "维修措施", "更换燃油泵"),
            Triple("发动机怠速不稳", "需要工具", "诊断仪"),
            Triple("发动机怠速不稳", "需要工具", "转速表"),
            # ================================================================
            # 发动机动力不足
            # ================================================================
            Triple("发动机动力不足", "属于系统", "动力系统"),
            Triple("发动机动力不足", "属于类别", "燃烧系统"),
            Triple("发动机动力不足", "表现为", "发动机沉闷，排气冒黑烟"),
            Triple("发动机动力不足", "表现为", "加速迟缓，超车困难"),
            Triple("发动机动力不足", "表现为", "上坡时动力明显不足，转速上不去"),
            Triple("发动机动力不足", "表现为", "急加速时进气管有嘶嘶声"),
            Triple("发动机动力不足", "由...引起", "空气滤清器堵塞"),
            Triple("发动机动力不足", "由...引起", "涡轮增压器故障"),
            Triple("发动机动力不足", "由...引起", "燃油系统压力不足"),
            Triple("发动机动力不足", "由...引起", "气缸压缩不足"),
            Triple("发动机动力不足", "维修措施", "更换空气滤清器"),
            Triple("发动机动力不足", "维修措施", "更换涡轮增压器"),
            Triple("发动机动力不足", "维修措施", "检修燃油系统"),
            Triple("发动机动力不足", "维修措施", "发动机大修"),
            Triple("发动机动力不足", "需要工具", "诊断仪"),
            Triple("发动机动力不足", "需要工具", "内窥镜"),
            Triple("发动机动力不足", "需要工具", "烟度计"),
            # ================================================================
            # 发动机过热
            # ================================================================
            Triple("发动机过热", "属于系统", "动力系统"),
            Triple("发动机过热", "属于类别", "冷却系统"),
            Triple("发动机过热", "表现为", "发动机舱冒出大量白色蒸汽"),
            Triple("发动机过热", "表现为", "水温表指针进入红色区域"),
            Triple("发动机过热", "表现为", "暖风不热"),
            Triple("发动机过热", "表现为", "冷却液液位下降快"),
            Triple("发动机过热", "由...引起", "冷却液泄漏"),
            Triple("发动机过热", "由...引起", "散热器堵塞"),
            Triple("发动机过热", "由...引起", "水泵故障"),
            Triple("发动机过热", "由...引起", "节温器失效"),
            Triple("发动机过热", "维修措施", "更换散热器"),
            Triple("发动机过热", "维修措施", "更换水泵"),
            Triple("发动机过热", "维修措施", "更换节温器"),
            Triple("发动机过热", "维修措施", "检修冷却系统管路"),
            Triple("发动机过热", "需要工具", "红外测温仪"),
            Triple("发动机过热", "需要工具", "冷却系统压力仪"),
            # ================================================================
            # 机油压力异常
            # ================================================================
            Triple("机油压力异常", "属于系统", "动力系统"),
            Triple("机油压力异常", "属于类别", "润滑系统"),
            Triple("机油压力异常", "表现为", "机油压力报警灯闪烁"),
            Triple("机油压力异常", "表现为", "发动机运行时异响增大"),
            Triple("机油压力异常", "表现为", "机油消耗量异常增加"),
            Triple("机油压力异常", "表现为", "排气管冒蓝烟"),
            Triple("机油压力异常", "由...引起", "机油泵磨损"),
            Triple("机油压力异常", "由...引起", "机油滤清器堵塞"),
            Triple("机油压力异常", "由...引起", "轴承间隙过大"),
            Triple("机油压力异常", "由...引起", "机油粘度不合适"),
            Triple("机油压力异常", "维修措施", "更换机油泵"),
            Triple("机油压力异常", "维修措施", "更换机油滤清器"),
            Triple("机油压力异常", "维修措施", "更换轴承"),
            Triple("机油压力异常", "维修措施", "更换合适粘度机油"),
            Triple("机油压力异常", "需要工具", "机油压力表"),
            # ================================================================
            # 离合器打滑
            # ================================================================
            Triple("离合器打滑", "属于系统", "传动系统"),
            Triple("离合器打滑", "属于类别", "离合器系统"),
            Triple("离合器打滑", "表现为", "起步时发动机转速升高但车速不提"),
            Triple("离合器打滑", "表现为", "爬坡时动力传递不畅"),
            Triple("离合器打滑", "表现为", "加速时有焦糊味"),
            Triple("离合器打滑", "表现为", "离合器踏板行程变长"),
            Triple("离合器打滑", "由...引起", "离合器片磨损"),
            Triple("离合器打滑", "由...引起", "压盘弹簧疲劳"),
            Triple("离合器打滑", "由...引起", "离合器踏板自由行程不当"),
            Triple("离合器打滑", "由...引起", "离合器油液不足"),
            Triple("离合器打滑", "维修措施", "更换离合器片"),
            Triple("离合器打滑", "维修措施", "更换压盘"),
            Triple("离合器打滑", "维修措施", "调整踏板行程"),
            Triple("离合器打滑", "维修措施", "补充离合器油"),
            Triple("离合器打滑", "需要工具", "诊断仪"),
            # ================================================================
            # 变速箱换挡顿挫
            # ================================================================
            Triple("变速箱换挡顿挫", "属于系统", "传动系统"),
            Triple("变速箱换挡顿挫", "属于类别", "变速箱系统"),
            Triple("变速箱换挡顿挫", "表现为", "换挡时车身明显闯动"),
            Triple("变速箱换挡顿挫", "表现为", "自动变速箱升档延迟"),
        ]

        # ---- 1. 构建节点词汇表 ----
        self.node_to_idx = self._build_vocab()
        self.idx_to_node = {idx: node for node, idx in self.node_to_idx.items()}

        # ---- 2. 构建关系词汇表 ----
        self._build_relation_vocab()

        # ---- 3. 构建边索引和边类型（用于 R-GCN） ----
        self.edge_index, self.edge_type = self._build_rgcn_edges()

        # ---- 4. 构建图遍历映射（用于推理管线） ----
        self._build_traversal_maps()

        # ---- 5. 构建标签 ----
        # 故障节点 = "由...引起" 关系的 tail（故障原因实体）
        self.fault_nodes = [
            t.tail for t in self.triples if t.relation == "由...引起"
        ]
        all_nodes = {t.head for t in self.triples} | {t.tail for t in self.triples}
        fault_set = set(self.fault_nodes)
        self.labels: Dict[str, int] = {
            node: (1 if node in fault_set else 0) for node in all_nodes
        }

        # ---- 6. 构建特征和标签张量 ----
        self.num_nodes = len(self.node_to_idx)
        self.x = torch.eye(self.num_nodes, dtype=torch.float)
        ordered_nodes = [self.idx_to_node[i] for i in range(self.num_nodes)]
        self.y = torch.tensor(
            [self.labels[node] for node in ordered_nodes], dtype=torch.long
        )

    # ================================================================
    # 构建方法
    # ================================================================

    def _build_vocab(self) -> Dict[str, int]:
        nodes = sorted({t.head for t in self.triples} | {t.tail for t in self.triples})
        return {node: idx for idx, node in enumerate(nodes)}

    def _build_relation_vocab(self) -> None:
        """构建关系类型词汇表。

        为每条关系分配正向索引（0..R-1）和反向索引（R..2R-1）。
        R-GCN 需要区分正向和反向边以正确学习方向性语义。
        """
        unique_relations = sorted({t.relation for t in self.triples})
        self.relation_list = unique_relations
        self.num_original_relations = len(unique_relations)
        # 总关系数 = 正向 + 反向
        self.num_relations = self.num_original_relations * 2

        self.relation_to_idx: Dict[str, int] = {
            rel: idx for idx, rel in enumerate(unique_relations)
        }
        self.idx_to_relation: Dict[int, str] = {
            idx: rel for rel, idx in self.relation_to_idx.items()
        }
        # 反向关系名称映射（仅供调试）
        self.idx_to_relation_rev: Dict[int, str] = {
            idx + self.num_original_relations: f"←{rel}"
            for rel, idx in self.relation_to_idx.items()
        }

    def _build_rgcn_edges(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """构建 R-GCN 所需的边索引和边类型张量。

        每条三元组生成两条边：
        - 正向边 (head → tail, type=relation_idx)
        - 反向边 (tail → head, type=relation_idx + num_original_relations)
        """
        edges: List[Tuple[int, int]] = []
        edge_types: List[int] = []
        R = self.num_original_relations

        for triple in self.triples:
            h = self.node_to_idx[triple.head]
            t = self.node_to_idx[triple.tail]
            r = self.relation_to_idx[triple.relation]

            # 正向边
            edges.append((h, t))
            edge_types.append(r)
            # 反向边（类型偏移 R，让 R-GCN 学习方向性）
            edges.append((t, h))
            edge_types.append(r + R)

        return (
            torch.tensor(edges, dtype=torch.long).t().contiguous(),
            torch.tensor(edge_types, dtype=torch.long),
        )

    def _build_traversal_maps(self) -> None:
        """构建图遍历映射，支持高效的推理查询。

        两种映射：
        - _forward:  relation → {head → [tails]}   正向查询
        - _backward: relation → {tail → [heads]}   反向查询
        """
        self._forward: Dict[str, Dict[str, List[str]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self._backward: Dict[str, Dict[str, List[str]]] = defaultdict(
            lambda: defaultdict(list)
        )

        for triple in self.triples:
            self._forward[triple.relation][triple.head].append(triple.tail)
            self._backward[triple.relation][triple.tail].append(triple.head)

        # 将 defaultdict 转为普通 dict（避免意外创建空条目）
        self._forward = {k: dict(v) for k, v in self._forward.items()}
        self._backward = {k: dict(v) for k, v in self._backward.items()}

    # ================================================================
    # 图遍历 API（用于推理管线）
    # ================================================================

    def get_forward(self, head: str, relation: str) -> List[str]:
        """获取与 head 通过 relation 关系相连的所有 tail 节点（正向）。

        示例:
            dataset.get_forward("发动机动力不足", "由...引起")
            → ["空气滤清器堵塞", "涡轮增压器故障", ...]
        """
        return list(self._forward.get(relation, {}).get(head, []))

    def get_backward(self, tail: str, relation: str) -> List[str]:
        """获取通过 relation 关系指向 tail 的所有 head 节点（反向）。

        示例:
            dataset.get_backward("加速迟缓，超车困难", "表现为")
            → ["发动机动力不足"]
        """
        return list(self._backward.get(relation, {}).get(tail, []))

    def get_symptom_nodes(self) -> List[str]:
        """获取所有症状描述节点（"表现为"关系的 tail）。"""
        symptoms = set()
        for triple in self.triples:
            if triple.relation == "表现为":
                symptoms.add(triple.tail)
        return sorted(symptoms)

    def get_fault_category_nodes(self) -> List[str]:
        """获取所有故障类别节点（三元组 head 中唯一的故障根节点名称）。

        这些是知识图谱中的核心故障概念（如"发动机动力不足"），
        它们通过"表现为"连接症状、通过"由...引起"连接具体原因。
        """
        fault_categories = set()
        for triple in self.triples:
            if triple.relation == "表现为":
                fault_categories.add(triple.head)
        return sorted(fault_categories)

    def get_fault_info(self, fault_node: str) -> Dict[str, List[str]]:
        """获取指定故障节点的完整诊断信息。

        Parameters
        ----------
        fault_node : str
            故障类别节点名称，如 "发动机动力不足"。

        Returns
        -------
        Dict[str, List[str]]
            {
                "symptoms": ["表现为"的 tail],
                "causes": ["由...引起"的 tail],
                "actions": ["维修措施"的 tail],
                "tools": ["需要工具"的 tail],
                "system": ["属于系统"的 tail],
                "category": ["属于类别"的 tail],
            }
        """
        return {
            "symptoms": self.get_forward(fault_node, "表现为"),
            "causes": self.get_forward(fault_node, "由...引起"),
            "actions": self.get_forward(fault_node, "维修措施"),
            "tools": self.get_forward(fault_node, "需要工具"),
            "system": self.get_forward(fault_node, "属于系统"),
            "category": self.get_forward(fault_node, "属于类别"),
        }

    # ================================================================
    # PyG Data 导出
    # ================================================================

    def to_data(self):
        """生成标准 PyG Data 对象（用于 FaultGCN 兼容）。"""
        from torch_geometric.data import Data
        return Data(x=self.x, edge_index=self.edge_index, y=self.y)

    def to_data_with_types(self):
        """生成带 edge_type 的 PyG Data 对象（用于 R-GCN）。

        Returns
        -------
        Data
            包含 x, edge_index, edge_type, y 的 PyG Data。
        """
        from torch_geometric.data import Data
        return Data(
            x=self.x,
            edge_index=self.edge_index,
            edge_type=self.edge_type,
            y=self.y,
        )
