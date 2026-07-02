"""内置示例知识图谱数据集 —— 小型工业故障知识图谱。

包含泵/电机/齿轮箱/压缩机的故障模式演示数据，
用于快速测试和原型验证。
"""

from __future__ import annotations

from typing import Dict, List

import torch

from src.core.types import Triple


class KGFaultDataset:
    """小型工业故障知识图谱，包含 13 条三元组。

    覆盖泵、电机、齿轮箱、压缩机的常见故障模式。
    用于快速原型验证，无需连接 ES。
    """

    def __init__(self) -> None:
        self.triples: List[Triple] = [
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
            Triple("变速箱换挡顿挫", "属于系统", "传动系统"),
            Triple("变速箱换挡顿挫", "属于类别", "变速箱系统"),
            Triple("变速箱换挡顿挫", "表现为", "换挡时车身明显闯动"),
            Triple("变速箱换挡顿挫", "表现为", "自动变速箱升档延迟"),
        ]

        # 根据 NER 6 类实体自动标注：
        #   FAULT_CAUSE（"由...引起"的 tail） → label=1
        #   其余（部件/症状/维修/系统/工具）   → label=0
        self.fault_nodes = [
            t.tail for t in self.triples if t.relation == "由...引起"
        ]
        all_nodes = {t.head for t in self.triples} | {t.tail for t in self.triples}
        fault_set = set(self.fault_nodes)
        self.labels: Dict[str, int] = {
            node: (1 if node in fault_set else 0) for node in all_nodes
        }

        self.node_to_idx = self._build_vocab()
        self.idx_to_node = {idx: node for node, idx in self.node_to_idx.items()}
        self.edge_index = self._build_edge_index()
        self.x = torch.eye(len(self.node_to_idx), dtype=torch.float)
        ordered_nodes = [self.idx_to_node[i] for i in range(len(self.idx_to_node))]
        self.y = torch.tensor([self.labels[node] for node in ordered_nodes], dtype=torch.long)

    def _build_vocab(self) -> Dict[str, int]:
        nodes = sorted({t.head for t in self.triples} | {t.tail for t in self.triples})
        return {node: idx for idx, node in enumerate(nodes)}

    def _build_edge_index(self) -> torch.Tensor:
        edges = []
        for triple in self.triples:
            h = self.node_to_idx[triple.head]
            t = self.node_to_idx[triple.tail]
            edges.append([h, t])
            edges.append([t, h])
        return torch.tensor(edges, dtype=torch.long).t().contiguous()

    def to_data(self):
        """生成 PyG Data 对象。"""
        from torch_geometric.data import Data
        return Data(x=self.x, edge_index=self.edge_index, y=self.y)
