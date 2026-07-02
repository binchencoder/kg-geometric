"""测试 FaultLabelBuilder.build_from_streamer —— 故障标签自动识别。"""

from __future__ import annotations

from typing import Dict, Iterator, List, Optional
from unittest.mock import MagicMock, patch

import pytest
import torch

with patch("src.core.config.logger", MagicMock()):
    from src.model.labels import FaultLabelBuilder


# ---------------------------------------------------------------------------
# Mock 工具
# ---------------------------------------------------------------------------

class MockVocab:
    """模拟 KGVocabulary。"""

    def __init__(self, entities: List[str]):
        self.entity2idx: Dict[str, int] = {e: i for i, e in enumerate(entities)}
        self.idx2entity: Dict[int, str] = {i: e for i, e in enumerate(entities)}

    @property
    def num_entities(self) -> int:
        return len(self.entity2idx)


class MockStreamer:
    """模拟 ESTripletStreamer.stream_triplets。"""

    def __init__(self, triplets: List[Dict[str, str]], batch_size: int = 3):
        self._triplets = triplets
        self._batch_size = batch_size

    def stream_triplets(
            self,
            head_field: str = "head_id",
            relation_field: str = "relation",
            tail_field: str = "tail_id",
            extra_filters: Optional[dict] = None,
            resume: bool = True,
    ) -> Iterator[List[dict]]:
        for i in range(0, len(self._triplets), self._batch_size):
            yield self._triplets[i:i + self._batch_size]


def make_streamer(*triplets: Dict[str, str]):
    return MockStreamer(list(triplets))


# ---------------------------------------------------------------------------
# 测试用例
# ---------------------------------------------------------------------------

class TestBuildFromStreamer:
    """核心测试：relation ∈ fault_relations → 收集 tail 作为故障节点。"""

    # === 正常匹配 ===

    def test_default_cause_relation_collects_tails(self):
        """默认 fault_relations 包含 "由...引起"，应收集其 tail 作为故障节点。"""
        vocab = MockVocab(["发动机", "轴承磨损", "密封失效", "泵_01", "振动过高"])
        streamer = make_streamer(
            {"head": "发动机", "relation": "由...引起", "tail": "轴承磨损"},
            {"head": "发动机", "relation": "由...引起", "tail": "密封失效"},
            {"head": "泵_01",  "relation": "存在症状",  "tail": "振动过高"},
        )
        builder = FaultLabelBuilder(vocab)
        y, fault_nodes = builder.build_from_streamer(streamer)

        assert set(fault_nodes) == {"轴承磨损", "密封失效"}
        # vocab: 发动机=0, 轴承磨损=1, 密封失效=2, 泵_01=3, 振动过高=4
        assert y.tolist() == [0, 1, 1, 0, 0]
        assert builder.num_faults == 2
        assert builder.num_normal == 3

    def test_type_is_relation_collects_fault_category(self):
        """"类型为" 关系 → 收集 tail "故障" 作为故障节点。"""
        vocab = MockVocab(["轴承磨损", "定子故障", "故障", "泵_01"])
        streamer = make_streamer(
            {"head": "轴承磨损", "relation": "类型为", "tail": "故障"},
            {"head": "定子故障", "relation": "类型为", "tail": "故障"},
            {"head": "泵_01",    "relation": "存在症状", "tail": "振动过高"},
        )
        builder = FaultLabelBuilder(vocab)
        y, fault_nodes = builder.build_from_streamer(streamer)

        assert fault_nodes == ["故障"]
        assert y.tolist() == [0, 0, 1, 0]

    def test_multiple_fault_relations(self):
        """多个 fault_relations 匹配时，收集所有 tail。"""
        vocab = MockVocab(["发动机", "积碳", "漏气", "电机"])
        streamer = make_streamer(
            {"head": "发动机", "relation": "由...引起", "tail": "积碳"},
            {"head": "发动机", "relation": "原因在于",  "tail": "漏气"},
            {"head": "电机",   "relation": "连接",     "tail": "发动机"},
        )
        builder = FaultLabelBuilder(
            vocab, fault_relations=["由...引起", "原因在于"]
        )
        y, fault_nodes = builder.build_from_streamer(streamer)

        assert set(fault_nodes) == {"积碳", "漏气"}
        assert y.tolist() == [0, 1, 1, 0]

    # === 边界情况 ===

    def test_empty_streamer(self):
        """空 streamer → 全 0。"""
        vocab = MockVocab(["A", "B", "C"])
        streamer = MockStreamer([])
        builder = FaultLabelBuilder(vocab)
        y, fault_nodes = builder.build_from_streamer(streamer)

        assert fault_nodes == []
        assert y.tolist() == [0, 0, 0]

    def test_streamer_no_match(self):
        """三元组存在但不匹配任何 fault_relations → 全 0。"""
        vocab = MockVocab(["发动机", "火花塞", "动力系统"])
        streamer = make_streamer(
            {"head": "发动机", "relation": "包含", "tail": "火花塞"},
            {"head": "发动机", "relation": "属于", "tail": "动力系统"},
        )
        builder = FaultLabelBuilder(vocab)
        y, fault_nodes = builder.build_from_streamer(streamer)

        assert fault_nodes == []
        assert y.tolist() == [0, 0, 0]

    def test_tail_not_in_vocab_skipped(self):
        """tail 不在 vocab 中时跳过，不影响结果。"""
        vocab = MockVocab(["泵_01", "轴承磨损"])
        streamer = make_streamer(
            {"head": "泵_01", "relation": "由...引起", "tail": "未知故障"},
            {"head": "泵_01", "relation": "由...引起", "tail": "轴承磨损"},
        )
        builder = FaultLabelBuilder(vocab)
        y, fault_nodes = builder.build_from_streamer(streamer)

        assert fault_nodes == ["轴承磨损"]
        assert y.tolist() == [0, 1]

    def test_same_tail_from_multiple_heads_dedup(self):
        """同一个 tail 被多条关系指向时应去重。"""
        vocab = MockVocab(["故障", "故障A", "故障B"])
        streamer = make_streamer(
            {"head": "故障A", "relation": "类型为", "tail": "故障"},
            {"head": "故障B", "relation": "类型为", "tail": "故障"},
        )
        builder = FaultLabelBuilder(vocab)
        y, fault_nodes = builder.build_from_streamer(streamer)

        assert fault_nodes == ["故障"]
        assert builder.num_faults == 1

    # === 自定义模式 ===

    def test_custom_relation_only(self):
        """自定义 fault_relations 应正确收集 tail 为故障节点。"""
        vocab = MockVocab(["传感器A", "电气故障", "机械故障", "控制器"])
        streamer = make_streamer(
            {"head": "传感器A", "relation": "故障类别", "tail": "电气故障"},
            {"head": "传感器A", "relation": "故障类别", "tail": "机械故障"},
            {"head": "控制器",  "relation": "控制",     "tail": "传感器A"},
        )
        builder = FaultLabelBuilder(
            vocab, fault_relations=["故障类别"]
        )
        y, fault_nodes = builder.build_from_streamer(streamer)

        assert set(fault_nodes) == {"电气故障", "机械故障"}
        assert y.tolist() == [0, 1, 1, 0]

    # === 输出结构 ===

    def test_y_tensor_shape(self):
        """y 的 shape 应与 vocab.num_entities 一致。"""
        vocab = MockVocab(["A", "B", "C", "D", "E", "故障"])
        streamer = make_streamer({"head": "A", "relation": "类型为", "tail": "故障"})
        builder = FaultLabelBuilder(vocab)
        y, _ = builder.build_from_streamer(streamer)

        assert isinstance(y, torch.Tensor)
        assert y.shape == (6,)
        assert y.dtype == torch.long

    def test_fault_mask(self):
        """fault_mask 应正确标记故障 tail 所在位置。"""
        vocab = MockVocab(["正常1", "故障", "正常2", "电气故障"])
        streamer = make_streamer(
            {"head": "X", "relation": "类型为",   "tail": "故障"},
            {"head": "Y", "relation": "故障类别", "tail": "电气故障"},
        )
        builder = FaultLabelBuilder(
            vocab, fault_relations=["类型为", "故障类别"]
        )
        builder.build_from_streamer(streamer)

        expected_mask = torch.tensor([False, True, False, True])
        assert torch.equal(builder.fault_mask, expected_mask)

    def test_fault_nodes_sorted(self):
        """fault_nodes 应有序排列。"""
        vocab = MockVocab(["设备", "原因C", "原因A", "原因B"])
        streamer = make_streamer(
            {"head": "设备", "relation": "由...引起", "tail": "原因C"},
            {"head": "设备", "relation": "由...引起", "tail": "原因A"},
            {"head": "设备", "relation": "由...引起", "tail": "原因B"},
        )
        builder = FaultLabelBuilder(vocab)
        _, fault_nodes = builder.build_from_streamer(streamer)

        assert fault_nodes == sorted(fault_nodes)

    # === 大 batch 测试 ===

    def test_large_batch(self):
        """大量三元组 + 多 batch 仍正确识别。"""
        vocab = MockVocab(
            [f"原因_{i}" for i in range(20)] +
            [f"正常_{i}" for i in range(30)]
        )
        triplets = []
        for i in range(0, 20, 2):  # 偶数索引为故障原因
            triplets.append({"head": "设备", "relation": "由...引起", "tail": f"原因_{i}"})
        for i in range(10):
            triplets.append({"head": f"正常_{i}", "relation": "连接", "tail": f"原因_{i*2}"})
        streamer = MockStreamer(triplets, batch_size=5)
        builder = FaultLabelBuilder(vocab)
        y, fault_nodes = builder.build_from_streamer(streamer)

        assert len(fault_nodes) == 10
        assert all(n.startswith("原因_") for n in fault_nodes)
        assert y.sum().item() == 10

    # === 真实场景模拟 ===

    def test_cvffad_style_collects_causes(self):
        """CVFFAD 模式："由...引起" 的 tail 即为故障原因。"""
        vocab = MockVocab([
            "发动机启动困难", "蓄电池电量不足", "启动机故障",
            "点火系统故障", "更换蓄电池", "万用表",
        ])
        streamer = make_streamer(
            {"head": "发动机启动困难", "relation": "由...引起", "tail": "蓄电池电量不足"},
            {"head": "发动机启动困难", "relation": "由...引起", "tail": "启动机故障"},
            {"head": "发动机启动困难", "relation": "由...引起", "tail": "点火系统故障"},
            {"head": "发动机启动困难", "relation": "维修措施",  "tail": "更换蓄电池"},
            {"head": "发动机启动困难", "relation": "需要工具",  "tail": "万用表"},
        )
        builder = FaultLabelBuilder(
            vocab, fault_relations=["由...引起"]
        )
        y, fault_nodes = builder.build_from_streamer(streamer)

        # tail 即故障原因，head 不是
        assert set(fault_nodes) == {"蓄电池电量不足", "启动机故障", "点火系统故障"}
        # vocab 顺序: 发动机启动困难=0, 蓄电池电量不足=1, 启动机故障=2, 点火系统故障=3, 更换蓄电池=4, 万用表=5
        assert y.tolist() == [0, 1, 1, 1, 0, 0]

    # === 返回结构不变性 ===

    def test_build_is_idempotent(self):
        """连续两次 build 不应累积。"""
        vocab = MockVocab(["设备", "原因A"])
        streamer = make_streamer({"head": "设备", "relation": "由...引起", "tail": "原因A"})
        builder = FaultLabelBuilder(vocab)

        y1, fn1 = builder.build_from_streamer(streamer)
        y2, fn2 = builder.build_from_streamer(streamer)

        assert torch.equal(y1, y2)
        assert fn1 == fn2

    # === 默认 relations 包含 "由...引起" 验证 ===

    def test_default_includes_cause_relation(self):
        """默认 fault_relations 包含 "由...引起"，并正确收集 tail。"""
        vocab = MockVocab(["设备A", "原因X", "原因Y"])
        streamer = make_streamer(
            {"head": "设备A", "relation": "由...引起", "tail": "原因X"},
            {"head": "设备A", "relation": "由...引起", "tail": "原因Y"},
        )
        builder = FaultLabelBuilder(vocab)  # 使用默认 fault_relations
        y, fault_nodes = builder.build_from_streamer(streamer)

        assert set(fault_nodes) == {"原因X", "原因Y"}
