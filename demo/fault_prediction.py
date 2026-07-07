import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch_geometric.data import HeteroData
from torch_geometric.nn import RGCNConv, Linear, HeteroConv, SAGEConv
from torch_geometric.nn.models.tgn import TGNMemory, IdentityMessage, LastAggregator, TimeEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error
import copy

# ===================== 全局配置 =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 健康状态映射
HEALTH_MAPPING = {
    0: "正常运行",
    1: "轻微过热",
    2: "严重过热",
    3: "过载故障"
}
HEALTH_NUM = 4
# 特征列表
FEATURE_LIST = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
FEATURE_NUM = len(FEATURE_LIST)
# 训练配置
EPOCHS = 100
LR = 1e-3
HIDDEN_DIM = 32


# ===================== 1. 数据加载与标签生成 =====================
def load_and_preprocess_dataset(csv_path):
    """加载变压器时序数据，生成健康状态标签"""
    df = pd.read_csv(csv_path)
    # 时间格式处理
    df["date"] = pd.to_datetime(df["date"])
    # 特征标准化
    df[FEATURE_LIST] = (df[FEATURE_LIST] - df[FEATURE_LIST].mean()) / df[FEATURE_LIST].std()

    # 生成健康状态标签（电力行业规则）
    health_labels = []
    for _, row in df.iterrows():
        ot = row["OT"]
        hufl = row["HUFL"]
        mufl = row["MUFL"]
        lufl = row["LUFL"]

        # 过载故障：主负载超过阈值，油温快速上升
        if hufl > 20 or mufl > 18 or lufl > 10:
            health_labels.append(3)
        # 严重过热：油温≥50℃
        elif ot >= 50:
            health_labels.append(2)
        # 轻微过热：40℃≤油温<50℃
        elif ot >= 40:
            health_labels.append(1)
        # 正常运行
        else:
            health_labels.append(0)

    df["health_label"] = health_labels
    # 生成未来3步油温预测标签（趋势预测）
    df["future_OT"] = df["OT"].shift(-3)
    df = df.dropna()
    return df


# ===================== 2. 构建电力变压器时序异构图知识图谱 =====================
def build_transformer_kg(df, transformer_id=0):
    data = HeteroData()
    slice_num = len(df)

    # 1. 节点1：transformer 电力变压器
    data["transformer"].x = torch.tensor([[transformer_id, 2, 110]], dtype=torch.float32)  # 编号、年限、电压等级

    # 2. 节点2：time_slice 时序运行切片
    data["time_slice"].x = torch.tensor(df[FEATURE_LIST].values, dtype=torch.float32)
    data["time_slice"].y_health = torch.tensor(df["health_label"].values, dtype=torch.long)
    data["time_slice"].y_future_ot = torch.tensor(df["future_OT"].values, dtype=torch.float32)
    # 顺序序号用于 TGN 时序编码，同时保存真实日期时间字符串用于展示
    data["time_slice"].time = torch.tensor(np.arange(slice_num), dtype=torch.float32).unsqueeze(1)
    data["time_slice"].date_str = df["date"].dt.strftime("%Y-%m-%d %H:%M:%S").tolist()

    # 3. 节点3：health_state 健康状态
    data["health_state"].x = torch.randn(HEALTH_NUM, 4)
    # 4. 节点4：feature_indicator 特征指标
    data["feature_indicator"].x = torch.randn(FEATURE_NUM, 2)

    # 构建边：transformer -> has_time_slice -> time_slice
    trans2slice = []
    for slice_id in range(slice_num):
        trans2slice.append([0, slice_id])
    data["transformer", "has_time_slice", "time_slice"].edge_index = torch.tensor(trans2slice).T
    # 时序边时间戳（TGN需要）
    data["transformer", "has_time_slice", "time_slice"].edge_time = data["time_slice"].time.squeeze()

    # 构建边：time_slice -> has_health_state -> health_state
    slice2health = []
    for slice_id, health_id in enumerate(df["health_label"].values):
        slice2health.append([slice_id, health_id])
    data["time_slice", "has_health_state", "health_state"].edge_index = torch.tensor(slice2health).T

    # 构建边：time_slice -> has_feature -> feature_indicator
    slice2feature = []
    for slice_id in range(slice_num):
        for feature_id in range(FEATURE_NUM):
            slice2feature.append([slice_id, feature_id])
    data["time_slice", "has_feature", "feature_indicator"].edge_index = torch.tensor(slice2feature).T

    # 行业先验知识边：health_state -> state_has_symbol -> feature_indicator
    state_feature_rule = [
        [1, 6],  # 轻微过热 → 特征：油温OT
        [2, 6],  # 严重过热 → 特征：油温OT
        [3, 0],  # 过载故障 → 特征：主负载HUFL
        [3, 2],  # 过载故障 → 特征：主负载MUFL
    ]
    data["health_state", "state_has_symbol", "feature_indicator"].edge_index = torch.tensor(state_feature_rule).T

    return data.to(DEVICE)


# ===================== 3. 模型1：R-GCN 故障诊断模型 =====================
class RGCNFaultDiagnosis(torch.nn.Module):
    def __init__(self, hidden_dim, out_dim):
        super().__init__()
        # 二分图卷积：health_state(源,4维) → time_slice(目标,FEATURE_NUM维)，输出 per-time_slice
        self.conv1 = RGCNConv((4, FEATURE_NUM), hidden_dim, num_relations=1)
        self.conv2 = RGCNConv((hidden_dim, hidden_dim), hidden_dim, num_relations=1)
        self.out_linear = Linear(hidden_dim, out_dim)

    def forward(self, x_dict, edge_index_dict):
        x_slice = x_dict["time_slice"]
        x_health = x_dict["health_state"]
        edge = edge_index_dict["time_slice", "has_health_state", "health_state"]
        # 翻转边方向：health_state → time_slice，使输出为每个 time_slice 的嵌入
        edge_rev = edge.flip(0)
        edge_type = torch.zeros(edge_rev.shape[1], dtype=torch.long, device=edge_rev.device)
        # source=health_state, target=time_slice → 输出 shape: [num_time_slices, hidden_dim]
        h1 = self.conv1((x_health, x_slice), edge_rev, edge_type=edge_type)
        h1 = F.relu(h1)
        h2 = self.conv2((h1, h1), edge_rev, edge_type=edge_type)
        logits = self.out_linear(h2)
        return logits, h2


# ===================== 4. 模型2：TGN 时序趋势预测模型 =====================
class TGN(torch.nn.Module):
    """Temporal Graph Network：基于节点记忆 + 时序编码 + 异构图卷积的时序预测模型"""

    def __init__(self, in_channels, hidden_channels, out_channels, edge_types,
                 num_layers, num_time_slices):
        super().__init__()
        # 余弦时序编码器（比线性编码更能捕获周期模式）
        self.time_enc = TimeEncoder(hidden_channels)
        # TGN 记忆模块：index 0=transformer, index 1..N=time_slice
        self.memory = TGNMemory(
            num_nodes=num_time_slices + 1,
            raw_msg_dim=hidden_channels,
            memory_dim=hidden_channels,
            time_dim=hidden_channels,
            message_module=IdentityMessage(hidden_channels, hidden_channels, hidden_channels),
            aggregator_module=LastAggregator(),
        )
        # PyG 默认将 last_update 创建为 Long，_reset_message_store 也将空时间戳初始化为 Long；
        # 但我们使用 Float 时间戳。为避免 eval() 触发 _update_memory 时 Long→Float 写入不匹配，
        # 这里将 last_update 改为 Float，并同步将 msg_store 中的空时间戳统一转为 Float。
        self.memory.register_buffer('last_update', self.memory.last_update.float())
        for _store in (self.memory.msg_s_store, self.memory.msg_d_store):
            for _j in range(self.memory.num_nodes):
                _s, _d, _t, _m = _store[_j]
                _store[_j] = (_s, _d, _t.float(), _m)
        # 节点特征投影
        self.node_proj = nn.ModuleDict({
            node_type: Linear(in_dim, hidden_channels)
            for node_type, in_dim in in_channels.items()
        })
        # 多层异构图卷积
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                edge_type: SAGEConv((-1, -1), hidden_channels)
                for edge_type in edge_types
            }, aggr='mean')
            self.convs.append(conv)
        # 输出投影
        self.out_proj = nn.ModuleDict({
            node_type: Linear(hidden_channels, out_channels)
            for node_type in in_channels.keys()
        })

    def forward(self, x_dict, edge_index_dict, edge_time_dict=None):
        device = x_dict["time_slice"].device
        # 投影到隐空间
        x_dict = {k: self.node_proj[k](v) for k, v in x_dict.items()}
        # TGN 时序记忆更新
        if edge_time_dict is not None:
            # 1) detach 记忆状态，断开与上一轮 forward/backward 计算图的连接，
            #    避免 "Trying to backward through the graph a second time" 错误。
            if hasattr(self.memory, 'detach'):
                self.memory.detach()
            if hasattr(self.memory, 'memory') and self.memory.memory is not None:
                self.memory.memory = self.memory.memory.detach()
            if hasattr(self.memory, 'last_update') and self.memory.last_update is not None:
                self.memory.last_update = self.memory.last_update.detach()
            # 2) 重置记忆与消息存储，保证本轮 forward 的因果性。
            self.memory.reset_state()
            # PyG TGNMemory._reset_message_store 将 msg_store 中的空时间戳初
            # 始化为 Long，与 self.last_update (Long) 保持一致。但若传入 Float
            # 时间戳，_compute_msg 中 cat(empty_Long, float_data) 会因 dtype
            # 不同而失败。这里将 msg_store 的空时间戳统一转为 Float 以避免冲突。
            for store in (self.memory.msg_s_store, self.memory.msg_d_store):
                for j in range(self.memory.num_nodes):
                    s, d, t, m = store[j]
                    store[j] = (s, d, t.float(), m)
            for edge_type, edge_time in edge_time_dict.items():
                src_type, _, dst_type = edge_type
                edge_index = edge_index_dict[edge_type]
                src, dst = edge_index[0], edge_index[1]
                # 映射索引：transformer→0, time_slice i→i+1
                src_mapped = torch.zeros(len(src), dtype=torch.long, device=device)
                dst_mapped = dst + 1
                # 按时序排序，保证时间因果性
                time_order = edge_time.argsort()
                src_mapped = src_mapped[time_order]
                dst_mapped = dst_mapped[time_order]
                t_sorted = edge_time[time_order].float()
                # 构造时序消息：源节点特征 + 时间编码
                time_emb = self.time_enc(t_sorted.unsqueeze(-1))
                raw_msg = x_dict[src_type][src[time_order]] + time_emb
                # 更新 TGN 节点记忆
                self.memory.update_state(src_mapped, dst_mapped, t_sorted, raw_msg)
            # 提取 time_slice 节点的最新记忆并注入特征
            num_slices = x_dict["time_slice"].shape[0]
            time_indices = torch.arange(1, num_slices + 1, device=device)
            mem_z, _ = self.memory(time_indices)
            x_dict["time_slice"] = x_dict["time_slice"] + mem_z
        # 逐层异构图卷积（HeteroConv 只输出目标节点，需保留仅作源节点的类型）
        for conv in self.convs:
            prev = x_dict
            x_dict = conv(x_dict, edge_index_dict)
            # 恢复未被更新的节点类型（如 transformer 仅作源节点，不会出现在输出中）
            for k, v in prev.items():
                if k not in x_dict:
                    x_dict[k] = v
            x_dict = {k: F.relu(v) for k, v in x_dict.items()}
        # 输出投影
        out_dict = {k: self.out_proj[k](v) for k, v in x_dict.items()}
        return out_dict


class TGNOilTemperaturePredict(torch.nn.Module):
    def __init__(self, hidden_dim, num_time_slices):
        super().__init__()
        self.gnn = TGN(
            in_channels={
                "transformer": 3,
                "time_slice": FEATURE_NUM,
                "health_state": 4,
                "feature_indicator": 2
            },
            hidden_channels=hidden_dim,
            out_channels=hidden_dim,
            edge_types=[
                ("transformer", "has_time_slice", "time_slice"),
                ("time_slice", "has_health_state", "health_state"),
                ("time_slice", "has_feature", "feature_indicator"),
                ("health_state", "state_has_symbol", "feature_indicator"),
            ],
            num_layers=2,
            num_time_slices=num_time_slices,
        )
        # 油温预测头
        self.ot_head = nn.Sequential(
            Linear(hidden_dim, 16),
            nn.ReLU(),
            Linear(16, 1)
        )
        # 故障风险预测头
        self.risk_head = nn.Sequential(
            Linear(hidden_dim, 16),
            nn.ReLU(),
            Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, hetero_data):
        emb_dict = self.gnn(
            x_dict=hetero_data.x_dict,
            edge_index_dict=hetero_data.edge_index_dict,
            edge_time_dict={("transformer", "has_time_slice", "time_slice"): hetero_data[
                "transformer", "has_time_slice", "time_slice"].edge_time}
        )
        slice_emb = emb_dict["time_slice"]
        future_ot = self.ot_head(slice_emb)
        fault_risk = self.risk_head(slice_emb)
        return future_ot, fault_risk, slice_emb


# ===================== 5. 完整训练循环 =====================
def train_two_models(kg_graph):
    # 划分训练/测试集
    total_num = kg_graph["time_slice"].x.shape[0]
    train_idx, test_idx = train_test_split(np.arange(total_num), test_size=0.2, random_state=42)
    train_idx = torch.tensor(train_idx, dtype=torch.long).to(DEVICE)
    test_idx = torch.tensor(test_idx, dtype=torch.long).to(DEVICE)

    # 初始化双模型
    num_time_slices = kg_graph["time_slice"].x.shape[0]
    diag_model = RGCNFaultDiagnosis(HIDDEN_DIM, HEALTH_NUM).to(DEVICE)
    tgn_model = TGNOilTemperaturePredict(HIDDEN_DIM, num_time_slices).to(DEVICE)
    # 优化器与损失函数
    optimizer = torch.optim.Adam(list(diag_model.parameters()) + list(tgn_model.parameters()), lr=LR)
    loss_cls = nn.CrossEntropyLoss()
    loss_ot = nn.MSELoss()
    loss_risk = nn.BCELoss()

    best_diag_acc = 0.0
    best_ot_mae = float("inf")
    best_diag_state = None
    best_tgn_state = None

    print("\n===== 开始联合训练 R-GCN故障诊断 + TGN时序趋势预测 =====")
    for epoch in range(EPOCHS):
        diag_model.train()
        tgn_model.train()
        optimizer.zero_grad()

        # 前向传播
        health_logits, _ = diag_model(kg_graph.x_dict, kg_graph.edge_index_dict)
        future_ot_pred, fault_risk_pred, _ = tgn_model(kg_graph)

        # 计算多任务损失
        loss1 = loss_cls(health_logits[train_idx], kg_graph["time_slice"].y_health[train_idx])
        loss2 = loss_ot(future_ot_pred.squeeze()[train_idx], kg_graph["time_slice"].y_future_ot[train_idx])
        # 风险标签：严重过热/过载故障为高风险1，其他为0
        risk_label = (kg_graph["time_slice"].y_health >= 2).float()
        loss3 = loss_risk(fault_risk_pred.squeeze()[train_idx], risk_label[train_idx])
        total_loss = loss1 + loss2 + loss3

        total_loss.backward()
        optimizer.step()

        # 评估精度（无需 eval()，TGN 切换会触发 msg_store 刷新导致二次 backward）
        with torch.no_grad():
            # 诊断精度
            pred_health = torch.argmax(health_logits, dim=1)
            diag_acc = accuracy_score(
                kg_graph["time_slice"].y_health[test_idx].cpu().numpy(),
                pred_health[test_idx].cpu().numpy()
            )
            # 油温预测MAE
            ot_mae = mean_absolute_error(
                kg_graph["time_slice"].y_future_ot[test_idx].cpu().numpy(),
                future_ot_pred.squeeze()[test_idx].cpu().numpy()
            )
            # 风险预测精度
            risk_pred = (fault_risk_pred.squeeze() >= 0.5).float()
            risk_acc = accuracy_score(
                risk_label[test_idx].cpu().numpy(),
                risk_pred[test_idx].cpu().numpy()
            )

        # 保存最优模型（state_dict 方式，避免 deepcopy 报错）
        if diag_acc > best_diag_acc:
            best_diag_acc = diag_acc
            best_diag_state = copy.deepcopy(diag_model.state_dict())
        if ot_mae < best_ot_mae:
            best_ot_mae = ot_mae
            best_tgn_state = copy.deepcopy(tgn_model.state_dict())

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch:{epoch + 1:3d} | TotalLoss:{total_loss:.4f} | DiagAcc:{diag_acc:.4f} | OT_MAE:{ot_mae:.4f} | RiskAcc:{risk_acc:.4f}")

    # 保存模型文件 & 恢复最优权重
    os.makedirs("./trained_models", exist_ok=True)
    torch.save(best_diag_state, "./trained_models/transformer_fault_diag_rgcn.pth")
    torch.save(best_tgn_state, "./trained_models/transformer_trend_tgn.pth")
    diag_model.load_state_dict(best_diag_state)
    tgn_model.load_state_dict(best_tgn_state)
    print(f"\n训练完成！最优故障诊断准确率: {best_diag_acc:.4f}，最优油温预测MAE: {best_ot_mae:.4f}")
    return diag_model, tgn_model, test_idx


# ===================== 6. 推理打印函数（完整输出推理链路） =====================
def full_inference_print(kg_data, diag_model, tgn_model, slice_idx):
    """对指定时序切片执行故障诊断+趋势预测，完整打印推理过程"""
    print("=" * 100)
    print(f"【变压器时序推理】切片ID：{slice_idx} | 时间：{kg_data['time_slice'].date_str[slice_idx]}")
    print("=" * 100)

    # 1. 提取切片基础信息
    slice_feat = kg_data["time_slice"].x[slice_idx].cpu().numpy()
    true_health = kg_data["time_slice"].y_health[slice_idx].item()
    true_future_ot = kg_data["time_slice"].y_future_ot[slice_idx].item()
    print(f"\n[1] 切片基础运行信息：")
    print(f"  真实健康状态：{HEALTH_MAPPING[true_health]}")
    print(f"  未来3步真实油温：{true_future_ot:.4f}℃")
    print(f"  核心运行特征：")
    for i, feat_name in enumerate(FEATURE_LIST):
        print(f"    {feat_name}: {slice_feat[i]:.4f}")

    # 2. R-GCN故障诊断
    with torch.no_grad():
        health_logits, _ = diag_model(kg_data.x_dict, kg_data.edge_index_dict)
        pred_health_logits = health_logits[slice_idx]
        pred_health = torch.argmax(pred_health_logits).item()
        pred_health_prob = F.softmax(pred_health_logits, dim=0).cpu().numpy()

    print(f"\n[2] R-GCN知识图谱故障诊断结果：")
    print(f"  预测健康状态：{HEALTH_MAPPING[pred_health]}")
    print(f"  各类别预测概率：")
    for label_id, state_name in HEALTH_MAPPING.items():
        print(f"    {state_name}: {pred_health_prob[label_id]:.2%}")

    # 3. 知识图谱行业规则匹配解释
    print(f"\n[3] 知识图谱行业规则匹配解释：")
    rules = []
    ot_val = slice_feat[6]
    hufl_val = slice_feat[0]
    if pred_health == 3 and hufl_val > 20:
        rules.append("规则1：主负载HUFL超标 → 图谱关联过载故障典型特征")
    if pred_health == 2 and ot_val >= 50:
        rules.append("规则2：油温OT≥50℃ → 图谱关联严重过热典型特征")
    if pred_health == 1 and 40 <= ot_val < 50:
        rules.append("规则3：油温40℃≤OT<50℃ → 图谱关联轻微过热典型特征")
    if not rules:
        rules.append("无异常特征，符合正常运行状态")
    for rule in rules:
        print(f"  {rule}")

    # 4. TGN时序趋势预测
    with torch.no_grad():
        future_ot_pred, fault_risk_pred, _ = tgn_model(kg_data)
        pred_future_ot = future_ot_pred[slice_idx].item()
        pred_fault_risk = fault_risk_pred[slice_idx].item()

    risk_level = "低风险" if pred_fault_risk < 0.3 else ("中风险" if pred_fault_risk < 0.7 else "高风险")
    print(f"\n[4] TGN时序图模型趋势预测结果：")
    print(
        f"  未来3步预测油温：{pred_future_ot:.4f}℃，真实油温：{true_future_ot:.4f}℃，误差：{abs(pred_future_ot - true_future_ot):.4f}℃")
    print(f"  未来故障发生概率：{pred_fault_risk:.2%}，风险等级：{risk_level}")

    # 5. 最终诊断结论
    print(f"\n[5] 最终运维建议：")
    if pred_health == 0 and pred_fault_risk < 0.3:
        print("  ✅ 设备运行正常，维持常规巡检周期")
    elif pred_health == 1 or pred_fault_risk >= 0.3:
        print("  ⚠️  设备轻微异常，缩短巡检周期，密切关注油温与负载变化")
    elif pred_health == 2 or pred_fault_risk >= 0.7:
        print("  ⚠️  设备严重过热，立即安排停电检修，检查绝缘与散热系统")
    elif pred_health == 3:
        print("  ❌  设备过载故障，立即降低负载，紧急停机检查")

    print("\n" + "=" * 100 + " 推理结束 " + "=" * 100 + "\n")


# ===================== 主运行入口 =====================
if __name__ == "__main__":
    # 1. 加载ETTh1小时级变压器数据集
    print("加载ETTh1电力变压器时序数据集...")
    df = load_and_preprocess_dataset("/home/binchen/Workspaces/PIE-Knowledge/故障诊断+故障预测/电力变压器数据集-ETDataset/ETDataset/ETT-small/ETTh2 (副本).csv")
    # 2. 构建变压器时序异构图知识图谱
    print("构建电力变压器时序运行知识图谱...")
    kg_graph = build_transformer_kg(df, transformer_id=0)

    # 3. 从训练好的模型目录 ./trained_models/ 加载权重，直接执行推理
    MODEL_DIR = "../trained_models"
    diag_path = f"{MODEL_DIR}/transformer_fault_diag_rgcn.pth"
    tgn_path = f"{MODEL_DIR}/transformer_trend_tgn.pth"

    num_time_slices = kg_graph["time_slice"].x.shape[0]
    diag_model = RGCNFaultDiagnosis(HIDDEN_DIM, HEALTH_NUM).to(DEVICE)
    tgn_model = TGNOilTemperaturePredict(HIDDEN_DIM, num_time_slices).to(DEVICE)

    print(f"加载故障诊断模型：{diag_path}")
    diag_model.load_state_dict(torch.load(diag_path, map_location=DEVICE, weights_only=True))
    diag_model.eval()
    print(f"加载时序趋势预测模型：{tgn_path}")
    tgn_model.load_state_dict(torch.load(tgn_path, map_location=DEVICE, weights_only=True))
    tgn_model.eval()

    # 保持与训练时一致的 train/test 划分，用于选取测试集样本
    total_num = kg_graph["time_slice"].x.shape[0]
    _, test_idx = train_test_split(np.arange(total_num), test_size=0.2, random_state=42)
    test_idx = torch.tensor(test_idx, dtype=torch.long).to(DEVICE)

    # 对测试集前3个样本执行完整推理，打印全部结果
    print("\n===== 开始批量推理测试集样本 =====")
    for i in range(3):
        slice_id = test_idx[i].item()
        full_inference_print(kg_graph, diag_model, tgn_model, slice_id)