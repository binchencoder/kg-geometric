# ==============================================
# RE-GCN 时序知识图谱链接预测 - 电力变压器故障场景
# 任务：时序尾实体预测 (h, r, ?, time_idx)
# 环境依赖：torch, numpy, collections
# 安装依赖: pip install torch numpy
# ==============================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
import random

# ====================== 全局配置参数 ======================
# 设备自动选择：有GPU用GPU，否则CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"【运行设备】{device}")

# 超参设置
EMBED_DIM = 64         # 实体、关系嵌入维度
HIDDEN_DIM = 64        # R-GCN卷积隐藏维度
TIME_SLICE_NUM = 8     # 总共有8个时间快照（t0~t7，代表不同巡检时段）
EPOCHS = 80            # 训练轮数
LR = 1e-3              # 学习率
BATCH_SIZE = 128       # 批次大小

# ====================== 1. 构建电力变压器时序知识图谱本体 & 模拟数据集 ======================
# 实体定义：电力变压器、油中气体、故障现象、故障类型
entity_list = [
    "变压器#1", "变压器#2", "变压器#3",
    "H2超标", "CH4超标", "C2H2超标", "CO超标",
    "低温过热", "高温过热", "电弧放电", "局部放电", "绝缘老化",
    "油温升高", "绕组变形", "铁芯多点接地"
]
# 关系定义
relation_list = [
    "发生现象",    # 变压器 -> 故障现象
    "诱发故障",    # 故障现象 -> 故障类型
    "表现症状",    # 故障 -> 现象
    "属于故障",    # 现象归属故障
    "伴随故障"     # 故障之间伴随发生
]

# 实体、关系映射id
ent2id = {ent: idx for idx, ent in enumerate(entity_list)}
rel2id = {rel: idx for idx, rel in enumerate(relation_list)}
num_entities = len(ent2id)
num_relations = len(rel2id)

print(f"【图谱本体信息】实体总数:{num_entities}, 关系总数:{num_relations}")
print("实体映射：", ent2id)
print("关系映射：", rel2id)

# 模拟时序四元组数据集 (head, rel, tail, time_slice)
# time_slice ∈ [0, TIME_SLICE_NUM-1]
raw_quads = []
random.seed(42)
np.random.seed(42)

# 人工构造基础事实，再随机扰动生成不同时间快照数据
base_facts = [
    ("变压器#1", "发生现象", "H2超标"),
    ("变压器#1", "发生现象", "C2H2超标"),
    ("变压器#2", "发生现象", "CH4超标"),
    ("变压器#3", "发生现象", "CO超标"),
    ("H2超标", "诱发故障", "局部放电"),
    ("C2H2超标", "诱发故障", "电弧放电"),
    ("CH4超标", "诱发故障", "低温过热"),
    ("CO超标", "诱发故障", "绝缘老化"),
    ("高温过热", "伴随故障", "油温升高"),
    ("电弧放电", "伴随故障", "绕组变形")
]

# 循环每个时间切片，动态增减事实模拟时序演化
for t in range(TIME_SLICE_NUM):
    # 基础事实全部存在
    for h, r, targ in base_facts:
        raw_quads.append((ent2id[h], rel2id[r], ent2id[targ], t))
    # 每隔2个时间步新增少量动态变化关系
    if t % 2 == 1:
        raw_quads.append((ent2id["变压器#1"], rel2id["发生现象"], ent2id["CO超标"], t))
    if t >= 4:
        raw_quads.append((ent2id["变压器#2"], rel2id["发生现象"], ent2id["H2超标"], t))

print(f"\n【原始四元组总量】{len(raw_quads)} 条时序事实")

# 数据集划分：按时间切片切分
# train: t0~t5; valid:t6; test:t7
train_quads = [q for q in raw_quads if q[3] <= 5]
valid_quads = [q for q in raw_quads if q[3] == 6]
test_quads  = [q for q in raw_quads if q[3] == 7]
print(f"训练集:{len(train_quads)} | 验证集:{len(valid_quads)} | 测试集:{len(test_quads)}")

# 构建【每个时间切片的多关系邻接矩阵】供R-GCN使用
# adj[time] -> dict: rel_id: [(h,t), (h,t)...]
def build_snapshot_adjacency(quad_list, total_time):
    snapshot_adj = [defaultdict(list) for _ in range(total_time)]
    for h, r, t, ts in quad_list:
        snapshot_adj[ts][r].append((h, t))
    return snapshot_adj

snapshot_adj = build_snapshot_adjacency(raw_quads, TIME_SLICE_NUM)

# ====================== 2. R-GCN 空间卷积模块（RE-GCN空间编码器） ======================
class RGCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_rel):
        super().__init__()
        self.in_dim = in_dim          # 输入特征维度
        self.out_dim = out_dim        # 输出特征维度
        self.num_rel = num_rel        # 关系总数量

        # 每种关系对应一个权重矩阵
        self.rel_weights = nn.ParameterList([
            nn.Parameter(torch.empty(in_dim, out_dim))
            for _ in range(num_rel)
        ])
        # 自环权重（实体自身特征保留）
        self.self_loop = nn.Parameter(torch.empty(in_dim, out_dim))

        # 参数初始化
        nn.init.xavier_uniform_(self.self_loop)
        for w in self.rel_weights:
            nn.init.xavier_uniform_(w)

    def forward(self, entity_emb, rel_edges):
        """
        entity_emb: [N, in_dim] 当前所有实体嵌入
        rel_edges: dict {rel_id: [(h,t)]} 当前时间切片内所有边
        return: [N, out_dim] 更新后的实体特征
        """
        N = entity_emb.shape[0]
        # 初始化聚合结果
        agg_out = torch.zeros(N, self.out_dim, device=device)

        # 1. 遍历每种关系，进行消息传递聚合
        for r_id, edge_list in rel_edges.items():
            if len(edge_list) == 0:
                continue
            edges = torch.tensor(edge_list, dtype=torch.long, device=device)
            heads = edges[:, 0]
            tails = edges[:, 1]
            # 取出头实体特征
            h_emb = entity_emb[heads]
            # 关系变换
            transformed = torch.matmul(h_emb, self.rel_weights[r_id])
            # 把消息聚合到尾实体上
            agg_out.index_add_(0, tails, transformed)

        # 2. 增加自连接特征
        self_feat = torch.matmul(entity_emb, self.self_loop)
        agg_out = agg_out + self_feat

        # 归一化 + 激活函数
        agg_out = F.normalize(agg_out, p=2, dim=-1)
        return F.relu(agg_out)

# ====================== 3. RE-GCN 完整模型主体 ======================
class RE_GCN(nn.Module):
    def __init__(self, num_ent, num_rel, embed_dim, hidden_dim):
        super().__init__()
        self.num_ent = num_ent
        self.num_rel = num_rel
        self.embed_dim = embed_dim

        # 实体初始嵌入
        self.ent_emb = nn.Embedding(num_ent, embed_dim)
        # 关系嵌入（用于DistMult打分）
        self.rel_emb = nn.Embedding(num_rel, embed_dim)

        # R-GCN卷积层
        self.rgcn_layer = RGCNLayer(embed_dim, hidden_dim, num_rel)
        # GRU时序单元：接收当前卷积特征，融合历史状态
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

        # 初始化权重
        nn.init.xavier_uniform_(self.ent_emb.weight)
        nn.init.xavier_uniform_(self.rel_emb.weight)

    def forward_snapshot(self, ts_adj, hidden_state):
        """
        处理单个时间快照
        ts_adj: 当前时间切片的邻接边
        hidden_state: GRU上一时刻状态 [N, hidden_dim]
        return: current_emb, new_hidden
        """
        # 步骤1：初始实体嵌入
        init_emb = self.ent_emb.weight.clone()
        # 步骤2：R-GCN空间卷积，捕捉当前时刻图谱结构信息
        spatial_emb = self.rgcn_layer(init_emb, ts_adj)
        # 步骤3：GRU融合时序信息，更新实体历史状态
        new_hidden = self.gru(spatial_emb, hidden_state)
        return spatial_emb, new_hidden

    def score_triple(self, h_idx, r_idx, t_idx, entity_emb):
        """
        DistMult 打分函数：计算 (h,r,t) 三元组置信度
        h_idx,r_idx,t_idx: 实体、关系id
        entity_emb: 某时刻经过图卷积后的实体特征矩阵
        return: 匹配分数，越高代表关系成立概率越大
        """
        h = entity_emb[h_idx]
        r = self.rel_emb(r_idx)
        t = entity_emb[t_idx]
        score = torch.sum(h * r * t, dim=-1)
        return score

# ====================== 4. 训练数据集采样器 ======================
class QuadDataset(torch.utils.data.Dataset):
    def __init__(self, quad_data):
        self.data = quad_data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        h, r, t, ts = self.data[idx]
        return torch.LongTensor([h, r, t, ts])

# ====================== 5. 模型训练入口函数 ======================
def train_re_gcn():
    # 初始化模型
    model = RE_GCN(num_entities, num_relations, EMBED_DIM, HIDDEN_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss()

    # 构建训练DataLoader
    train_dataset = QuadDataset(train_quads)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    print("\n==================== 开始训练 RE-GCN ====================")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        # GRU初始隐藏状态：全部置0 [实体数量, 隐藏维度]
        gru_hidden = torch.zeros(num_entities, HIDDEN_DIM, device=device)
        # 保存每个时间切片输出的实体Embedding，推理时使用
        time_entity_emb = {}

        # 按时间顺序遍历所有快照，模拟时序演进
        for time_step in range(TIME_SLICE_NUM):
            adj_now = snapshot_adj[time_step]
            # 前向传播，得到当前时刻实体表征 & 更新GRU状态
            emb_now, gru_hidden = model.forward_snapshot(adj_now, gru_hidden)
            time_entity_emb[time_step] = emb_now

        # 批次训练
        for batch in train_loader:
            batch = batch.to(device)
            h_batch = batch[:,0]
            r_batch = batch[:,1]
            t_batch = batch[:,2]
            ts_batch = batch[:,3]

            pos_scores = []
            for i in range(len(batch)):
                ts = int(ts_batch[i].item())
                emb_ts = time_entity_emb[ts]
                s = model.score_triple(h_batch[i], r_batch[i], t_batch[i], emb_ts)
                pos_scores.append(s)
            pos_scores = torch.stack(pos_scores)

            # 负采样：随机替换尾实体生成负样本
            neg_t = torch.randint(0, num_entities, size=t_batch.shape, device=device)
            neg_scores = []
            for i in range(len(batch)):
                ts = int(ts_batch[i].item())
                emb_ts = time_entity_emb[ts]
                s = model.score_triple(h_batch[i], r_batch[i], neg_t[i], emb_ts)
                neg_scores.append(s)
            neg_scores = torch.stack(neg_scores)

            # 构造标签：正样本=1，负样本=0
            all_scores = torch.cat([pos_scores, neg_scores])
            all_labels = torch.cat([
                torch.ones_like(pos_scores),
                torch.zeros_like(neg_scores)
            ])

            loss = loss_fn(all_scores, all_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        # 每10轮打印日志
        if (epoch + 1) % 10 == 0:
            print(f"【Epoch {epoch+1:3d}/{EPOCHS}】 训练损失 = {avg_loss:.4f}")

    print("==================== 训练完成 ====================\n")
    return model

# ====================== 6. 推理接口：时序链接预测（尾实体预测） ======================
def predict_tail(model, head_name, rel_name, time_slice, topk=5):
    """
    推理函数：(head, rel, ?, time_slice) 时序尾实体预测
    head_name: 头实体文本名称
    rel_name: 关系文本名称
    time_slice: 需要预测的时间步
    topk: 返回置信度最高的K个尾实体
    """
    model.eval()
    h_id = ent2id[head_name]
    r_id = rel2id[rel_name]

    # 前向推演到目标时间切片，获取该时刻实体embedding
    gru_hidden = torch.zeros(num_entities, HIDDEN_DIM, device=device)
    target_emb = None
    for ts in range(TIME_SLICE_NUM):
        adj_now = snapshot_adj[ts]
        emb_now, gru_hidden = model.forward_snapshot(adj_now, gru_hidden)
        if ts == time_slice:
            target_emb = emb_now
            break

    # 遍历全部候选实体打分
    score_list = []
    with torch.no_grad():
        for cand_t_id in range(num_entities):
            score = model.score_triple(
                torch.tensor(h_id, device=device),
                torch.tensor(r_id, device=device),
                torch.tensor(cand_t_id, device=device),
                target_emb
            )
            score_list.append((score.item(), cand_t_id))

    # 分数从高到低排序
    score_list.sort(reverse=True, key=lambda x:x[0])
    # id转回实体名称
    id2ent = {v:k for k,v in ent2id.items()}
    print(f"===== 时序链接预测结果【时间切片{time_slice}】 =====")
    print(f"查询三元组：({head_name}, {rel_name}, ? )")
    for rank, (scr, tid) in enumerate(score_list[:topk]):
        print(f"Rank{rank+1:2d} | 实体:{id2ent[tid]:<12} | 置信分数:{scr:.4f}")
    print("="*50 + "\n")

# ====================== 程序主入口 ======================
if __name__ == "__main__":
    # 启动训练
    trained_model = train_re_gcn()

    # ========= 案例1：时序推理 t=7时段：变压器#1 会发生什么现象 =========
    predict_tail(
        model=trained_model,
        head_name="变压器#1",
        rel_name="发生现象",
        time_slice=7,
        topk=6
    )

    # ========= 案例2：时序推理 t=7时段：H2超标 会诱发什么故障 =========
    predict_tail(
        model=trained_model,
        head_name="H2超标",
        rel_name="诱发故障",
        time_slice=7,
        topk=6
    )

    # ========= 案例3：时序推理 t=5时段：变压器#2 发生现象 =========
    predict_tail(
        model=trained_model,
        head_name="变压器#2",
        rel_name="发生现象",
        time_slice=5,
        topk=6
    )