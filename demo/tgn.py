import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import copy
from torch_geometric.data import Data
from torch_geometric.nn.models.tgn import (
    TGNMemory, TimeEncoder, IdentityMessage, LastAggregator
)

# ===================== 全局超参 =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HOUR_PER_DAY = 24
WINDOW_DAY = 30
WINDOW_STEP = WINDOW_DAY * HOUR_PER_DAY    # 固定窗口720
FEAT_RAW = 7                                # HUFL,HULL,MUFL,MULL,LUFL,LULL,OT
TRANS_STATIC_DIM = 3
FEAT_DIM = FEAT_RAW + TRANS_STATIC_DIM
HIDDEN_DIM = 32
EPOCHS = 20
LR = 1e-3
DATA_BLOCK_DIR = "./ett_month_blocks/"
TRANS_STATIC = torch.tensor([500.0, 2015.0, 1.0], dtype=torch.float32).to(DEVICE)

# 全局标准化参数（训练阶段统计，推理复用）
feat_mean = None
feat_std = None

# ===================== 1. 生成月度模拟CSV时序数据 =====================
def generate_month_block(block_id, save_path):
    block_hour = 720
    feat_list = []
    time_stamp_list = []
    for t in range(block_hour):
        hour_in_day = t % 24
        base_hufl = 30 + 12 * np.sin(hour_in_day / 24 * 2 * np.pi)
        base_ot = 35 + 10 * np.sin(hour_in_day / 24 * 2 * np.pi)
        feat = [
            base_hufl + np.random.randn()*1.2,
            np.random.uniform(8,15),
            np.random.uniform(25,42),
            np.random.uniform(6,13),
            np.random.uniform(1,7),
            np.random.uniform(0,2),
            base_ot + np.random.randn()*0.8
        ]
        feat_list.append(feat)
        time_stamp_list.append(block_id * 720 + t)
    df = pd.DataFrame(feat_list, columns=["HUFL","HULL","MUFL","MULL","LUFL","LULL","OT"])
    df["time_stamp"] = time_stamp_list
    df.to_csv(save_path, index=False)
    return save_path

# ===================== 2. 构造固定长度窗口图 =====================
def sample_window_graph(feature_array, time_array):
    g = Data()
    n_nodes = feature_array.shape[0]
    static_mat = TRANS_STATIC.repeat(n_nodes, 1)
    g.x = torch.cat([feature_array, static_mat], dim=-1)
    g.t = time_array.clone()
    src = torch.arange(0, n_nodes - 1)
    dst = torch.arange(1, n_nodes)
    g.edge_index = torch.stack([src, dst], dim=0).to(DEVICE)
    g.edge_t = g.t[:-1]
    g.raw_msg = feature_array[:-1].to(DEVICE)
    # 标签：每个节点i预测i+1原始7维特征
    g.y = feature_array[1:, :].clone()
    return g

# ===================== 3. TGN模型 =====================
class TransformerTGN(nn.Module):
    def __init__(self, hidden_dim, raw_msg_dim, window_size):
        super().__init__()
        self.window_size = window_size
        self.msg_module = IdentityMessage(
            raw_msg_dim=raw_msg_dim,
            memory_dim=hidden_dim,
            time_dim=hidden_dim
        )
        self.aggr_module = LastAggregator()
        self.memory = TGNMemory(
            num_nodes=window_size,
            raw_msg_dim=raw_msg_dim,
            memory_dim=hidden_dim,
            time_dim=hidden_dim,
            message_module=self.msg_module,
            aggregator_module=self.aggr_module
        )
        self.head = nn.Linear(hidden_dim, FEAT_RAW)

    def forward(self, g):
        n_ids = torch.arange(g.x.size(0)).to(DEVICE)
        mem_emb, _ = self.memory(n_ids)
        last_emb = mem_emb[-1:, :]
        pred = self.head(last_emb)
        return pred

# ===================== 4. 分批训练 =====================
def train_tgn(block_paths):
    global feat_mean, feat_std
    all_data = []
    for bp in block_paths:
        df = pd.read_csv(bp)
        arr = df[["HUFL","HULL","MUFL","MULL","LUFL","LULL","OT"]].values
        all_data.append(arr)
    all_data_np = np.concatenate(all_data, axis=0)
    feat_mean = torch.tensor(all_data_np.mean(axis=0), dtype=torch.float32).to(DEVICE)
    feat_std = torch.tensor(all_data_np.std(axis=0) + 1e-6, dtype=torch.float32).to(DEVICE)

    model = TransformerTGN(
        hidden_dim=HIDDEN_DIM,
        raw_msg_dim=FEAT_RAW,
        window_size=WINDOW_STEP
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    mse_loss = nn.MSELoss()

    print("="*65)
    print("TGN时序模型开始训练")
    print("="*65)
    for ep in range(EPOCHS):
        loss_sum = 0.0
        for bp in block_paths:
            df = pd.read_csv(bp)
            feat_np = df[["HUFL","HULL","MUFL","MULL","LUFL","LULL","OT"]].values
            time_np = df["time_stamp"].values
            feat_full = torch.tensor(feat_np, dtype=torch.float32).to(DEVICE)
            feat_full = (feat_full - feat_mean) / feat_std
            time_full = torch.tensor(time_np, dtype=torch.long).to(DEVICE)
            max_offset = len(df) - WINDOW_STEP
            if max_offset <= 0:
                continue
            offset = np.random.randint(0, max_offset)
            feat_window = feat_full[offset:offset+WINDOW_STEP]
            time_window = time_full[offset:offset+WINDOW_STEP]
            g = sample_window_graph(feat_window, time_window)

            model.train()
            optimizer.zero_grad()
            model.memory.reset_state()
            model.memory.update_state(
                src=g.edge_index[0],
                dst=g.edge_index[1],
                t=g.edge_t,
                raw_msg=g.raw_msg
            )
            pred = model(g)
            pred_target = g.y[-1:, :]
            loss = mse_loss(pred, pred_target)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            del g
            torch.cuda.empty_cache()
        avg_loss = loss_sum / len(block_paths)
        if (ep + 1) % 5 == 0:
            print(f"Epoch {ep+1:2d} 训练平均损失(MSE): {avg_loss:.6f}")
    torch.save(model.state_dict(), "tgn_ett_transformer.pth")
    print("\n训练完成，权重保存 tgn_ett_transformer.pth")
    return model

# ===================== 5.【重点修复推理逻辑】 =====================
def rolling_predict(model, latest_block_path, pred_hour=76):
    df = pd.read_csv(latest_block_path)
    feat_np = df[["HUFL","HULL","MUFL","MULL","LUFL","LULL","OT"]].values
    time_np = df["time_stamp"].values
    start_idx = len(feat_np) - WINDOW_STEP
    feat_window = torch.tensor(feat_np[start_idx:start_idx+WINDOW_STEP], dtype=torch.float32).to(DEVICE)
    time_window = torch.tensor(time_np[start_idx:start_idx+WINDOW_STEP], dtype=torch.long).to(DEVICE)
    feat_window = (feat_window - feat_mean) / feat_std

    curr_t = int(time_window[-1].item())
    result_list = []
    model.eval()
    print(f"\n===== 滚动自回归油温预测启动 =====")
    print(f"基准时间戳 t={curr_t}，向后连续预测{pred_hour}小时")
    print("-"*55)

    with torch.no_grad():
        for step in range(pred_hour):
            g = sample_window_graph(feat_window, time_window)
            # =========核心修复！每一步预测前，完整重置记忆 + 重新喂整条窗口序列
            model.memory.reset_state()
            model.memory.update_state(
                src=g.edge_index[0],
                dst=g.edge_index[1],
                t=g.edge_t,
                raw_msg=g.raw_msg
            )
            pred_feat_norm = model(g).squeeze()
            pred_feat = pred_feat_norm * feat_std + feat_mean
            curr_t += 1
            hufl, hull, mufl, mull, lufl, lull, ot = pred_feat.cpu().numpy()
            record = {"time": curr_t, "HUFL": round(hufl,2), "OT": round(ot,2)}
            result_list.append(record)
            print(f"第{step+1:2d}步 | t={curr_t:4d} | 顶层油温 OT={ot:.2f} ℃")

            # 滑动窗口更新
            feat_window = torch.cat([feat_window[1:, :], pred_feat_norm.unsqueeze(0)], dim=0)
            time_window = torch.cat([time_window[1:], torch.tensor([curr_t], dtype=torch.long, device=DEVICE)], dim=0)
            del g
    print("-"*55)
    torch.cuda.empty_cache()
    return result_list

# ===================== 主程序入口 =====================
if __name__ == "__main__":
    os.makedirs(DATA_BLOCK_DIR, exist_ok=True)
    block_list = []
    for month_id in range(12):
        filepath = os.path.join(DATA_BLOCK_DIR, f"month_{month_id}.csv")
        generate_month_block(month_id, filepath)
        block_list.append(filepath)
    print(f"成功生成{len(block_list)}个月度时序CSV文件")

    tgn_model = train_tgn(block_list)
    latest_file = block_list[-1]
    pred_result = rolling_predict(tgn_model, latest_file, pred_hour=76)
    target_window = pred_result[72:77]
    print("\n==========【业务目标区间：3天后14~18点预测汇总】==========")
    for item in target_window:
        t_stamp = item["time"]
        ot_val = item["OT"]
        alert = "⚠️ 高温过热故障预警" if ot_val > 50 else "✅ 油温运行正常"
        print(f"时间戳 t={t_stamp} | OT={ot_val:.2f}℃ | {alert}")
    print("="*70)
