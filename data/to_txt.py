import os

import h5py
import numpy as np
import pandas as pd


def _visit_datasets(name: str, obj) -> None:
    """h5py visititems 回调，将每个数据集转为 txt/csv。（通过闭包访问 out_dir）"""
    if not isinstance(obj, h5py.Dataset):
        return
    try:
        arr = obj[:]
    except Exception:
        return  # 跳过无法读取的数据集（如 reference）

    # 跳过标量和过大的数组（如 >2 维暂不处理）
    if arr.ndim == 0 or arr.ndim > 2:
        return

    # MATLAB 用列主序存储，转置还原行主序
    if arr.ndim == 2:
        arr = arr.T

    safe_name = name.replace("/", "_")
    out_path = _visit_datasets.out_dir  # type: ignore[attr-defined]
    print(f"  导出: {safe_name}  shape={arr.shape}")
    np.savetxt(os.path.join(out_path, f"{safe_name}.txt"), arr, fmt="%.4f")
    pd.DataFrame(arr).to_csv(os.path.join(out_path, f"{safe_name}.csv"), index=False, header=False)


def mat_to_text(mat_file, out_dir="./output/"):
    os.makedirs(out_dir, exist_ok=True)
    _visit_datasets.out_dir = out_dir  # 通过函数属性传递 out_dir，避免 lambda 闭包问题

    with h5py.File(mat_file, "r") as f:
        f.visititems(_visit_datasets)

    print(f"转换完成，输出至 {out_dir}")

# 使用
mat_to_text("/home/binchen/Workspaces/PIE-Knowledge/故障诊断+故障预测/dataset_cascades/ieee118/ieee118/raw/blist.mat")