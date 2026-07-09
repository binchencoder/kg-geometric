# KG Geometric Fault Diagnosis

这是一个使用 **PyTorch Geometric** 和 **知识图谱三元组** 实现的故障诊断最小示例，默认使用 **CPU** 环境。

## 环境安装

推荐使用 Conda：

```bash
conda env create -f environment.yml
conda activate kg-geometric-fault-diagnosis
```

## 运行示例

```bash
python fault_diagnosis.py
```

## 说明

- 图中的节点包括：设备、症状、故障原因
- 边来自知识图谱三元组，例如 `pump_01 --has_symptom--> vibration_high`
- 使用 `GCNConv` 学习节点表示，并进行节点级故障分类
- 最后输出节点的故障概率，作为诊断结果
