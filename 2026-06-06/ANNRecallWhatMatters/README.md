# ANN Search: Recall What Matters — 复现（Toy）

本目录复现论文 **“ANN Search: Recall What Matters”**（arXiv:2606.04522）的核心观点：

- 传统 ANN 评估使用 `Recall@k`（命中 true kNN 的比例）。
- 论文提出 `1/Ratio@k`（逆近似比）作为更贴近下游质量的度量：比较“检索到的近邻距离”与“真实 kNN 距离”的差异，而不是强依赖 ID 完全一致。

由于原论文实验覆盖多种真实数据集、ANN 算法与 RAG/分类任务，这里提供 **可运行的 toy pipeline**（接口与脚本齐全），用合成的高维聚类数据模拟“Recall 下降但下游质量保持”的现象，并对比 `Recall@k` 与 `1/Ratio@k`。

## Quickstart

```bash
pip install -r requirements.txt

# 1) 生成数据并构建索引（本项目的“train”=build index）
python train.py --out_dir outputs/run1

# 2) 评估并输出不同近似强度（投影维度）下的 Recall@k / 1-Ratio@k / LabelPrecision@k
python evaluate.py --run_dir outputs/run1
```

## Files

- `data.py`: 合成数据生成（高维高斯簇 + label）。
- `model.py`: 简单 ANN（随机投影 + 低维 brute-force），用于制造可控的近似误差。
- `train.py`: 生成数据 + 构建索引 + 保存。
- `evaluate.py`: brute-force 真实 kNN，对比 `Recall@k`、`1/Ratio@k`、以及下游代理指标 `LabelPrecision@k`。

## Notes

- 该复现强调论文核心结论与指标定义，非对原论文全部 ANN 算法/数据集的逐项复现。
- 若你希望我把 toy 数据替换为一个真实的开源向量检索数据集（如 BEIR/SCIDOCS 的 embedding 版本）并接入 FAISS/HNSWlib，我也可以继续补全。
