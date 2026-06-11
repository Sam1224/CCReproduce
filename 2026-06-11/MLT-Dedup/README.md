# MLT-Dedup (Toy Reproduction)

- Paper: **MLT-Dedup: Efficient Large-Scale Online Video Deduplication via Multi-Level Representations and Spatial-Temporal Matching**
- arXiv: https://arxiv.org/abs/2606.12215

## What’s implemented

这是一个 **toy but runnable** 的复现，覆盖论文的三段式闭环：

- **Multi-level representations（ML-VE 思想）**：同一条视频同时产生
  - clip-level embedding（用于检索 / 索引，模拟“稀疏可索引表征”），
  - frame-level embedding（用于精匹配 / 定位，模拟“细粒度证据”）。
- **Candidate retrieval**：用 clip embedding 做 top-K 召回，评估 Recall@5。
- **DiF-SiM（简化版）**：在 frame embedding 上加入差分特征 Δf_t，构造 similarity matrix，并用“对角线偏移搜索”定位最可能的拷贝片段（输出 overlap ratio + duplicate score）。

## Limitations

- 原论文的 ML-VE 基于 Swin-L + Perceiver Resampler，并结合大量 SSL/蒸馏/损失设计；本 toy 直接在合成帧特征上实现“多粒度 + 差分匹配”的关键机制。
- 原论文使用 HNSW 等 ANN 工程索引；本 toy 用 brute-force topK 代替（接口一致）。

## Quickstart

```bash
cd 2026-06-11/MLT-Dedup
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 训练（主要是阈值/参数校准）
python train.py --out_dir runs/mlt_dedup

# 测试（retrieval Recall@5 + duplicate classification + overlap localization）
python test.py --ckpt_dir runs/mlt_dedup
```

你会看到类似输出：

- Retrieval Recall@5（top-5 是否能召回真重复源视频）
- Duplicate classification：Precision/Recall/F1
- Overlap localization：预测 overlap ratio 与真值误差（MAE）