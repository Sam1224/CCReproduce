# DiffCold (Toy Reproduction)

- Paper: **DiffCold: A Diffusion-based Generative Model for Cold-Start Item Recommendation**
- arXiv: https://arxiv.org/abs/2606.12245

## What’s implemented

这是一个 **toy but runnable** 的 PyTorch 复现，聚焦论文的核心机制闭环：

- **冷启动 seesaw 的 toy 场景**：warm item 有行为 embedding（可训练/可用），cold item 仅有内容特征；评测同时报告 warm/cold/overall 的 Recall@20、NDCG@20。
- **Retrieval-enhanced Aggregator**：根据内容特征检索 top-k 相似 warm item，并对其 embedding 聚合作为扩散采样的初始化（避免纯噪声起点）。
- **Conditional DDPM on item embeddings**：用内容特征 + 聚合初始化作为条件，训练扩散模型去重建 warm item 的行为 embedding；推理时对 cold item 迭代去噪生成 embedding。
- **Alignment-style auxiliary loss（简化版）**：在训练中对预测的 x0 做 batch 内余弦对齐（InfoNCE 风格），鼓励生成分布贴近 warm embedding 分布。

## Limitations (vs. the full paper)

- 原论文在真实数据集、真实 backbone（MF/LightGCN/SimGCL）上评测并对 warm/cold 分布做更全面分析；这里用合成数据验证“检索增强起点 + 条件扩散生成”对 cold-start 的效果。
- 这里的“Representation Alignment”用简化的 batch 对齐损失替代论文中的更完整模拟/对比学习设计；但接口位置与训练逻辑保留，便于替换。

## Quickstart

```bash
cd 2026-06-11/DiffCold
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 训练并评测
python train.py --out_dir runs/diffcold
python test.py  --ckpt_dir runs/diffcold
```

运行结束会打印三组指标：

- **Backbone**：warm 用真实行为 embedding，cold 置零（模拟“无冷启动能力”）；
- **ContentProj**：用线性投影内容特征生成所有 item embedding（通常 cold ↑ 但 warm ↓，体现 seesaw）；
- **DiffCold**：warm 保持 backbone，cold 用扩散生成（通常 cold ↑ 且 warm 基本不降）。
