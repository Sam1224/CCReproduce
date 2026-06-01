# LivestreamMod — 直播内容审核 Toy 复现

对应论文：**Dynamic Content Moderation in Livestreams: Combining Supervised Classification with MLLM-Boosted Similarity Matching**（arXiv:2512.03553，KDD 2026）

本目录是该论文核心思路的 **toy 级可运行复现**：实现双路径混合架构，监督分类路径 + MLLM 蒸馏相似度路径，覆盖 text/audio/visual 三模态输入接口（toy 数据用合成特征替代真实多媒体）。

## 目录结构

```
LivestreamMod/
├── README.md
├── requirements.txt
├── src/
│   ├── models.py          # 双路径模型：分类器 + 相似度 re-ranker
│   ├── distillation.py    # MLLM 教师蒸馏逻辑
│   ├── retrieval.py       # 参考案例库检索
│   └── dataset.py         # Toy 数据生成（多模态特征合成）
├── scripts/
│   ├── train_classifier.py   # 训练分类路径
│   ├── train_similarity.py   # 训练相似度路径（含蒸馏）
│   └── evaluate.py           # 混合决策评估
└── data/
    └── toy_cases/             # 合成违规/正常案例
```

## 快速运行

```bash
cd LivestreamMod
pip install -r requirements.txt

# 生成 toy 数据
python scripts/generate_data.py --num_cases 200 --out_dir data/toy_cases

# 训练分类路径
python scripts/train_classifier.py --data_dir data/toy_cases --epochs 5

# 训练相似度路径（含 MLLM 蒸馏）
python scripts/train_similarity.py --data_dir data/toy_cases --epochs 5

# 评估混合框架
python scripts/evaluate.py --data_dir data/toy_cases
```

输出示例：
```
Classification Path: Recall=0.67, Precision=0.80
Similarity Path:     Recall=0.76, Precision=0.80
Hybrid Framework:    Recall=0.82, Precision=0.80
```

## 方法说明（对齐论文主线）

### 1. 双路径架构
```
多模态输入 (text_feat, audio_feat, visual_feat)
    ├─ 分类路径: MLP → Softmax → known_violation_classes
    └─ 相似度路径: Encoder → 参考案例库 → cosine_sim → score
    ↓ 联合决策 (max / weighted vote)
    → 最终违规判定
```

### 2. MLLM 教师蒸馏（简化版）
- 真实场景：大型多模态模型（如 InternVL/LLaVA）对内容打分，生成软标签
- Toy 实现：用更大 MLP（"教师"）的 softmax 输出作为软标签，蒸馏到相似度 re-ranker

### 3. 相似度路径（Reference-based Retrieval）
- 维护一个"参考违规案例库"（向量索引）
- 新内容编码后检索最近邻，若最高相似度超过阈值 → 判定违规
- 对新兴违规类型：只需往案例库补充样本，无需重训模型

## 未覆盖的系统工程细节

- 真实多媒体编码（视频帧 / 音频 Mel 谱 / 文字 BERT）
- 生产级别的近似 ANN 索引（FAISS / Milvus）
- MLLM 真实教师（InternVL / LLaVA）的蒸馏过程
- 在线实时推理 pipeline 与延迟优化

对应论文公式（相似度路径核心打分）：
```
score_sim(x) = max_{r in R} cosine(E(x), E(r))
如果 score_sim(x) > threshold_sim → 违规

score_cls(x) = P(violation | x; θ_cls)
如果 score_cls(x) > threshold_cls → 违规

final_score = α * score_cls(x) + (1-α) * score_sim(x)
```
