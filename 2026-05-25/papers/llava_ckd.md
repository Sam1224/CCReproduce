# LLaVA-CKD: Bottom-Up Cascaded Knowledge Distillation for Vision-Language Models

## 基本信息 / Basic Info

| Field | Value |
|-------|-------|
| **Title** | LLaVA-CKD: Bottom-Up Cascaded Knowledge Distillation for Vision-Language Models |
| **arXiv** | [2605.10641](https://arxiv.org/abs/2605.10641) |
| **Authors** | Nikolaos Gkalelis, Vasileios Mezaris |
| **Affiliations** | (Computer Vision lab, likely CERTH/ITI Greece) |
| **Date** | 2026-05-11 |
| **Bucket** | WEAK |
| **Total** | **55 / 100** |

---

## 故事弧 / Story Arc

> **问题:** 大型VLM（视觉语言模型）存储和计算需求过高，需要知识蒸馏获得小型学生模型。但直接从大型教师向小型学生蒸馏因容量差距（Capacity Gap）导致效果不佳。
>
> **方案:** LLaVA-CKD（Bottom-Up Cascaded Knowledge Distillation）：引入中间助手（Teaching Assistant, TA）模型，构建多级级联蒸馏链：教师 → TA → 学生，自底向上逐级传递知识，缩小每次蒸馏的容量差距。
>
> **基础:** 在LLaVA-KD框架（最新SOTA蒸馏基线）上构建，遵循LLaVA架构（冻结视觉编码器 + LLM骨干 + 视觉-语言连接器）。

---

## 方法概述 / Method Summary

**级联蒸馏链:**

```
Teacher VLM (Large)
     ↓ KD (LLaVA-KD)
Teaching Assistant VLM (Medium)  ← pre-trained via TinyLLaVA
     ↓ KD (LLaVA-KD)
Student VLM (Small)
```

- **预训练TA:** TA按照TinyLLaVA方法预训练，保证独立能力
- **逐级蒸馏:** 每对之间使用LLaVA-KD蒸馏目标
- **底向上训练:** 先蒸馏TA，再以TA为教师蒸馏学生

---

## 关键指标 / Key Metrics

| Benchmark | Metric | LLaVA-CKD | LLaVA-KD (direct) |
|-----------|--------|-----------|-------------------|
| VQA / MMBench | Accuracy | +↑ | direct teacher→student |

---

## 评分详情 / Score Breakdown

| Dimension | Score | Max | Justification |
|-----------|-------|-----|---------------|
| Innovation | 17 | 30 | 级联蒸馏为已知技术；应用于VLM是合理延伸，非突破性 |
| Experimental SOTA delta | 8 | 15 | 在LLaVA-KD基础上改进；差距数值有限 |
| Experimental quality / ablations | 9 | 15 | 消融TA配置；多个VQA基准验证 |
| Efficiency | 8 | 10 | 显著减小学生模型尺寸 |
| Generalization | 3 | 5 | VQA任务验证，其他任务迁移性待验证 |
| Domain relevance (ecom + governance) | 10 | 25 | 小型VLM对边端部署有价值，间接相关 |
| **Total** | **55** | **100** | — |

---

## 与本领域关联 / Domain Relevance

- **端侧部署:** 电商APP端侧VLM商品理解需要轻量级模型，蒸馏技术是关键
- **低成本内容审核:** 小模型服务于大规模内容审核流水线降低推理成本
