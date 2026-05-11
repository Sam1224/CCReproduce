# Cola-DLM: Continuous Latent Diffusion Language Model

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| 标题 | Continuous Latent Diffusion Language Model |
| 作者 | Hongcan Guo, Qinyu Zhao, Yian Zhao, Shen Nie, Rui Zhu, Qiushan Guo, Feng Wang, Tao Yang, Hengshuang Zhao, Guoqiang Wei, Yan Zeng |
| 机构 | ByteDance Seed |
| arXiv | https://arxiv.org/abs/2605.06548 |
| 提交日期 | 2026-05-07 |
| 领域标签 | 文本生成 · 扩散语言模型 · 层次潜在 · 大规模 · ByteDance |
| 桶类别 | WEAK |
| 综合评分 | **74 / 100** |

---

## 方法概述 (中文)

文本生成已经被自回归 Transformer 统治多年。**Cola-DLM** 试图用层次潜在扩散把"全局语义组织"和"局部文本现实化"两步解耦：

1. **Text VAE**：把文本映射到稳定的语义潜空间；
2. **Block-Causal DiT**：在潜空间做自回归式语义建模，但用扩散；
3. **条件解码**：把潜在轨迹转回 token 序列。

号称在多个 benchmark 显示出"强 scaling behavior"，最大跑到 2000 EFLOPs 训练量级（与 LLaMA-3 级别同量级）。论文 99 页 31 图，体量罕见。

---

## 故事线 (Story Arc)

> **现状不足：** 自回归 LM 把"语义规划"和"用词造句"耦合，难以分别 scale。
>
> **我们的解法：** Cola-DLM 层次潜在扩散——Text VAE 做潜空间，DiT 做语义生成，解码再现实化；scaling 行为更友好。

---

## 创新点分析

| 维度 | 描述 |
|------|------|
| 核心创新 | 把文本生成做成 hierarchical latent diffusion，把语义/词法解耦 |
| vs. 先前工作 | LLaDA 等 diffusion LM 在 token 级；Cola-DLM 在潜空间，scaling 更可控 |
| 工程价值 | ByteDance Seed 出品，到 2000 EFLOPs 训练 → 工业级路线 |

---

## 关键指标

- 多个 benchmark scaling 实验
- 2000 EFLOPs 训练量
- 详见论文 99 页正文

---

## 评分分解

| 维度 | 得分 | 满分 | 说明 |
|------|------|------|------|
| 创新性 | 25 | 30 | 层次潜在扩散文本 LM 是新路线 |
| 实验 SOTA delta | 12 | 15 | scaling 行为强，单点 SOTA 未公布 |
| 实验质量/消融 | 13 | 15 | 99 页 31 图 31 图体量罕见 |
| 效率 | 6 | 10 | 扩散生成本身慢 |
| 泛化性 | 4 | 5 | 通用文本生成 |
| 领域相关性 | 14 | 25 | 通用 LLM；电商场景可迁移做 SID/商品描述生成 |
| **Total** | **74** | **100** | |

---

## 代码与数据

- 项目页：https://hongcanguo.github.io/Cola-DLM/
- **本仓库提供完整 PyTorch 复现：** `code/ColaDLM/`
