# GESE toy reproduction

本目录提供一个 **toy but runnable** 的 PyTorch 复现，用于抽象论文：
**Generate to Explore, Select to Exploit: Aligning LLM-based Headline Generation with Personalized Recommendation**（简称 GESE）。

> 目标：用一个合成点击环境，把论文里“生成候选（exploration）→ 多样性奖励 → 轻量选择器（exploitation）→ 离线训练与在线选择解耦”的核心形状跑通。

## 实现了什么（核心对应关系）

- **候选标题生成（Generate to Explore）**：
  - `model.py/Generator`：条件生成器（user + item_topic → headline tokens），一次采样 K 个候选。
- **多样性奖励（Diversity reward）**：
  - `train.py` 中对同一 (user,item) 的 K 个候选计算平均 pairwise Jaccard distance，作为 set-level 奖励项加入 REINFORCE。
- **轻量 Selector（Select to Exploit）**：
  - `model.py/Selector`：小 MLP，根据 (user_id, item_topic, title_tokens) 预测 CTR proxy，在线从候选集中选 top-1。
- **离线训练 / 在线选择解耦**：
  - `train.py`：先训练生成器；冻结生成器后离线采样生成 logged candidates，并训练 selector。
  - `test.py`：在线阶段只做 `generator.sample()` + `selector.argmax()`，不再更新参数。

## Toy 环境说明

- 新闻只有一个 `item_topic`（如 tech / sports）。
- 用户有个性化偏好：
  - 对“标题风格 token”（如 funny / serious）的偏好。
  - 对“标题是否忠实表达 item_topic”的偏好。
- 我们用一个可控的 **CTR proxy**（sigmoid(logit)）作为点击率。

## 文件

- `data.py`：合成环境、token/vocab、reward(CTR proxy) 计算、数据集封装。
- `model.py`：生成器（带位置 mask 的小 RNN）+ selector（title encoder + MLP）。
- `train.py`：
  - 生成器：REINFORCE + diversity reward（exploration）。
  - selector：离线 supervised 回归 CTR（exploitation）。
- `test.py`：比较三种策略在 test interactions 上的平均 CTR proxy：
  - `static headline`：固定模板（无个性化）。
  - `only-generator`：只用生成器概率最大候选。
  - `GESE (selector)`：生成 K 个候选，用 selector 选 top-1。

## 运行

```bash
cd CCReproduce/2026-09-15/GESE
python train.py
python test.py
```

`train.py` 会在本目录保存：
- `generator.pt` / `selector.pt`
- `train_metrics.json`

## 预期现象

- 加了 diversity reward 后，生成器采样的候选在 style token 上更分散。
- `only-generator` 容易偏向“高概率但不一定对该用户最优”的风格。
- `GESE selector` 能在候选集中更好地挑到对用户 CTR 更高的标题（平均 CTR proxy 更高）。
