# Valley3 (Toy Reproduction)

本目录是对论文 **"Valley3: Scaling Omni Foundation Models for E-commerce" (arXiv:2605.01278)** 的可运行复现（**toy 级别**），实现了全模态（文本 + 图像 + 视频 + 音频）电商大模型的核心架构和四阶段训练管道。

## 核心复现内容

- **四模态编码器**：
  - `PatchCNN`：轻量 CNN 代替 ViT，图像 → patch tokens。
  - `VideoEncoder`：逐帧编码 + 时序池化，视频 → tokens。
  - `AudioEncoder`：CNN 处理 log-mel 频谱图，音频 → tokens。
  - `TextEmbedder`：词嵌入 + 位置编码。
- **模态对齐投影层 (ModalityProjector)**：将各模态投影至统一 LLM 隐藏维度。
- **轻量 Transformer LLM 主干 (OmniLLM)**：处理拼接的多模态 token 序列。
- **任务头**：分类 / VQA / 内容审核（二分类）/ 图文生成 / 音频分类。
- **四阶段训练演示**：
  - Stage 1 (audio_align)：仅训练音频编码器 + 投影层。
  - Stage 2 (cross_modal)：全模型跨模态 SFT。
  - Stage 3 (ecom_domain)：继续预训练，注入电商知识。
  - Stage 4 (full)：长上下文 fine-tune。

> **原论文与 toy 复现的主要差异**：
> - 原论文底层 LLM 为多十亿参数级语言模型；本复现用 4 层 Transformer。
> - 原论文使用大规模真实电商图文视频音频数据；本复现用合成 toy 数据。
> - 内容审核任务训练了涉黄/盗版/直播性暗示检测；本复现为简化二分类。

## 目录结构

```
Valley3/
├── valley3/
│   ├── __init__.py
│   ├── dataset.py   — 全模态电商 toy 数据集（5 类任务）
│   ├── model.py     — Valley3 全模型（4 模态编码 + OmniLLM + 任务头）
│   ├── losses.py    — 多任务损失（CE for cls/VQA/mod/audio, LM for caption）
│   └── metrics.py   — 任务级准确率
├── train.py         — 四阶段训练脚本
├── eval.py          — 评测脚本
├── requirements.txt
└── README.md
```

## 快速开始

```bash
pip install -r requirements.txt

# 训练（toy，约 3 分钟 CPU）
python train.py --epochs 12 --batch_size 32 --output_dir runs/valley3

# 评测
python eval.py --ckpt runs/valley3/model.pt
```

## 关键实现与论文对应

| 论文组件 | 代码实现 |
|---------|---------|
| Audio Encoder (Stage 1) | `valley3/model.py::AudioEncoder` |
| Cross-modal SFT (Stage 2) | `train.py::set_stage("cross_modal")` |
| E-commerce Domain Injection (Stage 3) | `train.py::set_stage("ecom_domain")` |
| Long-context fine-tune (Stage 4) | `train.py::set_stage("full")` |
| Content moderation head (涉黄/盗版) | `Valley3Model.mod_head` (binary head) |
| Caption generation head | `Valley3Model.cap_head` (token logits) |
| Multi-task routing | `Valley3Model.forward(task=...)` |
