# Valley3 — 电商全感官 MLLM Toy 复现

对应论文：**Valley3: Scaling Omni Foundation Models for E-commerce**（arXiv:2605.01278，ByteDance 2026）

官方代码：https://github.com/bytedance/Valley（已开源，含完整实现）

## 本目录定位

本目录是 Valley3 核心架构的 **toy 级学习性复现**，聚焦以下核心设计：
1. **Omni 多模态输入**：文本 + 图片（视觉 patch）+ 音频（Mel 谱）的统一编码与融合
2. **四阶段训练结构**：各阶段的 loss 设计与数据配置（toy 数据）
3. **可控推理（Thinking Modes）**：Non-thinking vs. Level-1/2/3 Thinking 的生成策略
4. **Agentic Search 接口**：简化版的工具调用接口

对于完整的生产级实现（Qwen3-VL 骨干、完整权重、大规模电商数据），请使用官方仓库。

## 目录结构

```
Valley3/
├── README.md
├── requirements.txt
├── src/
│   ├── audio_encoder.py       # Audio Transformer（简化版）
│   ├── vision_encoder.py      # Vision Encoder stub（接口对齐）
│   ├── omni_model.py          # Omni MLLM 主模型
│   ├── thinking_modes.py      # 可控推理模式（Non-thinking / Level 1-3）
│   └── agentic_search.py      # Agentic Search 工具调用接口
├── scripts/
│   ├── stage1_audio.py        # Stage 1: 音频理解预训练
│   ├── stage2_crossmodal.py   # Stage 2: 跨模态指令跟随
│   └── demo_inference.py      # 推理 demo（含 thinking mode 和 agentic search）
└── data/
    └── toy_ecom/              # 玩具电商数据（生成脚本内含）
```

## 快速运行

```bash
cd Valley3
pip install -r requirements.txt

# Stage 1: 音频对齐
python scripts/stage1_audio.py --epochs 3

# Stage 2: 跨模态指令跟随
python scripts/stage2_crossmodal.py --epochs 3

# 推理 demo（含 thinking mode 和 agentic search）
python scripts/demo_inference.py
```

## 方法概览（对齐论文主线）

### 架构
```
输入:  [文本 tokens] + [视觉 patches → Vision Encoder → MLP] 
      + [音频 Mel → Audio Transformer → MLP]
            ↓ 拼接到统一 token 序列
       LLM Backbone (Qwen3-VL, toy 中用小型 Transformer 替代)
            ↓
       可控推理 (think_level ∈ {0, 1, 2, 3})
            ↓
       输出 / Agentic Search (if needed)
```

### 四阶段训练（论文 Section 3）

| 阶段 | 目标 | 数据 | Loss |
|------|------|------|------|
| Stage 1 | 音频理解 | 音频-文本对 | MSE 重建 + 对比学习 |
| Stage 2 | 跨模态指令跟随 | omni 多模态指令数据 | 自回归 CE |
| Stage 3 | 电商域知识注入 | 电商 VQA / 属性 / 描述 | 自回归 CE |
| Stage 4 | 长上下文推理 | 长电商场景 | 自回归 CE |

### 可控推理公式

```
# Non-thinking (Level 0): 直接生成，无 <think> 标签
output = generate(prompt)

# Level 1-3 Thinking: 生成内部推理链后给出答案
# Level k 控制推理 token 预算（通过 max_think_tokens 参数）
output = generate("<think>" + reasoning_k + "</think>" + answer, 
                  max_think_tokens=budget_k)
```

### Agentic Search（论文 Section 4.3）

```
# 模型检测到需要外部信息时，生成工具调用
if model.needs_search(query):
    search_result = search_tool.retrieve(query.intent)
    augmented_context = context + search_result
    output = model.generate(augmented_context)
```

## 未实现细节（生产差异）

- Qwen3-VL 真实骨干替换（本复现使用小型 Transformer）
- 完整的电商预训练数据集（真实 SKU 图文音视频）
- Stage 3/4 的完整电商知识注入（用 toy 数据模拟）
- GRPO / RLHF 后训练（可控推理强化学习阶段）
- 真实 Agentic Search 的工具调用 API
