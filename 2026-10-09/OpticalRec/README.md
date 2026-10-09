# OpticalRec (Toy Reproduction)

- Paper: **OpticalRec: Unified Optical Vision-Language Representation for Multimodal Recommendation**
- arXiv: https://arxiv.org/abs/2610.05432

## What is implemented

这是一个 **toy but runnable** 的 PyTorch 复现，重点复现论文最核心的三个思想：

1. **视觉空间统一编码（visual-space unified encoding）**
   - 把商品文本元数据（title / brand / category / style / material / price）渲染成视觉 glyph。
   - 将商品图片与 glyph 文本拼成一张 “product card”。
   - 用单一视觉编码器直接对这张 card 编码，避免传统的 image/text 独立编码 + late fusion。

2. **与 late fusion 的对照实验接口**
   - `--mode optical`：使用 OpticalRec 风格的 unified optical encoding。
   - `--mode late_fusion`：使用独立图像编码 + 文本编码 + 拼接投影的基线。
   - 方便在 toy 数据上做最小可运行验证。

3. **完整可跑通 pipeline**
   - toy 商品目录生成
   - toy 用户历史/偏好交互生成
   - 渲染商品 card
   - 模型训练（BPR）
   - 测试指标（Recall@10 / NDCG@10）

## Files

- `data.py`: toy 数据构造、tokenizer、商品 card 渲染
- `model.py`: OpticalRec / LateFusion 两种编码器与推荐模型
- `train.py`: 训练脚本
- `test.py`: 测试脚本
- `render.py`: 导出样例商品 card，便于肉眼检查 unified optical 输入
- `requirements.txt`: 依赖

## Limitations (vs. the full paper)

- 原文使用冻结的大型 VLM（文中主实验使用 Qwen3-VL-8B）来实现“感知层 + 语义层”的双重注意力；这里为了保证可跑通与轻量化，使用的是 **小型 patch-transformer 视觉编码器** 与 **轻量文本编码器**。
- 原文在真实 Amazon Reviews 多个基准上报告 NDCG@20 / Recall@20 / HR@20；这里提供的是 **合成 toy 数据集** 上的 Recall@10 / NDCG@10。
- 原文强调“rendered text font / color / layout robustness”与更系统的理论分析；本复现保留统一编码思想和接口，但不声称完全覆盖原论文全部实验结论。
- 如果后续需要对接真实大模型权重，可在 `model.py` 中的 `FrozenVLMBackboneAdapter` 基础上替换为实际 VLM。

## Quickstart

```bash
cd 2026-10-09/OpticalRec
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 导出几张样例 unified product card
python render.py --out_dir samples

# 训练 OpticalRec 风格模型
python train.py --mode optical --epochs 8 --out_dir runs/optical

# 训练 late fusion 基线
python train.py --mode late_fusion --epochs 8 --out_dir runs/late_fusion

# 测试
python test.py --ckpt_dir runs/optical
python test.py --ckpt_dir runs/late_fusion
```

## Expected behavior

- 在默认 toy 数据构造下，`optical` 模式通常应不差于 `late_fusion`，因为 query 与商品语义的一部分确实通过“渲染后统一视觉编码”被更紧密耦合。
- 这是一个面向 **论文思想验证** 的最小闭环，而不是工业级 reproduction。

