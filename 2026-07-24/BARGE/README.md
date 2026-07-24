# BARGE (toy reproduction)

这是对 **Bridging the Structural Gap: Adapting Autoregressive Generation for Recommendation** 的一个 **toy but runnable** PyTorch 复现。

## 保留的论文核心思想

- **ICA**：先对历史 item 序列做上下文增强，恢复 item 级结构感。
- **HPR**：在层级语义路径上用用户表征对候选前缀做重排。
- **DPD**：构造两条互补的语义 ID 通道，并在推理时进行 OR-fusion 风格的联合纠错。

## 有意做的简化

- 语义 ID 深度从论文中的更复杂层级降成了 **2 层 code**；
- OSQ-VAE 改成了可复现的可学习正交旋转近似，而非完整工业量化链路；
- 数据使用合成的推荐 toy dataset，但接口保持为 `history -> target item -> hierarchical codes`。

## 文件说明

- `data.py`：合成 catalog、双路径语义 ID、train/val dataloader
- `model.py`：ICA、Prefix reranker、双路径解码器、plain/BARGE 解码
- `train.py`：训练脚本，输出 plain vs BARGE 的 Recall@1
- `test.py`：加载 checkpoint，报告 plain/BARGE recall 与 dual-path 命中占比

## 运行方式

```bash
python train.py --epochs 8 --batch-size 32
python test.py --checkpoint checkpoints/barge_toy.pt
```

## 结果解释

期望看到：

1. 训练后 `barge_recall@1` 高于 `plain_recall@1`；
2. `dual_path_ratio` 非 0，说明双路径纠错被真实触发；
3. 这是论文思想复现，不是工业级完整实现，未覆盖大规模量化码本训练与线上 serving 细节。
