# Vidu S2（Toy but Runnable）

该目录是论文 **Vidu S2: Real-Time Interactive, Editable, and Spatial Video Generation** 的一个 *toy-but-runnable* 复现实现（不做真实视频扩散生成，用小尺寸 synthetic frame tensor 来模拟）。

本实现刻意保留论文中最核心、最“系统接口化”的能力：

- **实时 / 流式生成（streaming rollout）**：逐帧 `step()` 推出下一帧；模型内部维护 **avatar/stream state**（GRU hidden state）。
- **Reference-conditioned editing**：通过参考（reference）来控制属性（至少两类：`clothes`、`style`；同时也支持 `background`）。
- **可中途更新 reference**：流式过程中可在任意时刻调用 `update_reference()` 切换参考，后续帧会立刻跟随新的参考属性。

> 注意：这里的“空间视频生成”用 16x16 的合成图像表达：背景 / 衣服 / 风格分别对应不同空间区域的颜色块（从而可定义可解释的 attribute-control 指标）。

## 代码结构

- `data.py`：合成视频序列数据集 + 可解释渲染器/属性解码器（用于 test 指标）。
- `model.py`：带 StreamState 的流式模型（支持 reference 更新与 attribute mixing）。
- `train.py`：训练两个模型：
  - `baseline`：**不具备 reference editing 能力**（忽略 edit flags / reference）。
  - `editing`：具备 reference-conditioned editing 能力。
- `test.py`：在包含编辑与 reference 更新的序列上对比 baseline vs editing 的：
  - 重建误差（MSE）
  - 属性控制准确率（background / clothes / style）

## 运行

在本目录下执行：

```bash
python train.py --cpu --epochs 2 --samples 256
python test.py  --cpu --samples 128
```

正常情况下你会看到：

- editing 模型在 `clothes/style` 的属性控制准确率显著高于 baseline
- 当 reference 在序列中途发生更新时，editing 模型能更快切换到新的参考属性

## 与论文概念的对应（Toy 抽象）

- **Real-time interactive / stream state**：`StreamState(h, ref_attr_ids)`；`step()` 逐帧更新 `h`。
- **Reference-conditioned editing**：对每帧的 `(bg, clothes, style)`，用 `edit_flags` 决定从 stream prompt 取还是从 reference 取（mix in embedding space）。
- **Mid-stream reference update**：序列中途更新 `ref_attr_ids`，后续帧的 edited attributes 立刻变化。

本 toy 实现不包含论文中的真实视频生成 backbone、3D/空间建模、真实感损失与推理加速工程；但保留了接口与时序逻辑，便于扩展到真实模型。