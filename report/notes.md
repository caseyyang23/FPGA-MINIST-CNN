# 课程大作业 1 报告笔记

题目：基于 FPGA 的轻量化 CNN 加速器设计及其 MNIST 手写体图像识别实现

## 1. 网络与量化结果

网络结构：

- 输入：MNIST 灰度图像，1x28x28
- Conv1：3x3，1 -> 8，padding=1
- ReLU1 + MaxPool1：2x2，输出 8x14x14
- Conv2：3x3，8 -> 16，padding=1
- ReLU2 + MaxPool2：2x2，输出 16x7x7
- FC1：16x7x7 -> 32
- ReLU3
- FC2：32 -> 10

训练与量化结果：

| 项目 | 结果 |
| --- | --- |
| 参数量 | 26698 |
| 浮点模型测试准确率 | 98.49% |
| int8 量化模型测试准确率 | 98.47% |
| 训练 epoch | 8 |
| batch size | 128 |
| 学习率 | 0.001 |
| 输入预处理 | ToTensor only，无 Normalize |

结论：浮点模型准确率达到课程要求的 98% 以上；int8 量化后准确率仅下降约 0.02 个百分点，量化方案可用。

## 2. 8 位定点量化方案

- 输入图像：按 `INPUT_SCALE = 1/127` 量化到 int8。
- 权重：每层 per-tensor symmetric int8 量化。
- bias：int32 量化，scale 等于输入 scale 与权重 scale 的乘积。
- 激活：ReLU 后裁剪到 0 到 127，使用整数乘法加右移完成 requantization。

量化参数来自 `hls/data/quant_summary.json`：

| 层 | 权重 scale | 激活 scale | 定点乘子 | 右移 |
| --- | ---: | ---: | ---: | ---: |
| Conv1 | 0.0062853919 | 0.0278832837 | 29779 | 24 |
| Conv2 | 0.0043370268 | 0.1090867651 | 18599 | 24 |
| FC1 | 0.0045013517 | 0.3405605526 | 24190 | 24 |
| FC2 | 0.0044079432 | 输出 int32 logits | - | - |

## 3. HLS 架构

HLS 顶层函数：`cnn_accel`

模块划分：

- `conv1_layer`：3x3 卷积，1 输入通道，8 输出通道
- `pool1_layer`：2x2 最大池化
- `conv2_layer`：3x3 卷积，8 输入通道，16 输出通道
- `pool2_layer`：2x2 最大池化
- `fc1_layer`：全连接层，784 -> 32
- `fc2_layer`：全连接层，32 -> 10，输出 int32 logits
- `cnn_accel`：调用各层并完成 argmax 输出标签

顶层接口：

- `image`：`m_axi` 输入图像接口，长度 784
- `label`：4 bit 输出标签
- `control`：AXI-Lite 控制接口，包含 start/done/return 等信号

## 4. C Simulation 验证

solution1 和 solution2 均使用 10 张量化 MNIST 测试图像逐层验证。

solution1 日志：

`report/hls_baseline/solution1/csim/report/cnn_top_csim.log`

solution2 日志：

`report/hls_baseline/solution2/csim/report/cnn_accel_csim.log`

验证结果：

| 项目 | solution1 | solution2 |
| --- | --- | --- |
| 测试图片数量 | 10 | 10 |
| Conv1 逐层对比 | 10/10 PASS | 10/10 PASS |
| Pool1 逐层对比 | 10/10 PASS | 10/10 PASS |
| Conv2 逐层对比 | 10/10 PASS | 10/10 PASS |
| Pool2 逐层对比 | 10/10 PASS | 10/10 PASS |
| FC1 逐层对比 | 10/10 PASS | 10/10 PASS |
| FC2 logits 逐层对比 | 10/10 PASS | 10/10 PASS |
| Top 输出 vs 量化软件预测 | 10/10 PASS | 10/10 PASS |
| Top 输出 vs 真实标签 | 10/10 PASS | 10/10 PASS |
| 逐层允许误差 | <= +/-1 | <= +/-1 |

结论：两个版本都满足“至少 10 张量化图像仿真、逐层验证中间结果误差 <= +/-1”的要求。

## 5. HLS 综合结果对比

solution1 是 baseline 版本，solution2 是 3x3 kernel 展开优化版本。

| 指标 | solution1 baseline | solution2 optimized | 变化 |
| --- | ---: | ---: | ---: |
| Latency cycles | 354430 | 76031 | 降低约 78.5% |
| 100 MHz 单张延迟 | 3.544 ms | 0.760 ms | 降低约 78.5% |
| 100 MHz 等效吞吐量 | 约 282 images/s | 约 1315 images/s | 提升约 4.66x |
| Estimated timing | 7.300 ns | 7.300 ns | 均满足 10 ns |
| Interval | 354431 | 76032 | 仍接近 latency |

资源对比：

| 资源 | solution1 使用量 | solution1 利用率 | solution2 使用量 | solution2 利用率 |
| --- | ---: | ---: | ---: | ---: |
| BRAM_18K | 26 / 280 | 9% | 84 / 280 | 30% |
| DSP | 16 / 220 | 7% | 116 / 220 | 52% |
| FF | 5274 / 106400 | 4% | 19695 / 106400 | 18% |
| LUT | 7505 / 53200 | 14% | 22045 / 53200 | 41% |

主要模块延迟对比：

| 模块 | solution1 cycles | solution2 cycles | 分析 |
| --- | ---: | ---: | --- |
| Conv1 | 56467 | 56467 | 当前未明显改善，后续可进一步检查 conv1 是否受接口/循环结构限制 |
| Conv2 | 269696 | 12632 | 大幅改善，是 solution2 的主要加速来源 |
| FC1 | 25472 | 4129 | 明显改善，factor=4 部分展开有效 |
| Pool1 | 1573 | 1577 | 基本不变 |
| Pool2 | 789 | 789 | 基本不变 |
| FC2 | 410 | 410 | 基本不变 |
| Argmax | 12 | 12 | 可忽略 |

结论：solution2 通过对卷积核计算循环展开，把 3x3 MAC 从顺序计算改为并行乘加阵列，显著降低了 Conv2 延迟，并使整体 latency 从 354430 cycles 降到 76031 cycles，速度提升约 4.66 倍。这正好对应题目要求中的“3x3 乘加阵列”。代价是 DSP、LUT 和 BRAM 使用显著上升，但资源仍未超限，timing estimate 仍满足 100 MHz，因此 solution2 是有效优化版本。

## 6. 当前是否达到大作业要求

已经达到的部分：

- 网络结构符合轻量 CNN 要求：两卷积层 + 两全连接层。
- 浮点准确率 98.49%，达到准确率 >= 98% 要求。
- 完成 int8 权重/激活量化，并导出 HLS 可读头文件。
- HLS 顶层具备 `ap_ctrl_hs` 控制语义，接口包含输入图像和输出标签。
- 包含卷积、池化、ReLU、全连接、片上 buffer 和控制流程。
- C Simulation 使用 10 张量化图像逐层验证，所有层 PASS。
- HLS 综合已生成 RTL，并给出 latency、throughput、BRAM/DSP/FF/LUT 资源估计。
- solution2 已实现 3x3 kernel 并行展开，体现了 3x3 乘加阵列优化。

仍需补充：

- Word 报告正式排版。
- 架构图、仿真日志截图、综合报告截图或波形截图。
- 如果老师要求 Vivado implementation/place-route 后的最终资源，还需要继续跑 Vivado 实现。

## 7. Solution3 建议

solution2 已经足够好，不建议继续在 solution2 上盲目大规模展开。

原因：

- DSP 已从 7% 增加到 52%，继续展开 Conv2 输入通道可能导致资源快速上升。
- Interval 仍接近 latency，说明当前架构仍是“一张图处理完再处理下一张图”，不是多图流水。
- 后续优化应展示“性能-资源权衡”，而不是单纯追求更低 latency。

solution3 可二选一：

1. 资源友好版：保留 solution2 的主要加速结构，但降低部分 unroll 或限制乘法资源，尝试减少 DSP/BRAM/LUT，同时观察 latency 退化幅度。
2. 小幅继续优化版：只针对 FC1 做轻量优化，例如把输入循环改为一维循环，并使用 factor=4 或 factor=8 的部分展开；不要继续完整展开 Conv2 的输入通道。

建议优先做资源友好版，因为 solution2 的性能已经提升约 4.66x，报告中展示性能与资源权衡会更完整。
