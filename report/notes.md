# 课程大作业 1 报告笔记

题目：基于 FPGA 的轻量化 CNN 加速器设计及其 MNIST 手写体图像识别实现

## 1. 网络模型设计

网络结构：

- 输入：MNIST 灰度图像，尺寸 1x28x28
- Conv1：3x3，输入通道 1，输出通道 8，padding=1
- ReLU1
- MaxPool1：2x2，输出 8x14x14
- Conv2：3x3，输入通道 8，输出通道 16，padding=1
- ReLU2
- MaxPool2：2x2，输出 16x7x7
- FC1：16x7x7 -> 32
- ReLU3
- FC2：32 -> 10

训练结果：

| 项目 | 结果 |
| --- | --- |
| 参数量 | 26698 |
| 浮点模型测试准确率 | 98.49% |
| 量化模型测试准确率 | 98.47% |
| 训练 epoch | 8 |
| batch size | 128 |
| 学习率 | 0.001 |
| 输入预处理 | ToTensor only，无 Normalize |

结论：浮点模型准确率达到课程要求的 98% 以上。int8 量化后准确率下降约 0.02 个百分点，量化损失很小。

## 2. 8 位定点量化方案

- 输入图像：按 `INPUT_SCALE = 1/127` 量化到 int8。
- 权重：每层 per-tensor symmetric int8 量化。
- bias：int32 量化，scale 等于输入 scale 与权重 scale 的乘积。
- 激活：ReLU 后裁剪到 0 到 127，并使用整数乘法加右移完成 requantization。

量化参数来自 `hls/data/quant_summary.json`：

| 层 | 权重 scale | 激活 scale | 定点乘子 | 右移 |
| --- | --- | --- | --- | --- |
| Conv1 | 0.0062853919 | 0.0278832837 | 29779 | 24 |
| Conv2 | 0.0043370268 | 0.1090867651 | 18599 | 24 |
| FC1 | 0.0045013517 | 0.3405605526 | 24190 | 24 |
| FC2 | 0.0044079432 | 无 ReLU requant，输出 int32 logits | - | - |

## 3. 硬件架构

HLS 顶层函数：`cnn_accel`

顶层接口：

- `image`：MNIST 输入图像，`m_axi` 读接口，长度 784
- `label`：输出分类标签，4 bit
- `control`：AXI-Lite 控制接口，包含 start/done/return 等控制信号

模块划分：

- `conv1_layer`：3x3 卷积，1 输入通道，8 输出通道
- `pool1_layer`：2x2 最大池化
- `conv2_layer`：3x3 卷积，8 输入通道，16 输出通道
- `pool2_layer`：2x2 最大池化
- `fc1_layer`：全连接层，784 -> 32
- `fc2_layer`：全连接层，32 -> 10，输出 int32 logits
- `cnn_accel`：调用各层并完成 argmax 输出标签

当前 baseline 版本已经拆成独立 layer 函数，因此 testbench 可以逐层验证中间结果。

## 4. Testbench 和仿真

测试数据：

- `hls/data/test_images.h`：10 张量化 MNIST 测试图像
- `hls/data/test_labels.h`：真实标签和 `GOLDEN_PRED_LABELS`
- `hls/data/golden_layers.h`：逐层 golden 输出
- `hls/data/weights.h`：int8 权重和 int32 bias

C Simulation 结果来自：

`report/hls_baseline/solution1/csim/report/cnn_top_csim.log`

| 项目 | 结果 |
| --- | --- |
| 测试图片数量 | 10 |
| Conv1 逐层对比 | 10/10 PASS |
| Pool1 逐层对比 | 10/10 PASS |
| Conv2 逐层对比 | 10/10 PASS |
| Pool2 逐层对比 | 10/10 PASS |
| FC1 逐层对比 | 10/10 PASS |
| FC2 logits 逐层对比 | 10/10 PASS |
| Top 输出 vs 量化软件预测 | 10/10 PASS |
| Top 输出 vs 真实标签 | 10/10 PASS |
| 逐层允许误差 | <= +/-1 |

10 张测试图像结果：

| 图片编号 | 真实标签 | 量化 golden 标签 | HLS 输出 | 是否正确 |
| --- | --- | --- | --- | --- |
| 0 | 7 | 7 | 7 | PASS |
| 1 | 2 | 2 | 2 | PASS |
| 2 | 1 | 1 | 1 | PASS |
| 3 | 0 | 0 | 0 | PASS |
| 4 | 4 | 4 | 4 | PASS |
| 5 | 1 | 1 | 1 | PASS |
| 6 | 4 | 4 | 4 | PASS |
| 7 | 9 | 9 | 9 | PASS |
| 8 | 5 | 5 | 5 | PASS |
| 9 | 9 | 9 | 9 | PASS |

结论：C Simulation 已经满足“加载至少 10 张量化图像进行仿真，逐层验证中间结果与软件模型误差 <= +/-1”的要求。

## 5. Baseline 性能分析

综合报告来自：

`report/hls_baseline/solution1/syn/report/cnn_accel_csynth.rpt`

目标时钟：10 ns，即 100 MHz。

| 项目 | 结果 |
| --- | --- |
| 单张图像延迟 | 354430 cycles |
| 100 MHz 下单张延迟 | 3.5443 ms |
| 100 MHz 下等效吞吐量 | 约 282.1 images/s |
| HLS Estimated Clock | 7.300 ns |
| 是否满足 100 MHz 时序估计 | 是，7.300 ns < 10 ns |

主要模块延迟：

| 模块 / 循环 | Latency cycles | 占总延迟比例 |
| --- | ---: | ---: |
| conv2_layer / conv2 loop | 269696 | 约 76.1% |
| conv1_layer | 56467 | 约 15.9% |
| fc1_loop | 25472 | 约 7.2% |
| pool1 | 1573 | 约 0.4% |
| pool2 | 789 | 约 0.2% |
| fc2_loop | 410 | 约 0.1% |
| argmax | 12 | 可忽略 |

资源利用率：

| 资源 | 使用量 | 可用量 | 利用率 |
| --- | ---: | ---: | ---: |
| BRAM_18K | 26 | 280 | 9% |
| DSP | 16 | 220 | 7% |
| FF | 5274 | 106400 | 4% |
| LUT | 7505 | 53200 | 14% |

存储需求估算：

| 项目 | 结果 |
| --- | --- |
| int8 权重数量 | 26632 bytes |
| int32 bias 数量 | 264 bytes |
| 权重 + bias 总量 | 26896 bytes，约 26.3 KiB |
| 最大单层特征图 | Conv1，8x28x28 = 6272 bytes |
| 全部中间特征图 buffer 合计 | 约 10224 bytes，约 10.0 KiB |

## 6. 是否达到大作业要求

已经达到的部分：

- 网络结构符合轻量 CNN 要求：两层卷积 + 两层全连接。
- 浮点准确率 98.49%，达到准确率 >= 98% 要求。
- 完成 int8 权重/激活量化，并导出 HLS 可读头文件。
- HLS 顶层具备时钟、复位、启动、完成等 `ap_ctrl_hs` 控制语义，接口包含输入图像和输出标签。
- 包含卷积、池化、ReLU、全连接、片上 buffer 和控制流程。
- C Simulation 已用 10 张量化图像逐层验证，所有层 PASS。
- HLS 综合已生成 Verilog/VHDL RTL，并给出 latency、throughput、BRAM/DSP/FF/LUT 资源估计。

仍需补充或注意的部分：

- Word 报告还需要正式排版，并加入架构图、关键代码截图、仿真日志截图或波形截图。
- 当前资源是 HLS synthesis estimate，不是 Vivado implementation/place-route 后的最终资源。如果老师要求 FPGA 综合/实现资源，应继续跑 Vivado implementation。
- baseline 延迟较高，主要瓶颈是 `conv2_layer`，占总延迟约 76.1%。
- 当前 baseline 还没有体现充分的 3x3 乘加阵列并行，下一版 solution2 应优先通过源码 pragma 优化卷积 kernel。

## 7. Solution2 优化计划

优化优先级：

1. `conv2_layer`：最大瓶颈，优先优化 3x3 kernel 内部并行。
2. `conv1_layer`：第二大瓶颈，同样适合 3x3 kernel unroll。
3. `fc1_layer`：延迟约 25472 cycles，可先做 factor=4 的部分展开。
4. pool 和 argmax 延迟很小，暂不优化。

第一轮优化策略：

- 在 `conv1_layer` 和 `conv2_layer` 中对 3x3 kernel 循环加 `#pragma HLS UNROLL`。
- 暂不完全展开 `conv2_layer` 的输入通道 `C1_OUT=8`，避免 DSP/布线资源突然升高。
- 在 `fc1_layer` 的输入累加循环上使用 `#pragma HLS UNROLL factor=4`。
- 优先使用源码 pragma，避免主要优化信息隐藏在 GUI directives 中。

预期：

- `conv2_layer` latency 应明显下降。
- DSP/LUT 使用会上升，但 baseline 资源余量较大：DSP 7%，LUT 14%，仍有优化空间。
- 如果 conv2 改善不明显，下一轮应考虑对 feature map 或权重数组做 partition，解决读端口瓶颈。
