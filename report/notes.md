# 课程大作业 1 笔记与完成度检查

题目：基于 FPGA 的轻量化 CNN 加速器设计及其 MNIST 手写体图像识别实现

本文档用于整理当前工程进展、报告可用数据、待办事项，以及需要队友补充的 FPGA 板端系统集成/上板测试部分。最终 Word 报告可以以本文件为基础扩写和排版。

---

## 0. 当前总体结论

目前工程已经完成了软件训练、8 位定点量化、HLS 加速器设计、C Simulation 逐层验证、solution1 baseline 综合、solution2 优化综合、solution3 进一步优化综合、solution3 C/RTL 协同仿真，以及 solution3 Vivado 后端 implementation。

已达到的核心要求：

- 轻量 CNN 结构：两层卷积 + 两层全连接。
- MNIST 浮点模型准确率：98.49%，满足 >= 98%。
- int8 量化模型准确率：98.47%，量化后准确率下降很小。
- 已导出 HLS 可读的权重、测试图像、标签和逐层 golden 数据。
- HLS 设计包含卷积、ReLU、池化、全连接、片上 buffer、AXI-Lite 控制接口和 m_axi 图像输入接口。
- Testbench 加载 10 张量化图像，逐层验证中间结果，误差容限 <= +/-1。
- solution2 相比 solution1 latency 从 354430 cycles 降到 76031 cycles，加速约 4.66 倍。
- solution3 相比 solution1 latency 从 354430 cycles 降到 30203 cycles，加速约 11.74 倍；相比 solution2 继续加速约 2.52 倍。
- solution3 RTL cosim 使用 Vivado XSIM 验证 Verilog RTL，状态为 Pass，平均 latency 为 30161 cycles。
- solution3 Vivado implementation timing met，post-implementation clock period 为 9.884 ns，WNS=0.116 ns。

还需要补充的核心内容：

- Word 报告正式排版。
- 架构图、仿真波形/日志截图、综合报告截图。
- 队友负责的 FPGA 板端系统集成、bitstream、板端测试或演示截图。

---

## 1. 课程要求 Checklist

### 1. 网络模型设计

课程要求：

- 设计轻量 CNN，例如两卷积层 + 两全连接层。
- 在 PyTorch/TensorFlow 上训练 MNIST。
- 准确率 >= 98%。
- 对权重和特征图进行 8 位定点量化。
- 导出硬件可读格式。

当前完成情况：

| 检查项 | 状态 | 当前证据 |
| --- | --- | --- |
| 两卷积层 + 两全连接层 CNN | 已完成 | `software/train_mnist.py` 中 `TinyMNISTCNN` |
| PyTorch 训练 MNIST | 已完成 | 使用 `torchvision.datasets.MNIST` |
| 浮点准确率 >= 98% | 已完成 | float32 test accuracy = 98.49% |
| 8 位权重量化 | 已完成 | `software/quantize_export.py`，导出 `hls/data/weights.h` |
| 8 位特征图量化 | 已完成 | conv/pool/fc1 激活使用 qint8，ReLU 后限制在 0~127 |
| 量化模型准确率评估 | 已完成 | int8 test accuracy = 98.47% |
| 硬件可读格式导出 | 已完成 | `weights.h`、`test_images.h`、`test_labels.h`、`golden_layers.h` |

TODO：

- 在 Word 报告中加入训练日志截图或 `train_history.json` 曲线图。
- 在 Word 报告中说明输入不做 Normalize，便于将 MNIST 像素直接量化为 8 bit。

### 2. 硬件架构

课程要求：

- 包含卷积引擎，3x3 乘加阵列。
- 包含 2x2 最大池化单元。
- 包含 ReLU 激活单元。
- 包含全连接加速模块。
- 包含控制状态机及片上缓冲器。
- 支持流水线或时分复用调度。

当前完成情况：

| 检查项 | 状态 | 当前证据 |
| --- | --- | --- |
| 3x3 卷积引擎 | 已完成 | `conv1_layer`、`conv2_layer` |
| 3x3 乘加阵列优化 | 已完成 | solution2 对 kernel 维度 `UNROLL`，对应 3x3 并行 MAC |
| 2x2 最大池化 | 已完成 | `pool1_layer`、`pool2_layer` |
| ReLU 激活 | 已完成 | `clamp_relu_requant` 中完成 ReLU + requant |
| 全连接加速模块 | 已完成 | `fc1_layer`、`fc2_layer` |
| 控制状态机 | 已完成 HLS 版本 | `cnn_accel` 顶层由 HLS 生成 `ap_ctrl_hs` 控制协议 |
| 片上缓冲器 | 已完成 | `conv1`、`pool1`、`conv2`、`pool2`、`fc1`、`logits` 均为片上数组 |
| 流水线/时分复用 | 已完成 | 多个循环使用 `#pragma HLS PIPELINE II=1`，层间采用时分复用调度 |

TODO：

- 报告中绘制顶层架构图：输入图像 -> Conv1/ReLU -> Pool1 -> Conv2/ReLU -> Pool2 -> FC1/ReLU -> FC2 -> Argmax。
- 报告中说明 solution1 为顺序/低并行 baseline，solution2 为 3x3 MAC 展开优化版，solution3 为输入片上 buffer 优化版。

### 3. RTL/HLS 实现与 Testbench

课程要求：

- 使用 Verilog/VHDL 或 HLS 编写各模块。
- 顶层接口含时钟、复位、启动、输入数据、输出标签、完成标志。
- 编写 testbench。
- 加载至少 10 张量化图像进行仿真。
- 逐层验证中间结果与软件模型误差 <= +/-1。

当前完成情况：

| 检查项 | 状态 | 当前证据 |
| --- | --- | --- |
| HLS 实现 | 已完成 | `hls/cnn_accel.cpp`、`hls/cnn_accel.h` |
| 顶层函数 | 已完成 | `cnn_accel` |
| 顶层控制接口 | 已完成 | `s_axilite port=return` 生成 start/done/idle/ready 控制 |
| 输入图像接口 | 已完成 | `m_axi port=image` |
| 输出标签接口 | 已完成 | `s_axilite port=label` |
| Testbench | 已完成 | `hls/tb_cnn.cpp` |
| 至少 10 张量化图像 | 已完成 | `NUM_TEST_IMAGES = 10` |
| 逐层 golden 对比 | 已完成 | Conv1、Pool1、Conv2、Pool2、FC1、FC2 logits 均对比 |
| 误差容限 <= +/-1 | 已完成 | `GOLDEN_TOLERANCE` |

C Simulation 结果：

| 项目 | solution1 | solution2 | solution3 |
| --- | --- | --- | --- |
| 测试图像数量 | 10 | 10 | 10 |
| Conv1 逐层对比 | 10/10 PASS | 10/10 PASS | 10/10 PASS |
| Pool1 逐层对比 | 10/10 PASS | 10/10 PASS | 10/10 PASS |
| Conv2 逐层对比 | 10/10 PASS | 10/10 PASS | 10/10 PASS |
| Pool2 逐层对比 | 10/10 PASS | 10/10 PASS | 10/10 PASS |
| FC1 逐层对比 | 10/10 PASS | 10/10 PASS | 10/10 PASS |
| FC2 logits 逐层对比 | 10/10 PASS | 10/10 PASS | 10/10 PASS |
| Top 输出 vs 量化软件预测 | 10/10 PASS | 10/10 PASS | 10/10 PASS |
| Top 输出 vs 真实标签 | 10/10 PASS | 10/10 PASS | 10/10 PASS |

TODO：

- 在 Word 报告中加入 C Simulation 日志截图。
- 如果老师要求 RTL 仿真波形，需要从 Vitis/Vivado 导出波形截图。

### 4. 性能分析

课程要求：

- 统计单张图像识别延迟，单位为周期数。
- 计算 @100MHz 等效吞吐量。
- 计算权重与特征图存储需求。
- 若 FPGA 综合，报告 LUT/FF/DSP/BRAM 资源利用率。

当前完成情况：

| 检查项 | 状态 | 当前证据 |
| --- | --- | --- |
| solution1 latency | 已完成 | 354430 cycles |
| solution2 latency | 已完成 | 76031 cycles |
| solution3 latency | 已完成 | 30203 cycles |
| solution3 RTL cosim | 已完成 | Verilog Pass，avg latency 30161 cycles，avg interval 30150 cycles |
| @100MHz 单张延迟 | 已完成 | solution1 约 3.544 ms，solution2 约 0.760 ms，solution3 约 0.302 ms |
| @100MHz 吞吐量 | 已完成 | solution1 约 282 images/s，solution2 约 1315 images/s，solution3 C Synthesis 约 3311 images/s，solution3 RTL cosim 约 3316 images/s |
| 资源利用率 | 已完成 | solution1/solution2/solution3 已有 BRAM/DSP/FF/LUT |
| Vivado implementation | 已完成 | LUT 26.60%，FF 12.53%，DSP 59.09%，BRAM 18.21%，WNS=0.116 ns |
| 权重存储需求 | 已计算 | 约 26.7 KB，如果权重 int8、bias int32 |
| 特征图存储需求 | 已计算 | 约 22.7 KB，如果主要中间特征图 int8，logits int32 |
| solution3 性能 | 已完成 | C Simulation PASS，C Synthesis latency 30203 cycles |

TODO：

- 队友补充 FPGA 板端系统集成、bitstream 或上板测试结果。
- 如果队友使用不同 FPGA 平台，需要补充器件型号、时钟频率、资源总量。

### 5. 提交成果

课程要求：

- Word 报告，包含网络结构、量化方案、架构图、仿真波形、准确率。
- 源码，包括 RTL/HLS 和训练脚本。

当前完成情况：

| 检查项 | 状态 | 当前证据 |
| --- | --- | --- |
| 训练脚本 | 已完成 | `software/train_mnist.py` |
| 量化导出脚本 | 已完成 | `software/quantize_export.py` |
| 量化评估脚本/逻辑 | 已完成 | `quantize_export.py` 内 float/int8 accuracy |
| HLS 源码 | 已完成 | `hls/cnn_accel.cpp`、`hls/cnn_accel.h` |
| HLS Testbench | 已完成 | `hls/tb_cnn.cpp` |
| HLS 数据头文件 | 已完成 | `hls/data/*.h` |
| README | 已完成初稿 | `README.md` |
| Word 报告 | 未完成 | 需要根据本 notes 整理 |
| 架构图 | 未完成 | 需要绘图 |
| 仿真波形/截图 | 未完成 | 需要从 Vitis/Vivado 截图 |
| 队友上板部分 | 未完成 | 预留章节 |

---

## 2. 网络与训练结果

### 2.1 网络结构

本项目采用轻量化 CNN，用于 MNIST 28x28 灰度手写数字识别。网络结构如下：

| 层次 | 参数 | 输出尺寸 |
| --- | --- | --- |
| Input | MNIST 灰度图像 | 1 x 28 x 28 |
| Conv1 + ReLU | 3x3, 1 -> 8, padding=1 | 8 x 28 x 28 |
| MaxPool1 | 2x2, stride=2 | 8 x 14 x 14 |
| Conv2 + ReLU | 3x3, 8 -> 16, padding=1 | 16 x 14 x 14 |
| MaxPool2 | 2x2, stride=2 | 16 x 7 x 7 |
| Flatten | - | 784 |
| FC1 + ReLU | 784 -> 32 | 32 |
| FC2 | 32 -> 10 | 10 |
| Argmax | 10 类分类 | label |

模型参数量为 26698。该结构规模较小，适合 HLS 实现和 FPGA 资源约束，同时仍能在 MNIST 数据集上达到 98% 以上准确率。

### 2.2 训练设置

训练脚本：`software/train_mnist.py`

训练配置：

| 项目 | 设置 |
| --- | --- |
| 框架 | PyTorch |
| 数据集 | MNIST |
| 输入预处理 | `transforms.ToTensor()`，不做 Normalize |
| epoch | 8 |
| batch size | 128 |
| optimizer | Adam |
| learning rate | 0.001 |
| checkpoint | `software/checkpoints/mnist_cnn_fp32.pt` |
| history | `software/checkpoints/train_history.json` |

不对输入做 Normalize 的原因是：硬件端更容易处理原始像素范围。MNIST 像素由 `[0,1]` 直接量化到 8 bit，便于在 HLS 中使用整数计算。

### 2.3 准确率结果

| 模型 | 测试准确率 |
| --- | ---: |
| float32 PyTorch 模型 | 98.49% |
| int8 量化模型 | 98.47% |

结论：浮点模型满足课程要求的准确率 >= 98%。int8 量化后准确率仅下降约 0.02 个百分点，说明当前量化方案对模型精度影响较小，适合用于硬件部署。

---

## 3. 8 位定点量化方案

### 3.1 量化策略

本项目采用 per-tensor symmetric int8 量化。权重量化为 `qint8_t`，bias 量化为 `qint32_t`，中间激活在 ReLU 后限制到 0 到 127。

量化基本公式：

```text
q = round(x / scale)
x_approx = q * scale
```

卷积和全连接计算时，乘加累加器使用 int32，随后通过整数乘法和右移完成 requantization：

```text
acc_int32 -> ReLU -> multiplier -> right shift -> qint8
```

这种方式与 HLS 中的整数硬件实现一致，避免 Python golden 和 HLS 之间由于浮点近似造成额外误差。

### 3.2 量化参数

量化参数来自 `hls/data/quant_summary.json`。

| 层 | 权重 scale | 激活 scale | 定点乘子 | 右移 |
| --- | ---: | ---: | ---: | ---: |
| Conv1 | 0.0062853919 | 0.0278832837 | 29779 | 24 |
| Conv2 | 0.0043370268 | 0.1090867651 | 18599 | 24 |
| FC1 | 0.0045013517 | 0.3405605526 | 24190 | 24 |
| FC2 | 0.0044079432 | 输出 int32 logits | - | - |

### 3.3 存储需求

权重数量：

| 层 | 权重数量 | bias 数量 |
| --- | ---: | ---: |
| Conv1 | 8 x 1 x 3 x 3 = 72 | 8 |
| Conv2 | 16 x 8 x 3 x 3 = 1152 | 16 |
| FC1 | 32 x 784 = 25088 | 32 |
| FC2 | 10 x 32 = 320 | 10 |
| 合计 | 26632 | 66 |

若权重使用 int8、bias 使用 int32，则权重和 bias 总存储约为：

```text
26632 bytes + 66 * 4 bytes = 26896 bytes，约 26.3 KB
```

主要特征图存储：

| Buffer | 尺寸 | int8 存储 |
| --- | ---: | ---: |
| input image | 1 x 28 x 28 | 784 B |
| conv1 | 8 x 28 x 28 | 6272 B |
| pool1 | 8 x 14 x 14 | 1568 B |
| conv2 | 16 x 14 x 14 | 3136 B |
| pool2 | 16 x 7 x 7 | 784 B |
| fc1 | 32 | 32 B |
| logits | 10 x int32 | 40 B |

若中间结果全部保留，主要特征图 buffer 合计约 12.6 KB。若再考虑 HLS 数组分块和 BRAM 映射，实际 BRAM 使用会高于纯数据字节数。

---

## 4. HLS 架构设计

### 4.1 顶层架构

HLS 顶层函数为：

```cpp
void cnn_accel(const qint8_t image[MNIST_PIXELS], ap_uint<4> &label);
```

顶层接口：

| 接口 | 类型 | 说明 |
| --- | --- | --- |
| `image` | `m_axi` | 输入 784 个 int8 像素 |
| `label` | `s_axilite` | 输出 0~9 预测标签 |
| `return` | `s_axilite` | HLS 控制接口，生成 start/done/idle/ready |

HLS 模块划分：

| 模块 | 功能 |
| --- | --- |
| `conv1_layer` | 第一层 3x3 卷积 + ReLU + requant |
| `pool1_layer` | 第一层 2x2 最大池化 |
| `conv2_layer` | 第二层 3x3 卷积 + ReLU + requant |
| `pool2_layer` | 第二层 2x2 最大池化 |
| `fc1_layer` | 第一层全连接 + ReLU + requant |
| `fc2_layer` | 第二层全连接，输出 int32 logits |
| `cnn_accel` | 顶层调度和 argmax 分类 |

### 4.2 工作流程

```text
量化图像输入
  -> Conv1 3x3 + ReLU + requant
  -> Pool1 2x2 max
  -> Conv2 3x3 + ReLU + requant
  -> Pool2 2x2 max
  -> FC1 + ReLU + requant
  -> FC2 logits
  -> Argmax
  -> 输出预测标签
```

当前设计采用层间时分复用调度。每一层完成后将中间结果写入片上 buffer，下一层再从 buffer 中读取。优点是结构清晰、便于逐层验证；不足是 `Interval` 接近 `Latency`，说明当前不是多图并行流水架构。

---

## 5. 关键电路设计

### 5.1 卷积计算核

卷积核大小为 3x3。baseline 中卷积乘加以较低并行度执行，solution2 中对 `ky/kx` 两个 kernel 维度进行 `UNROLL`，将 3x3 乘加改为并行 MAC 阵列。

solution2 的核心优化：

```cpp
#pragma HLS ARRAY_PARTITION variable=W_CONV2 complete dim=3
#pragma HLS ARRAY_PARTITION variable=W_CONV2 complete dim=4
#pragma HLS UNROLL
```

该优化显著降低 `conv2_layer` 延迟，是 solution2 加速的主要来源。

### 5.2 池化单元

池化单元实现 2x2 最大池化，每次读取四个输入值并比较得到最大值。由于池化计算量较小，在总 latency 中占比不高，因此目前没有作为主要优化目标。

### 5.3 ReLU 与 requant 单元

ReLU 与 requant 合并在 `clamp_relu_requant` 中：

```text
if acc <= 0: output = 0
else: output = clamp((acc * multiplier + rounding) >> shift, 0, 127)
```

这样可以在 int32 累加后直接输出 int8 激活，减少中间浮点计算和额外数据格式转换。

### 5.4 全连接模块

FC1 输入维度为 784，输出维度为 32，是除卷积外较明显的计算热点。solution2 中对 FC1 输入循环做 factor=4 部分展开，使 FC1 latency 从 25472 cycles 降到 4129 cycles。

solution3 中暂时将 FC1 展开从 factor=4 调整为 factor=2，用于观察资源下降与 latency 回退之间的权衡。

### 5.5 片上缓冲器

当前片上 buffer 包括：

- `conv1[C1_OUT][IMG_H][IMG_W]`
- `pool1[C1_OUT][P1_H][P1_W]`
- `conv2[C2_OUT][P1_H][P1_W]`
- `pool2[C2_OUT][P2_H][P2_W]`
- `fc1[FC1_OUT]`
- `logits[NUM_CLASSES]`

solution3 额外在 `conv1_layer` 内加入 `image_buf[IMG_H][IMG_W]`，用于将 AXI 输入图像搬到片上 buffer 后再进行 3x3 卷积读取，目标是减少 conv1 受外部接口读访问限制的影响。

### 5.6 通信接口与控制

当前 HLS 顶层使用 AXI 接口：

- `m_axi` 用于读取输入图像。
- `s_axilite` 用于控制寄存器和输出标签。

HLS 会根据 `s_axilite port=return` 自动生成控制状态机，包含 start、done、idle、ready 等信号，满足课程对顶层启动和完成标志的要求。

---

## 6. 实验结果与讨论

### 6.1 C Simulation 结果

当前 testbench 使用 10 张量化 MNIST 测试图像。每张图像依次执行各层函数，并与软件量化模型导出的 golden 数据对比。

逐层验证内容：

- Conv1 vs `GOLDEN_CONV1`
- Pool1 vs `GOLDEN_POOL1`
- Conv2 vs `GOLDEN_CONV2`
- Pool2 vs `GOLDEN_POOL2`
- FC1 vs `GOLDEN_FC1`
- FC2 logits vs `GOLDEN_FC2_LOGITS`
- Top label vs `GOLDEN_PRED_LABELS`
- Top label vs `TEST_LABELS`

solution1、solution2 和 solution3 均为 10/10 PASS，说明 HLS 计算结果与软件 int8 量化模型一致。solution3 的日志位于 `report/hls_baseline/solution3/csim/report/cnn_accel_csim.log`。

TODO：

- Word 报告中加入仿真日志截图。

### 6.2 C/RTL 协同仿真结果

solution3 进一步进行了 C/RTL cosimulation，仿真工具为 Vivado XSIM，验证对象为 HLS 生成的 Verilog RTL。报告位于 `report/hls_baseline/solution3/sim/report/cnn_accel_cosim.rpt`。

| RTL | Status | Latency min | Latency avg | Latency max | Interval min | Interval avg | Interval max | Total execution |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Verilog | Pass | 30158 | 30161 | 30162 | 30147 | 30150 | 30151 | 301517 |

在 100 MHz 时钟下，平均单张图像推理延迟约为：

```text
30161 cycles / 100 MHz = 0.30161 ms
```

对应等效吞吐量约为：

```text
100 MHz / 30161 cycles = 3315.54 images/s
```

结论：C/RTL cosimulation 结果为 Pass，说明 HLS 生成的 Verilog RTL 与 C 模型功能一致。RTL cosim 的平均 latency 30161 cycles 与 HLS C Synthesis 估计的 30203 cycles 非常接近，说明 HLS latency 估计可信。

### 6.3 HLS 综合与 Vivado Implementation 对比

solution1 为 baseline，solution2 为 3x3 kernel 展开优化版，solution3 为输入片上 buffer + FC1 资源权衡版。前三行资源来自 HLS C Synthesis；最后一行来自 Vivado 后端 implementation，因此最后一行更接近真实 FPGA 布局布线后的资源与时序。

| Version | Main optimization | Latency/cycles | Latency/ms @100MHz | Throughput/images/s | Timing | LUT | FF | DSP | BRAM |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| solution1 | baseline | 354430 | 3.544 | 282 | met | 14% | 4% | 7% | 9% |
| solution2 | 3x3 kernel unroll / partition | 76031 | 0.760 | 1315 | met | 41.4% | 18.5% | 52.7% | 30% |
| solution3 | final optimized HLS | 30203 | 0.302 | 3311 | met | 61.4% | 24.1% | 52.7% | 17.9% |
| solution3 implementation | Vivado backend implementation + RTL cosim | 30161 RTL cosim avg | 0.302 | 3316 | met, WNS=0.116 ns | 26.6% | 12.5% | 59.1% | 18.2% |

注：solution1、solution2、solution3 三行的 latency 和资源来自 HLS C Synthesis 报告；`solution3 implementation` 行的 latency 来自 C/RTL cosimulation 平均值，资源和 WNS 来自 Vivado 后端 implementation。两类报告的资源估计方法不同，因此 implementation 行资源数值与 HLS C Synthesis 行不完全一致。

Vivado implementation 关键结果：

| 项目 | 结果 |
| --- | ---: |
| LUT | 14151 / 53200，26.60% |
| FF | 13335 / 106400，12.53% |
| DSP | 130 / 220，59.09% |
| BRAM | 51 / 280，18.21% |
| Post-implementation clock period | 9.884 ns |
| WNS | 0.116 ns |
| Timing | met |

资源利用率：

| 资源 | solution1 | solution2 | solution3 |
| --- | ---: | ---: | ---: |
| BRAM_18K | 26 / 280，9% | 84 / 280，30% | 50 / 280，17% |
| DSP | 16 / 220，7% | 116 / 220，52% | 116 / 220，52% |
| FF | 5274 / 106400，4% | 19695 / 106400，18% | 25626 / 106400，24% |
| LUT | 7505 / 53200，14% | 22045 / 53200，41% | 32671 / 53200，61% |

主要模块 latency：

| 模块 | solution1 cycles | solution2 cycles | solution3 cycles | 分析 |
| --- | ---: | ---: | ---: | --- |
| Conv1 | 56467 | 56467 | 7086 | s3 通过片上 `image_buf` 改善输入并行读取 |
| Conv2 | 269696 | 12632 | 12632 | s2 的 3x3 kernel 展开带来主要加速 |
| FC1 | 25472 | 4129 | 7681 | s2 factor=4 最快；s3 factor=2 降低 BRAM 但延迟回升 |
| Pool1 | 1573 | 1577 | 1577 | 基本不变 |
| Pool2 | 789 | 789 | 789 | 基本不变 |
| FC2 | 410 | 410 | 410 | 基本不变 |
| Argmax | 12 | 12 | 12 | 占比很小 |

结论：solution2 通过对卷积核计算循环展开，将 3x3 MAC 从顺序计算改为并行乘加阵列，显著降低了 Conv2 延迟，并使整体 latency 从 354430 cycles 降到 76031 cycles。solution3 进一步将输入图像搬入完全分块的片上 `image_buf`，解决 Conv1 并行读端口瓶颈，使整体 latency 降到 30203 cycles。Vivado 后端 implementation 显示 timing met，WNS 为 0.116 ns，说明当前最终 HLS 版本不仅在 HLS 估计阶段可行，在后端布局布线后也满足 100 MHz 时序。

### 6.4 三轮优化路径总结

三轮优化过程如下：

| 版本 | 主要改动 | 结果 |
| --- | --- | --- |
| solution1 | baseline，优先保证功能正确和逐层验证 | latency 354430 cycles，资源低，但 Conv2 占 76.1% |
| solution2 | 展开 Conv1/Conv2 的 3x3 kernel 维度，FC1 factor=4 | Conv2 从 269696 降到 12632 cycles，整体加速 4.66x |
| solution3 | Conv1 增加 `image_buf[28][28]` 且 complete partition，FC1 factor=2 | Conv1 从 56467 降到 7086 cycles，整体加速到 11.74x |

需要注意的是，solution3 的 `Interval` 仍接近 `Latency`，说明当前架构仍是一张图处理完再处理下一张图，还没有形成多图层间流水。Vivado backend implementation 已经通过时序检查，WNS=0.116 ns；后续队友进行板端系统集成和上板测试时，需要重点关注数据搬运、bitstream 下载、板端输入输出验证以及 DSP 使用率较高带来的平台余量问题。

TODO：

- Word 报告中补架构图、C Simulation 截图和综合报告截图。
- 队友补板端系统集成、bitstream 下载、板端测试截图和板端输入输出验证结果。
- 最终报告中明确区分 HLS C Synthesis 结果和 Vivado implementation 结果。

---

## 7. Word 报告建议结构

### 7.1 引言

已完成素材：

- 题目背景：CNN 在图像识别中计算量大，FPGA 适合进行并行乘加和定点加速。
- 实验目标：设计轻量 CNN，训练 MNIST，量化为 int8，并使用 HLS 实现 FPGA 加速器。
- 当前方案优势：模型小、准确率达标、量化精度损失小、HLS 逐层验证完整，solution3 相比 baseline 加速约 11.74x。
- 当前方案不足：层间尚未做多图流水，Interval 接近 Latency；solution3 implementation 的 DSP 使用率约 59.09%，继续大规模并行化的余量有限；板端系统集成和上板测试结果待队友补充。

TODO：

- 加入课程要求原文的简要概括。
- 加入队友负责的板卡/FPGA 平台介绍。

### 7.2 文献综述

TODO：

- 补近 5 年 FPGA CNN 加速器、低比特量化、HLS 深度学习部署相关研究。
- 可按地区组织：美国/欧洲/日韩/中国。
- 注意课程报告中可以写综述性文字，不需要过度堆论文。

可写方向：

- 美国：Xilinx/AMD Vitis AI、FINN、DNN accelerator、低比特量化部署。
- 欧洲：FINN、hls4ml、边缘 AI 加速。
- 日韩：低功耗边缘 AI、FPGA/ASIC CNN 加速。
- 中国：面向边缘智能的 FPGA CNN 加速、国产 FPGA 和嵌入式 AI 部署。

### 7.3 算法设计

已完成素材：

- 网络结构表。
- 训练配置。
- float32 和 int8 准确率。
- 量化公式。
- golden 数据导出流程。

TODO：

- 加训练曲线图。
- 加量化前后准确率对比图。

### 7.4 架构设计

已完成素材：

- HLS 顶层接口。
- 模块划分。
- 数据流说明。
- solution1/solution2/solution3 设计区别。

TODO：

- 画架构图。
- 画数据流图或状态流程图。
- 队友补 FPGA 平台整体系统图，例如 PS/PL 或外设接口。

### 7.5 关键电路设计

已完成素材：

- 3x3 卷积 MAC 阵列。
- 2x2 max pooling。
- ReLU + requant。
- FC1/FC2。
- 片上 buffer。
- AXI 接口和 HLS 控制。

TODO：

- 加关键代码截图。
- 如果有 RTL schematic 或 Vivado block design，由队友补充。

### 7.6 实验结果及讨论

已完成素材：

- C Simulation 逐层验证。
- solution1/solution2/solution3 latency 和资源对比。
- solution2 speedup = 4.66x，solution3 speedup = 11.74x。
- solution3 RTL cosim Verilog Pass，平均 latency = 30161 cycles。
- Vivado implementation timing met，WNS = 0.116 ns。

TODO：

- 补仿真日志截图。
- 补综合报告截图。
- 队友补上板测试、板端运行截图或串口输出。
- 与已有研究对比时，需要说明本项目是课程级轻量模型，重点在完整流程和 HLS 优化，不直接追求大型模型最高性能。

### 7.7 总结与展望

已完成素材：

- 完成了从 PyTorch 训练、int8 量化、HLS 实现、逐层验证到综合优化的流程。
- solution2 通过 3x3 MAC 展开显著降低延迟。
- 当前不足是多图流水不足、DSP 使用较高、板端系统集成和上板测试待补。

TODO：

- 根据队友上板结果补最终结论。
- 展望可以写：更细粒度流水、双缓冲、输入通道并行度自动搜索、权重量化 per-channel、稀疏化、部署到更大数据集。

---

## 8. 队友预留部分

以下内容主要由负责 FPGA 板端系统集成和上板验证的队友补充。Vivado 后端 implementation 已经有一版结果，可直接写入报告；板卡信息和实际板端测试仍需补充。

### 8.1 FPGA 平台信息

TODO：

- FPGA/开发板型号：
- FPGA 器件：
- 时钟频率：
- 工具版本：
- 是否使用 Zynq PS/PL：
- 数据输入输出方式：

### 8.2 Vivado Implementation 结果

| 指标 | 结果 |
| --- | ---: |
| LUT | 14151 / 53200，26.60% |
| FF | 13335 / 106400，12.53% |
| DSP | 130 / 220，59.09% |
| BRAM | 51 / 280，18.21% |
| Post-implementation clock period | 9.884 ns |
| WNS | 0.116 ns |
| Timing | met |

TODO：

- 如果队友在完整 Vivado 工程中重新集成 IP，需要确认完整系统 implementation 资源和 WNS 是否与 HLS export implementation 一致。
- 若完整系统包含 PS、AXI interconnect、DMA 或 BRAM 控制器，应另列完整系统资源。

### 8.3 上板测试结果

TODO：

- bitstream 是否生成成功：
- 板端是否能启动：
- 输入测试图像数量：
- 板端预测结果：
- 板端单张图像延迟：
- 板端吞吐量：
- 与 HLS 估计是否一致：

### 8.4 队友报告撰写空间

TODO：

在 Word 报告中建议单独加入一节：

```text
6.x FPGA 实现与上板验证
```

该节由队友补充，建议包括：

- Vivado block design 或系统框图。
- HLS IP 集成方式。
- AXI 接口连接方式。
- 时钟和复位设计。
- bitstream 生成结果。
- 上板测试流程。
- 板端测试截图。
- 与 HLS C Synthesis、RTL cosim、Vivado implementation 结果的差异分析。

---

## 9. 最终提交前 Checklist

- [x] 训练脚本完成。
- [x] 量化导出脚本完成。
- [x] HLS 源码完成。
- [x] Testbench 完成。
- [x] 10 张量化图像逐层验证完成。
- [x] solution1 baseline 综合完成。
- [x] solution2 优化综合完成。
- [x] solution2 已 push 到 GitHub。
- [x] solution2 源码已另存为 `cnn_accel_solution2.cpp/.h`。
- [x] solution3 C Simulation。
- [x] solution3 C Synthesis。
- [x] solution3 C/RTL cosimulation，Verilog Pass。
- [x] solution3 Vivado backend implementation，timing met，WNS=0.116 ns。
- [x] 确定当前 HLS synthesis 最优版本为 solution3。
- [ ] Word 报告排版。
- [ ] 架构图。
- [ ] 仿真截图。
- [ ] 综合报告截图。
- [ ] 队友补 FPGA 板端系统集成/上板结果。
- [ ] 最终 GitHub 代码检查。
- [ ] 最终提交压缩包或仓库链接。
