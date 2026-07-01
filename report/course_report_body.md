# 基于 FPGA 的轻量化 CNN 加速器设计及其 MNIST 手写体图像识别实现

## 1. 引言

卷积神经网络在图像分类任务中具有较高准确率，但卷积层和全连接层包含大量乘加运算。若直接在通用 CPU 上执行，计算并行度有限；若使用 GPU，吞吐能力较强，但功耗、成本和系统复杂度不一定适合嵌入式场景。FPGA 具有可重构、并行度可定制、片上存储结构灵活和定点运算效率高等特点，适合实现面向特定网络的小型 CNN 加速器。

本课程大作业的目标是完成一个“基于 FPGA 的轻量化 CNN 加速器设计及其 MNIST 手写体图像识别实现”。按课程要求，设计需要完成以下工作：在 PyTorch/TensorFlow 中训练轻量 CNN，准确率达到 98% 以上；将权重和特征图量化为 8 位定点格式，并导出硬件可读数据；使用 HLS 或 RTL 实现卷积、池化、ReLU、全连接、控制状态机和片上缓冲器；编写 testbench，加载至少 10 张量化图像进行逐层验证；最后给出延迟、吞吐量、存储需求和 FPGA 资源利用率。

本设计采用自定义 HLS 加速器方案，而不是直接调用通用 DPU。这样可以更清楚地展示 CNN 各层如何映射为硬件计算模块，也便于对卷积瓶颈进行针对性优化。整个流程分为三步：首先在 PyTorch 中训练两层卷积加两层全连接的轻量 CNN；然后用 int8 权重、int8 激活和 int32 累加器完成量化导出；最后用 Vitis HLS 编写可综合 C++，并通过三轮 solution 逐步优化性能。

本项目的三轮 HLS 版本如下：

| 版本 | 目的 | 主要方法 |
| --- | --- | --- |
| solution1 | 功能正确的 baseline | 保持较低并行度，优先完成逐层验证和综合 |
| solution2 | 优化最大瓶颈 Conv2 | 对 3x3 kernel 维度展开，形成 3x3 乘加阵列 |
| solution3 | 继续优化 Conv1 并保留资源分析 | 输入图像搬入片上 buffer，改善 Conv1 读端口限制；FC1 展开因子从 4 调整为 2 |

从结果看，solution1 的单张图像延迟为 354430 cycles；solution2 降到 76031 cycles；solution3 进一步降到 30203 cycles。三轮优化清楚体现了“先保证正确，再定位瓶颈，再通过 HLS pragma 改善并行度和片上数据访问”的设计过程。

## 2. 文献综述

近 5 年，FPGA 上的 CNN 加速研究主要集中在低比特量化、HLS 自动生成、片上数据复用、流水线调度和 DSP 利用率优化等方向。

美国及国际合作团队在 HLS 机器学习部署方面有较多代表性工作。hls4ml 将神经网络模型转换为 FPGA HLS 设计，强调低延迟、低功耗和可配置定点推理。其面向卷积网络的工作展示了剪枝、量化感知训练和 HLS 映射可以显著降低资源消耗，并实现微秒级推理延迟。2022 年之后，hls4ml 也扩展到 RNN、Transformer 等结构，说明 HLS 已经成为 FPGA 机器学习部署的重要方法之一。

欧洲相关研究常关注边缘场景中的混合精度、低功耗和可靠部署。混合精度 CNN 加速器通过为不同层或不同运算选择不同位宽，在准确率、延迟和资源之间进行折中。航天、工业检测等边缘应用也常采用 8 位量化、通道剪枝和片上缓存复用，以减少外部存储访问。

日韩和中国相关研究中，常见方向包括固定点 CNN 流水线、systolic array、DSP 细粒度映射和 Vitis AI 部署。很多工作表明，CNN 加速器性能不只取决于乘法器数量，还受片上 buffer 读写端口、数组分块、循环依赖和布线压力影响。该结论与本项目实验结果一致：solution2 展开 3x3 kernel 后 Conv2 大幅加速，但 solution2 中 Conv1 仍未改善，说明仅增加 MAC 并行度还不够，还需要处理输入数据读端口和缓冲方式。

总体来看，当前 FPGA CNN 加速的主流方法是：算法侧采用轻量网络和低比特量化，硬件侧采用片上缓存、循环展开、数组 partition、pipeline 和 dataflow 等技术。本项目规模较小，但完整覆盖了训练、量化、HLS 实现、逐层验证和综合优化流程，能够作为课程实验中的基础 CNN 加速器设计。

## 3. 算法设计

### 3.1 模型构建

本项目采用适合 MNIST 的轻量 CNN。输入为 1x28x28 灰度图像，网络包含两层卷积、两层池化和两层全连接。

| 层 | 参数 | 输出尺寸 |
| --- | --- | --- |
| Input | MNIST 灰度图像 | 1 x 28 x 28 |
| Conv1 + ReLU | 3x3, 1 -> 8, padding=1 | 8 x 28 x 28 |
| MaxPool1 | 2x2, stride=2 | 8 x 14 x 14 |
| Conv2 + ReLU | 3x3, 8 -> 16, padding=1 | 16 x 14 x 14 |
| MaxPool2 | 2x2, stride=2 | 16 x 7 x 7 |
| Flatten | - | 784 |
| FC1 + ReLU | 784 -> 32 | 32 |
| FC2 | 32 -> 10 | 10 |

模型参数量为 26698。该结构比大型 CNN 更容易部署到 FPGA，但仍保留卷积、池化和全连接等典型 CNN 结构，适合作为课程加速器设计对象。

### 3.2 参数训练

训练脚本为 `software/train_mnist.py`，使用 PyTorch 和 torchvision 读取 MNIST 数据集。输入预处理只使用 `transforms.ToTensor()`，不使用 Normalize。这样做是为了让输入像素保持在 `[0,1]` 范围，后续可以直接量化为 8 位整数，简化 HLS 端输入处理。

训练配置如下：

| 项目 | 设置 |
| --- | --- |
| 框架 | PyTorch |
| epoch | 8 |
| batch size | 128 |
| optimizer | Adam |
| learning rate | 0.001 |
| checkpoint | `software/checkpoints/mnist_cnn_fp32.pt` |
| history | `software/checkpoints/train_history.json` |

训练脚本保存 best checkpoint，并记录每轮 loss、train accuracy 和 test accuracy。最终浮点模型测试准确率为 98.49%，满足课程要求。

### 3.3 8 位定点量化

量化导出脚本为 `software/quantize_export.py`。本设计采用 per-tensor symmetric int8 量化：

- 输入图像按 `INPUT_SCALE = 1/127` 量化为 int8。
- 权重使用 int8。
- bias 使用 int32，scale 为输入 scale 与权重 scale 的乘积。
- 卷积和全连接的累加器使用 int32。
- Conv1、Conv2 和 FC1 输出经过 ReLU 后重新量化为 int8，范围限制为 0 到 127。
- FC2 输出保留为 int32 logits，用于最终 argmax。

requantization 使用整数乘法和右移实现：

```text
scaled = (acc_int32 * multiplier + rounding) >> shift
output = clamp(scaled, 0, 127)
```

该逻辑与 HLS 中 `clamp_relu_requant` 函数一致，因此软件 golden 与硬件 C Simulation 之间能够保持逐层一致。

量化后模型测试准确率为 98.47%，仅比 float32 低约 0.02 个百分点。说明 8 位定点方案在本 MNIST 模型上精度损失很小。

### 3.4 硬件数据导出

量化脚本导出以下文件：

| 文件 | 作用 |
| --- | --- |
| `hls/data/weights.h` | int8 权重、int32 bias、量化参数 |
| `hls/data/test_images.h` | 10 张量化测试图像 |
| `hls/data/test_labels.h` | 真实标签和量化模型预测标签 |
| `hls/data/golden_layers.h` | 软件量化模型逐层输出 |
| `hls/data/quant_summary.json` | float/int8 准确率与量化参数摘要 |

`golden_layers.h` 是逐层验证的关键。testbench 不只检查最终分类标签，还检查 Conv1、Pool1、Conv2、Pool2、FC1 和 FC2 logits 的中间结果，从而满足课程要求中“逐层验证中间结果与软件模型误差 <= +/-1”的条件。

## 4. 架构设计

### 4.1 顶层架构

HLS 顶层函数为：

```cpp
void cnn_accel(const qint8_t image[MNIST_PIXELS], ap_uint<4> &label);
```

顶层接口如下：

| 接口 | HLS 类型 | 作用 |
| --- | --- | --- |
| `image` | `m_axi` | 读取 784 个 int8 输入像素 |
| `label` | `s_axilite` | 输出 0 到 9 的预测标签 |
| `return` | `s_axilite` | 生成 start/done/idle/ready 等控制信号 |

HLS 会基于 `s_axilite port=return` 生成控制状态机。综合后的 RTL 顶层具备时钟、复位、启动、完成标志和输出标签等信号，满足课程对顶层接口的要求。

### 4.2 模块划分

代码将 CNN 拆成独立 layer 函数：

| 模块 | 功能 |
| --- | --- |
| `conv1_layer` | Conv1 + ReLU + requant |
| `pool1_layer` | 2x2 最大池化 |
| `conv2_layer` | Conv2 + ReLU + requant |
| `pool2_layer` | 2x2 最大池化 |
| `fc1_layer` | FC1 + ReLU + requant |
| `fc2_layer` | FC2，输出 int32 logits |
| `cnn_accel` | 顶层调度、片上 buffer 管理和 argmax |

数据流如下：

```text
image
  -> Conv1/ReLU/Requant
  -> Pool1
  -> Conv2/ReLU/Requant
  -> Pool2
  -> FC1/ReLU/Requant
  -> FC2 logits
  -> Argmax
  -> label
```

这种写法的好处是功能验证和综合优化都比较清楚。testbench 可以单独调用各层做 golden 对比，而综合时仍以 `cnn_accel` 作为 top function。

### 4.3 平台及语言

软件训练使用 Python、PyTorch 和 conda 环境 `ML`。硬件设计使用 Vitis HLS 2022.2，目标器件为 `xc7z020-clg400-1`，目标时钟周期为 10 ns，即 100 MHz。HLS C++ 综合后生成 Verilog RTL。

FPGA 实际上板综合、Vivado implementation 和板端测试由队友负责。本文当前报告的资源和时序数据来自 Vitis HLS C Synthesis，后续 Word 报告中应由队友补充 implementation 后的 LUT/FF/DSP/BRAM、WNS/TNS、bitstream 和板端测试结果。

## 5. 关键电路设计

### 5.1 卷积计算核

卷积层是本设计的主要计算热点。Conv1 计算量为 8 个输出通道、28x28 输出空间和 3x3 kernel；Conv2 计算量为 16 个输出通道、14x14 输出空间、8 个输入通道和 3x3 kernel。因此 Conv2 在 baseline 中成为最大瓶颈。

solution1 中卷积以普通嵌套循环为主，主要目标是功能正确。综合结果显示，Conv2 延迟为 269696 cycles，占总延迟约 76.1%。

solution2 针对 Conv2 做第一轮优化：对 3x3 kernel 的 `ky` 和 `kx` 循环展开，并对权重 kernel 维度 complete partition。这样每个输入通道内部的 3x3 乘加可以并行执行，对应课程要求中的“3x3 乘加阵列”。优化后 Conv2 延迟从 269696 cycles 降到 12632 cycles。

solution3 发现 solution2 中 Conv1 仍为 56467 cycles，说明 Conv1 的 3x3 展开没有真正转化为有效吞吐。原因是 Conv1 直接从 `m_axi image` 输入数组读取像素，多个并行乘法对输入读端口形成压力。solution3 在 `conv1_layer` 中新增 `image_buf[28][28]`，先将输入图像搬入片上 buffer，并对两个维度 complete partition，再从 `image_buf` 中读取 3x3 邻域。优化后 Conv1 延迟从 56467 cycles 降到 7086 cycles。

### 5.2 ReLU 与定点重量化电路

ReLU 与 requant 合并在 `clamp_relu_requant` 中完成。该函数先将小于等于 0 的累加值置 0，再执行整数乘法、rounding 和右移，最后裁剪到 0 到 127。这样每层输出都保持为 int8 激活，减少片上存储和后续计算开销。

### 5.3 池化单元

池化单元实现 2x2 最大池化，每次读取 4 个 int8 激活值并输出最大值。Pool1 和 Pool2 的延迟分别约为 1577 cycles 和 789 cycles，在三轮设计中基本不变，占总延迟比例较低，因此不是主要优化对象。

### 5.4 全连接模块

FC1 输入维度为 16x7x7=784，输出维度为 32。solution1 中 FC1 延迟为 25472 cycles。solution2 对 FC1 的输入累加循环做 factor=4 部分展开，延迟降到 4129 cycles，但 BRAM 使用上升明显。solution3 将 FC1 展开因子调整为 factor=2，FC1 延迟回升到 7681 cycles，但 BRAM 从 solution2 的 84 个 BRAM_18K 降到 50 个。该结果体现了性能和资源之间的权衡。

FC2 输入为 32，输出为 10，延迟约 410 cycles；argmax 延迟约 12 cycles，二者对总延迟影响很小。

### 5.5 片上缓冲器

顶层使用静态数组保存中间特征图：

- `conv1[C1_OUT][IMG_H][IMG_W]`
- `pool1[C1_OUT][P1_H][P1_W]`
- `conv2[C2_OUT][P1_H][P1_W]`
- `pool2[C2_OUT][P2_H][P2_W]`
- `fc1[FC1_OUT]`
- `logits[NUM_CLASSES]`

solution3 额外加入 `image_buf[IMG_H][IMG_W]`，用于改善 Conv1 的并行读访问。片上 buffer 减少了对外部存储的重复访问，但数组 partition 会提高 LUT、FF 或 BRAM 使用率，需要结合综合结果选择合适并行度。

### 5.6 测试电路

测试电路为 `hls/tb_cnn.cpp`。testbench 对每张图像依次执行各层，并与软件 golden 数据对比。最终又调用顶层 `cnn_accel`，检查输出是否等于量化模型预测标签和真实标签。

逐层验证范围包括：

- Conv1 vs `GOLDEN_CONV1`
- Pool1 vs `GOLDEN_POOL1`
- Conv2 vs `GOLDEN_CONV2`
- Pool2 vs `GOLDEN_POOL2`
- FC1 vs `GOLDEN_FC1`
- FC2 logits vs `GOLDEN_FC2_LOGITS`
- Top label vs `GOLDEN_PRED_LABELS`
- Top label vs `TEST_LABELS`

三轮 solution 的 C Simulation 均为 10/10 PASS。

## 6. 实验结果及讨论

### 6.1 训练与量化结果

| 项目 | 结果 |
| --- | ---: |
| 模型参数量 | 26698 |
| float32 测试准确率 | 98.49% |
| int8 量化测试准确率 | 98.47% |
| 量化精度下降 | 约 0.02 个百分点 |

该结果说明本设计在算法准确率上满足课程要求，并且 int8 定点量化没有造成明显精度损失。

### 6.2 C Simulation 结果

三轮 solution 均使用 10 张量化 MNIST 测试图像进行 C Simulation。结果如下：

| 验证项目 | solution1 | solution2 | solution3 |
| --- | --- | --- | --- |
| Conv1 | 10/10 PASS | 10/10 PASS | 10/10 PASS |
| Pool1 | 10/10 PASS | 10/10 PASS | 10/10 PASS |
| Conv2 | 10/10 PASS | 10/10 PASS | 10/10 PASS |
| Pool2 | 10/10 PASS | 10/10 PASS | 10/10 PASS |
| FC1 | 10/10 PASS | 10/10 PASS | 10/10 PASS |
| FC2 logits | 10/10 PASS | 10/10 PASS | 10/10 PASS |
| Top vs 量化软件预测 | 10/10 PASS | 10/10 PASS | 10/10 PASS |
| Top vs 真实标签 | 10/10 PASS | 10/10 PASS | 10/10 PASS |

这说明三轮 HLS 优化没有改变模型数值行为，硬件 C++ 输出与软件 int8 模型保持一致。

### 6.3 HLS 综合性能对比

目标时钟为 10 ns，即 100 MHz。三轮综合结果如下：

| 指标 | solution1 baseline | solution2 3x3 MAC 展开 | solution3 输入 buffer + FC1 调整 |
| --- | ---: | ---: | ---: |
| Latency | 354430 cycles | 76031 cycles | 30203 cycles |
| Interval | 354431 | 76032 | 30204 |
| @100MHz 单张延迟 | 3.544 ms | 0.760 ms | 0.302 ms |
| @100MHz 吞吐量 | 约 282 images/s | 约 1315 images/s | 约 3311 images/s |
| 相对 solution1 加速比 | 1.00x | 4.66x | 11.74x |
| 相对上一版加速比 | - | 4.66x | 2.52x |
| Estimated timing | 7.300 ns | 7.300 ns | 7.300 ns |

solution2 的核心收益来自 Conv2：3x3 kernel 展开后，最大瓶颈从 269696 cycles 降到 12632 cycles。solution3 的核心收益来自 Conv1：输入图像先进入片上完全分块 buffer 后，Conv1 从 56467 cycles 降到 7086 cycles。最终 solution3 单张图像延迟为 0.302 ms，等效吞吐量约 3311 images/s。

### 6.4 资源利用率对比

目标器件为 `xc7z020-clg400-1`。资源利用率如下：

| 资源 | solution1 | solution2 | solution3 |
| --- | ---: | ---: | ---: |
| BRAM_18K | 26 / 280，9% | 84 / 280，30% | 50 / 280，17% |
| DSP | 16 / 220，7% | 116 / 220，52% | 116 / 220，52% |
| FF | 5274 / 106400，4% | 19695 / 106400，18% | 25626 / 106400，24% |
| LUT | 7505 / 53200，14% | 22045 / 53200，41% | 32671 / 53200，61% |

资源变化反映了三轮优化的取舍。solution2 增加大量 DSP 和 BRAM，用并行乘加换取 Conv2 加速。solution3 保持 DSP 数量不变，但由于 `image_buf` complete partition 和更复杂的本地读数据通路，LUT 和 FF 上升；同时 FC1 展开因子从 4 降到 2，使 BRAM 从 84 降到 50。总体看，solution3 性能最好，但 LUT 已达到 61%，继续增加并行度需要谨慎。

### 6.5 主要模块延迟对比

| 模块 | solution1 | solution2 | solution3 | 说明 |
| --- | ---: | ---: | ---: | --- |
| Conv1 | 56467 | 56467 | 7086 | s3 通过片上 `image_buf` 解决输入读端口限制 |
| Pool1 | 1573 | 1577 | 1577 | 基本不变 |
| Conv2 | 269696 | 12632 | 12632 | s2 的 3x3 kernel 展开带来主要加速 |
| Pool2 | 789 | 789 | 789 | 基本不变 |
| FC1 | 25472 | 4129 | 7681 | s2 factor=4 最快；s3 factor=2 降低 BRAM 但延迟回升 |
| FC2 | 410 | 410 | 410 | 基本不变 |
| Argmax | 12 | 12 | 12 | 占比很小 |

这一表格说明优化路径是有针对性的。solution1 的最大瓶颈是 Conv2，因此 solution2 先优化 Conv2。solution2 之后 Conv1 成为最大剩余瓶颈，因此 solution3 再优化 Conv1。FC1 在 s3 中没有追求最低延迟，而是用于资源权衡。

### 6.6 存储需求

权重数量为：

| 层 | 权重数量 | bias 数量 |
| --- | ---: | ---: |
| Conv1 | 72 | 8 |
| Conv2 | 1152 | 16 |
| FC1 | 25088 | 32 |
| FC2 | 320 | 10 |
| 合计 | 26632 | 66 |

若权重使用 int8，bias 使用 int32，则权重和 bias 总存储约为：

```text
26632 bytes + 66 * 4 bytes = 26896 bytes，约 26.3 KiB
```

主要特征图 buffer 包括 Conv1、Pool1、Conv2、Pool2、FC1 和 logits。按纯数据量估算，中间特征图约 12.6 KiB；实际 HLS BRAM 使用还受数组 partition、端口需求和存储器映射方式影响。

### 6.7 存在问题与改进方向

当前设计已经满足课程大作业的主要功能与性能分析要求，但仍存在一些限制：

1. `Interval` 仍接近 `Latency`。这说明当前架构仍是“一张图处理完再处理下一张图”，不是多张图层间 dataflow 流水。
2. solution3 的 LUT 已达到 61%。后续若继续展开输入通道或增加并行度，可能导致 LUT 或布线压力过高。
3. 当前结果来自 HLS C Synthesis。最终 FPGA 上板综合、Vivado implementation、bitstream 和板端测试仍需由队友补充。
4. 当前 testbench 使用 10 张图像，满足课程最低要求；若时间允许，可以增加更多测试样本提高验证覆盖面。

后续改进可以从两个方向进行。若目标是进一步降低 latency，可以尝试层间 dataflow、双缓冲和更细粒度输入通道并行；若目标是降低资源，可以回退部分 array partition 或降低展开因子，形成资源友好版。考虑到 solution3 已经达到 11.74 倍加速，后续更适合围绕“性能-资源权衡”和“实际上板可实现性”继续完善。

### 6.8 与相关工作的比较

与 hls4ml、FINN 或 Vitis AI DPU 等成熟工具链相比，本项目没有追求大型网络或极限吞吐，而是针对课程要求实现了完整、可解释、可逐层验证的小型 CNN 加速器。成熟工具链通常拥有更完善的自动量化、图优化、dataflow pipeline 和 IP 集成能力；本项目则更强调从 PyTorch 训练到 HLS 代码、从 golden 数据导出到逐层验证、从 baseline 到多轮 pragma 优化的完整学习过程。

从实验结果看，solution3 在 100 MHz 下达到约 0.302 ms 单张延迟和约 3311 images/s 等效吞吐量。对于 MNIST 这类小规模任务，该性能已经足够完成实时识别演示。与工业级加速器相比，后续差距主要在自动化优化、多图流水、板端数据搬运和系统集成。

## 7. 总结及展望

本项目完成了一个基于 FPGA/HLS 的轻量化 CNN 加速器设计。软件端使用 PyTorch 训练 MNIST 模型，浮点准确率达到 98.49%；随后采用 int8 权重和激活完成定点量化，量化准确率达到 98.47%。硬件端使用 Vitis HLS 实现卷积、池化、ReLU、全连接、片上 buffer、AXI 接口和 argmax。testbench 使用 10 张量化图像逐层验证，三轮 solution 均与软件 golden 数据匹配。

性能优化方面，solution1 作为 baseline，延迟为 354430 cycles。solution2 对 3x3 kernel 进行展开，将 Conv2 延迟从 269696 cycles 降到 12632 cycles，整体延迟降到 76031 cycles。solution3 在 Conv1 中加入片上输入 buffer，将 Conv1 延迟从 56467 cycles 降到 7086 cycles，整体延迟进一步降到 30203 cycles。相对于 baseline，solution3 加速约 11.74 倍，@100MHz 等效吞吐量约 3311 images/s。

下一步工作主要包括：补充 Word 报告中的架构图、仿真截图和综合报告截图；由队友完成 Vivado implementation、bitstream 生成和板端测试；若继续优化，可研究 dataflow pipeline、双缓冲、更系统的资源约束和自动化 pragma 搜索。最终报告中应明确区分 HLS C Synthesis 结果和队友后续提供的实际 FPGA implementation 结果。

## 参考文献 TODO

正式 Word 报告中建议补充以下方向的参考文献：

1. hls4ml 及其 CNN/RNN/Transformer FPGA 部署工作。
2. FINN 或类似量化神经网络 FPGA 数据流框架。
3. Vitis AI / DPU 在 Zynq 或 Zynq UltraScale+ 上的 CNN 部署案例。
4. 近年 int8/混合精度 CNN FPGA 加速器论文。
5. HLS 中 array partition、loop unroll、pipeline、dataflow 对 FPGA CNN 性能影响的相关研究。
