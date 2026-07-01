# mnist_fpga_cnn

基于 FPGA 的轻量化 CNN 加速器设计及 MNIST 手写体图像识别实现。

## Directory

```text
mnist_fpga_cnn/
├─ software/              # PyTorch training, quantization, evaluation
├─ hls/                   # Vitis HLS accelerator and C++ testbench
├─ vivado/                # Vivado project files or exported RTL integration files
├─ report/                # Report notes and figures
└─ README.md
```

## Suggested Workflow

1. 训练浮点模型：

```powershell
python software/train_mnist.py --epochs 5
```

2. 导出 8-bit 定点权重、测试图像和逐层 golden 数据：

```powershell
python software/quantize_export.py --checkpoint software/checkpoints/mnist_cnn_fp32.pt --num-test 10
```

3. 验证量化模型精度：

```powershell
python software/eval_quant.py --checkpoint software/checkpoints/mnist_cnn_fp32.pt
```

4. 在 Vitis HLS 中加入：

```text
hls/cnn_accel.cpp
hls/cnn_accel.h
hls/tb_cnn.cpp
hls/data/*.h
```

Top function 设置为 `cnn_accel`。HLS 的 `ap_ctrl_hs` 控制接口会生成 `ap_clk`、`ap_rst`、`ap_start`、`ap_done` 等 RTL 顶层信号。

## Network

默认网络结构：

```text
Input 1x28x28
Conv3x3(1 -> 8), ReLU, MaxPool2x2
Conv3x3(8 -> 16), ReLU, MaxPool2x2
FC(16*7*7 -> 32), ReLU
FC(32 -> 10)
```

目标：MNIST 测试集准确率达到 98% 以上，并导出硬件可读的 int8 权重、int8 激活和 int32 bias。

## Report Checklist

- 网络结构和参数量
- 训练曲线与测试准确率
- 8-bit 定点量化方案和量化精度
- CNN 加速器架构图
- 顶层接口说明
- Testbench 仿真结果，至少 10 张量化图像
- 逐层误差检查，目标误差不超过 ±1
- 单张图片识别延迟、100 MHz 下吞吐量
- 权重和特征图片上存储需求
- 若完成综合，加入 LUT/FF/DSP/BRAM 资源利用率
