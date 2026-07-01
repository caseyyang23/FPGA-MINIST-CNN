from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from train_mnist import ROOT, TinyMNISTCNN, make_loaders


INT8_MAX = 127
FIXED_SHIFT = 24


def load_checkpoint(checkpoint: Path, device: torch.device) -> TinyMNISTCNN:
    model = TinyMNISTCNN().to(device)
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(saved["model_state"])
    model.eval()
    return model


def quantize_symmetric(tensor: torch.Tensor) -> tuple[np.ndarray, float]:
    max_abs = float(tensor.detach().abs().max().cpu())
    scale = max(max_abs / INT8_MAX, 1e-8)
    q = torch.clamp(torch.round(tensor.detach().cpu() / scale), -INT8_MAX, INT8_MAX)
    return q.to(torch.int8).numpy(), scale


def quantize_bias(bias: torch.Tensor, input_scale: float, weight_scale: float) -> np.ndarray:
    denom = max(input_scale * weight_scale, 1e-12)
    q = torch.round(bias.detach().cpu() / denom)
    return q.to(torch.int32).numpy()


def fixed_multiplier(real_multiplier: float, shift: int = FIXED_SHIFT) -> tuple[int, int]:
    mult = int(round(real_multiplier * (1 << shift)))
    return max(mult, 1), shift


def collect_activation_scales(
    model: TinyMNISTCNN,
    loader,
    device: torch.device,
    calib_batches: int,
) -> dict[str, float]:
    max_abs = {
        "conv1_relu": 0.0,
        "conv2_relu": 0.0,
        "fc1_relu": 0.0,
    }

    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(loader):
            if batch_idx >= calib_batches:
                break
            images = images.to(device)
            _, layers = model(images, return_layers=True)
            for name in max_abs:
                max_abs[name] = max(max_abs[name], float(layers[name].abs().max().cpu()))

    return {name: max(value / INT8_MAX, 1e-8) for name, value in max_abs.items()}


def build_quantized_bundle(
    model: TinyMNISTCNN,
    calib_loader,
    device: torch.device,
    calib_batches: int,
) -> dict[str, Any]:
    input_scale = 1.0 / INT8_MAX
    act_scales = collect_activation_scales(model, calib_loader, device, calib_batches)

    w_conv1, sw_conv1 = quantize_symmetric(model.conv1.weight)
    w_conv2, sw_conv2 = quantize_symmetric(model.conv2.weight)
    w_fc1, sw_fc1 = quantize_symmetric(model.fc1.weight)
    w_fc2, sw_fc2 = quantize_symmetric(model.fc2.weight)

    b_conv1 = quantize_bias(model.conv1.bias, input_scale, sw_conv1)
    b_conv2 = quantize_bias(model.conv2.bias, act_scales["conv1_relu"], sw_conv2)
    b_fc1 = quantize_bias(model.fc1.bias, act_scales["conv2_relu"], sw_fc1)
    b_fc2 = quantize_bias(model.fc2.bias, act_scales["fc1_relu"], sw_fc2)

    m_conv1, s_conv1 = fixed_multiplier(input_scale * sw_conv1 / act_scales["conv1_relu"])
    m_conv2, s_conv2 = fixed_multiplier(act_scales["conv1_relu"] * sw_conv2 / act_scales["conv2_relu"])
    m_fc1, s_fc1 = fixed_multiplier(act_scales["conv2_relu"] * sw_fc1 / act_scales["fc1_relu"])

    return {
        "w_conv1": w_conv1,
        "w_conv2": w_conv2,
        "w_fc1": w_fc1,
        "w_fc2": w_fc2,
        "b_conv1": b_conv1,
        "b_conv2": b_conv2,
        "b_fc1": b_fc1,
        "b_fc2": b_fc2,
        "params": {
            "input_scale": input_scale,
            "weight_scales": {
                "conv1": sw_conv1,
                "conv2": sw_conv2,
                "fc1": sw_fc1,
                "fc2": sw_fc2,
            },
            "activation_scales": act_scales,
            "requant": {
                "conv1": {"mult": m_conv1, "shift": s_conv1},
                "conv2": {"mult": m_conv2, "shift": s_conv2},
                "fc1": {"mult": m_fc1, "shift": s_fc1},
            },
        },
    }


def torch_tensor(array: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(array.astype(np.float32)).to(device)


def requant_relu(acc: torch.Tensor, mult: int, shift: int) -> torch.Tensor:
    acc_i64 = torch.round(acc).to(torch.int64)
    acc_i64 = torch.clamp(acc_i64, min=0)

    rounding = 1 << (shift - 1)
    scaled = (acc_i64 * int(mult) + rounding) >> int(shift)

    return torch.clamp(scaled, 0, 127).to(torch.int64)


def quantized_forward_torch(
    images: torch.Tensor,
    bundle: dict[str, Any],
    device: torch.device,
    return_layers: bool = False,
):
    params = bundle["params"]
    qx = torch.clamp(torch.round(images.to(device) / params["input_scale"]), -128.0, 127.0)

    w_conv1 = torch_tensor(bundle["w_conv1"], device)
    w_conv2 = torch_tensor(bundle["w_conv2"], device)
    w_fc1 = torch_tensor(bundle["w_fc1"], device)
    w_fc2 = torch_tensor(bundle["w_fc2"], device)
    b_conv1 = torch_tensor(bundle["b_conv1"], device)
    b_conv2 = torch_tensor(bundle["b_conv2"], device)
    b_fc1 = torch_tensor(bundle["b_fc1"], device)
    b_fc2 = torch_tensor(bundle["b_fc2"], device)

    rq = params["requant"]
    conv1_acc = F.conv2d(qx, w_conv1, b_conv1, padding=1)
    conv1_q = requant_relu(conv1_acc, rq["conv1"]["mult"], rq["conv1"]["shift"])
    pool1_q = F.max_pool2d(conv1_q.to(torch.float32), kernel_size=2, stride=2).to(torch.int64)

    conv2_acc = F.conv2d(pool1_q.to(torch.float32), w_conv2, b_conv2, padding=1)
    conv2_q = requant_relu(conv2_acc, rq["conv2"]["mult"], rq["conv2"]["shift"])
    pool2_q = F.max_pool2d(conv2_q.to(torch.float32), kernel_size=2, stride=2).to(torch.int64)

    flat = pool2_q.flatten(1).to(torch.float32)
    fc1_acc = F.linear(flat, w_fc1, b_fc1)
    fc1_q = requant_relu(fc1_acc, rq["fc1"]["mult"], rq["fc1"]["shift"])

    logits_q = F.linear(fc1_q.to(torch.float32), w_fc2, b_fc2)
    pred = logits_q.argmax(dim=1)

    if not return_layers:
        return pred

    return pred, {
        "conv1": conv1_q,
        "pool1": pool1_q,
        "conv2": conv2_q,
        "pool2": pool2_q,
        "fc1": fc1_q,
        "fc2_logits": logits_q,
    }


@torch.no_grad()
def evaluate_float(
    model: TinyMNISTCNN,
    loader,
    device: torch.device,
) -> float:
    correct = 0
    total = 0

    model.eval()

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        pred = logits.argmax(dim=1)

        correct += (pred == labels).sum().item()
        total += labels.numel()

    return correct / total


@torch.no_grad()
def evaluate_quantized(
    loader,
    bundle: dict[str, Any],
    device: torch.device,
) -> float:
    correct = 0
    total = 0

    for images, labels in loader:
        labels = labels.to(device)
        pred = quantized_forward_torch(images, bundle, device, return_layers=False)

        correct += (pred == labels).sum().item()
        total += labels.numel()

    return correct / total


def format_c_values(array: np.ndarray, per_line: int = 16) -> str:
    flat = array.reshape(-1)
    values = [str(int(v)) for v in flat]
    lines = []
    for idx in range(0, len(values), per_line):
        lines.append("    " + ", ".join(values[idx : idx + per_line]))
    return "{\n" + ",\n".join(lines) + "\n}"


def c_array(ctype: str, name: str, dims: str, array: np.ndarray) -> str:
    return f"static const {ctype} {name}{dims} = {format_c_values(array)};\n"


def write_weights_header(bundle: dict[str, Any], output: Path) -> None:
    params = bundle["params"]
    output.parent.mkdir(parents=True, exist_ok=True)

    text = f"""#ifndef WEIGHTS_H
#define WEIGHTS_H

#include "ap_int.h"

typedef ap_int<8> qint8_t;
typedef ap_int<32> qint32_t;

static const int IMG_H = 28;
static const int IMG_W = 28;
static const int MNIST_PIXELS = IMG_H * IMG_W;
static const int K = 3;
static const int C1_OUT = 8;
static const int C2_OUT = 16;
static const int P1_H = 14;
static const int P1_W = 14;
static const int P2_H = 7;
static const int P2_W = 7;
static const int FC_IN = C2_OUT * P2_H * P2_W;
static const int FC1_OUT = 32;
static const int NUM_CLASSES = 10;

static const int M_CONV1 = {params["requant"]["conv1"]["mult"]};
static const int S_CONV1 = {params["requant"]["conv1"]["shift"]};
static const int M_CONV2 = {params["requant"]["conv2"]["mult"]};
static const int S_CONV2 = {params["requant"]["conv2"]["shift"]};
static const int M_FC1 = {params["requant"]["fc1"]["mult"]};
static const int S_FC1 = {params["requant"]["fc1"]["shift"]};

// Floating-point scales are kept for documentation and debugging.
static const float INPUT_SCALE = {params["input_scale"]:.10e}f;
static const float W_CONV1_SCALE = {params["weight_scales"]["conv1"]:.10e}f;
static const float W_CONV2_SCALE = {params["weight_scales"]["conv2"]:.10e}f;
static const float W_FC1_SCALE = {params["weight_scales"]["fc1"]:.10e}f;
static const float W_FC2_SCALE = {params["weight_scales"]["fc2"]:.10e}f;
static const float A_CONV1_SCALE = {params["activation_scales"]["conv1_relu"]:.10e}f;
static const float A_CONV2_SCALE = {params["activation_scales"]["conv2_relu"]:.10e}f;
static const float A_FC1_SCALE = {params["activation_scales"]["fc1_relu"]:.10e}f;

"""
    text += c_array("qint8_t", "W_CONV1", "[C1_OUT][1][K][K]", bundle["w_conv1"])
    text += c_array("qint8_t", "W_CONV2", "[C2_OUT][C1_OUT][K][K]", bundle["w_conv2"])
    text += c_array("qint8_t", "W_FC1", "[FC1_OUT][FC_IN]", bundle["w_fc1"])
    text += c_array("qint8_t", "W_FC2", "[NUM_CLASSES][FC1_OUT]", bundle["w_fc2"])
    text += c_array("qint32_t", "B_CONV1", "[C1_OUT]", bundle["b_conv1"])
    text += c_array("qint32_t", "B_CONV2", "[C2_OUT]", bundle["b_conv2"])
    text += c_array("qint32_t", "B_FC1", "[FC1_OUT]", bundle["b_fc1"])
    text += c_array("qint32_t", "B_FC2", "[NUM_CLASSES]", bundle["b_fc2"])
    text += "\n#endif\n"
    output.write_text(text, encoding="utf-8")


def write_test_headers(
    images: torch.Tensor,
    labels: torch.Tensor,
    bundle: dict[str, Any],
    data_dir: Path,
    device: torch.device,
) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    params = bundle["params"]
    q_images = torch.clamp(torch.round(images / params["input_scale"]), -128, 127).to(torch.int8)
    q_images_np = q_images.cpu().numpy().reshape(images.shape[0], -1)
    labels_np = labels.cpu().numpy().astype(np.int32)
    pred, layers = quantized_forward_torch(images, bundle, device, return_layers=True)
    pred_np = pred.cpu().numpy().astype(np.int32)

    test_images = f"""#ifndef TEST_IMAGES_H
#define TEST_IMAGES_H

#include "weights.h"

static const int NUM_TEST_IMAGES = {images.shape[0]};
{c_array("qint8_t", "TEST_IMAGES", "[NUM_TEST_IMAGES][MNIST_PIXELS]", q_images_np)}

#endif
"""
    (data_dir / "test_images.h").write_text(test_images, encoding="utf-8")

    test_labels = f"""#ifndef TEST_LABELS_H
#define TEST_LABELS_H

#include "weights.h"
#include "test_images.h"

{c_array("int", "TEST_LABELS", "[NUM_TEST_IMAGES]", labels_np)}
{c_array("int", "GOLDEN_PRED_LABELS", "[NUM_TEST_IMAGES]", pred_np)}

#endif
"""
    (data_dir / "test_labels.h").write_text(test_labels, encoding="utf-8")

    golden_text = """#ifndef GOLDEN_LAYERS_H
#define GOLDEN_LAYERS_H

#include "weights.h"
#include "test_images.h"

static const int GOLDEN_TOLERANCE = 1;

"""
    golden_text += c_array(
        "qint8_t",
        "GOLDEN_CONV1",
        "[NUM_TEST_IMAGES][C1_OUT][IMG_H][IMG_W]",
        layers["conv1"].cpu().numpy().astype(np.int8),
    )
    golden_text += c_array(
        "qint8_t",
        "GOLDEN_POOL1",
        "[NUM_TEST_IMAGES][C1_OUT][P1_H][P1_W]",
        layers["pool1"].cpu().numpy().astype(np.int8),
    )
    golden_text += c_array(
        "qint8_t",
        "GOLDEN_CONV2",
        "[NUM_TEST_IMAGES][C2_OUT][P1_H][P1_W]",
        layers["conv2"].cpu().numpy().astype(np.int8),
    )
    golden_text += c_array(
        "qint8_t",
        "GOLDEN_POOL2",
        "[NUM_TEST_IMAGES][C2_OUT][P2_H][P2_W]",
        layers["pool2"].cpu().numpy().astype(np.int8),
    )
    golden_text += c_array(
        "qint8_t",
        "GOLDEN_FC1",
        "[NUM_TEST_IMAGES][FC1_OUT]",
        layers["fc1"].cpu().numpy().astype(np.int8),
    )
    golden_text += c_array(
        "qint32_t",
        "GOLDEN_FC2_LOGITS",
        "[NUM_TEST_IMAGES][NUM_CLASSES]",
        layers["fc2_logits"].cpu().numpy().astype(np.int32),
    )
    golden_text += "\n#endif\n"
    (data_dir / "golden_layers.h").write_text(golden_text, encoding="utf-8")


def save_npz(bundle: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    arrays = {key: value for key, value in bundle.items() if key != "params"}
    arrays["params_json"] = np.array(json.dumps(bundle["params"], indent=2))
    np.savez_compressed(output, **arrays)


def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    train_loader, test_loader = make_loaders(
        batch_size=args.batch_size,
        data_dir=args.data_dir,
        num_workers=args.num_workers,
    )
    model = load_checkpoint(args.checkpoint, device)
    bundle = build_quantized_bundle(model, train_loader, device, args.calib_batches)

    float_acc = evaluate_float(model, test_loader, device)
    int8_acc = evaluate_quantized(test_loader, bundle, device)

    print(f"float32 test accuracy: {float_acc * 100:.2f}%")
    print(f"int8 quantized test accuracy: {int8_acc * 100:.2f}%")

    write_weights_header(bundle, args.hls_data_dir / "weights.h")

    images_list = []
    labels_list = []
    for images, labels in test_loader:
        images_list.append(images)
        labels_list.append(labels)
        if sum(batch.shape[0] for batch in images_list) >= args.num_test:
            break

    test_images = torch.cat(images_list, dim=0)[: args.num_test]
    test_labels = torch.cat(labels_list, dim=0)[: args.num_test]
    write_test_headers(test_images, test_labels, bundle, args.hls_data_dir, device)
    save_npz(bundle, args.output_npz)

    summary = {
        "float32_accuracy": float_acc,
        "int8_accuracy": int8_acc,
        "checkpoint": str(args.checkpoint),
        "hls_data_dir": str(args.hls_data_dir),
        "num_test": args.num_test,
        "calib_batches": args.calib_batches,
        "params": bundle["params"],
    }
    summary_path = args.hls_data_dir / "quant_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"wrote HLS headers to: {args.hls_data_dir}")
    print(f"saved quantized bundle: {args.output_npz}")
    print(f"saved quantization summary: {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export int8 MNIST CNN weights and golden data for HLS.")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "software" / "checkpoints" / "mnist_cnn_fp32.pt")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--calib-batches", type=int, default=20)
    parser.add_argument("--num-test", type=int, default=10)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "software" / "data")
    parser.add_argument("--hls-data-dir", type=Path, default=ROOT / "hls" / "data")
    parser.add_argument("--output-npz", type=Path, default=ROOT / "software" / "checkpoints" / "mnist_cnn_int8_export.npz")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
