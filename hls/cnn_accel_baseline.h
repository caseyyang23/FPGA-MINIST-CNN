#ifndef CNN_ACCEL_H
#define CNN_ACCEL_H

#include "ap_int.h"
#include "data/weights.h"

void conv1_layer(
    const qint8_t image[MNIST_PIXELS],
    qint8_t conv1[C1_OUT][IMG_H][IMG_W]
);

void pool1_layer(
    const qint8_t conv1[C1_OUT][IMG_H][IMG_W],
    qint8_t pool1[C1_OUT][P1_H][P1_W]
);

void conv2_layer(
    const qint8_t pool1[C1_OUT][P1_H][P1_W],
    qint8_t conv2[C2_OUT][P1_H][P1_W]
);

void pool2_layer(
    const qint8_t conv2[C2_OUT][P1_H][P1_W],
    qint8_t pool2[C2_OUT][P2_H][P2_W]
);

void fc1_layer(
    const qint8_t pool2[C2_OUT][P2_H][P2_W],
    qint8_t fc1[FC1_OUT]
);

void fc2_layer(
    const qint8_t fc1[FC1_OUT],
    qint32_t logits[NUM_CLASSES]
);

void cnn_accel(const qint8_t image[MNIST_PIXELS], ap_uint<4> &label);

#endif
