#include "cnn_accel.h"

static qint8_t clamp_relu_requant(qint32_t acc, int mult, int shift) {
    if (acc <= 0) {
        return 0;
    }

    ap_int<64> scaled = (ap_int<64>)acc * mult;
    scaled += (ap_int<64>)1 << (shift - 1);
    scaled >>= shift;

    if (scaled > 127) {
        return 127;
    }
    if (scaled < 0) {
        return 0;
    }
    return (qint8_t)scaled;
}

void conv1_layer(
    const qint8_t image[MNIST_PIXELS],
    qint8_t conv1[C1_OUT][IMG_H][IMG_W]
) {
#pragma HLS ARRAY_PARTITION variable=W_CONV1 complete dim=3
#pragma HLS ARRAY_PARTITION variable=W_CONV1 complete dim=4

conv1_loop:
    for (int oc = 0; oc < C1_OUT; ++oc) {
        for (int y = 0; y < IMG_H; ++y) {
            for (int x = 0; x < IMG_W; ++x) {
#pragma HLS PIPELINE II=1
                qint32_t acc = B_CONV1[oc];
                for (int ky = 0; ky < K; ++ky) {
#pragma HLS UNROLL
                    for (int kx = 0; kx < K; ++kx) {
#pragma HLS UNROLL
                        int iy = y + ky - 1;
                        int ix = x + kx - 1;
                        if (iy >= 0 && iy < IMG_H && ix >= 0 && ix < IMG_W) {
                            qint8_t pix = image[iy * IMG_W + ix];
                            acc += (qint32_t)pix * W_CONV1[oc][0][ky][kx];
                        }
                    }
                }
                conv1[oc][y][x] = clamp_relu_requant(acc, M_CONV1, S_CONV1);
            }
        }
    }
}

void pool1_layer(
    const qint8_t conv1[C1_OUT][IMG_H][IMG_W],
    qint8_t pool1[C1_OUT][P1_H][P1_W]
) {
pool1_loop:
    for (int c = 0; c < C1_OUT; ++c) {
        for (int y = 0; y < P1_H; ++y) {
            for (int x = 0; x < P1_W; ++x) {
#pragma HLS PIPELINE II=1
                qint8_t m = conv1[c][2 * y][2 * x];
                qint8_t v1 = conv1[c][2 * y][2 * x + 1];
                qint8_t v2 = conv1[c][2 * y + 1][2 * x];
                qint8_t v3 = conv1[c][2 * y + 1][2 * x + 1];
                if (v1 > m) m = v1;
                if (v2 > m) m = v2;
                if (v3 > m) m = v3;
                pool1[c][y][x] = m;
            }
        }
    }
}

void conv2_layer(
    const qint8_t pool1[C1_OUT][P1_H][P1_W],
    qint8_t conv2[C2_OUT][P1_H][P1_W]
) {
#pragma HLS ARRAY_PARTITION variable=W_CONV2 complete dim=3
#pragma HLS ARRAY_PARTITION variable=W_CONV2 complete dim=4

conv2_loop:
    for (int oc = 0; oc < C2_OUT; ++oc) {
        for (int y = 0; y < P1_H; ++y) {
            for (int x = 0; x < P1_W; ++x) {
#pragma HLS PIPELINE II=1
                qint32_t acc = B_CONV2[oc];
                for (int ic = 0; ic < C1_OUT; ++ic) {
                    for (int ky = 0; ky < K; ++ky) {
#pragma HLS UNROLL
                        for (int kx = 0; kx < K; ++kx) {
#pragma HLS UNROLL
                            int iy = y + ky - 1;
                            int ix = x + kx - 1;
                            if (iy >= 0 && iy < P1_H && ix >= 0 && ix < P1_W) {
                                acc += (qint32_t)pool1[ic][iy][ix] * W_CONV2[oc][ic][ky][kx];
                            }
                        }
                    }
                }
                conv2[oc][y][x] = clamp_relu_requant(acc, M_CONV2, S_CONV2);
            }
        }
    }
}

void pool2_layer(
    const qint8_t conv2[C2_OUT][P1_H][P1_W],
    qint8_t pool2[C2_OUT][P2_H][P2_W]
) {
pool2_loop:
    for (int c = 0; c < C2_OUT; ++c) {
        for (int y = 0; y < P2_H; ++y) {
            for (int x = 0; x < P2_W; ++x) {
#pragma HLS PIPELINE II=1
                qint8_t m = conv2[c][2 * y][2 * x];
                qint8_t v1 = conv2[c][2 * y][2 * x + 1];
                qint8_t v2 = conv2[c][2 * y + 1][2 * x];
                qint8_t v3 = conv2[c][2 * y + 1][2 * x + 1];
                if (v1 > m) m = v1;
                if (v2 > m) m = v2;
                if (v3 > m) m = v3;
                pool2[c][y][x] = m;
            }
        }
    }
}

void fc1_layer(
    const qint8_t pool2[C2_OUT][P2_H][P2_W],
    qint8_t fc1[FC1_OUT]
) {
#pragma HLS ARRAY_PARTITION variable=fc1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=pool2 cyclic factor=4 dim=3
#pragma HLS ARRAY_PARTITION variable=W_FC1 cyclic factor=4 dim=2

fc1_loop:
    for (int o = 0; o < FC1_OUT; ++o) {
        qint32_t acc = B_FC1[o];
        for (int c = 0; c < C2_OUT; ++c) {
            for (int y = 0; y < P2_H; ++y) {
#pragma HLS PIPELINE II=1
                for (int x = 0; x < P2_W; ++x) {
#pragma HLS UNROLL factor=4
                    int idx = c * P2_H * P2_W + y * P2_W + x;
                    acc += (qint32_t)pool2[c][y][x] * W_FC1[o][idx];
                }
            }
        }
        fc1[o] = clamp_relu_requant(acc, M_FC1, S_FC1);
    }
}

void fc2_layer(
    const qint8_t fc1[FC1_OUT],
    qint32_t logits[NUM_CLASSES]
) {
fc2_loop:
    for (int o = 0; o < NUM_CLASSES; ++o) {
        qint32_t score = B_FC2[o];
        for (int i = 0; i < FC1_OUT; ++i) {
#pragma HLS PIPELINE II=1
            score += (qint32_t)fc1[i] * W_FC2[o][i];
        }
        logits[o] = score;
    }
}

void cnn_accel(const qint8_t image[MNIST_PIXELS], ap_uint<4> &label) {
#pragma HLS INTERFACE m_axi port=image offset=slave bundle=gmem depth=MNIST_PIXELS
#pragma HLS INTERFACE s_axilite port=image bundle=control
#pragma HLS INTERFACE s_axilite port=label bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    static qint8_t conv1[C1_OUT][IMG_H][IMG_W];
    static qint8_t pool1[C1_OUT][P1_H][P1_W];
    static qint8_t conv2[C2_OUT][P1_H][P1_W];
    static qint8_t pool2[C2_OUT][P2_H][P2_W];
    static qint8_t fc1[FC1_OUT];
    static qint32_t logits[NUM_CLASSES];

    conv1_layer(image, conv1);
    pool1_layer(conv1, pool1);
    conv2_layer(pool1, conv2);
    pool2_layer(conv2, pool2);
    fc1_layer(pool2, fc1);
    fc2_layer(fc1, logits);

    qint32_t best_score = logits[0];
    ap_uint<4> best_label = 0;

argmax_loop:
    for (int o = 1; o < NUM_CLASSES; ++o) {
#pragma HLS PIPELINE II=1
        if (logits[o] > best_score) {
            best_score = logits[o];
            best_label = o;
        }
    }

    label = best_label;
}
