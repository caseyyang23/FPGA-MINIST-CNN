#include <cstdlib>
#include <iostream>

#include "cnn_accel.h"
#include "data/golden_layers.h"
#include "data/test_images.h"
#include "data/test_labels.h"

template <typename T>
static bool check_values(
    const char *name,
    int image_idx,
    const T *actual,
    const T *expected,
    int count
) {
    for (int i = 0; i < count; ++i) {
        int diff = (int)actual[i] - (int)expected[i];
        if (diff < 0) {
            diff = -diff;
        }

        if (diff > GOLDEN_TOLERANCE) {
            std::cout << "image " << image_idx
                      << " " << name
                      << " FAIL index=" << i
                      << " actual=" << (int)actual[i]
                      << " expected=" << (int)expected[i]
                      << " diff=" << diff
                      << std::endl;
            return false;
        }
    }

    std::cout << "image " << image_idx << " " << name << " PASS" << std::endl;
    return true;
}

int main() {
    int layer_match = 0;
    int quant_match = 0;
    int label_match = 0;

    static qint8_t conv1[C1_OUT][IMG_H][IMG_W];
    static qint8_t pool1[C1_OUT][P1_H][P1_W];
    static qint8_t conv2[C2_OUT][P1_H][P1_W];
    static qint8_t pool2[C2_OUT][P2_H][P2_W];
    static qint8_t fc1[FC1_OUT];
    static qint32_t logits[NUM_CLASSES];

    for (int n = 0; n < NUM_TEST_IMAGES; ++n) {
        bool layers_ok = true;

        conv1_layer(TEST_IMAGES[n], conv1);
        layers_ok &= check_values(
            "conv1",
            n,
            &conv1[0][0][0],
            &GOLDEN_CONV1[n][0][0][0],
            C1_OUT * IMG_H * IMG_W
        );

        pool1_layer(conv1, pool1);
        layers_ok &= check_values(
            "pool1",
            n,
            &pool1[0][0][0],
            &GOLDEN_POOL1[n][0][0][0],
            C1_OUT * P1_H * P1_W
        );

        conv2_layer(pool1, conv2);
        layers_ok &= check_values(
            "conv2",
            n,
            &conv2[0][0][0],
            &GOLDEN_CONV2[n][0][0][0],
            C2_OUT * P1_H * P1_W
        );

        pool2_layer(conv2, pool2);
        layers_ok &= check_values(
            "pool2",
            n,
            &pool2[0][0][0],
            &GOLDEN_POOL2[n][0][0][0],
            C2_OUT * P2_H * P2_W
        );

        fc1_layer(pool2, fc1);
        layers_ok &= check_values(
            "fc1",
            n,
            &fc1[0],
            &GOLDEN_FC1[n][0],
            FC1_OUT
        );

        fc2_layer(fc1, logits);
        layers_ok &= check_values(
            "fc2_logits",
            n,
            &logits[0],
            &GOLDEN_FC2_LOGITS[n][0],
            NUM_CLASSES
        );

        layer_match += layers_ok ? 1 : 0;

        ap_uint<4> pred = 0;
        cnn_accel(TEST_IMAGES[n], pred);

        int expected_label = TEST_LABELS[n];
        int expected_quant = GOLDEN_PRED_LABELS[n];
        bool match_quant = ((int)pred == expected_quant);
        bool match_label = ((int)pred == expected_label);
        quant_match += match_quant ? 1 : 0;
        label_match += match_label ? 1 : 0;

        std::cout << "image " << n
                  << " top pred=" << (int)pred
                  << " golden_pred=" << expected_quant
                  << " label=" << expected_label
                  << " quant_match=" << (match_quant ? "PASS" : "FAIL")
                  << " label_match=" << (match_label ? "PASS" : "FAIL")
                  << std::endl;
    }

    std::cout << "layer matched " << layer_match << " / " << NUM_TEST_IMAGES << std::endl;
    std::cout << "matched quantized model " << quant_match << " / " << NUM_TEST_IMAGES << std::endl;
    std::cout << "matched true labels " << label_match << " / " << NUM_TEST_IMAGES << std::endl;

    return (layer_match == NUM_TEST_IMAGES && quant_match == NUM_TEST_IMAGES)
        ? EXIT_SUCCESS
        : EXIT_FAILURE;
}
