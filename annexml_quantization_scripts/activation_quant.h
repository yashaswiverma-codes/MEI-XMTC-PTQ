#pragma once

#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdint>

// ============================================================
// Symmetric activation
// ============================================================

struct QuantizedActivation {

    std::vector<int8_t> q;

    float scale;
};

// ============================================================
// Asymmetric activation
// ============================================================

struct QuantizedActivationAsym {

    std::vector<int8_t> q;

    float scale;

    int zero_point;
};

// ============================================================
// Symmetric activation quantization
// ============================================================

inline QuantizedActivation quantize_activation_sym(
    const float* x,
    int n)
{
    QuantizedActivation qa;

    float absmax = 0.0f;

    for(int i = 0; i < n; i++)
        absmax =
            std::max(
                absmax,
                std::fabs(x[i]));

    if(absmax == 0.0f)
        absmax = 1.0f;

    qa.scale =
        absmax / 127.0f;

    qa.q.resize(n);

    for(int i = 0; i < n; i++) {

        int v =
            int(std::round(
                x[i] / qa.scale));

        v =
            std::max(
                -127,
                std::min(127, v));

        qa.q[i] =
            int8_t(v);
    }

    return qa;
}

// ============================================================
// Asymmetric activation quantization
// ============================================================

inline QuantizedActivationAsym quantize_activation_asym(
    const float* x,
    int n)
{
    QuantizedActivationAsym qa;

    float xmin = x[0];

    float xmax = x[0];

    for(int i = 1; i < n; i++) {

        xmin =
            std::min(
                xmin,
                x[i]);

        xmax =
            std::max(
                xmax,
                x[i]);
    }

    float range =
        xmax - xmin;

    if(range == 0.0f)
        range = 1.0f;

    qa.scale =
        range / 255.0f;

    qa.zero_point =
        int(std::round(
            -xmin / qa.scale));

    qa.q.resize(n);

    for(int i = 0; i < n; i++) {

        int v =
            int(std::round(
                x[i] / qa.scale));

        v += qa.zero_point;

        v =
            std::max(
                0,
                std::min(255, v));

        qa.q[i] =
            int8_t(v);
    }

    return qa;
}
