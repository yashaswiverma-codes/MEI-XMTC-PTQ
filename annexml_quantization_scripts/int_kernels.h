#pragma once

#include <immintrin.h>
#include <cstdint>
#include <cmath>
#include <algorithm>

// ============================================================
// INT8 symmetric dot product
// ============================================================

inline int32_t dot_int8_sym(
    const int8_t* w,
    const int8_t* x,
    int n)
{
    int32_t acc = 0;

    for(int i = 0; i < n; i++)
        acc +=
            int32_t(w[i]) *
            int32_t(x[i]);

    return acc;
}

// ============================================================
// INT8 asymmetric dot product
// ============================================================

inline int32_t dot_int8_asym(
    const int8_t* w,
    const int8_t* x,
    int n,
    int zw,
    int zx)
{
    int32_t acc = 0;

    for(int i = 0; i < n; i++)
    {
        acc +=
            (int32_t(w[i]) - zw) *
            (int32_t(x[i]) - zx);
    }

    return acc;
}

// ============================================================
// Group-wise INT8 dot
// ============================================================

inline float dot_group_int8(
    const int8_t* w,
    const int8_t* x,
    const float* sw,
    const float* sx,
    int n,
    int group_size)
{
    float result = 0.0f;

    int ng =
        (n + group_size - 1)
        / group_size;

    for(int g = 0; g < ng; g++)
    {
        int start =
            g * group_size;

        int end =
            std::min(
                start + group_size,
                n);

        int32_t acc = 0;

        for(int i = start;
            i < end;
            i++)
        {
            acc +=
                int32_t(w[i]) *
                int32_t(x[i]);
        }

        result +=
            sw[g] *
            sx[g] *
            float(acc);
    }

    return result;
}

// ============================================================
// INT4 unpack helpers
// ============================================================

inline int8_t unpack_low_int4(
    uint8_t v)
{
    int8_t x =
        v & 0xF;

    if(x >= 8)
        x -= 16;

    return x;
}

inline int8_t unpack_high_int4(
    uint8_t v)
{
    int8_t x =
        (v >> 4) & 0xF;

    if(x >= 8)
        x -= 16;

    return x;
}

// ============================================================
// TRUE packed INT4 dot
// ============================================================

inline int32_t dot_int4(
    const uint8_t* w,
    const int8_t* x,
    int n)
{
    int32_t acc = 0;

    for(int i = 0;
        i < n;
        i += 2)
    {
        uint8_t packed =
            w[i / 2];

        int8_t w0 =
            unpack_low_int4(
                packed);

        int8_t w1 =
            unpack_high_int4(
                packed);

        acc +=
            int32_t(w0) *
            int32_t(x[i]);

        if(i + 1 < n)
        {
            acc +=
                int32_t(w1) *
                int32_t(x[i + 1]);
        }
    }

    return acc;
}
