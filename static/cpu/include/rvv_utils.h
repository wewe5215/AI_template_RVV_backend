#include <iomanip>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <riscv_vector.h>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <ostream>
#include <string>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <random>
#include <vector>
#include <cstring>
#include <bitset>
#include <time.h>
#include <fstream>
#include <assert.h>
#include <cfloat>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#if defined(__has_builtin)
  #define COMPILER_HAS_BUILTIN(builtin) __has_builtin(builtin)
#else
  #define COMPILER_HAS_BUILTIN(builtin) 0
#endif

#if COMPILER_HAS_BUILTIN(__builtin_unpredictable)
  #define UNPREDICTABLE(condition) (__builtin_unpredictable(!!(condition)))
#elif defined(__GNUC__) && (__GNUC__ >= 9) && !defined(__INTEL_COMPILER)
  #define UNPREDICTABLE(condition) (__builtin_expect_with_probability(!!(condition), 0, 0.5))
#else
  #define UNPREDICTABLE(condition) (!!(condition))
#endif

#if defined(__GNUC__)
  #define LIKELY(condition) (__builtin_expect(!!(condition), 1))
  #define UNLIKELY(condition) (__builtin_expect(!!(condition), 0))
#else
  #define LIKELY(condition) (!!(condition))
  #define UNLIKELY(condition) (!!(condition))
#endif
static size_t min(size_t a, size_t b) {
  return UNPREDICTABLE(b < a) ? b : a;
}
static inline int zero_max(int x) {
    return x & ~(x >> 31);
}
 static size_t divide_round_up(size_t n, size_t q) {
  return UNPREDICTABLE(n % q == 0) ? n / q : n / q + 1;
}

 static size_t round_up(size_t n, size_t q) {
  return divide_round_up(n, q) * q;
}
typedef void (*f32_gemm_input_T_N_M_pruning)(
    size_t mr/*mr_block_size*/,
    size_t nc/*nr_block_size*/,
    size_t kc/*in_ch << 2*/,
    const float*  a,
    size_t w_stride/*in_ch << 2*/,
    const float*  w,
    const float*  bias,
    float*  c,
    size_t cm_stride/*batch_size * height * width << 2*/,
    size_t cn_stride/*nr << 2*/,
    uint16_t* indice,
    size_t nr);

struct function_context {
    float* input;
    float* bias;
    float* weight;
    float* pruned_weight;
    float* output;
    size_t input_channel;
    size_t output_channel;
    size_t output_height;
    size_t output_width;
    uint32_t mr;
    uint32_t nr;
    uint32_t* im2col_packing;
    uint16_t* indice;
    f32_gemm_input_T_N_M_pruning microkernel;
    const size_t a_stride;
    const size_t cm_stride;
    const size_t cn_stride;
    const size_t k_scaled;
    const size_t w_stride;
};

void conv2d_columnwise_pruning_vector(function_context* context, \
    size_t mr_block_start, size_t nr_block_start, size_t mr_block_size, size_t nr_block_size){
    uint32_t nr = context->nr;
    uint32_t w_stride = context->w_stride;
    context->microkernel(
        mr_block_size,
        nr_block_size,
        context->k_scaled, // group_input_channels << log2(float)
        (const float*) ((uintptr_t) context->im2col_packing + nr_block_start * context->a_stride),
        w_stride, // group_input_channels) << log2(4) = group_input_channels * 4
        (const float*) ((uintptr_t) context->pruned_weight + mr_block_start * context->w_stride),
        (const float*)((uintptr_t) context->bias + (mr_block_start << 2)),
        (float*) ((uintptr_t) context->output + mr_block_start * context->cm_stride + (nr_block_start << 2/*=log2_output_element_size*/)),
        context->cm_stride, // group_output_channels << log2_output_element_size
        context->cn_stride, // nr << log2_output_element_size
        (uint16_t*)((uintptr_t) context->indice + (mr_block_start / context->mr) * (w_stride >> 1)),
        nr
    );
}

typedef void (*xnn_pack_transpose_x4v_input_T)(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  uint32_t* weights,
  const void* scale,
  uint32_t* packed_weights,
  size_t extra_bytes,
  const void* params);
void xnn_x32_pack_transpose_ukernel_x4v__rvv_u8(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  uint32_t* weights,
  const void* scale,
  uint32_t* packed_weights,
  size_t extra_bytes,
  const void* params);

typedef void (*xnn_pack_f32_with_im2col_input_T_nr)(
  uint32_t batch_size,
  const size_t input_height,
  const size_t input_width,
  size_t group_input_channels,
  const int output_height,
  const int output_width,
  const size_t kernel_height,
  const size_t kernel_width,
  const size_t stride_height,
  const size_t stride_width,
  const int dilation_height,
  const int dilation_width,
  const int input_padding_top,
  const int input_padding_left,
  uint32_t* input,
  uint32_t* output,
  const int nr);

void im2col_local_avgpool_s2_d1_p0_with_pack_1x(
  uint32_t batch_size,
  const size_t input_height,
  const size_t input_width,
  size_t group_input_channels,
  const int output_height,
  const int output_width,
  const size_t kernel_height,
  const size_t kernel_width,
  const size_t stride_height,
  const size_t stride_width,
  const int dilation_height,
  const int dilation_width,
  const int input_padding_top,
  const int input_padding_left,
  uint32_t* input,
  uint32_t* output,
  const int nr);

void im2col_local_avgpool_s2_d1_p0_with_pack_2x(
  uint32_t batch_size,
  const size_t input_height,
  const size_t input_width,
  size_t group_input_channels,
  const int output_height,
  const int output_width,
  const size_t kernel_height,
  const size_t kernel_width,
  const size_t stride_height,
  const size_t stride_width,
  const int dilation_height,
  const int dilation_width,
  const int input_padding_top,
  const int input_padding_left,
  uint32_t* input,
  uint32_t* output,
  const int nr);

void im2col_local_avgpool_s2_d1_p0_with_pack_4x(
  uint32_t batch_size,
  const size_t input_height,
  const size_t input_width,
  size_t group_input_channels,
  const int output_height,
  const int output_width,
  const size_t kernel_height,
  const size_t kernel_width,
  const size_t stride_height,
  const size_t stride_width,
  const int dilation_height,
  const int dilation_width,
  const int input_padding_top,
  const int input_padding_left,
  uint32_t* input,
  uint32_t* output,
  const int nr);
  
void xnn_x32_packa_in_T_gemm_im2col_s2_d1_p0_x1v(
  uint32_t batch_size,
  const size_t input_height,
  const size_t input_width,
  size_t group_input_channels,
  const int output_height,
  const int output_width,
  const size_t kernel_height,
  const size_t kernel_width,
  const size_t stride_height,
  const size_t stride_width,
  const int dilation_height,
  const int dilation_width,
  const int input_padding_top,
  const int input_padding_left,
  uint32_t* input,
  uint32_t* output,
  const int nr);

void xnn_x32_packa_in_T_gemm_im2col_s2_d1_p0_x2v(
  uint32_t batch_size,
  const size_t input_height,
  const size_t input_width,
  size_t group_input_channels,
  const int output_height,
  const int output_width,
  const size_t kernel_height,
  const size_t kernel_width,
  const size_t stride_height,
  const size_t stride_width,
  const int dilation_height,
  const int dilation_width,
  const int input_padding_top,
  const int input_padding_left,
  uint32_t* input,
  uint32_t* output,
  const int nr);

void xnn_x32_packa_in_T_gemm_im2col_s2_d1_p0_x4v(
  uint32_t batch_size,
  const size_t input_height,
  const size_t input_width,
  size_t group_input_channels,
  const int output_height,
  const int output_width,
  const size_t kernel_height,
  const size_t kernel_width,
  const size_t stride_height,
  const size_t stride_width,
  const int dilation_height,
  const int dilation_width,
  const int input_padding_top,
  const int input_padding_left,
  uint32_t* input,
  uint32_t* output,
  const int nr);

typedef void (*xnn_pack_f32_with_im2col_input_T)(
  uint32_t batch_size,
  const size_t input_height,
  const size_t input_width,
  size_t group_input_channels,
  const int output_height,
  const int output_width,
  const size_t kernel_height,
  const size_t kernel_width,
  const size_t stride_height,
  const size_t stride_width,
  const int dilation_height,
  const int dilation_width,
  const int input_padding_top,
  const int input_padding_left,
  uint32_t* input,
  uint32_t* output);

// nr = vsetvlmax_e32m1()
void xnn_x32_packa_in_T_im2col_pooling_s2_d1_x2v(
  uint32_t batch_size,
  const size_t input_height,
  const size_t input_width,
  size_t group_input_channels,
  const int output_height,
  const int output_width,
  const size_t kernel_height,
  const size_t kernel_width,
  const size_t stride_height,
  const size_t stride_width,
  const int dilation_height,
  const int dilation_width,
  const int input_padding_top,
  const int input_padding_left,
  uint32_t* input,
  uint32_t* output);

void xnn_x32_packa_in_T_gemm_im2col_s2_d1_x1v(
  uint32_t batch_size,
  const size_t input_height,
  const size_t input_width,
  size_t group_input_channels,
  const int output_height,
  const int output_width,
  const size_t kernel_height,
  const size_t kernel_width,
  const size_t stride_height,
  const size_t stride_width,
  const int dilation_height,
  const int dilation_width,
  const int input_padding_top,
  const int input_padding_left,
  uint32_t* input,
  uint32_t* output);

void xnn_x32_packa_in_T_gemm_im2col_s1_d1_1x1v(
  uint32_t batch_size,
  const size_t input_height,
  const size_t input_width,
  size_t group_input_channels,
  const int output_height,
  const int output_width,
  const size_t kernel_height,
  const size_t kernel_width,
  const size_t stride_height,
  const size_t stride_width,
  const int dilation_height,
  const int dilation_width,
  const int input_padding_top,
  const int input_padding_left,
  uint32_t* input,
  uint32_t* output);

// nr = vsetvlmax_e32m2()
void xnn_x32_packa_in_T_gemm_im2col_s2_d1_x2v(
  uint32_t batch_size,
  const size_t input_height,
  const size_t input_width,
  size_t group_input_channels,
  const int output_height,
  const int output_width,
  const size_t kernel_height,
  const size_t kernel_width,
  const size_t stride_height,
  const size_t stride_width,
  const int dilation_height,
  const int dilation_width,
  const int input_padding_top,
  const int input_padding_left,
  uint32_t* input,
  uint32_t* output);

void xnn_x32_packa_in_T_gemm_im2col_s1_d1_1x2v(
  uint32_t batch_size,
  const size_t input_height,
  const size_t input_width,
  size_t group_input_channels,
  const int output_height,
  const int output_width,
  const size_t kernel_height,
  const size_t kernel_width,
  const size_t stride_height,
  const size_t stride_width,
  const int dilation_height,
  const int dilation_width,
  const int input_padding_top,
  const int input_padding_left,
  uint32_t* input,
  uint32_t* output);

void xnn_x32_packa_in_T_gemm_im2col_s1_d1_2x2v(
  uint32_t batch_size,
  const size_t input_height,
  const size_t input_width,
  size_t group_input_channels,
  const int output_height,
  const int output_width,
  const size_t kernel_height,
  const size_t kernel_width,
  const size_t stride_height,
  const size_t stride_width,
  const int dilation_height,
  const int dilation_width,
  const int input_padding_top,
  const int input_padding_left,
  uint32_t* input,
  uint32_t* output);

// nr = vsetvlmax_e32m4()
void xnn_x32_packa_in_T_gemm_im2col_s2_d1_1x4v(
  uint32_t batch_size,
  const size_t input_height,
  const size_t input_width,
  size_t group_input_channels,
  const int output_height,
  const int output_width,
  const size_t kernel_height,
  const size_t kernel_width,
  const size_t stride_height,
  const size_t stride_width,
  const int dilation_height,
  const int dilation_width,
  const int input_padding_top,
  const int input_padding_left,
  uint32_t* input,
  uint32_t* output);

void xnn_x32_packa_in_T_gemm_im2col_s2_d1_2x4v(
  uint32_t batch_size,
  const size_t input_height,
  const size_t input_width,
  size_t group_input_channels,
  const int output_height,
  const int output_width,
  const size_t kernel_height,
  const size_t kernel_width,
  const size_t stride_height,
  const size_t stride_width,
  const int dilation_height,
  const int dilation_width,
  const int input_padding_top,
  const int input_padding_left,
  uint32_t* input,
  uint32_t* output);

void xnn_x32_packa_in_T_gemm_im2col_s2_d1(
  uint32_t batch_size,
  const size_t input_height,
  const size_t input_width,
  size_t group_input_channels,
  const int output_height,
  const int output_width,
  const size_t kernel_height,
  const size_t kernel_width,
  const size_t stride_height,
  const size_t stride_width,
  const int dilation_height,
  const int dilation_width,
  const int input_padding_top,
  const int input_padding_left,
  uint32_t* input,
  uint32_t* output);

void xnn_x32_packa_in_T_gemm_im2col_s1_d1_1x4v(
  uint32_t batch_size,
  const size_t input_height,
  const size_t input_width,
  size_t group_input_channels,
  const int output_height,
  const int output_width,
  const size_t kernel_height,
  const size_t kernel_width,
  const size_t stride_height,
  const size_t stride_width,
  const int dilation_height,
  const int dilation_width,
  const int input_padding_top,
  const int input_padding_left,
  uint32_t* input,
  uint32_t* output);

void xnn_x32_packa_in_T_gemm_im2col_s1_d1_2x4v(
  uint32_t batch_size,
  const size_t input_height,
  const size_t input_width,
  size_t group_input_channels,
  const int output_height,
  const int output_width,
  const size_t kernel_height,
  const size_t kernel_width,
  const size_t stride_height,
  const size_t stride_width,
  const int dilation_height,
  const int dilation_width,
  const int input_padding_top,
  const int input_padding_left,
  uint32_t* input,
  uint32_t* output);

void xnn_x32_packa_in_T_gemm_im2col_s1_d1_4x4v(
  uint32_t batch_size,
  const size_t input_height,
  const size_t input_width,
  size_t group_input_channels,
  const int output_height,
  const int output_width,
  const size_t kernel_height,
  const size_t kernel_width,
  const size_t stride_height,
  const size_t stride_width,
  const int dilation_height,
  const int dilation_width,
  const int input_padding_top,
  const int input_padding_left,
  uint32_t* input,
  uint32_t* output);

// implemented by memcpy
void xnn_x32_packa_in_T_gemm_im2col_s1_d1_4v(
  uint32_t batch_size,
  const size_t input_height,
  const size_t input_width,
  size_t group_input_channels,
  const int output_height,
  const int output_width,
  const size_t kernel_height,
  const size_t kernel_width,
  const size_t stride_height,
  const size_t stride_width,
  const int dilation_height,
  const int dilation_width,
  const int input_padding_top,
  const int input_padding_left,
  uint32_t* input,
  uint32_t* output);

// nr = vsetvlmax_e32m8()
void xnn_x32_packa_in_T_gemm_im2col_s2_d1_x8v(
  uint32_t batch_size,
  const size_t input_height,
  const size_t input_width,
  size_t group_input_channels,
  const int output_height,
  const int output_width,
  const size_t kernel_height,
  const size_t kernel_width,
  const size_t stride_height,
  const size_t stride_width,
  const int dilation_height,
  const int dilation_width,
  const int input_padding_top,
  const int input_padding_left,
  uint32_t* input,
  uint32_t* output);

void xnn_x32_packa_in_T_gemm_im2col_s1_d1_1x8v(
  uint32_t batch_size,
  const size_t input_height,
  const size_t input_width,
  size_t group_input_channels,
  const int output_height,
  const int output_width,
  const size_t kernel_height,
  const size_t kernel_width,
  const size_t stride_height,
  const size_t stride_width,
  const int dilation_height,
  const int dilation_width,
  const int input_padding_top,
  const int input_padding_left,
  uint32_t* input,
  uint32_t* output);

void xnn_x32_packa_in_T_gemm_im2col_s1_d1_2x8v(
  uint32_t batch_size,
  const size_t input_height,
  const size_t input_width,
  size_t group_input_channels,
  const int output_height,
  const int output_width,
  const size_t kernel_height,
  const size_t kernel_width,
  const size_t stride_height,
  const size_t stride_width,
  const int dilation_height,
  const int dilation_width,
  const int input_padding_top,
  const int input_padding_left,
  uint32_t* input,
  uint32_t* output);

void xnn_x32_packa_in_T_gemm_im2col_s1_d1_4x8v(
  uint32_t batch_size,
  const size_t input_height,
  const size_t input_width,
  size_t group_input_channels,
  const int output_height,
  const int output_width,
  const size_t kernel_height,
  const size_t kernel_width,
  const size_t stride_height,
  const size_t stride_width,
  const int dilation_height,
  const int dilation_width,
  const int input_padding_top,
  const int input_padding_left,
  uint32_t* input,
  uint32_t* output);

void xnn_x32_packa_in_T_gemm_im2col_s1_d1_8x8v(
  uint32_t batch_size,
  const size_t input_height,
  const size_t input_width,
  size_t group_input_channels,
  const int output_height,
  const int output_width,
  const size_t kernel_height,
  const size_t kernel_width,
  const size_t stride_height,
  const size_t stride_width,
  const int dilation_height,
  const int dilation_width,
  const int input_padding_top,
  const int input_padding_left,
  uint32_t* input,
  uint32_t* output);