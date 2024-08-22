
#include <cstdio>
#include <stdexcept>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>
#include "xnnpack.h"
#include "logging.h"
#include <functional>
#include <random>
#include <stdint.h>
#include <cstddef> // For size_t
#include <cstring> // For memcpy


void avg_pool2d_54(
    void* in_ptr,
    void* out_ptr,
    int64_t* batch,
    int64_t* in_ch,
    int64_t* in_h,
    int64_t* in_w,
    int64_t* out_batch,
    int64_t* out_h,
    int64_t* out_w,
    int64_t kernel_h,
    int64_t kernel_w,
    int64_t stride,
    int64_t pad,
    pthreadpool* pthreadpool_
    ) {
  
  int64_t NI = *batch;
  int64_t HI = *in_h;
  int64_t WI = *in_w;
  int64_t CI = *in_ch;
  int64_t CO = *in_ch;
  int64_t KH = kernel_h;
  int64_t KW = kernel_w;
  int64_t SH = stride;
  int64_t SW = stride;
  int64_t PH = pad;
  int64_t PW = pad;
  int64_t NO = NI;
  int64_t HO = (HI + PH + PH - KH) / SH + 1;
  int64_t WO = (WI + PW + PW - KW) / SW + 1;
  *out_batch = NO;
  *out_h = HO;
  *out_w = WO;
  
  
    xnn_operator_t op_avg = nullptr;
    const xnn_status status = xnn_create_average_pooling2d_nhwc_f32(
      PH, PW, PH, PW, KH, KW, SH, SW, 
      -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(), 
      /*flags=*/0, &op_avg);

    CHECK_EQ(xnn_status_success, status);
    CHECK_NE(nullptr, op_avg);
    std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op(op_avg, xnn_delete_operator);

    size_t w_size = 0;
    size_t w_align = 0;
    CHECK_EQ(
      xnn_status_success,
      xnn_reshape_average_pooling2d_nhwc_f32(
        op_avg, NI, HI, WI,
        CI, /*input_pixel_stride=*/CI, /*output_pixel_stride=*/CO,
        &w_size, &w_align,
        /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
        /*threadpool=*/pthreadpool_));
    CHECK_LE(w_align, 16);
    std::vector<char> workspace_vector(w_size + w_align + 16);
    void* maybe_aligned_workspace = workspace_vector.data();
    void* aligned_workspace =     (void*)((intptr_t)maybe_aligned_workspace + w_align - (intptr_t)maybe_aligned_workspace % w_align);
    CHECK_EQ(xnn_status_success, xnn_setup_average_pooling2d_nhwc_f32(op_avg, aligned_workspace, (float*)(in_ptr), (float*)(out_ptr)));
    CHECK_EQ(xnn_status_success, xnn_run_operator(op_avg, /*threadpool=*/pthreadpool_));
  return;
  throw std::runtime_error(
      "Unsupported workload for this conv2d specialization."
  );
}