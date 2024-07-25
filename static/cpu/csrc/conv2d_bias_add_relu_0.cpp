
#include <cstdio>
#include <stdexcept>
#include "xnnpack.h"
#include "../include/logging.h"
void conv2d_bias_add_relu_0 (
    void* in_ptr,
    void* weight_ptr,
    void* out_ptr,

    void* bias_ptr,
    void* res_ptr,

    uint8_t* workspace,
    int64_t* batch,
    int64_t* out_ch,
    int64_t* in_ch,
    int64_t* kernel_h,
    int64_t* kernel_w,
    int64_t* in_h,
    int64_t* in_w,
    int64_t* out_batch,
    int64_t* out_h,
    int64_t* out_w,
    int strideh,
    int dilationh,
    int padh,
    int stridew,
    int dilationw,
    int padw
  ) {

  
  int64_t NI = *batch;
  int64_t HI = *in_h;
  int64_t WI = *in_w;
  int64_t CI = *in_ch;
  int64_t CO = *out_ch;
  int64_t KH = *kernel_h;
  int64_t KW = *kernel_w;
  int64_t SH = strideh;
  int64_t SW = stridew;
  int64_t DH = dilationh;
  int64_t DW = dilationw;
  int64_t PH = padh;
  int64_t PW = padw;
  int64_t KHEff = (KH - 1) * DH + 1;
  int64_t KWEff = (KW - 1) * DW + 1;
  int64_t NO = NI;
  int64_t HO = (HI + PH + PH - KHEff) / SH + 1;
  int64_t WO = (WI + PW + PW - KWEff) / SW + 1;
  *out_batch = NO;
  *out_h = HO;
  *out_w = WO;
  *out_ch = CO;

  int i32_batch = *batch;
  int i32_in_h = *in_h;
  int i32_in_w = *in_w;
  int i32_in_ch = *in_ch;
  int i32_out_ch = *out_ch;
  int i32_kernel_h = *kernel_h;
  int i32_kernel_w = *kernel_w;
  int i32_out_batch = *out_batch;
  int i32_out_h = *out_h;
  int i32_out_w = *out_w;
  xnn_operator_t op_conv = nullptr;
  const xnn_status status_init = xnn_initialize(nullptr);
  CHECK_EQ(status_init, xnn_status_success);
  // for conv2d_bias
  const xnn_status status = xnn_create_convolution2d_nhwc_f32(
    PH, PW, PH, PW, i32_kernel_h, i32_kernel_w,
    SH, SW, DH, DW, 1, CI,
    CO, 1 * CI, 1 * CO, (float*)(weight_ptr), (float*)(bias_ptr),
    0, std::numeric_limits<float>::infinity(),
    /*flags=*/0, nullptr, nullptr, &op_conv);
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op_conv(op_conv, xnn_delete_operator);

  CHECK_EQ(status, xnn_status_success);
  CHECK_NE(op_conv, nullptr);
  size_t workspace_size = SIZE_MAX;
  size_t workspace_alignment = SIZE_MAX;
  CHECK_EQ(
    xnn_reshape_convolution2d_nhwc_f32(
      op_conv, i32_batch, i32_in_h, i32_in_w,
      &workspace_size, &workspace_alignment,
      /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
      /*threadpool=*/nullptr), xnn_status_success);
  CHECK_EQ(workspace_size, 0);
  CHECK_EQ(workspace_alignment, 1);
  CHECK_EQ(xnn_setup_convolution2d_nhwc_f32(
      op_conv, 
      /*workspace=*/nullptr, 
      (float*)(in_ptr), 
      (float*)(out_ptr)), xnn_status_success);

  CHECK_EQ(xnn_run_operator(op_conv, /*threadpool=*/nullptr), xnn_status_success);
  // for add
  xnn_operator_t op_add = nullptr;
  CHECK_EQ(xnn_status_success, xnn_create_add_nd_f32(0, std::numeric_limits<float>::infinity(), 0, &op_add));
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op_add(op_add, xnn_delete_operator);

  const size_t a_shape[] = { (size_t)i32_out_batch, (size_t)i32_out_h, (size_t)i32_out_w, (size_t)i32_out_ch};
  const size_t b_shape[] = { (size_t)i32_out_batch, (size_t)i32_out_h, (size_t)i32_out_w, (size_t)i32_out_ch};
  CHECK_EQ(
    xnn_status_success, xnn_reshape_add_nd_f32(
                          op_add, 4, a_shape, 4, b_shape,
                          /*threadpool=*/nullptr));

  CHECK_EQ(
    xnn_status_success, xnn_setup_add_nd_f32(op_add, (float*)(res_ptr), (float*)(out_ptr), (float*)(out_ptr)));

  CHECK_EQ(xnn_status_success, xnn_run_operator(op_add, /*threadpool=*/nullptr));

  return;
}