#include "rvv_utils.h"

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
  const void* params)
{
  assert(g != 0);
  assert(nc != 0);
  assert(kc != 0);
  assert(nr == __riscv_vsetvlmax_e32m4());
  assert(kr == 1);
  assert(sr == 1);
  assert(weights != NULL);
  assert(packed_weights != NULL);

  uint32_t* out = packed_weights;
  size_t kc_bstride = kc * 4;

  do {
    const uint32_t* w0 = weights;
    size_t n = nc;
    // NC main loop: process multiple of NR
    for (;n >= nr; n -= nr) {
      size_t vlmax = __riscv_vsetvlmax_e32m4();

      uint32_t* out0 = out;
      size_t k = kc;
      // vlsseg8, LMUL must <= 1
      vlmax = __riscv_vsetvlmax_e32m1();
      // Pack 8 x nr weights
      for (; k >= 8; k -= 8) {
        uint32_t* out1 = out0 + nr;
        uint32_t* out2 = out1 + nr;
        uint32_t* out3 = out2 + nr;
        uint32_t* out4 = out3 + nr;
        uint32_t* out5 = out4 + nr;
        uint32_t* out6 = out5 + nr;
        uint32_t* out7 = out6 + nr;
        // When vlsseg8, LMUL is contraint to 1. We need to use multiple of load & store.
        const uint32_t* w_ptr = w0;
        size_t remaining_n = nr;
        do {
          vuint32m1x8_t v_w_m1x8 = __riscv_vlsseg8e32_v_u32m1x8(w_ptr, kc_bstride, vlmax);
          w_ptr += kc * vlmax;
          vuint32m1_t v_w0 = __riscv_vget_v_u32m1x8_u32m1(v_w_m1x8, 0);
          __riscv_vse32_v_u32m1(out0, v_w0, vlmax); out0 += vlmax;
          vuint32m1_t v_w1 = __riscv_vget_v_u32m1x8_u32m1(v_w_m1x8, 1);
          __riscv_vse32_v_u32m1(out1, v_w1, vlmax); out1 += vlmax;
          vuint32m1_t v_w2 = __riscv_vget_v_u32m1x8_u32m1(v_w_m1x8, 2);
          __riscv_vse32_v_u32m1(out2, v_w2, vlmax); out2 += vlmax;
          vuint32m1_t v_w3 = __riscv_vget_v_u32m1x8_u32m1(v_w_m1x8, 3);
          __riscv_vse32_v_u32m1(out3, v_w3, vlmax); out3 += vlmax;
          vuint32m1_t v_w4 = __riscv_vget_v_u32m1x8_u32m1(v_w_m1x8, 4);
          __riscv_vse32_v_u32m1(out4, v_w4, vlmax); out4 += vlmax;
          vuint32m1_t v_w5 = __riscv_vget_v_u32m1x8_u32m1(v_w_m1x8, 5);
          __riscv_vse32_v_u32m1(out5, v_w5, vlmax); out5 += vlmax;
          vuint32m1_t v_w6 = __riscv_vget_v_u32m1x8_u32m1(v_w_m1x8, 6);
          __riscv_vse32_v_u32m1(out6, v_w6, vlmax); out6 += vlmax;
          vuint32m1_t v_w7 = __riscv_vget_v_u32m1x8_u32m1(v_w_m1x8, 7);
          __riscv_vse32_v_u32m1(out7, v_w7, vlmax); out7 += vlmax;
          remaining_n -= vlmax;
        } while(remaining_n > 0);
        out0 = out7;
        w0 += 8;
      }
      // vlsseg4, LMUL must <= 2
      vlmax = __riscv_vsetvlmax_e32m2();
      // Pack 4 x nr weights
      for (; k >= 4; k -= 4) {
        uint32_t* out1 = out0 + nr;
        uint32_t* out2 = out1 + nr;
        uint32_t* out3 = out2 + nr;
        // When vlsseg4, LMUL is contraint to 2. We need to use multiple of load & store.
        const uint32_t* w_ptr = w0;
        size_t remaining_n = nr;
        do {
          vuint32m2x4_t v_w_m2x4 = __riscv_vlsseg4e32_v_u32m2x4(w_ptr, kc_bstride, vlmax);
          w_ptr += kc * vlmax;
          vuint32m2_t v_w0 = __riscv_vget_v_u32m2x4_u32m2(v_w_m2x4, 0);
          __riscv_vse32_v_u32m2(out0, v_w0, vlmax); out0 += vlmax;
          vuint32m2_t v_w1 = __riscv_vget_v_u32m2x4_u32m2(v_w_m2x4, 1);
          __riscv_vse32_v_u32m2(out1, v_w1, vlmax); out1 += vlmax;
          vuint32m2_t v_w2 = __riscv_vget_v_u32m2x4_u32m2(v_w_m2x4, 2);
          __riscv_vse32_v_u32m2(out2, v_w2, vlmax); out2 += vlmax;
          vuint32m2_t v_w3 = __riscv_vget_v_u32m2x4_u32m2(v_w_m2x4, 3);
          __riscv_vse32_v_u32m2(out3, v_w3, vlmax); out3 += vlmax;
          remaining_n -= vlmax;
        } while(remaining_n > 0);
        out0 = out3;
        w0 += 4;
      }
      vlmax = __riscv_vsetvlmax_e32m4();
      // Pack nr weights
      for (; k >= 1; k -= 1) {
        vuint32m4_t v_w = __riscv_vlse32_v_u32m4(w0, kc_bstride, vlmax);
        __riscv_vse32_v_u32m4(out0, v_w, vlmax);
        out0 += vlmax;
        w0 += 1;
      }
      out = (uint32_t*) ((uintptr_t) out0 + extra_bytes);
      w0 += (nr - 1) * kc;
    }
    // NC remainder: process n < NR
    if (n > 0) {
      size_t vl = __riscv_vsetvl_e32m4(n);

      size_t vlmax;
      uint32_t* out0 = out;
      size_t k = kc;
      // vlsseg8, LMUL must <= 1
      vlmax = __riscv_vsetvlmax_e32m1();
      // Pack 8 x n weights
      for (; k >= 8; k -= 8) {
        uint32_t* out1 = out0 + nr;
        uint32_t* out2 = out1 + nr;
        uint32_t* out3 = out2 + nr;
        uint32_t* out4 = out3 + nr;
        uint32_t* out5 = out4 + nr;
        uint32_t* out6 = out5 + nr;
        uint32_t* out7 = out6 + nr;
        // When vlsseg8, LMUL is contraint to 1. We need to use multiple of load & store.
        const uint32_t* w_ptr = w0;
        unsigned char remaining_blocks = 4;
        size_t remaining_n = n;
        do {
          size_t vl;
          if XNN_LIKELY(remaining_n >= vlmax) {
            vl = vlmax;
          } else {
            vl = __riscv_vsetvl_e32m1(remaining_n);
          }
          vuint32m1x8_t v_w_m1x8 = __riscv_vlsseg8e32_v_u32m1x8(w_ptr, kc_bstride, vl);
          w_ptr += kc * vl;
          vuint32m1_t v_w0 = __riscv_vget_v_u32m1x8_u32m1(v_w_m1x8, 0);
          __riscv_vse32_v_u32m1(out0, v_w0, vl); out0 += vlmax;
          vuint32m1_t v_w1 = __riscv_vget_v_u32m1x8_u32m1(v_w_m1x8, 1);
          __riscv_vse32_v_u32m1(out1, v_w1, vl); out1 += vlmax;
          vuint32m1_t v_w2 = __riscv_vget_v_u32m1x8_u32m1(v_w_m1x8, 2);
          __riscv_vse32_v_u32m1(out2, v_w2, vl); out2 += vlmax;
          vuint32m1_t v_w3 = __riscv_vget_v_u32m1x8_u32m1(v_w_m1x8, 3);
          __riscv_vse32_v_u32m1(out3, v_w3, vl); out3 += vlmax;
          vuint32m1_t v_w4 = __riscv_vget_v_u32m1x8_u32m1(v_w_m1x8, 4);
          __riscv_vse32_v_u32m1(out4, v_w4, vl); out4 += vlmax;
          vuint32m1_t v_w5 = __riscv_vget_v_u32m1x8_u32m1(v_w_m1x8, 5);
          __riscv_vse32_v_u32m1(out5, v_w5, vl); out5 += vlmax;
          vuint32m1_t v_w6 = __riscv_vget_v_u32m1x8_u32m1(v_w_m1x8, 6);
          __riscv_vse32_v_u32m1(out6, v_w6, vl); out6 += vlmax;
          vuint32m1_t v_w7 = __riscv_vget_v_u32m1x8_u32m1(v_w_m1x8, 7);
          __riscv_vse32_v_u32m1(out7, v_w7, vl); out7 += vlmax;
          remaining_n -= vl;
          remaining_blocks--;
        } while(remaining_n > 0);
        out0 = out7 + remaining_blocks * vlmax;
        w0 += 8;
      }
      // vlsseg4, LMUL must <= 2
      vlmax = __riscv_vsetvlmax_e32m2();
      // Pack 4 x n weights
      for (; k >= 4; k -= 4) {
        uint32_t* out1 = out0 + nr;
        uint32_t* out2 = out1 + nr;
        uint32_t* out3 = out2 + nr;
        // When vlsseg4, LMUL is contraint to 2. We need to use multiple of load & store.
        const uint32_t* w_ptr = w0;
        unsigned char remaining_blocks = 2;
        size_t remaining_n = n;
        do {
          size_t vl;
          if XNN_LIKELY(remaining_n >= vlmax) {
            vl = vlmax;
          } else {
            vl = __riscv_vsetvl_e32m2(remaining_n);
          }
          vuint32m2x4_t v_w_m2x4 = __riscv_vlsseg4e32_v_u32m2x4(w_ptr, kc_bstride, vl);
          w_ptr += kc * vl;
          vuint32m2_t v_w0 = __riscv_vget_v_u32m2x4_u32m2(v_w_m2x4, 0);
          __riscv_vse32_v_u32m2(out0, v_w0, vl); 
          out0 += vlmax;
          vuint32m2_t v_w1 = __riscv_vget_v_u32m2x4_u32m2(v_w_m2x4, 1);
          __riscv_vse32_v_u32m2(out1, v_w1, vl); 
          out1 += vlmax;
          vuint32m2_t v_w2 = __riscv_vget_v_u32m2x4_u32m2(v_w_m2x4, 2);
          __riscv_vse32_v_u32m2(out2, v_w2, vl); 
          out2 += vlmax;
          vuint32m2_t v_w3 = __riscv_vget_v_u32m2x4_u32m2(v_w_m2x4, 3);
          __riscv_vse32_v_u32m2(out3, v_w3, vl); 
          out3 += vlmax;
          remaining_n -= vl;
          remaining_blocks--;
        } while(remaining_n > 0);
        out0 = out3 + remaining_blocks * vlmax;
        w0 += 4;
      }
      vlmax = __riscv_vsetvlmax_e32m4();
      vl = __riscv_vsetvl_e32m4(n);
      // Pack n weights
      for (; k >= 1; k -= 1) {
        vuint32m4_t v_w = __riscv_vlse32_v_u32m4(w0, kc_bstride, vl);
        __riscv_vse32_v_u32m4(out0, v_w, vl);
        out0 += vlmax;
        w0 += 1;
      }
      out = (uint32_t*) ((uintptr_t) out0 + extra_bytes);
      w0 += (nr - 1) * kc; 
    }
    weights += nc * kc;
  } while (--g != 0);
}
void im2col_local_avgpool_s2_d1_p0_with_pack_1x(uint32_t batch_size, const size_t input_height, const size_t input_width, size_t group_input_channels, \
  const int output_height, const int output_width,
  const size_t kernel_height, const size_t kernel_width, const size_t stride_height, const size_t stride_width, \
  const int dilation_height, const int dilation_width, const int input_padding_top,const int input_padding_left, \
  uint32_t* input, uint32_t* output, const int nr){
  const size_t output_size = output_height * output_width;
  const size_t batch_output_size = group_input_channels * output_size * batch_size;
  
  const size_t input_size = input_height*input_width*batch_size;
  uint32_t* in_ptr = input;
  uint32_t* out_ptr = output;
  const int vlmax = __riscv_vsetvlmax_e32m1();
  int input_stride;
  int remainder = 0;
  int input_cursor = 0;
  int out_h, out_w, batch;
  int im2col_cur = 0;
  int batch_cur = 0;
  int output_cur = 0;
  while(output_cur < batch_output_size){
      int out_h = output_cur / output_width;
      int out_w = output_cur % output_width;
      batch_cur = output_cur / output_size;
      int base = batch_cur * output_size;
      in_ptr = input + input_cursor;
      for(int k_h = 0; k_h < kernel_height; k_h++){
          for(int k_w = 0; k_w < kernel_width; k_w++){
              uint32_t* in_ptr_now = in_ptr + k_w;
              int vl = min(vlmax, output_width - out_w);
              vuint32m1_t v_w0 = __riscv_vlse32_v_u32m1 (in_ptr_now, stride_width << 2, vl);
              __riscv_vse32_v_u32m1(out_ptr, v_w0, vl);
              out_ptr += vl;
              int cur = vl;
              uint32_t* in_ptr_now_rem = in_ptr_now + ((output_width - out_w) << 1) + input_width;
              while(cur < nr){
                  int vl_rem = min(nr - cur, output_width);
                  int vl_for_vector = -(output_cur + cur < batch_output_size) & vl_rem;
                  v_w0 = __riscv_vlse32_v_u32m1 (in_ptr_now_rem, stride_width << 2, vl_for_vector);
                  __riscv_vse32_v_u32m1(out_ptr, v_w0, vl_for_vector);
                  out_ptr += vl_rem;
                  cur += vl_rem;
                  in_ptr_now_rem += input_width << 1;
              }
          }
          in_ptr += input_width;
      }
      output_cur += nr;
      int sliding_window_row_diff = output_cur / output_width - out_h;
      input_cursor += (
          -(sliding_window_row_diff > 0) & 
              (
                  ((output_width - out_w) << 1) + input_width \
                  + (sliding_window_row_diff - 1) * (input_width << 1)
              )
          ) \
          + (((output_cur % output_width) - (-(sliding_window_row_diff == 0) & out_w)) << 1);
  }
}

void im2col_local_avgpool_s2_d1_p0_with_pack_2x(uint32_t batch_size, const size_t input_height, const size_t input_width, size_t group_input_channels, \
  const int output_height, const int output_width,
  const size_t kernel_height, const size_t kernel_width, const size_t stride_height, const size_t stride_width, \
  const int dilation_height, const int dilation_width, const int input_padding_top,const int input_padding_left, \
  uint32_t* input, uint32_t* output, const int nr){
  const size_t output_size = output_height * output_width;
  const size_t batch_output_size = group_input_channels * output_size * batch_size;
  
  const size_t input_size = input_height*input_width*batch_size;
  uint32_t* in_ptr = input;
  uint32_t* out_ptr = output;
  const int vlmax = __riscv_vsetvlmax_e32m2();
  int input_stride;
  int remainder = 0;
  int input_cursor = 0;
  int out_h, out_w, batch;
  int im2col_cur = 0;
  int batch_cur = 0;
  int output_cur = 0;
  while(output_cur < batch_output_size){
      int out_h = output_cur / output_width;
      int out_w = output_cur % output_width;
      batch_cur = output_cur / output_size;
      int base = batch_cur * output_size;
      in_ptr = input + input_cursor;
      for(int k_h = 0; k_h < kernel_height; k_h++){
          for(int k_w = 0; k_w < kernel_width; k_w++){
              uint32_t* in_ptr_now = in_ptr + k_w;
              int vl = min(vlmax, output_width - out_w);
              vuint32m2_t v_w0 = __riscv_vlse32_v_u32m2 (in_ptr_now, stride_width << 2, vl);
              __riscv_vse32_v_u32m2(out_ptr, v_w0, vl);
              out_ptr += vl;
              int cur = vl;
              uint32_t* in_ptr_now_rem = in_ptr_now + ((output_width - out_w) << 1) + input_width;
              while(cur < nr){
                  int vl_rem = min(nr - cur, output_width);
                  int vl_for_vector = -(output_cur + cur < batch_output_size) & vl_rem;
                  v_w0 = __riscv_vlse32_v_u32m2 (in_ptr_now_rem, stride_width << 2, vl_for_vector);
                  __riscv_vse32_v_u32m2(out_ptr, v_w0, vl_for_vector);
                  out_ptr += vl_rem;
                  cur += vl_rem;
                  in_ptr_now_rem += input_width << 1;
              }
          }
          in_ptr += input_width;
      }
      output_cur += nr;
      int sliding_window_row_diff = output_cur / output_width - out_h;
      input_cursor += (
          -(sliding_window_row_diff > 0) & 
              (
                  ((output_width - out_w) << 1) + input_width \
                  + (sliding_window_row_diff - 1) * (input_width << 1)
              )
          ) \
          + (((output_cur % output_width) - (-(sliding_window_row_diff == 0) & out_w)) << 1);
  }
}

void im2col_local_avgpool_s2_d1_p0_with_pack_4x(uint32_t batch_size, const size_t input_height, const size_t input_width, size_t group_input_channels, \
  const int output_height, const int output_width,
  const size_t kernel_height, const size_t kernel_width, const size_t stride_height, const size_t stride_width, \
  const int dilation_height, const int dilation_width, const int input_padding_top,const int input_padding_left, \
  uint32_t* input, uint32_t* output, const int nr){
  const size_t output_size = output_height * output_width;
  const size_t batch_output_size = group_input_channels * output_size * batch_size;
  
  const size_t input_size = input_height*input_width*batch_size;
  uint32_t* in_ptr = input;
  uint32_t* out_ptr = output;
  const int vlmax = __riscv_vsetvlmax_e32m4();
  int input_stride;
  int remainder = 0;
  int input_cursor = 0;
  int out_h, out_w, batch;
  int im2col_cur = 0;
  int batch_cur = 0;
  int output_cur = 0;
  while(output_cur < batch_output_size){
      int out_h = output_cur / output_width;
      int out_w = output_cur % output_width;
      batch_cur = output_cur / output_size;
      int base = batch_cur * output_size;
      in_ptr = input + input_cursor;
      for(int k_h = 0; k_h < kernel_height; k_h++){
          for(int k_w = 0; k_w < kernel_width; k_w++){
              uint32_t* in_ptr_now = in_ptr + k_w;
              int vl = min(vlmax, output_width - out_w);
              vuint32m4_t v_w0 = __riscv_vlse32_v_u32m4 (in_ptr_now, stride_width << 2, vl);
              __riscv_vse32_v_u32m4(out_ptr, v_w0, vl);
              out_ptr += vl;
              int cur = vl;
              uint32_t* in_ptr_now_rem = in_ptr_now + ((output_width - out_w) << 1) + input_width;
              while(cur < nr){
                  int vl_rem = min(nr - cur, output_width);
                  int vl_for_vector = -(output_cur + cur < batch_output_size) & vl_rem;
                  v_w0 = __riscv_vlse32_v_u32m4 (in_ptr_now_rem, stride_width << 2, vl_for_vector);
                  __riscv_vse32_v_u32m4(out_ptr, v_w0, vl_for_vector);
                  out_ptr += vl_rem;
                  cur += vl_rem;
                  in_ptr_now_rem += input_width << 1;
              }
          }
          in_ptr += input_width;
      }
      output_cur += nr;
      int sliding_window_row_diff = output_cur / output_width - out_h;
      input_cursor += (
          -(sliding_window_row_diff > 0) & 
              (
                  ((output_width - out_w) << 1) + input_width \
                  + (sliding_window_row_diff - 1) * (input_width << 1)
              )
          ) \
          + (((output_cur % output_width) - (-(sliding_window_row_diff == 0) & out_w)) << 1);
  }
}
void xnn_x32_packa_in_T_gemm_im2col_s2_d1_p0_x1v(uint32_t batch_size, const size_t input_height, const size_t input_width, size_t group_input_channels, \
  const int output_height, const int output_width,
  const size_t kernel_height, const size_t kernel_width, const size_t stride_height, const size_t stride_width, \
  const int dilation_height, const int dilation_width, const int input_padding_top,const int input_padding_left, \
  uint32_t* input, uint32_t* output, const int nr){
  const size_t output_size = output_height * output_width;
  const size_t batch_output_size = output_size * batch_size;
  
  const size_t input_size = input_height*input_width*batch_size;
  uint32_t* in_ptr = input;
  uint32_t* out_ptr = output;
  const int vlmax = __riscv_vsetvlmax_e32m1();
  int input_stride;
  int remainder = 0;
  int input_cursor = 0;
  int out_h, out_w, batch;
  int im2col_cur = 0;
  int batch_cur = 0;
  int output_cur = 0;
  while(output_cur < batch_output_size){
      int out_h = output_cur / output_width;
      int out_w = output_cur % output_width;
      batch_cur = output_cur / output_size;
      int base = batch_cur * output_size;
      in_ptr = input + input_cursor;
      for(int k_h = 0; k_h < kernel_height; k_h++){
          for(int k_w = 0; k_w < kernel_width; k_w++){
              uint32_t* in_ptr_now = in_ptr;
              for(size_t in_ch = 0; in_ch < group_input_channels; in_ch++){
                  int vl = min(vlmax, output_width - out_w);
                  vuint32m1_t v_w0 = __riscv_vlse32_v_u32m1 (in_ptr_now, stride_width << 2, vl);
                  __riscv_vse32_v_u32m1(out_ptr, v_w0, vl);
                  out_ptr += vl;
                  int cur = vl;
                  uint32_t* in_ptr_now_rem = in_ptr_now + ((output_width - out_w) << 1) + input_width;
                  while(cur < nr){
                      int vl_rem = min(nr - cur, output_width);
                      int vl_for_vector = -(output_cur + cur < batch_output_size) & vl_rem;
                      v_w0 = __riscv_vlse32_v_u32m1 (in_ptr_now_rem, stride_width << 2, vl_for_vector);
                      __riscv_vse32_v_u32m1(out_ptr, v_w0, vl_for_vector);
                      out_ptr += vl_rem;
                      cur += vl_rem;
                      in_ptr_now_rem += input_width << 1;
                  }
                  in_ptr_now += input_size;
              }
          }
          in_ptr += input_width;
      }
      output_cur += nr;
      int sliding_window_row_diff = output_cur / output_width - out_h;
      input_cursor += (
          -(sliding_window_row_diff > 0) & 
              (
                  ((output_width - out_w) << 1) + input_width \
                  + (sliding_window_row_diff - 1) * (input_width << 1)
              )
          ) \
          + (((output_cur % output_width) - (-(sliding_window_row_diff == 0) & out_w)) << 1);
  }
}

void xnn_x32_packa_in_T_gemm_im2col_s2_d1_p0_x2v(uint32_t batch_size, const size_t input_height, const size_t input_width, size_t group_input_channels, \
  const int output_height, const int output_width,
  const size_t kernel_height, const size_t kernel_width, const size_t stride_height, const size_t stride_width, \
  const int dilation_height, const int dilation_width, const int input_padding_top,const int input_padding_left, \
  uint32_t* input, uint32_t* output, const int nr){
  const size_t output_size = output_height * output_width;
  const size_t batch_output_size = output_size * batch_size;
  
  const size_t input_size = input_height*input_width*batch_size;
  uint32_t* in_ptr = input;
  uint32_t* out_ptr = output;
  const int vlmax = __riscv_vsetvlmax_e32m2();
  int input_stride;
  int remainder = 0;
  int input_cursor = 0;
  int out_h, out_w, batch;
  int im2col_cur = 0;
  int batch_cur = 0;
  int output_cur = 0;
  while(output_cur < batch_output_size){
      int out_h = output_cur / output_width;
      int out_w = output_cur % output_width;
      batch_cur = output_cur / output_size;
      int base = batch_cur * output_size;
      in_ptr = input + input_cursor;
      for(int k_h = 0; k_h < kernel_height; k_h++){
          for(int k_w = 0; k_w < kernel_width; k_w++){
              uint32_t* in_ptr_now = in_ptr;
              for(size_t in_ch = 0; in_ch < group_input_channels; in_ch++){
                  int vl = min(vlmax, output_width - out_w);
                  vuint32m2_t v_w0 = __riscv_vlse32_v_u32m2 (in_ptr_now, stride_width << 2, vl);
                  __riscv_vse32_v_u32m2(out_ptr, v_w0, vl);
                  out_ptr += vl;
                  int cur = vl;
                  uint32_t* in_ptr_now_rem = in_ptr_now + ((output_width - out_w) << 1) + input_width;
                  while(cur < nr){
                      int vl_rem = min(nr - cur, output_width);
                      int vl_for_vector = -(output_cur + cur < batch_output_size) & vl_rem;
                      v_w0 = __riscv_vlse32_v_u32m2 (in_ptr_now_rem, stride_width << 2, vl_for_vector);
                      __riscv_vse32_v_u32m2(out_ptr, v_w0, vl_for_vector);
                      out_ptr += vl_rem;
                      cur += vl_rem;
                      in_ptr_now_rem += input_width << 1;
                  }
                  in_ptr_now += input_size;
              }
          }
          in_ptr += input_width;
      }
      output_cur += nr;
      int sliding_window_row_diff = output_cur / output_width - out_h;
      input_cursor += (
          -(sliding_window_row_diff > 0) & 
              (
                  ((output_width - out_w) << 1) + input_width \
                  + (sliding_window_row_diff - 1) * (input_width << 1)
              )
          ) \
          + (((output_cur % output_width) - (-(sliding_window_row_diff == 0) & out_w)) << 1);
  }
}

void xnn_x32_packa_in_T_gemm_im2col_s2_d1_p0_x4v(uint32_t batch_size, const size_t input_height, const size_t input_width, size_t group_input_channels, \
  const int output_height, const int output_width,
  const size_t kernel_height, const size_t kernel_width, const size_t stride_height, const size_t stride_width, \
  const int dilation_height, const int dilation_width, const int input_padding_top,const int input_padding_left, \
  uint32_t* input, uint32_t* output, const int nr){
  const size_t output_size = output_height * output_width;
  const size_t batch_output_size = output_size * batch_size;
  
  const size_t input_size = input_height*input_width*batch_size;
  uint32_t* in_ptr = input;
  uint32_t* out_ptr = output;
  const int vlmax = __riscv_vsetvlmax_e32m4();
  int input_stride;
  int remainder = 0;
  int input_cursor = 0;
  int out_h, out_w, batch;
  int im2col_cur = 0;
  int batch_cur = 0;
  int output_cur = 0;
  while(output_cur < batch_output_size){
      int out_h = output_cur / output_width;
      int out_w = output_cur % output_width;
      batch_cur = output_cur / output_size;
      int base = batch_cur * output_size;
      in_ptr = input + input_cursor;
      for(int k_h = 0; k_h < kernel_height; k_h++){
          for(int k_w = 0; k_w < kernel_width; k_w++){
              uint32_t* in_ptr_now = in_ptr;
              for(size_t in_ch = 0; in_ch < group_input_channels; in_ch++){
                  int vl = min(vlmax, output_width - out_w);
                  vuint32m4_t v_w0 = __riscv_vlse32_v_u32m4 (in_ptr_now, stride_width << 2, vl);
                  __riscv_vse32_v_u32m4(out_ptr, v_w0, vl);
                  out_ptr += vl;
                  int cur = vl;
                  uint32_t* in_ptr_now_rem = in_ptr_now + ((output_width - out_w) << 1) + input_width;
                  while(cur < nr){
                      int vl_rem = min(nr - cur, output_width);
                      int vl_for_vector = -(output_cur + cur < batch_output_size) & vl_rem;
                      v_w0 = __riscv_vlse32_v_u32m4 (in_ptr_now_rem, stride_width << 2, vl_for_vector);
                      __riscv_vse32_v_u32m4(out_ptr, v_w0, vl_for_vector);
                      out_ptr += vl_rem;
                      cur += vl_rem;
                      in_ptr_now_rem += input_width << 1;
                  }
                  in_ptr_now += input_size;
              }
          }
          in_ptr += input_width;
      }
      output_cur += nr;
      int sliding_window_row_diff = output_cur / output_width - out_h;
      input_cursor += (
          -(sliding_window_row_diff > 0) & 
              (
                  ((output_width - out_w) << 1) + input_width \
                  + (sliding_window_row_diff - 1) * (input_width << 1)
              )
          ) \
          + (((output_cur % output_width) - (-(sliding_window_row_diff == 0) & out_w)) << 1);
  }
}

void xnn_x32_packa_in_T_im2col_pooling_s2_d1_x2v(uint32_t batch_size, const size_t input_height, const size_t input_width, size_t group_input_channels, \
  const int output_height, const int output_width,
  const size_t kernel_height, const size_t kernel_width, const size_t stride_height, const size_t stride_width, \
  const int dilation_height, const int dilation_width, const int input_padding_top,const int input_padding_left, \
  uint32_t* input, uint32_t* output){
  const size_t output_size = output_height*output_width;
  uint32_t* in_ptr = input;
  uint32_t* in_ptr_rem = input;
  uint32_t* out_ptr = output;
  const int vlmax = __riscv_vsetvlmax_e32m2();
  int output_cur = 0;
  int input_cursor = 0;
  int valid_height = input_padding_top + input_height - 1;
  int last_stride = kernel_height - 1 + (output_height - 1)*2;
  int k_h_padding_end = last_stride - valid_height;
  int remainder = 0;
  for(size_t in_ch = 0; in_ch < group_input_channels; in_ch++){
      for(size_t batch = 0; batch < batch_size; batch++){
          int base = in_ch * batch_size * input_height*input_width + batch * input_height * input_width;
          int output_padding_top_stride = ((input_padding_top + 1) >> 1) * output_width;
          int output_padding_down_stride = output_padding_top_stride;
          while(output_padding_top_stride > 0){
              int out_h = output_cur / output_width;
              int out_w = output_cur % output_width;
              input_cursor = base + (out_w << 1);
              remainder = -(output_width - out_w < vlmax) & (vlmax - output_width + out_w);
              int padded_k_h = zero_max(input_padding_top - (out_h << 1) - (-(remainder > 0) & ((out_h + 1) << 1)));
              in_ptr = input + input_cursor;
              in_ptr_rem = input + base + ((output_cur + remainder) / output_width) * input_width;
              vuint32m2_t init = __riscv_vmv_v_x_u32m2 (0xff, vlmax);
              for(int i = 0; i < vlmax * kernel_width * padded_k_h; i += vlmax){
                  __riscv_vse32_v_u32m2(out_ptr, init, vlmax); out_ptr += vlmax;
              }
              for(int k_h = padded_k_h; k_h < kernel_height; k_h++){
                  int padded = -(remainder > 0 && k_h < input_padding_top - (out_h << 1));
                  for(int k_w = 0; k_w < kernel_width; k_w++){
                      int width_padding_start = -(out_w == 0) & ((zero_max(input_padding_left-k_w) + 1) >> 1);
                      int width_padding_end = -((output_cur + vlmax) / output_width != out_h) & ((zero_max(k_w + ((output_width-1) << 1) - (input_padding_left + input_width-1)) + 1) >> 1);
                      int input_offset_with_pad = stride_width * width_padding_start - (input_padding_left - k_w);
                      int input_offset_with_pad_cond = -(k_w < input_padding_left);
                      int input_cur_offset = (input_offset_with_pad_cond & input_offset_with_pad) + \
                                              (~input_offset_with_pad_cond & (k_w - input_padding_left));
                      // for remainder
                      int width_padding_start_rem = -(remainder > 0) & ((zero_max(input_padding_left-k_w) + 1) >> 1);
                      input_offset_with_pad = stride_width * width_padding_start_rem - (input_padding_left - k_w);
                      int input_cur_offset_rem = (input_offset_with_pad_cond & input_offset_with_pad) + \
                                              (~input_offset_with_pad_cond & (k_w - input_padding_left));
                      uint32_t* in_ptr_now = in_ptr + input_cur_offset;
                      uint32_t* in_ptr_remainder = in_ptr_rem + input_cur_offset_rem;
                      *out_ptr = 0xff;
                      out_ptr += width_padding_start;
                      int vl = vlmax-width_padding_end-width_padding_start-remainder;
                      vuint32m2_t v_w0 = __riscv_vlse32_v_u32m2 (in_ptr_now, stride_width << 2, ~padded & vl);
                      __riscv_vse32_v_u32m2(out_ptr, v_w0, ~padded & vl);
                      out_ptr += vl;
                      *out_ptr = 0xff;
                      out_ptr += width_padding_end;
                      // for remainder
                      *out_ptr = 0xff;
                      out_ptr += width_padding_start_rem;
                      vl = zero_max(remainder - width_padding_start_rem);
                      v_w0 = __riscv_vlse32_v_u32m2 (in_ptr_remainder, stride_width << 2, vl);
                      __riscv_vse32_v_u32m2(out_ptr, v_w0, vl);
                      out_ptr += vl;
                  }
                  in_ptr += ~padded & input_width;
                  in_ptr_rem += -(remainder > 0) & input_width;
              }
              output_padding_top_stride -= vlmax;
              output_cur += vlmax;
          }
          input_cursor = base + input_width + (remainder << 1);
          while(((input_padding_top + 1) >> 1) + input_cursor / input_width + kernel_height - base / input_width <= valid_height){
              int out_h = output_cur / output_width;
              int out_w = output_cur % output_width;
              int input_cursor_rem = base + input_width + ((output_cur + remainder) / output_width) * (input_width << 1);
              in_ptr = input + input_cursor;
              in_ptr_rem = input + input_cursor_rem;
              remainder = -(output_width - out_w < vlmax) & (vlmax - output_width + out_w);
              for(int k_h = 0; k_h < kernel_height; k_h++){
                  int padded = -(remainder > 0 && ((input_padding_top + 1) >> 1) + input_cursor_rem / input_width + k_h - base / input_width > valid_height);
                  for(int k_w = 0; k_w < kernel_width; k_w++){
                      int width_padding_start = -(out_w == 0) & ((zero_max(input_padding_left-k_w) + 1) >> 1);
                      int width_padding_end = -((output_cur + vlmax) / output_width != out_h) & ((zero_max(k_w + ((output_width-1) << 1) - (input_padding_left + input_width-1)) + 1) >> 1);
                      int input_offset_with_pad = stride_width * width_padding_start - (input_padding_left - k_w);
                      int input_offset_with_pad_cond = -(k_w < input_padding_left);
                      int input_cur_offset = (input_offset_with_pad_cond & input_offset_with_pad) + \
                                              (~input_offset_with_pad_cond & (k_w - input_padding_left));
                      // for remainder
                      int width_padding_start_rem = -(remainder > 0) & ((zero_max(input_padding_left-k_w) + 1) >> 1);
                      input_offset_with_pad = stride_width * width_padding_start_rem - (input_padding_left - k_w);
                      int input_cur_offset_rem = (input_offset_with_pad_cond & input_offset_with_pad) + \
                                              (~input_offset_with_pad_cond & (k_w - input_padding_left));
                      uint32_t* in_ptr_now = in_ptr + input_cur_offset;
                      uint32_t* in_ptr_remainder = in_ptr_rem + input_cur_offset_rem;
                      *out_ptr = 0xff;
                      out_ptr += width_padding_start;
                      int vl = vlmax-width_padding_end-width_padding_start-remainder;
                      vuint32m2_t v_w0 = __riscv_vlse32_v_u32m2 (in_ptr_now, stride_width << 2, vl);
                      __riscv_vse32_v_u32m2(out_ptr, v_w0, vl);
                      out_ptr += vl;
                      *out_ptr = 0xff;
                      out_ptr += width_padding_end;
                      // for remainder
                      *out_ptr = 0xff;
                      out_ptr += width_padding_start_rem;
                      vl = zero_max(remainder - width_padding_start_rem);
                      // std::cout << "vl for remainder = " << vl << "\n";
                      v_w0 = __riscv_vlse32_v_u32m2 (in_ptr_remainder, stride_width << 2, ~padded & vl);
                      __riscv_vse32_v_u32m2(out_ptr, v_w0, ~padded & vl);
                      out_ptr += vl;
                  }
                  in_ptr += input_width;
                  in_ptr_rem += input_width;
              }
              output_padding_top_stride -= vlmax;
              output_cur += vlmax;
              input_cursor = base + input_width + ((input_width << 1) * (output_cur / output_width - ((input_padding_top + 1) >> 1))) + ((output_cur % output_width) << 1);
          }
          while(output_cur < output_size){
              int out_h = output_cur / output_width;
              int out_w = output_cur % output_width;
              input_cursor = base + input_width + ((input_width << 1) * (out_h - ((input_padding_top + 1) >> 1))) + (out_w << 1);
              int input_cursor_rem = base + input_width + ((input_width << 1) * (out_h - 1));
              in_ptr = input + input_cursor;
              remainder = -(output_width - out_w < vlmax) & (vlmax - output_width + out_w);
              for(int k_h = 0; k_h < kernel_height-k_h_padding_end; k_h++){
                  for(int k_w = 0; k_w < kernel_width; k_w++){
                      int width_padding_start = -(out_w == 0) & ((zero_max(input_padding_left-k_w) + 1) >> 1);
                      int width_padding_end = -((output_cur + vlmax) / output_width != out_h) & ((zero_max(k_w + ((output_width-1) << 1) - (input_padding_left + input_width-1)) + 1) >> 1);
                      int input_offset_with_pad = stride_width * width_padding_start - (input_padding_left - k_w);
                      int input_offset_with_pad_cond = -(k_w < input_padding_left);
                      int input_cur_offset = (input_offset_with_pad_cond & input_offset_with_pad) + \
                                              (~input_offset_with_pad_cond & (k_w - input_padding_left));
                      uint32_t* in_ptr_now = in_ptr + input_cur_offset;
                      *out_ptr = 0xff;
                      out_ptr += width_padding_start;
                      int vl = vlmax-width_padding_end-width_padding_start-remainder;
                      vuint32m2_t v_w0 = __riscv_vlse32_v_u32m2 (in_ptr_now, stride_width << 2, vl);
                      __riscv_vse32_v_u32m2(out_ptr, v_w0, vl);
                      out_ptr += vl;
                      *out_ptr = 0xff;
                      out_ptr += width_padding_end;
                  }
                  in_ptr += output_width << 1;
              }
              vuint32m2_t init = __riscv_vmv_v_x_u32m2 (0xff, vlmax);
              for(int i = 0; i < k_h_padding_end * vlmax * kernel_width; i += vlmax){
                  __riscv_vse32_v_u32m2(out_ptr, init, vlmax); out_ptr += vlmax;
              }
              output_padding_top_stride -= vlmax;
              output_cur += vlmax;
          }
          output_cur = 0;
      }
  }
}

void xnn_x32_packa_in_T_gemm_im2col_s2_d1_x1v(uint32_t batch_size, const size_t input_height, const size_t input_width, size_t group_input_channels, \
  const int output_height, const int output_width,
  const size_t kernel_height, const size_t kernel_width, const size_t stride_height, const size_t stride_width, \
  const int dilation_height, const int dilation_width, const int input_padding_top,const int input_padding_left, \
  uint32_t* input, uint32_t* output){
  const size_t output_size = output_height*output_width;
  const size_t input_size = batch_size * input_height*input_width;
  uint32_t* in_ptr = input;
  uint32_t* in_ptr_rem = input;
  uint32_t* out_ptr = output;
  const int vlmax = __riscv_vsetvlmax_e32m1();
  int output_cur = 0;
  int input_cursor = 0;
  int valid_height = input_padding_top + input_height - 1;
  int last_stride = kernel_height - 1 + (output_height - 1)*2;
  int k_h_padding_end = last_stride - valid_height;
  for(size_t batch = 0; batch < batch_size; batch++){
      int base = batch * input_height*input_width;
      int output_padding_top_stride = ((input_padding_top + 1) >> 1) * output_width;
      int output_padding_down_stride = output_padding_top_stride;
      while(output_padding_top_stride > 0){
          int out_h = output_cur / output_width;
          int out_w = output_cur % output_width;
          input_cursor = base + (out_w << 1);
          int padded_k_h = input_padding_top - (out_h << 1);
          in_ptr = input + input_cursor;
          out_ptr += vlmax * group_input_channels * kernel_width * padded_k_h;
          for(int k_h = padded_k_h; k_h < kernel_height; k_h++){
              for(int k_w = 0; k_w < kernel_width; k_w++){
                  int width_padding_start = -(out_w == 0) & ((zero_max(input_padding_left-k_w) + 1) >> 1);
                  int width_padding_end = -((output_cur + vlmax) / output_width != out_h) & ((zero_max(k_w + ((output_width-1) << 1) - (input_padding_left + input_width-1)) + 1) >> 1);
                  int input_offset_with_pad = stride_width * width_padding_start - (input_padding_left - k_w);
                  int input_offset_with_pad_cond = -(k_w < input_padding_left);
                  int input_cur_offset = (input_offset_with_pad_cond & input_offset_with_pad) + \
                                          (~input_offset_with_pad_cond & (k_w - input_padding_left));
                  uint32_t* in_ptr_now = in_ptr + input_cur_offset;
                  for(size_t in_ch = 0; in_ch < group_input_channels; in_ch++){
                      out_ptr += width_padding_start;
                      int vl = vlmax-width_padding_end-width_padding_start;
                      // std::cout << "vl = " << vl << "\n";
                      vuint32m1_t v_w0 = __riscv_vlse32_v_u32m1 (in_ptr_now, stride_width << 2, vl);
                      __riscv_vse32_v_u32m1(out_ptr, v_w0, vl);
                      out_ptr += vl + width_padding_end;
                      in_ptr_now += input_size;
                  }
              }
              in_ptr += (output_width << 1);
          }
          output_padding_top_stride -= vlmax;
          output_cur += vlmax;
      }
      input_cursor = base + input_width;
      while(((input_padding_top + 1) >> 1) + input_cursor / input_width + kernel_height - base / input_width <= valid_height){
          int out_h = output_cur / output_width;
          int out_w = output_cur % output_width;
          int input_cursor_rem = base + input_width + ((input_width << 1) * (out_h - 1));
          in_ptr = input + input_cursor;
          for(int k_h = 0; k_h < kernel_height; k_h++){
              for(int k_w = 0; k_w < kernel_width; k_w++){
                  int width_padding_start = -(out_w == 0) & ((zero_max(input_padding_left-k_w) + 1) >> 1);
                  int width_padding_end = -((output_cur + vlmax) / output_width != out_h) & ((zero_max(k_w + ((output_width-1) << 1) - (input_padding_left + input_width-1)) + 1) >> 1);
                  int input_offset_with_pad = stride_width * width_padding_start - (input_padding_left - k_w);
                  int input_offset_with_pad_cond = -(k_w < input_padding_left);
                  int input_cur_offset = (input_offset_with_pad_cond & input_offset_with_pad) + \
                                          (~input_offset_with_pad_cond & (k_w - input_padding_left));
                  uint32_t* in_ptr_now = in_ptr + input_cur_offset;
                  for(size_t in_ch = 0; in_ch < group_input_channels; in_ch++){
                      out_ptr += width_padding_start;
                      int vl = vlmax-width_padding_end-width_padding_start;
                      vuint32m1_t v_w0 = __riscv_vlse32_v_u32m1 (in_ptr_now, stride_width << 2, vl);
                      __riscv_vse32_v_u32m1(out_ptr, v_w0, vl);
                      out_ptr += vl + width_padding_end;
                      in_ptr_now += input_size;
                  }
              }
              in_ptr += output_width << 1;
          }
          output_padding_top_stride -= vlmax;
          output_cur += vlmax;
          input_cursor = base + input_width + ((input_width << 1) * (output_cur / output_width - 2)) + ((output_cur % output_width) << 1);
      }
      while(output_cur < output_size){
          int out_h = output_cur / output_width;
          int out_w = output_cur % output_width;
          input_cursor = base + input_width + ((input_width << 1) * (out_h - 2)) + (out_w << 1);
          int input_cursor_rem = base + input_width + ((input_width << 1) * (out_h - 1));
          in_ptr = input + input_cursor;
          for(int k_h = 0; k_h < kernel_height-k_h_padding_end; k_h++){
              for(int k_w = 0; k_w < kernel_width; k_w++){
                  int width_padding_start = -(out_w == 0) & ((zero_max(input_padding_left-k_w) + 1) >> 1);
                  int width_padding_end = -((output_cur + vlmax) / output_width != out_h) & ((zero_max(k_w + ((output_width-1) << 1) - (input_padding_left + input_width-1)) + 1) >> 1);
                  int input_offset_with_pad = stride_width * width_padding_start - (input_padding_left - k_w);
                  int input_offset_with_pad_cond = -(k_w < input_padding_left);
                  int input_cur_offset = (input_offset_with_pad_cond & input_offset_with_pad) + \
                                          (~input_offset_with_pad_cond & (k_w - input_padding_left));
                  uint32_t* in_ptr_now = in_ptr + input_cur_offset;
                  for(size_t in_ch = 0; in_ch < group_input_channels; in_ch++){
                      out_ptr += width_padding_start;
                      int vl = vlmax-width_padding_end-width_padding_start;
                      vuint32m1_t v_w0 = __riscv_vlse32_v_u32m1 (in_ptr_now, stride_width << 2, vl);
                      __riscv_vse32_v_u32m1(out_ptr, v_w0, vl);
                      out_ptr += vl + width_padding_end;
                      in_ptr_now += input_size;
                  }
              }
              in_ptr += output_width << 1;
          }
          out_ptr += k_h_padding_end * vlmax * group_input_channels * kernel_width;
          output_padding_top_stride -= vlmax;
          output_cur += vlmax;
      }
      output_cur = 0;
  }
}

void xnn_x32_packa_in_T_gemm_im2col_s1_d1_1x1v(uint32_t batch_size, const size_t input_height, const size_t input_width, size_t group_input_channels, \
  const int output_height, const int output_width,
  const size_t kernel_height, const size_t kernel_width, const size_t stride_height, const size_t stride_width, \
  const int dilation_height, const int dilation_width, const int input_padding_top,const int input_padding_left, \
  uint32_t* input, uint32_t* output){
  const size_t output_size = output_height*output_width;
  const size_t input_size = input_height*input_width*batch_size;
  uint32_t* in_ptr = input;
  uint32_t* out_ptr = output;
  const int vlmax = __riscv_vsetvlmax_e32m1();
  int valid_height = input_padding_top + input_height;
  int valid_width = input_padding_left + input_width;
  int height_padding_start = input_padding_top;
  int height_padding_end = zero_max(output_height-valid_height);
  int output_padding_top_stride = height_padding_start * output_width;
  int output_padding_down_stride = height_padding_end * output_width;
  int width_padding_start;
  int width_padding_end;
  int input_stride;
  int remainder = 0;
  int input_cursor = 0;
  int out_h, out_w, batch;
  int im2col_cur = 0;
  int batch_cur = 0;
  int output_cur = 0;
  batch = im2col_cur / output_size;
  // top_pad
  /*
      output_cur < vlmax * ceil(output_width / vlmax)
  --> output_cur < vlmax * ((output_width + vlmax - 1) / vlmax)
  */ 
  int input_cursor_real_part = 0;
  for(batch = 0; batch < batch_size; batch ++){
      while(output_padding_top_stride){
          out_h = output_cur / output_width;
          out_w = output_cur % output_width;
          int is_whole_stride_padded_part = -(output_padding_top_stride/vlmax >= 1);
          remainder = ~is_whole_stride_padded_part & output_padding_top_stride;
          uint32_t* in_ptr_now = in_ptr + input_cursor;
          uint32_t* in_ptr_now_real_part = in_ptr + output_size*batch + (-(remainder == 0) & input_cursor_real_part);
          // std::cout << "first: output_cur = " << output_cur << ", remainder = " << remainder << "\n";
          out_ptr += (is_whole_stride_padded_part & (min(vlmax, output_width) * group_input_channels * kernel_width));
          output_padding_top_stride -= (is_whole_stride_padded_part & vlmax);
          for(int k_h = is_whole_stride_padded_part & input_padding_top; k_h < kernel_height; k_h++){
              int moved_input_ptr_step = 0;
              int moved_input_ptr_real_step = 0;
              int is_partial_stride_padded_part = -(output_padding_top_stride && zero_max(input_padding_top-k_h));
              for(int k_w = 0; k_w < kernel_width; k_w++){
                  int output_padding_left = zero_max(input_padding_left-k_w) & (-((output_cur + remainder) % output_width == 0));
                  int output_padding_right = zero_max(k_w + output_width-valid_width) & (-((output_cur + vlmax) % output_width == 0));
                  int remainder_padding_left = output_padding_left & (-(remainder > 0 && out_w == 0));
                  int remainder_padding_right = (-(remainder > 0) & zero_max(k_w + output_width-valid_width));
                  int vl;
                  for(int in_ch = 0; in_ch < group_input_channels; in_ch++){
                      // |---- pad_top ---- |  --> output_padding_top_stride 
                      // |- real -| -pad_r -|  --> remainder_padding_right(for k_w == 2 when kernel_width = 3)
                      out_ptr += output_padding_left & -(remainder > 0 && out_w == 0);
                      vl = remainder \
                          - (is_partial_stride_padded_part & (output_padding_top_stride - remainder_padding_left - remainder_padding_right)) \
                          - remainder_padding_right - remainder_padding_left;
                      // std::cout << "output_padding_left = " << output_padding_left << ", vl = " << vl << "\n";
                      vuint32m1_t v_w0 = __riscv_vle32_v_u32m1(in_ptr_now, vl);
                      __riscv_vse32_v_u32m1(out_ptr, v_w0, vl);
                      out_ptr += remainder - remainder_padding_left; // including remainder_padding_right
                      out_ptr += output_padding_left;
                      vl = vlmax - remainder - output_padding_left - output_padding_right;
                      v_w0 = __riscv_vle32_v_u32m1(in_ptr_now_real_part, vl);
                      __riscv_vse32_v_u32m1(out_ptr, v_w0, vl);
                      out_ptr += vl + output_padding_right;
                      in_ptr_now += input_size;
                      in_ptr_now_real_part += input_size;
                  }
                  in_ptr_now = in_ptr_now - input_size*group_input_channels + 1 - (output_padding_left & (-(out_w == 0) || is_partial_stride_padded_part));
                  in_ptr_now_real_part = in_ptr_now_real_part - input_size*group_input_channels + 1 - (output_padding_left & (-((output_cur + remainder) % output_width == 0)));
                  moved_input_ptr_step += (1 - (output_padding_left & (-(out_w == 0) || is_partial_stride_padded_part)));
                  moved_input_ptr_real_step += (1 - (output_padding_left & (-((output_cur + remainder) % output_width == 0))));
              }
              in_ptr_now = in_ptr_now - moved_input_ptr_step - (is_partial_stride_padded_part & (output_padding_top_stride + (input_padding_left & -(out_w != 0)))) + input_width;
              in_ptr_now_real_part = in_ptr_now_real_part - moved_input_ptr_real_step + input_width;
              output_padding_top_stride -= (is_partial_stride_padded_part & output_padding_top_stride);
          }
          // todo : input_offset, end height padding
          int input_offset = -(output_padding_top_stride == 0 && output_width % vlmax != 0) & (vlmax - remainder + (-(vlmax - remainder < output_width) & -input_padding_left));
          input_cursor = input_cursor + input_offset;
          input_cursor_real_part += (vlmax - remainder - (-(output_cur == 0) & input_padding_left));
          im2col_cur += vlmax;
          output_cur += vlmax;
          // std::cout << "first: input_offset= " << input_offset << "\n";
      }
      // middle part
      while((output_cur + (-((output_cur + vlmax) % output_width) & vlmax)) / output_width < output_height - input_padding_top){
          out_h = output_cur / output_width;
          out_w = output_cur % output_width;
          // -(out_h < input_padding_top) --> for the condition with padding
          // 0xFFFFFFFF if cond=1, else 0x00000000
          const int r = (output_cur + vlmax) % output_width;
          remainder = ((-(out_h != (output_cur + vlmax) / output_width)) & \
                  (((-(r != 0)) & (vlmax - r)) + ((-(r == 0)) & zero_max(vlmax - output_width))));
          int k_h = 0, k_w = 0;
          int input_offset = 0;
          uint32_t* in_ptr_now = in_ptr + input_cursor;
          // std::cout << "middle: output_cur = " << output_cur << ", r = " << r << ", remainder = " << remainder << "\n";
          // || -- pad -- | -- input ----- ||                         --> pad before second load/store
          // || -- remainder ------------- || -- pad -- | - input -|| --> pad before second load/store
          int is_in_left_part = -((out_w == 0) || (remainder > 0));
          // || --- input ---   |-- pad -- ||                         --> pad after second load/store
          // || -- remainder -- |-- pad -- || ------ input ------- || --> pad before second load/store
          int is_in_right_part1 = -((out_h != (output_cur + vlmax) / output_width));
          int is_in_right_part2 = -((output_cur + vlmax) % output_width > 0);
          for(k_h = 0; k_h < kernel_height; k_h++){
              int moved_input_ptr_step = 0;
              for(k_w = 0; k_w < kernel_width; k_w++){
                  int output_padding_left = (is_in_left_part & zero_max(input_padding_left-k_w));
                  int remainder_padding_left = output_padding_left & (-(remainder > 0 && out_w == 0));
                  int remainder_padding_right = (-(remainder > 0) & zero_max(k_w + output_width-valid_width));
                  int output_padding_right = (-(remainder == 0 || (remainder > 0 && ~is_in_right_part2)) & is_in_right_part1 & zero_max(k_w + output_width-valid_width));
                  int vl;
                  // std::cout << "remainder_padding_right = " << output_padding_right << "\n";
                  for(int in_ch = 0; in_ch < group_input_channels; in_ch++){
                      out_ptr += remainder_padding_left;
                      vl = zero_max(remainder - remainder_padding_right - remainder_padding_left);
                      vuint32m1_t v_w0 = __riscv_vle32_v_u32m1(in_ptr_now, vl);
                      __riscv_vse32_v_u32m1(out_ptr, v_w0, vl);
                      // remainder with padding
                      out_ptr = out_ptr + remainder - remainder_padding_left + output_padding_left;
                      // in_ptr_now + vl - (output_width - output_padding_left - remainder_padding_right + input_width)
                      // two segments
                      int is_added = vl + ((output_padding_left + remainder_padding_right) & -(remainder > 0));
                      in_ptr_now = in_ptr_now + is_added;
                      // std::cout << "vl for remainder = " << vl << "\n";
                      vl = vlmax - remainder - output_padding_left - output_padding_right;
                      // std::cout << "vl = " << vl << "\n";
                      v_w0 = __riscv_vle32_v_u32m1(in_ptr_now, vl);
                      __riscv_vse32_v_u32m1(out_ptr, v_w0, vl);

                      out_ptr += vl + output_padding_right;
                      in_ptr_now = in_ptr_now - is_added + input_size;
                  }
                  // at the start of each stride, we need to take padding_left into consideration
                  in_ptr_now = in_ptr_now - input_size*group_input_channels + 1 - (output_padding_left & (-(out_w == 0)));
                  moved_input_ptr_step += (1 - (output_padding_left & (-(out_w == 0))));
              }
              // std::cout << "moved_input_ptr_step = " << moved_input_ptr_step << "\n";
              in_ptr_now = in_ptr_now - moved_input_ptr_step + input_width;
          }
          // || -- pad -- | -- input ----- ||
          //      --> + len(input)
          // || -- remainder ------------- || -- pad -- | - input -|| 
          //      --> + remainder - (output_width - input_padding_left) + input_width + (vlmax - remainder - input_padding_left) \
                      = remainder - output_width + input_padding_left + input_width + vlmax - remainder - input_padding_left \
                      = vlmax
          // || ----------- input -------- ||
          //      --> + vlmax
          // || --- pad( == remainder) --- || -- pad -- | - input -||
          int cond = -(input_cursor % input_width == 0);
          input_offset = ((~cond & (vlmax + (~is_in_right_part2 & input_padding_left))) | \
                          (cond & (vlmax - remainder - input_padding_left + (output_width * ((remainder + (-((remainder + input_padding_left) % output_width == 0) & 1)) / output_width)))));
          input_cursor = input_cursor + input_offset;
          // std::cout << "middle: input_offset= " << input_offset << "\n";
          // std::cout << "middle: input_cursor= " << input_cursor << "\n";
          im2col_cur += vlmax;
          output_cur += vlmax;
      }

      // end part
      while(output_cur < output_size){
          out_h = output_cur / output_width;
          out_w = output_cur % output_width;
          int is_next_batch = -(im2col_cur + vlmax > output_size*(batch+1) && batch + 1 != batch_size);
          // std::cout << "is_next_batch = " << (is_next_batch & 1) << "\n";
          const int r = (output_cur + vlmax) % output_width;
          remainder = ((-(out_h != (output_cur + vlmax) / output_width)) & \
                  (((-(r != 0)) & (vlmax - r)) + ((-(r == 0)) & zero_max(vlmax - output_width))));
          int k_h = 0, k_w = 0;
          int input_offset = 0;
          uint32_t* in_ptr_now = in_ptr + input_cursor;
          // std::cout << "end: output_cur = " << output_cur << ", r = " << r << ", remainder = " << remainder << "\n";
          int is_in_left_part = -((out_w == 0) || (remainder > 0));
          int is_in_right_part1 = -(vlmax - remainder >= output_width || (remainder == 0 && (out_h != (output_cur + vlmax) / output_width))); 
          int is_in_right_part2 = -((output_cur + vlmax) % output_width > 0);
          int is_whole_stride_padded = ~is_next_batch & -(out_h >= output_height - input_padding_top);
          int last_part_in_this_batch = output_size % vlmax;
          for(k_h = 0; k_h < kernel_height - (is_whole_stride_padded & input_padding_top); k_h++){
              int moved_input_ptr_step = 0;
              for(k_w = 0; k_w < kernel_width; k_w++){
                  int output_padding_left = is_in_left_part & zero_max(input_padding_left-k_w);
                  int output_padding_right = is_in_right_part1 & zero_max(k_w + output_width-valid_width);
                  int remainder_padding_left = output_padding_left & (-(remainder > 0 && out_w == 0));
                  int remainder_padding_right = zero_max(k_w + output_width-valid_width) & (-(remainder > 0 && out_h != (output_cur + vlmax) / output_width));
                  int vl;
                  // std::cout << "k_h = " << k_h << ", k_w = " << k_w << ", output_padding_left = " << output_padding_left << "\n";
                  // std::cout << "output_padding_right = " << output_padding_right << "\n";
                  for(int in_ch = 0; in_ch < group_input_channels; in_ch++){
                      int is_padded_remainder = -(
                          (zero_max(k_h - input_padding_top)) && out_h >= output_height - input_padding_top
                      );
                      vl = zero_max(remainder - remainder_padding_right);
                      out_ptr += remainder_padding_left;
                      vuint32m1_t v_w0 = __riscv_vle32_v_u32m1(in_ptr_now, ~is_padded_remainder & vl);
                      __riscv_vse32_v_u32m1(out_ptr, v_w0, ~is_padded_remainder & vl);
                      // remainder with padding
                      out_ptr = out_ptr + remainder - remainder_padding_left + output_padding_left;
                      // in_ptr_now + vl - (output_width - output_padding_left - output_padding_right + input_width)
                      // two segments
                      int is_added = vl + ((output_padding_left + remainder_padding_right) & -(remainder > 0));
                      in_ptr_now = in_ptr_now + is_added;
                      int is_padded = -(
                                      ((zero_max(k_h - input_padding_top)) & -(((output_cur + vlmax) / output_width + 1 > output_height - input_padding_top))) \
                                      || ((output_cur + remainder >= output_size) && (batch + 1 >= batch_size)) \
                                      || (zero_max(input_padding_top-k_h) & -(output_cur + remainder >= output_size && batch + 1 != batch_size))
                                      );
                      int is_in_next_batch = -(output_cur + remainder >= output_size && batch + 1 != batch_size);
                      vl = vlmax - remainder - output_padding_left - output_padding_right;
                      v_w0 = __riscv_vle32_v_u32m1(in_ptr_now, (~is_padded | is_padded_remainder) & vl);
                      __riscv_vse32_v_u32m1(out_ptr, v_w0, (~is_padded | is_padded_remainder) & vl);

                      out_ptr += (vl + output_padding_right);
                      in_ptr_now = in_ptr_now - is_added + input_size;
                  }
                  // at the start of each stride, we need to take padding_left into consideration
                  in_ptr_now = in_ptr_now - input_size*group_input_channels + 1 - (output_padding_left & (-(out_w == 0)));
                  moved_input_ptr_step += (1 - (output_padding_left & (-(out_w == 0))));
              }
              // std::cout << "moved_input_ptr_step = " << moved_input_ptr_step << "\n";
              in_ptr_now = in_ptr_now - moved_input_ptr_step + input_width;
          }
          int cond = -(input_cursor % input_width == 0);
          input_offset = ((~cond & (vlmax + (~is_in_right_part2 & input_padding_left))) | \
                              ((cond) & (vlmax - remainder - input_padding_left)));
          input_cursor = input_cursor + input_offset;
          out_ptr += (is_whole_stride_padded & min(vlmax, output_width) * group_input_channels * kernel_width);
          im2col_cur += vlmax;
          output_cur += vlmax;
          // std::cout << "end: input_offset = " << input_offset << "\n";

      }
      int last_part_in_this_batch = (output_size*(batch + 1)) % vlmax;
      int finished_part_in_next_batch = -(last_part_in_this_batch > 0) & (vlmax - last_part_in_this_batch);
      output_cur = finished_part_in_next_batch;
      output_padding_top_stride = zero_max(height_padding_start * output_width - finished_part_in_next_batch);
      // std::cout << "end: finished_part_in_next_batch = " << finished_part_in_next_batch << "\n";
      input_cursor = output_size*(batch + 1);// + ((vlmax - remainder) % output_width);// output_size = output_width*output_height = input_width*input_height
      input_cursor_real_part = finished_part_in_next_batch - (finished_part_in_next_batch + output_width - 1) / output_width;
      // std::cout << "output_padding_top_stride = " << output_padding_top_stride << ", input_cursor = " << input_cursor << "\n";
  }
}

void xnn_x32_packa_in_T_gemm_im2col_s2_d1_x2v(uint32_t batch_size, const size_t input_height, const size_t input_width, size_t group_input_channels, \
  const int output_height, const int output_width,
  const size_t kernel_height, const size_t kernel_width, const size_t stride_height, const size_t stride_width, \
  const int dilation_height, const int dilation_width, const int input_padding_top,const int input_padding_left, \
  uint32_t* input, uint32_t* output){
  const size_t output_size = output_height*output_width;
  const size_t input_size = batch_size * input_height*input_width;
  uint32_t* in_ptr = input;
  uint32_t* in_ptr_rem = input;
  uint32_t* out_ptr = output;
  const int vlmax = __riscv_vsetvlmax_e32m2();
  int output_cur = 0;
  int input_cursor = 0;
  int valid_height = input_padding_top + input_height - 1;
  int last_stride = kernel_height - 1 + (output_height - 1)*2;
  int k_h_padding_end = last_stride - valid_height;
  for(size_t batch = 0; batch < batch_size; batch++){
      int base = batch * input_height*input_width;
      int output_padding_top_stride = ((input_padding_top + 1) >> 1) * output_width;
      int output_padding_down_stride = output_padding_top_stride;
      while(output_padding_top_stride > 0){
          int out_h = output_cur / output_width;
          int out_w = output_cur % output_width;
          input_cursor = base + (out_w << 1);
          int padded_k_h = input_padding_top - (out_h << 1);
          in_ptr = input + input_cursor;
          out_ptr += vlmax * group_input_channels * kernel_width * padded_k_h;
          for(int k_h = padded_k_h; k_h < kernel_height; k_h++){
              for(int k_w = 0; k_w < kernel_width; k_w++){
                  int width_padding_start = -(out_w == 0) & ((zero_max(input_padding_left-k_w) + 1) >> 1);
                  int width_padding_end = -((output_cur + vlmax) / output_width != out_h) & ((zero_max(k_w + ((output_width-1) << 1) - (input_padding_left + input_width-1)) + 1) >> 1);
                  int input_offset_with_pad = stride_width * width_padding_start - (input_padding_left - k_w);
                  int input_offset_with_pad_cond = -(k_w < input_padding_left);
                  int input_cur_offset = (input_offset_with_pad_cond & input_offset_with_pad) + \
                                          (~input_offset_with_pad_cond & (k_w - input_padding_left));
                  uint32_t* in_ptr_now = in_ptr + input_cur_offset;
                  for(size_t in_ch = 0; in_ch < group_input_channels; in_ch++){
                      out_ptr += width_padding_start;
                      int vl = vlmax-width_padding_end-width_padding_start;
                      // std::cout << "vl = " << vl << "\n";
                      vuint32m2_t v_w0 = __riscv_vlse32_v_u32m2 (in_ptr_now, stride_width << 2, vl);
                      __riscv_vse32_v_u32m2(out_ptr, v_w0, vl);
                      out_ptr += vl + width_padding_end;
                      in_ptr_now += input_size;
                  }
              }
              in_ptr += (output_width << 1);
          }
          output_padding_top_stride -= vlmax;
          output_cur += vlmax;
      }
      input_cursor = base + input_width;
      while(((input_padding_top + 1) >> 1) + input_cursor / input_width + kernel_height - base / input_width <= valid_height){
          int out_h = output_cur / output_width;
          int out_w = output_cur % output_width;
          int input_cursor_rem = base + input_width + ((input_width << 1) * (out_h - 1));
          in_ptr = input + input_cursor;
          for(int k_h = 0; k_h < kernel_height; k_h++){
              for(int k_w = 0; k_w < kernel_width; k_w++){
                  int width_padding_start = -(out_w == 0) & ((zero_max(input_padding_left-k_w) + 1) >> 1);
                  int width_padding_end = -((output_cur + vlmax) / output_width != out_h) & ((zero_max(k_w + ((output_width-1) << 1) - (input_padding_left + input_width-1)) + 1) >> 1);
                  int input_offset_with_pad = stride_width * width_padding_start - (input_padding_left - k_w);
                  int input_offset_with_pad_cond = -(k_w < input_padding_left);
                  int input_cur_offset = (input_offset_with_pad_cond & input_offset_with_pad) + \
                                          (~input_offset_with_pad_cond & (k_w - input_padding_left));
                  uint32_t* in_ptr_now = in_ptr + input_cur_offset;
                  for(size_t in_ch = 0; in_ch < group_input_channels; in_ch++){
                      out_ptr += width_padding_start;
                      int vl = vlmax-width_padding_end-width_padding_start;
                      vuint32m2_t v_w0 = __riscv_vlse32_v_u32m2 (in_ptr_now, stride_width << 2, vl);
                      __riscv_vse32_v_u32m2(out_ptr, v_w0, vl);
                      out_ptr += vl + width_padding_end;
                      in_ptr_now += input_size;
                  }
              }
              in_ptr += output_width << 1;
          }
          output_padding_top_stride -= vlmax;
          output_cur += vlmax;
          input_cursor = base + input_width + ((input_width << 1) * (output_cur / output_width - 2)) + ((output_cur % output_width) << 1);
      }
      while(output_cur < output_size){
          int out_h = output_cur / output_width;
          int out_w = output_cur % output_width;
          input_cursor = base + input_width + ((input_width << 1) * (out_h - 2)) + (out_w << 1);
          int input_cursor_rem = base + input_width + ((input_width << 1) * (out_h - 1));
          in_ptr = input + input_cursor;
          for(int k_h = 0; k_h < kernel_height-k_h_padding_end; k_h++){
              for(int k_w = 0; k_w < kernel_width; k_w++){
                  int width_padding_start = -(out_w == 0) & ((zero_max(input_padding_left-k_w) + 1) >> 1);
                  int width_padding_end = -((output_cur + vlmax) / output_width != out_h) & ((zero_max(k_w + ((output_width-1) << 1) - (input_padding_left + input_width-1)) + 1) >> 1);
                  int input_offset_with_pad = stride_width * width_padding_start - (input_padding_left - k_w);
                  int input_offset_with_pad_cond = -(k_w < input_padding_left);
                  int input_cur_offset = (input_offset_with_pad_cond & input_offset_with_pad) + \
                                          (~input_offset_with_pad_cond & (k_w - input_padding_left));
                  uint32_t* in_ptr_now = in_ptr + input_cur_offset;
                  for(size_t in_ch = 0; in_ch < group_input_channels; in_ch++){
                      out_ptr += width_padding_start;
                      int vl = vlmax-width_padding_end-width_padding_start;
                      vuint32m2_t v_w0 = __riscv_vlse32_v_u32m2 (in_ptr_now, stride_width << 2, vl);
                      __riscv_vse32_v_u32m2(out_ptr, v_w0, vl);
                      out_ptr += vl + width_padding_end;
                      in_ptr_now += input_size;
                  }
              }
              in_ptr += output_width << 1;
          }
          out_ptr += k_h_padding_end * vlmax * group_input_channels * kernel_width;
          output_padding_top_stride -= vlmax;
          output_cur += vlmax;
      }
      output_cur = 0;
  }
}

void xnn_x32_packa_in_T_gemm_im2col_s1_d1_1x2v(uint32_t batch_size, const size_t input_height, const size_t input_width, size_t group_input_channels, \
  const int output_height, const int output_width,
  const size_t kernel_height, const size_t kernel_width, const size_t stride_height, const size_t stride_width, \
  const int dilation_height, const int dilation_width, const int input_padding_top,const int input_padding_left, \
  uint32_t* input, uint32_t* output){
  const size_t output_size = output_height*output_width;
  const size_t input_size = input_height*input_width*batch_size;
  uint32_t* in_ptr = input;
  uint32_t* out_ptr = output;
  const int nr = __riscv_vsetvlmax_e32m2();
  const int vlmax = __riscv_vsetvlmax_e32m1();
  int valid_height = input_padding_top + input_height;
  int valid_width = input_padding_left + input_width;
  int height_padding_start = input_padding_top;
  int height_padding_end = zero_max(output_height-valid_height);
  int output_padding_top_stride = height_padding_start * output_width;
  int output_padding_down_stride = height_padding_end * output_width;
  int width_padding_start;
  int width_padding_end;
  int input_stride;
  int remainder = 0;
  int input_cursor = 0;
  int out_h, out_w, batch;
  int im2col_cur = 0;
  int output_cur = 0;

  for(batch = 0; batch < batch_size; batch++){
  // top_pad
  /*
      im2col_cur < nr * ceil(output_width / nr)
  --> im2col_cur < nr * ((output_width + nr - 1) / nr)
  */ 
      while(output_padding_top_stride){
          out_h = output_cur / output_width;
          out_w = output_cur % output_width;
          remainder = output_padding_top_stride;
          uint32_t* in_ptr_now = in_ptr + input_cursor;
          for(int k_h = 0; k_h < kernel_height; k_h++){
              int moved_input_ptr_step = 0;
              int is_partial_stride_padded_part = -(output_padding_top_stride && zero_max(input_padding_top-k_h));
              for(int k_w = 0; k_w < kernel_width; k_w++){
                  int output_padding_left = zero_max(input_padding_left-k_w);
                  int output_padding_right = zero_max(k_w + output_width-valid_width);
                  int remainder_padding_left = output_padding_left & (-(remainder > 0 && out_w == 0));
                  int remainder_padding_right = output_padding_right & (-(remainder > 0));
                  int vl;
                  for(int in_ch = 0; in_ch < group_input_channels; in_ch++){
                      out_ptr += remainder_padding_left;
                      vl = remainder \
                          - (is_partial_stride_padded_part & (output_padding_top_stride - remainder_padding_left - remainder_padding_right)) \
                          - remainder_padding_right - remainder_padding_left;
                      vuint32m1_t v_w0 = __riscv_vle32_v_u32m1(in_ptr_now, vl);
                      __riscv_vse32_v_u32m1(out_ptr, v_w0, vl);
                      out_ptr += remainder - remainder_padding_left; // including remainder_padding_right
                      int is_added = vl + ((output_padding_left + remainder_padding_right) & -(vl > 0));
                      in_ptr_now = in_ptr_now + is_added;
                      for(int cur_vl = 0; cur_vl < nr - remainder; cur_vl += output_width){
                          out_ptr += output_padding_left;
                          int is_whole_stride = -(nr - remainder - cur_vl >= output_width);
                          int segment_padding_right = (output_padding_right & is_whole_stride);
                          vl = (is_whole_stride & (output_width - output_padding_left - segment_padding_right)) + \
                              (~is_whole_stride & nr - remainder - cur_vl - output_padding_left);
                          v_w0 = __riscv_vle32_v_u32m1(in_ptr_now, vl);
                          __riscv_vse32_v_u32m1(out_ptr, v_w0, vl);
                          in_ptr_now += vl + ((output_padding_left + segment_padding_right) & is_whole_stride);
                          out_ptr += vl + segment_padding_right;
                          // + vl - (output_width - output_padding_left - remainder_padding_right) + output_width = vl + output_padding_left
                          is_added += vl + ((output_padding_left + segment_padding_right) & is_whole_stride);
                      }
                      in_ptr_now += input_size - is_added;
                  }
                  in_ptr_now = in_ptr_now - input_size*group_input_channels + 1 - (output_padding_left & (-(out_w == 0) || is_partial_stride_padded_part));
                  moved_input_ptr_step += (1 - (output_padding_left & (-(out_w == 0) || is_partial_stride_padded_part)));
              }
              in_ptr_now = in_ptr_now - moved_input_ptr_step - (is_partial_stride_padded_part & (output_padding_top_stride + (input_padding_left & -(out_w != 0)))) + input_width;
              output_padding_top_stride -= (is_partial_stride_padded_part & output_padding_top_stride);
          }
          int input_offset = input_width * ((nr - remainder) / input_width) + ((nr - remainder) % input_width - input_padding_left);
          input_cursor = input_cursor + input_offset;
          im2col_cur += nr;
          output_cur += nr;
      }
      // middle part
      while((output_cur + nr) / output_width < output_height - input_padding_top){
          out_h = output_cur / output_width;
          out_w = output_cur % output_width;
          remainder = output_width - ((nr - remainder) % output_width);
          int k_h = 0, k_w = 0;
          int input_offset = 0;
          uint32_t* in_ptr_now = in_ptr + input_cursor;
          int is_in_left_part = -((out_w == 0) || (remainder > 0));
          int is_in_right_part1 = -((out_h != (output_cur + nr) / output_width));
          int is_in_right_part2 = -((output_cur + nr) % output_width > 0);
          for(k_h = 0; k_h < kernel_height; k_h++){
              int moved_input_ptr_step = 0;
              for(k_w = 0; k_w < kernel_width; k_w++){
                  int output_padding_left = (zero_max(input_padding_left-k_w));
                  int remainder_padding_left = output_padding_left & (-(remainder > 0 && out_w == 0));
                  int remainder_padding_right = (-(remainder > 0) & zero_max(k_w + output_width-valid_width));
                  int output_padding_right = zero_max(k_w + output_width-valid_width);
                  int vl;
                  for(int in_ch = 0; in_ch < group_input_channels; in_ch++){
                      out_ptr += remainder_padding_left;
                      vl = zero_max(remainder - remainder_padding_right - remainder_padding_left);
                      vuint32m1_t v_w0 = __riscv_vle32_v_u32m1(in_ptr_now, vl);
                      __riscv_vse32_v_u32m1(out_ptr, v_w0, vl);
                      // remainder with padding
                      out_ptr = out_ptr + remainder - remainder_padding_left;
                      // in_ptr_now + vl - (output_width - output_padding_left - remainder_padding_right + input_width)
                      // two segments
                      int is_added = vl + ((output_padding_left + remainder_padding_right) & -(remainder > 0));
                      in_ptr_now = in_ptr_now + is_added;
                      // std::cout << "vl for remainder = " << vl << "\n";
                      for(int cur_vl = 0; cur_vl < nr - remainder; cur_vl += output_width){
                          out_ptr += output_padding_left;
                          int is_whole_stride = -(nr - remainder - cur_vl >= output_width);
                          int segment_padding_right = (output_padding_right & is_whole_stride);
                          // std::cout << "vl = " << vl << "\n";
                          vl = (is_whole_stride & (output_width - output_padding_left - segment_padding_right)) + \
                              (~is_whole_stride & nr - remainder - cur_vl - output_padding_left);
                          v_w0 = __riscv_vle32_v_u32m1(in_ptr_now, vl);
                          __riscv_vse32_v_u32m1(out_ptr, v_w0, vl);
                          in_ptr_now += vl + ((output_padding_left + segment_padding_right) & is_whole_stride);
                          out_ptr += vl + segment_padding_right;
                          // + vl - (output_width - output_padding_left - remainder_padding_right) + output_width = vl + output_padding_left
                          is_added += vl + ((output_padding_left + segment_padding_right) & is_whole_stride);
                      }
                      in_ptr_now = in_ptr_now - is_added + input_size;
                  }
                  in_ptr_now = in_ptr_now - input_size*group_input_channels + 1 - (output_padding_left & (-(out_w == 0)));
                  moved_input_ptr_step += (1 - (output_padding_left & (-(out_w == 0))));
              }
              in_ptr_now = in_ptr_now - moved_input_ptr_step + input_width;
          }
          int cond = -(input_cursor % input_width == 0);
          input_offset = ((~cond & (nr + (~is_in_right_part2 & input_padding_left))) | \
                          (cond & (nr - remainder - input_padding_left + (output_width * ((remainder + (-((remainder + input_padding_left) % output_width == 0) & 1)) / output_width)))));
          input_cursor = input_cursor + input_offset;
          im2col_cur += nr;
          output_cur += nr;
      }
      // end part
      while(output_cur < output_size){
          out_h = output_cur / output_width;
          out_w = output_cur % output_width;
          int is_next_batch = -(im2col_cur + vlmax > output_size*(batch+1) && batch + 1 != batch_size);
          const int r = (output_cur + nr) % output_width;
          remainder = output_width - ((nr - remainder) % output_width);
          int k_h = 0, k_w = 0;
          int input_offset = 0;
          uint32_t* in_ptr_now = in_ptr + input_cursor;
          int is_in_left_part = -((out_w == 0) || (remainder > 0));
          int is_in_right_part1 = -((out_h != (output_cur + nr) / output_width));
          int is_in_right_part2 = -((output_cur + nr) % output_width > 0);
          int is_whole_stride_padded = ~is_next_batch & -(out_h >= output_height - input_padding_top);            int last_part_in_this_batch = output_size % vlmax;
          for(k_h = 0; k_h < kernel_height - (is_whole_stride_padded & input_padding_top); k_h++){
              int moved_input_ptr_step = 0;
              for(k_w = 0; k_w < kernel_width; k_w++){
                  int output_padding_left = zero_max(input_padding_left-k_w);
                  int output_padding_right = zero_max(k_w + output_width-valid_width);
                  int remainder_padding_left = output_padding_left & (-(remainder > 0 && out_w == 0));
                  int remainder_padding_right = (-(remainder > 0) & zero_max(k_w + output_width-valid_width));
                  int vl;
                  for(int in_ch = 0; in_ch < group_input_channels; in_ch++){
                      int is_padded_remainder = -((zero_max(k_h - input_padding_top)) & -(output_size - output_width * input_padding_top <= output_cur));
                      out_ptr += remainder_padding_left;
                      vl = zero_max(remainder - remainder_padding_right - remainder_padding_left);
                      vuint32m1_t v_w0 = __riscv_vle32_v_u32m1(in_ptr_now, ~is_padded_remainder & vl);
                      __riscv_vse32_v_u32m1(out_ptr, v_w0, ~is_padded_remainder & vl);
                      // remainder with padding
                      out_ptr = out_ptr + remainder - remainder_padding_left;
                      // in_ptr_now + vl - (output_width - output_padding_left - remainder_padding_right + input_width)
                      // two segments
                      int is_added = vl + ((output_padding_left + remainder_padding_right) & -(remainder > 0));
                      in_ptr_now = in_ptr_now + is_added;
                      for(int cur_vl = 0; cur_vl < nr - remainder; cur_vl += output_width){
                          out_ptr += output_padding_left;
                          int is_whole_stride = -(nr - remainder - cur_vl >= output_width);
                          int segment_padding_right = (output_padding_right & is_whole_stride);
                          int current_output_offset = output_cur + remainder + cur_vl;
                          // std::cout << "vl = " << vl << "\n";
                          vl = (is_whole_stride & (output_width - output_padding_left - segment_padding_right)) + \
                              (~is_whole_stride & nr - remainder - cur_vl - output_padding_left);
                          int is_exceed_boarder = -(current_output_offset >= output_size);
                          int is_exceed_batch_output_size = -(batch + 1 >= batch_size) & is_exceed_boarder;
                          int is_padded_end = -(zero_max(k_h - input_padding_top) & -(output_size - output_width * input_padding_top <= current_output_offset && current_output_offset < output_size));
                          int is_next_batch_padding_top = -(zero_max(input_padding_top-k_h) & -(batch + 1 != batch_size && (current_output_offset % output_size < output_width*input_padding_top)) & is_exceed_boarder);
                          v_w0 = __riscv_vle32_v_u32m1(in_ptr_now, (~is_padded_end & ~is_exceed_batch_output_size & ~is_next_batch_padding_top) & vl);
                          __riscv_vse32_v_u32m1(out_ptr, v_w0, (~is_padded_end & ~is_exceed_batch_output_size & ~is_next_batch_padding_top) & vl);
                          in_ptr_now += vl + ((output_padding_left + segment_padding_right) & is_whole_stride);
                          out_ptr += vl + segment_padding_right;
                          // + vl - (output_width - output_padding_left - remainder_padding_right) + output_width = vl + output_padding_left
                          is_added += vl + ((output_padding_left + segment_padding_right) & is_whole_stride);
                      }
                      in_ptr_now = in_ptr_now - is_added + input_size;
                  }
                  // at the start of each stride, we need to take padding_left into consideration
                  in_ptr_now = in_ptr_now - input_size*group_input_channels + 1 - (output_padding_left & (-(out_w == 0)));
                  moved_input_ptr_step += (1 - (output_padding_left & (-(out_w == 0))));
              }
              // std::cout << "moved_input_ptr_step = " << moved_input_ptr_step << "\n";
              in_ptr_now = in_ptr_now - moved_input_ptr_step + input_width;
          }
          int cond = -(input_cursor % input_width == 0);
          input_offset = ((~cond & (nr + (~is_in_right_part2 & input_padding_left))) | \
                              (cond & nr));
          input_cursor = input_cursor + input_offset;
          out_ptr += (is_whole_stride_padded & min(vlmax, output_width) * group_input_channels * kernel_width);
          im2col_cur += nr;
          output_cur += nr;
      }
      int last_part_in_this_batch = (output_size*(batch + 1)) % nr;
      int finished_part_in_next_batch = -(last_part_in_this_batch > 0) & (nr - last_part_in_this_batch);
      output_cur = finished_part_in_next_batch;
      output_padding_top_stride = zero_max(height_padding_start * output_width - finished_part_in_next_batch);
  }
}

void xnn_x32_packa_in_T_gemm_im2col_s1_d1_2x2v(uint32_t batch_size, const size_t input_height, const size_t input_width, size_t group_input_channels, \
  const int output_height, const int output_width,
  const size_t kernel_height, const size_t kernel_width, const size_t stride_height, const size_t stride_width, \
  const int dilation_height, const int dilation_width, const int input_padding_top,const int input_padding_left, \
  uint32_t* input, uint32_t* output){
  const size_t output_size = output_height*output_width;
  const size_t input_size = input_height*input_width*batch_size;
  uint32_t* in_ptr = input;
  uint32_t* out_ptr = output;
  const int vlmax = __riscv_vsetvlmax_e32m2();
  int valid_height = input_padding_top + input_height;
  int valid_width = input_padding_left + input_width;
  int height_padding_start = input_padding_top;
  int height_padding_end = zero_max(output_height-valid_height);
  int output_padding_top_stride = height_padding_start * output_width;
  int output_padding_down_stride = height_padding_end * output_width;
  int width_padding_start;
  int width_padding_end;
  int input_stride;
  int remainder = 0;
  int input_cursor = 0;
  int out_h, out_w, batch;
  int im2col_cur = 0;
  int batch_cur = 0;
  int output_cur = 0;
  batch = im2col_cur / output_size;
  // top_pad
  /*
      output_cur < vlmax * ceil(output_width / vlmax)
  --> output_cur < vlmax * ((output_width + vlmax - 1) / vlmax)
  */ 
  int input_cursor_real_part = 0;
  for(batch = 0; batch < batch_size; batch ++){
      while(output_padding_top_stride){
          out_h = output_cur / output_width;
          out_w = output_cur % output_width;
          int is_whole_stride_padded_part = -(output_padding_top_stride/vlmax >= 1);
          remainder = ~is_whole_stride_padded_part & output_padding_top_stride;
          uint32_t* in_ptr_now = in_ptr + input_cursor;
          uint32_t* in_ptr_now_real_part = in_ptr + output_size*batch + (-(remainder == 0) & input_cursor_real_part);
          // std::cout << "first: output_cur = " << output_cur << ", remainder = " << remainder << "\n";
          out_ptr += (is_whole_stride_padded_part & (min(vlmax, output_width) * group_input_channels * kernel_width));
          output_padding_top_stride -= (is_whole_stride_padded_part & vlmax);
          for(int k_h = is_whole_stride_padded_part & input_padding_top; k_h < kernel_height; k_h++){
              int moved_input_ptr_step = 0;
              int moved_input_ptr_real_step = 0;
              int is_partial_stride_padded_part = -(output_padding_top_stride && zero_max(input_padding_top-k_h));
              for(int k_w = 0; k_w < kernel_width; k_w++){
                  int output_padding_left = zero_max(input_padding_left-k_w) & (-((output_cur + remainder) % output_width == 0));
                  int output_padding_right = zero_max(k_w + output_width-valid_width) & (-((output_cur + vlmax) % output_width == 0));
                  int remainder_padding_left = output_padding_left & (-(remainder > 0 && out_w == 0));
                  int remainder_padding_right = (-(remainder > 0) & zero_max(k_w + output_width-valid_width));
                  int vl;
                  for(int in_ch = 0; in_ch < group_input_channels; in_ch++){
                      out_ptr += output_padding_left & -(remainder > 0 && out_w == 0);
                      vl = remainder \
                          - (is_partial_stride_padded_part & (output_padding_top_stride - remainder_padding_left - remainder_padding_right)) \
                          - remainder_padding_right - remainder_padding_left;
                      // std::cout << "output_padding_left = " << output_padding_left << ", vl = " << vl << "\n";
                      vuint32m2_t v_w0 = __riscv_vle32_v_u32m2(in_ptr_now, vl);
                      __riscv_vse32_v_u32m2(out_ptr, v_w0, vl);
                      out_ptr += remainder - remainder_padding_left; // including remainder_padding_right
                      out_ptr += output_padding_left;
                      vl = vlmax - remainder - output_padding_left - output_padding_right;
                      v_w0 = __riscv_vle32_v_u32m2(in_ptr_now_real_part, vl);
                      __riscv_vse32_v_u32m2(out_ptr, v_w0, vl);
                      out_ptr += vl + output_padding_right;
                      in_ptr_now += input_size;
                      in_ptr_now_real_part += input_size;
                  }
                  in_ptr_now = in_ptr_now - input_size*group_input_channels + 1 - (output_padding_left & (-(out_w == 0) || is_partial_stride_padded_part));
                  in_ptr_now_real_part = in_ptr_now_real_part - input_size*group_input_channels + 1 - (output_padding_left & (-((output_cur + remainder) % output_width == 0)));
                  moved_input_ptr_step += (1 - (output_padding_left & (-(out_w == 0) || is_partial_stride_padded_part)));
                  moved_input_ptr_real_step += (1 - (output_padding_left & (-((output_cur + remainder) % output_width == 0))));
              }
              in_ptr_now = in_ptr_now - moved_input_ptr_step - (is_partial_stride_padded_part & (output_padding_top_stride + (input_padding_left & -(out_w != 0)))) + input_width;
              in_ptr_now_real_part = in_ptr_now_real_part - moved_input_ptr_real_step + input_width;
              output_padding_top_stride -= (is_partial_stride_padded_part & output_padding_top_stride);
          }
          // todo : input_offset, end height padding
          int input_offset = -(output_padding_top_stride == 0 && output_width % vlmax != 0) & (vlmax - remainder + (-(vlmax - remainder < output_width) & -input_padding_left));
          input_cursor = input_cursor + input_offset;
          input_cursor_real_part += (vlmax - remainder - (-(output_cur == 0) & input_padding_left));
          im2col_cur += vlmax;
          output_cur += vlmax;
          // std::cout << "first: input_offset= " << input_offset << "\n";
      }
      // middle part
      while(((output_cur + vlmax) % output_width != 0 && (output_cur + vlmax) / output_width < output_height - input_padding_top) || \
            ((output_cur + vlmax) % output_width == 0 && (output_cur + vlmax) / output_width <= output_height - input_padding_top)){
          out_h = output_cur / output_width;
          out_w = output_cur % output_width;
          // -(out_h < input_padding_top) --> for the condition with padding
          // 0xFFFFFFFF if cond=1, else 0x00000000
          const int r = (output_cur + vlmax) % output_width;
          remainder = ((-(out_h != (output_cur + vlmax) / output_width)) & \
                  (((-(r != 0)) & (vlmax - r)) + ((-(r == 0)) & zero_max(vlmax - output_width))));
          int k_h = 0, k_w = 0;
          int input_offset = 0;
          uint32_t* in_ptr_now = in_ptr + input_cursor;
          // std::cout << "middle: output_cur = " << output_cur << ", r = " << r << ", remainder = " << remainder << "\n";
          // || -- pad -- | -- input ----- ||                         --> pad before second load/store
          // || -- remainder ------------- || -- pad -- | - input -|| --> pad before second load/store
          int is_in_left_part = -((out_w == 0) || (remainder > 0));
          // || --- input ---   |-- pad -- ||                         --> pad after second load/store
          // || -- remainder -- |-- pad -- || ------ input ------- || --> pad before second load/store
          int is_in_right_part1 = -((out_h != (output_cur + vlmax) / output_width));
          int is_in_right_part2 = -((output_cur + vlmax) % output_width > 0);
          for(k_h = 0; k_h < kernel_height; k_h++){
              int moved_input_ptr_step = 0;
              for(k_w = 0; k_w < kernel_width; k_w++){
                  int output_padding_left = (is_in_left_part & zero_max(input_padding_left-k_w));
                  int remainder_padding_left = output_padding_left & (-(remainder > 0 && out_w == 0));
                  int remainder_padding_right = (-(remainder > 0) & zero_max(k_w + output_width-valid_width));
                  int output_padding_right = (-(remainder == 0 || (remainder > 0 && ~is_in_right_part2)) & is_in_right_part1 & zero_max(k_w + output_width-valid_width));
                  int vl;
                  // std::cout << "remainder_padding_right = " << output_padding_right << "\n";
                  for(int in_ch = 0; in_ch < group_input_channels; in_ch++){
                      out_ptr += remainder_padding_left;
                      vl = zero_max(remainder - remainder_padding_right - remainder_padding_left);
                      vuint32m2_t v_w0 = __riscv_vle32_v_u32m2(in_ptr_now, vl);
                      __riscv_vse32_v_u32m2(out_ptr, v_w0, vl);
                      // remainder with padding
                      out_ptr = out_ptr + remainder - remainder_padding_left + output_padding_left;
                      // in_ptr_now + vl - (output_width - output_padding_left - remainder_padding_right + input_width)
                      // two segments
                      int is_added = vl + ((output_padding_left + remainder_padding_right) & -(remainder > 0));
                      in_ptr_now = in_ptr_now + is_added;
                      // std::cout << "vl for remainder = " << vl << "\n";
                      vl = vlmax - remainder - output_padding_left - output_padding_right;
                      // std::cout << "vl = " << vl << "\n";
                      v_w0 = __riscv_vle32_v_u32m2(in_ptr_now, vl);
                      __riscv_vse32_v_u32m2(out_ptr, v_w0, vl);

                      out_ptr += vl + output_padding_right;
                      in_ptr_now = in_ptr_now - is_added + input_size;
                  }
                  // at the start of each stride, we need to take padding_left into consideration
                  in_ptr_now = in_ptr_now - input_size*group_input_channels + 1 - (output_padding_left & (-(out_w == 0)));
                  moved_input_ptr_step += (1 - (output_padding_left & (-(out_w == 0))));
              }
              // std::cout << "moved_input_ptr_step = " << moved_input_ptr_step << "\n";
              in_ptr_now = in_ptr_now - moved_input_ptr_step + input_width;
          }
          // || -- pad -- | -- input ----- ||
          //      --> + len(input)
          // || -- remainder ------------- || -- pad -- | - input -|| 
          //      --> + remainder - (output_width - input_padding_left) + input_width + (vlmax - remainder - input_padding_left) \
                      = remainder - output_width + input_padding_left + input_width + vlmax - remainder - input_padding_left \
                      = vlmax
          // || ----------- input -------- ||
          //      --> + vlmax
          // || --- pad( == remainder) --- || -- pad -- | - input -||
          int cond = -(input_cursor % input_width == 0);
          input_offset = ((~cond & (vlmax + (~is_in_right_part2 & input_padding_left))) | \
                          (cond & (vlmax - remainder - input_padding_left + (output_width * ((remainder + (-((remainder + input_padding_left) % output_width == 0) & 1)) / output_width)))));
          input_cursor = input_cursor + input_offset;
          // std::cout << "middle: input_offset= " << input_offset << "\n";
          // std::cout << "middle: input_cursor= " << input_cursor << "\n";
          im2col_cur += vlmax;
          output_cur += vlmax;
      }

      // end part
      while(output_cur < output_size){
          out_h = output_cur / output_width;
          out_w = output_cur % output_width;
          int is_next_batch = -(im2col_cur + vlmax > output_size*(batch+1) && batch + 1 != batch_size);
          // std::cout << "is_next_batch = " << (is_next_batch & 1) << "\n";
          const int r = (output_cur + vlmax) % output_width;
          remainder = ((-(out_h != (output_cur + vlmax) / output_width)) & \
                  (((-(r != 0)) & (vlmax - r)) + ((-(r == 0)) & zero_max(vlmax - output_width))));
          int k_h = 0, k_w = 0;
          int input_offset = 0;
          uint32_t* in_ptr_now = in_ptr + input_cursor;
          // std::cout << "end: output_cur = " << output_cur << ", r = " << r << ", remainder = " << remainder << "\n";
          int is_in_left_part = -((out_w == 0) || (remainder > 0));
          int is_in_right_part1 = -(vlmax - remainder >= output_width || (remainder == 0 && (out_h != (output_cur + vlmax) / output_width))); 
          int is_in_right_part2 = -((output_cur + vlmax) % output_width > 0);
          int is_whole_stride_padded = ~is_next_batch & -(out_h >= output_height - input_padding_top);
          int last_part_in_this_batch = output_size % vlmax;
          for(k_h = 0; k_h < kernel_height - (is_whole_stride_padded & input_padding_top); k_h++){
              int moved_input_ptr_step = 0;
              for(k_w = 0; k_w < kernel_width; k_w++){
                  int output_padding_left = is_in_left_part & zero_max(input_padding_left-k_w);
                  int output_padding_right = is_in_right_part1 & zero_max(k_w + output_width-valid_width);
                  int remainder_padding_left = output_padding_left & (-(remainder > 0 && out_w == 0));
                  int remainder_padding_right = zero_max(k_w + output_width-valid_width) & (-(remainder > 0 && out_h != (output_cur + vlmax) / output_width));
                  int vl;
                  // std::cout << "k_h = " << k_h << ", k_w = " << k_w << ", output_padding_left = " << output_padding_left << "\n";
                  // std::cout << "output_padding_right = " << output_padding_right << "\n";
                  for(int in_ch = 0; in_ch < group_input_channels; in_ch++){
                      int is_padded_remainder = -(
                          (zero_max(k_h - input_padding_top)) && out_h >= output_height - input_padding_top
                      );
                      vl = zero_max(remainder - remainder_padding_right - remainder_padding_left);
                      out_ptr += remainder_padding_left;
                      vuint32m2_t v_w0 = __riscv_vle32_v_u32m2(in_ptr_now, ~is_padded_remainder & vl);
                      __riscv_vse32_v_u32m2(out_ptr, v_w0, ~is_padded_remainder & vl);
                      // remainder with padding
                      out_ptr = out_ptr + remainder - remainder_padding_left + output_padding_left;
                      // in_ptr_now + vl - (output_width - output_padding_left - output_padding_right + input_width)
                      // two segments
                      int is_added = vl + ((output_padding_left + remainder_padding_right) & -(remainder > 0));
                      in_ptr_now = in_ptr_now + is_added;
                      int is_padded = -(
                                      ((zero_max(k_h - input_padding_top)) & -(((output_cur + vlmax) / output_width + 1 > output_height - input_padding_top))) \
                                      || ((output_cur + remainder >= output_size) && (batch + 1 >= batch_size)) \
                                      || (zero_max(input_padding_top-k_h) & -(output_cur + remainder >= output_size && batch + 1 != batch_size))
                                      );
                      int is_in_next_batch = -(output_cur + remainder >= output_size && batch + 1 != batch_size);
                      vl = vlmax - remainder - output_padding_left - output_padding_right;
                      v_w0 = __riscv_vle32_v_u32m2(in_ptr_now, (~is_padded | is_padded_remainder) & vl);
                      __riscv_vse32_v_u32m2(out_ptr, v_w0, (~is_padded | is_padded_remainder) & vl);

                      out_ptr += (vl + output_padding_right);
                      in_ptr_now = in_ptr_now - is_added + input_size;
                  }
                  // at the start of each stride, we need to take padding_left into consideration
                  in_ptr_now = in_ptr_now - input_size*group_input_channels + 1 - (output_padding_left & (-(out_w == 0)));
                  moved_input_ptr_step += (1 - (output_padding_left & (-(out_w == 0))));
              }
              // std::cout << "moved_input_ptr_step = " << moved_input_ptr_step << "\n";
              in_ptr_now = in_ptr_now - moved_input_ptr_step + input_width;
          }
          int cond = -(input_cursor % input_width == 0);
          input_offset = ((~cond & (vlmax + (~is_in_right_part2 & input_padding_left))) | \
                              ((cond) & (vlmax - remainder - input_padding_left + (output_width * ((remainder + (-((remainder + input_padding_left) % output_width == 0) & 1)) / output_width)))));
          input_cursor = input_cursor + input_offset;
          out_ptr += (is_whole_stride_padded & min(vlmax, output_width) * group_input_channels * kernel_width);
          im2col_cur += vlmax;
          output_cur += vlmax;
          // std::cout << "end: input_offset = " << input_offset << "\n";

      }
      int last_part_in_this_batch = (output_size*(batch + 1)) % vlmax;
      int finished_part_in_next_batch = -(last_part_in_this_batch > 0) & (vlmax - last_part_in_this_batch);
      output_cur = finished_part_in_next_batch;
      output_padding_top_stride = zero_max(height_padding_start * output_width - finished_part_in_next_batch);
      // std::cout << "end: finished_part_in_next_batch = " << finished_part_in_next_batch << "\n";
      input_cursor = output_size*(batch + 1);// + ((vlmax - remainder) % output_width);// output_size = output_width*output_height = input_width*input_height
      input_cursor_real_part = finished_part_in_next_batch - (finished_part_in_next_batch + output_width - 1) / output_width;
      // std::cout << "output_padding_top_stride = " << output_padding_top_stride << ", input_cursor = " << input_cursor << "\n";
  }
}

void xnn_x32_packa_in_T_gemm_im2col_s2_d1_1x4v(uint32_t batch_size, const size_t input_height, const size_t input_width, size_t group_input_channels, \
  const int output_height, const int output_width,
  const size_t kernel_height, const size_t kernel_width, const size_t stride_height, const size_t stride_width, \
  const int dilation_height, const int dilation_width, const int input_padding_top,const int input_padding_left, \
  uint32_t* input, uint32_t* output){
  const size_t output_size = output_height*output_width;
  const size_t input_size = batch_size * input_height*input_width;
  uint32_t* in_ptr = input;
  uint32_t* in_ptr_rem = input;
  uint32_t* out_ptr = output;
  const int nr = __riscv_vsetvlmax_e32m4();
  const int vlmax = __riscv_vsetvlmax_e32m1();
  int output_cur = 0;
  int im2col_cur = 0;
  int input_cursor = 0;
  int valid_height = input_padding_top + input_height - 1;
  int last_stride = kernel_height - 1 + (output_height - 1)*2;
  int k_h_padding_end = last_stride - valid_height;
  int remainder = 0;
  int output_padding_top_stride = ((input_padding_top + 1) >> 1) * output_width;
  const int output_padding_top_stride_tmp = output_padding_top_stride;
  for(size_t batch = 0; batch < batch_size; batch++){
      int base = batch * input_height*input_width;
      while(output_padding_top_stride > 0){
          int out_h = output_cur / output_width;
          int out_w = output_cur % output_width;
          input_cursor = base + (out_w << 1);
          remainder = -(output_width - out_w < nr) & (nr - output_width + out_w);
          // replace `* stride` to `<< 1` by the fact that stride = 2
          int padded_k_h = zero_max(input_padding_top - (out_h << 1) - (-(remainder > 0) & ((out_h + 1) << 1)));
          in_ptr = input + input_cursor;
          in_ptr_rem = input + base + (stride_width - input_padding_top + padded_k_h) * input_width;
          out_ptr += nr * group_input_channels * kernel_width * padded_k_h;
          for(int k_h = padded_k_h; k_h < kernel_height; k_h++){
              int padded = -(remainder > 0 && k_h < input_padding_top - (out_h << 1));
              for(int k_w = 0; k_w < kernel_width; k_w++){
                  int width_padding_start = -(out_w == 0) & ((zero_max(input_padding_left-k_w) + 1) >> 1);
                  int width_padding_end = -((output_cur + nr) / output_width != out_h) & ((zero_max(k_w + ((output_width-1) << 1) - (input_padding_left + input_width-1)) + 1) >> 1);
                  int input_offset_with_pad = stride_width * width_padding_start - (input_padding_left - k_w);
                  int input_offset_with_pad_cond = -(k_w < input_padding_left);
                  int input_cur_offset = (input_offset_with_pad_cond & input_offset_with_pad) + \
                                          (~input_offset_with_pad_cond & (k_w - input_padding_left));
                  // for remainder
                  int width_padding_start_rem = (zero_max(input_padding_left-k_w) + 1) >> 1;
                  input_offset_with_pad = stride_width * width_padding_start_rem - (input_padding_left - k_w);
                  int input_cur_offset_rem = (input_offset_with_pad_cond & input_offset_with_pad) + \
                                          (~input_offset_with_pad_cond & (k_w - input_padding_left));
                  uint32_t* in_ptr_now = in_ptr + input_cur_offset;
                  uint32_t* in_ptr_remainder = in_ptr_rem + input_cur_offset_rem;
                  for(size_t in_ch = 0; in_ch < group_input_channels; in_ch++){
                      out_ptr += width_padding_start;
                      int vl = nr-width_padding_end-width_padding_start-remainder;
                      // std::cout << "vl = " << vl << "\n";
                      vuint32m1_t v_w0 = __riscv_vlse32_v_u32m1 (in_ptr_now, stride_width << 2, ~padded & vl);
                      __riscv_vse32_v_u32m1(out_ptr, v_w0, ~padded & vl);
                      out_ptr += vl + width_padding_end;
                      // for remainder
                      int cur = nr - remainder;
                      int cnt = 0;
                      while(cur < nr){
                          int width_padding_end_rem = -(nr - cur >= output_width) & ((zero_max(k_w + ((output_width-1) << 1) - (input_padding_left + input_width-1)) + 1) >> 1);
                          out_ptr += width_padding_start_rem;
                          vl = min(nr - cur, output_width) - width_padding_start_rem - width_padding_end_rem;
                          v_w0 = __riscv_vlse32_v_u32m1 (in_ptr_remainder + (cnt * (input_width << 1)), stride_width << 2, vl);
                          __riscv_vse32_v_u32m1(out_ptr, v_w0, vl);
                          out_ptr += vl + width_padding_end_rem;
                          cnt ++;
                          cur += output_width;
                      }
                      in_ptr_now += input_size;
                      in_ptr_remainder += input_size;
                  }
              }
              in_ptr += ~padded & input_width;
              in_ptr_rem += -(remainder > 0) & input_width;
          }
          output_padding_top_stride -= nr;
          output_cur += nr;
      }
      input_cursor = base + input_width + ((input_width << 1) * (output_cur / output_width - ((input_padding_top + 1) >> 1))) + ((output_cur % output_width) << 1);
      while(output_cur < output_size){
          int out_h = output_cur / output_width;
          int out_w = output_cur % output_width;
          input_cursor = base + input_width + ((input_width << 1) * (out_h - ((input_padding_top + 1) >> 1))) + (out_w << 1);
          int input_cursor_rem = input_cursor - (out_w << 1) + (input_width << 1);
          in_ptr = input + input_cursor;
          in_ptr_rem = input + input_cursor_rem;
          remainder = -(output_width - out_w < nr) & (nr - output_width + out_w);
          for(int k_h = 0; k_h < kernel_height-k_h_padding_end; k_h++){
              for(int k_w = 0; k_w < kernel_width; k_w++){
                  int width_padding_start = -(out_w == 0) & ((zero_max(input_padding_left-k_w) + 1) >> 1);
                  int width_padding_end = -((output_cur + nr) / output_width != out_h) & ((zero_max(k_w + ((output_width-1) << 1) - (input_padding_left + input_width-1)) + 1) >> 1);
                  int input_offset_with_pad = stride_width * width_padding_start - (input_padding_left - k_w);
                  int input_offset_with_pad_cond = -(k_w < input_padding_left);
                  int input_cur_offset = (input_offset_with_pad_cond & input_offset_with_pad) + \
                                          (~input_offset_with_pad_cond & (k_w - input_padding_left));
                  int width_padding_start_rem = -(remainder > 0) & ((zero_max(input_padding_left-k_w) + 1) >> 1);
                  input_offset_with_pad = stride_width * width_padding_start_rem - (input_padding_left - k_w);
                  int input_cur_offset_rem = (input_offset_with_pad_cond & input_offset_with_pad) + \
                                          (~input_offset_with_pad_cond & (k_w - input_padding_left));
                  uint32_t* in_ptr_now = in_ptr + input_cur_offset;
                  uint32_t* in_ptr_remainder = in_ptr_rem + input_cur_offset_rem;
                  for(size_t in_ch = 0; in_ch < group_input_channels; in_ch++){
                      out_ptr += width_padding_start;
                      int vl = nr-width_padding_end-width_padding_start-remainder;
                      vuint32m1_t v_w0 = __riscv_vlse32_v_u32m1 (in_ptr_now, stride_width << 2, vl);
                      __riscv_vse32_v_u32m1(out_ptr, v_w0, vl);
                      out_ptr += vl + width_padding_end;
                      int cur = nr - remainder;
                      int cnt = 0;
                      while(cur < nr){
                          int exceed_boundary = -(output_cur + cur >= output_size && (batch + 1 == batch_size));
                          int padded_rem = -(k_h < zero_max(input_padding_top - ((((output_cur + cur) % (output_size)) / output_width) << 1)));
                          int width_padding_end_rem = -(nr - cur >= output_width) & ((zero_max(k_w + ((output_width-1) << 1) - (input_padding_left + input_width-1)) + 1) >> 1);
                          out_ptr += width_padding_start_rem;
                          vl = min(nr - cur, output_width) - width_padding_start_rem - width_padding_end_rem;
                          v_w0 = __riscv_vlse32_v_u32m1 (in_ptr_remainder + (cnt * (input_width << 1)), stride_width << 2, (~padded_rem & ~exceed_boundary) & vl);
                          __riscv_vse32_v_u32m1(out_ptr, v_w0, (~padded_rem & ~exceed_boundary) & vl);
                          out_ptr += vl + width_padding_end_rem;
                          cnt ++;
                          cur += output_width;
                      }
                      in_ptr_now += input_size;
                      in_ptr_remainder += input_size;
                  }
              }
              in_ptr += output_width << 1;
              in_ptr_rem += -(remainder > 0) & input_width;
          }
          out_ptr += k_h_padding_end * nr * group_input_channels * kernel_width;
          output_cur += nr;
      }
      int last_part_in_this_batch = (output_size*(batch + 1)) % nr;
      int finished_part_in_next_batch = -(last_part_in_this_batch > 0) & (nr - last_part_in_this_batch);
      output_cur = finished_part_in_next_batch;
      output_padding_top_stride = zero_max(output_padding_top_stride_tmp - finished_part_in_next_batch);
  }
}


void xnn_x32_packa_in_T_gemm_im2col_s2_d1_2x4v(uint32_t batch_size, const size_t input_height, const size_t input_width, size_t group_input_channels, \
  const int output_height, const int output_width,
  const size_t kernel_height, const size_t kernel_width, const size_t stride_height, const size_t stride_width, \
  const int dilation_height, const int dilation_width, const int input_padding_top,const int input_padding_left, \
  uint32_t* input, uint32_t* output){
  const size_t output_size = output_height*output_width;
  const size_t input_size = batch_size * input_height*input_width;
  uint32_t* in_ptr = input;
  uint32_t* in_ptr_rem = input;
  uint32_t* out_ptr = output;
  const int nr = __riscv_vsetvlmax_e32m4();
  const int vlmax = __riscv_vsetvlmax_e32m2();
  int output_cur = 0;
  int im2col_cur = 0;
  int input_cursor = 0;
  int valid_height = input_padding_top + input_height - 1;
  int last_stride = kernel_height - 1 + (output_height - 1)*2;
  int k_h_padding_end = last_stride - valid_height;
  int remainder = 0;
  int output_padding_top_stride = ((input_padding_top + 1) >> 1) * output_width;
  const int output_padding_top_stride_tmp = output_padding_top_stride;
  for(size_t batch = 0; batch < batch_size; batch++){
      int base = batch * input_height*input_width;
      while(output_padding_top_stride > 0){
          int out_h = output_cur / output_width;
          int out_w = output_cur % output_width;
          input_cursor = base + (out_w << 1);
          remainder = -(output_width - out_w < nr) & (nr - output_width + out_w);
          // replace `* stride` to `<< 1` by the fact that stride = 2
          int padded_k_h = zero_max(input_padding_top - (out_h << 1) - (-(remainder > 0) & ((out_h + 1) << 1)));
          in_ptr = input + input_cursor;
          in_ptr_rem = input + base + (stride_width - input_padding_top + padded_k_h) * input_width;
          out_ptr += nr * group_input_channels * kernel_width * padded_k_h;
          for(int k_h = padded_k_h; k_h < kernel_height; k_h++){
              int padded = -(remainder > 0 && k_h < input_padding_top - (out_h << 1));
              for(int k_w = 0; k_w < kernel_width; k_w++){
                  int width_padding_start = -(out_w == 0) & ((zero_max(input_padding_left-k_w) + 1) >> 1);
                  int width_padding_end = -((output_cur + nr) / output_width != out_h) & ((zero_max(k_w + ((output_width-1) << 1) - (input_padding_left + input_width-1)) + 1) >> 1);
                  int input_offset_with_pad = stride_width * width_padding_start - (input_padding_left - k_w);
                  int input_offset_with_pad_cond = -(k_w < input_padding_left);
                  int input_cur_offset = (input_offset_with_pad_cond & input_offset_with_pad) + \
                                          (~input_offset_with_pad_cond & (k_w - input_padding_left));
                  // for remainder
                  int width_padding_start_rem = (zero_max(input_padding_left-k_w) + 1) >> 1;
                  input_offset_with_pad = stride_width * width_padding_start_rem - (input_padding_left - k_w);
                  int input_cur_offset_rem = (input_offset_with_pad_cond & input_offset_with_pad) + \
                                          (~input_offset_with_pad_cond & (k_w - input_padding_left));
                  uint32_t* in_ptr_now = in_ptr + input_cur_offset;
                  uint32_t* in_ptr_remainder = in_ptr_rem + input_cur_offset_rem;
                  for(size_t in_ch = 0; in_ch < group_input_channels; in_ch++){
                      out_ptr += width_padding_start;
                      int vl = nr-width_padding_end-width_padding_start-remainder;
                      // std::cout << "vl = " << vl << "\n";
                      vuint32m2_t v_w0 = __riscv_vlse32_v_u32m2 (in_ptr_now, stride_width << 2, ~padded & vl);
                      __riscv_vse32_v_u32m2(out_ptr, v_w0, ~padded & vl);
                      out_ptr += vl + width_padding_end;
                      // for remainder
                      int cur = nr - remainder;
                      int cnt = 0;
                      while(cur < nr){
                          int width_padding_end_rem = -(nr - cur >= output_width) & ((zero_max(k_w + ((output_width-1) << 1) - (input_padding_left + input_width-1)) + 1) >> 1);
                          out_ptr += width_padding_start_rem;
                          vl = min(nr - cur, output_width) - width_padding_start_rem - width_padding_end_rem;
                          v_w0 = __riscv_vlse32_v_u32m2 (in_ptr_remainder + (cnt * (input_width << 1)), stride_width << 2, vl);
                          __riscv_vse32_v_u32m2(out_ptr, v_w0, vl);
                          out_ptr += vl + width_padding_end_rem;
                          cnt ++;
                          cur += output_width;
                      }
                      in_ptr_now += input_size;
                      in_ptr_remainder += input_size;
                  }
              }
              in_ptr += ~padded & input_width;
              in_ptr_rem += -(remainder > 0) & input_width;
          }
          output_padding_top_stride -= nr;
          output_cur += nr;
      }
      input_cursor = base + input_width + ((input_width << 1) * (output_cur / output_width - ((input_padding_top + 1) >> 1))) + ((output_cur % output_width) << 1);
      while((((input_padding_top + 1) >> 1) + input_cursor / input_width + kernel_height - base / input_width <= valid_height && k_h_padding_end != 0) || \
      (((input_padding_top + 1) >> 1) + input_cursor / input_width + kernel_height - base / input_width < valid_height && k_h_padding_end == 0)){
          int out_h = output_cur / output_width;
          int out_w = output_cur % output_width;
          remainder = -(output_width - out_w < nr) & (nr - output_width + out_w);
          int input_cursor_rem = input_cursor - (out_w << 1) + (input_width << 1);
          in_ptr = input + input_cursor;
          in_ptr_rem = input + input_cursor_rem;
          for(int k_h = 0; k_h < kernel_height; k_h++){
              int padded = -(remainder > 0 && ((input_padding_top + 1) >> 1) + input_cursor_rem / input_width + k_h - base / input_width >= valid_height && k_h_padding_end > 0);
              for(int k_w = 0; k_w < kernel_width; k_w++){
                  int width_padding_start = -(out_w == 0) & ((zero_max(input_padding_left-k_w) + 1) >> 1);
                  int width_padding_end = -((output_cur + nr) / output_width != out_h) & ((zero_max(k_w + ((output_width-1) << 1) - (input_padding_left + input_width-1)) + 1) >> 1);
                  int input_offset_with_pad = stride_width * width_padding_start - (input_padding_left - k_w);
                  int input_offset_with_pad_cond = -(k_w < input_padding_left);
                  int input_cur_offset = (input_offset_with_pad_cond & input_offset_with_pad) + \
                                          (~input_offset_with_pad_cond & (k_w - input_padding_left));
                  // for remainder
                  int width_padding_start_rem = -(remainder > 0) & ((zero_max(input_padding_left-k_w) + 1) >> 1);
                  input_offset_with_pad = stride_width * width_padding_start_rem - (input_padding_left - k_w);
                  int input_cur_offset_rem = (input_offset_with_pad_cond & input_offset_with_pad) + \
                                          (~input_offset_with_pad_cond & (k_w - input_padding_left));
                  uint32_t* in_ptr_now = in_ptr + input_cur_offset;
                  uint32_t* in_ptr_remainder = in_ptr_rem + input_cur_offset_rem;
                  for(size_t in_ch = 0; in_ch < group_input_channels; in_ch++){
                      out_ptr += width_padding_start;
                      int vl = nr-width_padding_end-width_padding_start-remainder;
                      vuint32m2_t v_w0 = __riscv_vlse32_v_u32m2 (in_ptr_now, stride_width << 2, vl);
                      __riscv_vse32_v_u32m2(out_ptr, v_w0, vl);
                      out_ptr += vl + width_padding_end;
                      int cur = nr - remainder;
                      int cnt = 0;
                      while(cur < nr){
                          int exceed_boundary = -(output_cur + cur >= output_size && (batch + 1 == batch_size));
                          int width_padding_end_rem = -(nr - cur >= output_width) & ((zero_max(k_w + ((output_width-1) << 1) - (input_padding_left + input_width-1)) + 1) >> 1);
                          out_ptr += width_padding_start_rem;
                          vl = min(nr - cur, output_width) - width_padding_start_rem - width_padding_end_rem;
                          v_w0 = __riscv_vlse32_v_u32m2 (in_ptr_remainder + (cnt * (input_width << 1)), stride_width << 2, ~exceed_boundary & vl);
                          __riscv_vse32_v_u32m2(out_ptr, v_w0, ~exceed_boundary & vl);
                          out_ptr += vl + width_padding_end_rem;
                          cnt ++;
                          cur += output_width;
                      }
                      in_ptr_now += input_size;
                      in_ptr_remainder += input_size;
                  }
              }
              in_ptr += input_width;
              in_ptr_rem += input_width;
          }
          output_padding_top_stride -= nr;
          output_cur += nr;
          input_cursor = base + input_width + ((input_width << 1) * (output_cur / output_width - ((input_padding_top + 1) >> 1))) + ((output_cur % output_width) << 1);
      }
      while(output_cur < output_size){
          int out_h = output_cur / output_width;
          int out_w = output_cur % output_width;
          input_cursor = base + input_width + ((input_width << 1) * (out_h - ((input_padding_top + 1) >> 1))) + (out_w << 1);
          int input_cursor_rem = input_cursor - (out_w << 1) + (input_width << 1);
          in_ptr = input + input_cursor;
          in_ptr_rem = input + input_cursor_rem;
          remainder = -(output_width - out_w < nr) & (nr - output_width + out_w);
          for(int k_h = 0; k_h < kernel_height-k_h_padding_end; k_h++){
              for(int k_w = 0; k_w < kernel_width; k_w++){
                  int width_padding_start = -(out_w == 0) & ((zero_max(input_padding_left-k_w) + 1) >> 1);
                  int width_padding_end = -((output_cur + nr) / output_width != out_h) & ((zero_max(k_w + ((output_width-1) << 1) - (input_padding_left + input_width-1)) + 1) >> 1);
                  int input_offset_with_pad = stride_width * width_padding_start - (input_padding_left - k_w);
                  int input_offset_with_pad_cond = -(k_w < input_padding_left);
                  int input_cur_offset = (input_offset_with_pad_cond & input_offset_with_pad) + \
                                          (~input_offset_with_pad_cond & (k_w - input_padding_left));
                  int width_padding_start_rem = -(remainder > 0) & ((zero_max(input_padding_left-k_w) + 1) >> 1);
                  input_offset_with_pad = stride_width * width_padding_start_rem - (input_padding_left - k_w);
                  int input_cur_offset_rem = (input_offset_with_pad_cond & input_offset_with_pad) + \
                                          (~input_offset_with_pad_cond & (k_w - input_padding_left));
                  uint32_t* in_ptr_now = in_ptr + input_cur_offset;
                  uint32_t* in_ptr_remainder = in_ptr_rem + input_cur_offset_rem;
                  for(size_t in_ch = 0; in_ch < group_input_channels; in_ch++){
                      out_ptr += width_padding_start;
                      int vl = nr-width_padding_end-width_padding_start-remainder;
                      vuint32m2_t v_w0 = __riscv_vlse32_v_u32m2 (in_ptr_now, stride_width << 2, vl);
                      __riscv_vse32_v_u32m2(out_ptr, v_w0, vl);
                      out_ptr += vl + width_padding_end;
                      int cur = nr - remainder;
                      int cnt = 0;
                      while(cur < nr){
                          int exceed_boundary = -(output_cur + cur >= output_size && (batch + 1 == batch_size));
                          int padded_rem = -(k_h < zero_max(input_padding_top - ((((output_cur + cur) % (output_size)) / output_width) << 1)));
                          int width_padding_end_rem = -(nr - cur >= output_width) & ((zero_max(k_w + ((output_width-1) << 1) - (input_padding_left + input_width-1)) + 1) >> 1);
                          out_ptr += width_padding_start_rem;
                          vl = min(nr - cur, output_width) - width_padding_start_rem - width_padding_end_rem;
                          v_w0 = __riscv_vlse32_v_u32m2 (in_ptr_remainder + (cnt * (input_width << 1)), stride_width << 2, (~padded_rem & ~exceed_boundary) & vl);
                          __riscv_vse32_v_u32m2(out_ptr, v_w0, (~padded_rem & ~exceed_boundary) & vl);
                          out_ptr += vl + width_padding_end_rem;
                          cnt ++;
                          cur += output_width;
                      }
                      in_ptr_now += input_size;
                      in_ptr_remainder += input_size;
                  }
              }
              in_ptr += output_width << 1;
              in_ptr_rem += -(remainder > 0) & input_width;
          }
          out_ptr += k_h_padding_end * nr * group_input_channels * kernel_width;
          output_cur += nr;
      }
      int last_part_in_this_batch = (output_size*(batch + 1)) % nr;
      int finished_part_in_next_batch = -(last_part_in_this_batch > 0) & (nr - last_part_in_this_batch);
      output_cur = finished_part_in_next_batch;
      output_padding_top_stride = zero_max(output_padding_top_stride_tmp - finished_part_in_next_batch);
  }
}

void xnn_x32_packa_in_T_gemm_im2col_s2_d1(uint32_t batch_size, const size_t input_height, const size_t input_width, size_t group_input_channels, \
  const int output_height, const int output_width,
  const size_t kernel_height, const size_t kernel_width, const size_t stride_height, const size_t stride_width, \
  const int dilation_height, const int dilation_width, const int input_padding_top,const int input_padding_left, \
  uint32_t* input, uint32_t* output){
  const size_t output_size = output_height*output_width;
  const size_t input_size = batch_size * input_height*input_width;
  uint32_t* in_ptr = input;
  uint32_t* in_ptr_rem = input;
  uint32_t* out_ptr = output;
  const int vlmax = __riscv_vsetvlmax_e32m4();
  int output_cur = 0;
  int input_cursor = 0;
  int valid_height = input_padding_top + input_height - 1;
  int last_stride = kernel_height - 1 + (output_height - 1)*2;
  int k_h_padding_end = last_stride - valid_height;
  int remainder = 0;
  int output_padding_top_stride = ((input_padding_top + 1) >> 1) * output_width;
  const int output_padding_top_stride_tmp = output_padding_top_stride;
  for(size_t batch = 0; batch < batch_size; batch++){
      int base = batch * input_height*input_width;
      while(output_padding_top_stride > 0){
          int out_h = output_cur / output_width;
          int out_w = output_cur % output_width;
          input_cursor = base + (out_w << 1);
          remainder = -(output_width - out_w < vlmax) & (vlmax - output_width + out_w);
          // replace `* stride` to `<< 1` by the fact that stride = 2
          int padded_k_h = zero_max(input_padding_top - (out_h << 1) - (-(remainder > 0) & ((out_h + 1) << 1)));
          in_ptr = input + input_cursor;
          in_ptr_rem = input + base + (stride_width - input_padding_top + padded_k_h) * input_width;
          out_ptr += vlmax * group_input_channels * kernel_width * padded_k_h;
          for(int k_h = padded_k_h; k_h < kernel_height; k_h++){
              int padded = -(remainder > 0 && k_h < input_padding_top - (out_h << 1));
              for(int k_w = 0; k_w < kernel_width; k_w++){
                  int width_padding_start = -(out_w == 0) & ((zero_max(input_padding_left-k_w) + 1) >> 1);
                  int width_padding_end = -((output_cur + vlmax) / output_width != out_h) & ((zero_max(k_w + ((output_width-1) << 1) - (input_padding_left + input_width-1)) + 1) >> 1);
                  int input_offset_with_pad = stride_width * width_padding_start - (input_padding_left - k_w);
                  int input_offset_with_pad_cond = -(k_w < input_padding_left);
                  int input_cur_offset = (input_offset_with_pad_cond & input_offset_with_pad) + \
                                          (~input_offset_with_pad_cond & (k_w - input_padding_left));
                  // for remainder
                  int width_padding_start_rem = -(remainder > 0) & ((zero_max(input_padding_left-k_w) + 1) >> 1);
                  input_offset_with_pad = stride_width * width_padding_start_rem - (input_padding_left - k_w);
                  int input_cur_offset_rem = (input_offset_with_pad_cond & input_offset_with_pad) + \
                                          (~input_offset_with_pad_cond & (k_w - input_padding_left));
                  uint32_t* in_ptr_now = in_ptr + input_cur_offset;
                  uint32_t* in_ptr_remainder = in_ptr_rem + input_cur_offset_rem;
                  for(size_t in_ch = 0; in_ch < group_input_channels; in_ch++){
                      out_ptr += width_padding_start;
                      int vl = vlmax-width_padding_end-width_padding_start-remainder;
                      // std::cout << "vl = " << vl << "\n";
                      vuint32m4_t v_w0 = __riscv_vlse32_v_u32m4 (in_ptr_now, stride_width << 2, ~padded & vl);
                      __riscv_vse32_v_u32m4(out_ptr, v_w0, ~padded & vl);
                      out_ptr += vl + width_padding_end;
                      // for remainder
                      out_ptr += width_padding_start_rem;
                      vl = zero_max(remainder - width_padding_start_rem);
                      // std::cout << "vl for remainder = " << vl << "\n";
                      v_w0 = __riscv_vlse32_v_u32m4 (in_ptr_remainder, stride_width << 2, vl);
                      __riscv_vse32_v_u32m4(out_ptr, v_w0, vl);
                      out_ptr += vl;
                      in_ptr_now += input_size;
                      in_ptr_remainder += input_size;
                  }
              }
              in_ptr += ~padded & input_width;
              in_ptr_rem += -(remainder > 0) & input_width;
          }
          output_padding_top_stride -= vlmax;
          output_cur += vlmax;
      }
      input_cursor = base + input_width + (remainder << 1);
      while((((input_padding_top + 1) >> 1) + input_cursor / input_width + kernel_height - base / input_width <= valid_height && k_h_padding_end != 0) || \
      (((input_padding_top + 1) >> 1) + input_cursor / input_width + kernel_height - base / input_width < valid_height && k_h_padding_end == 0)){
          int out_h = output_cur / output_width;
          int out_w = output_cur % output_width;
          remainder = -(output_width - out_w < vlmax) & (vlmax - output_width + out_w);
          int input_cursor_rem = input_cursor - (out_w << 1) + (input_width << 1);
          in_ptr = input + input_cursor;
          in_ptr_rem = input + input_cursor_rem;
          for(int k_h = 0; k_h < kernel_height; k_h++){
              int padded = -(remainder > 0 && ((input_padding_top + 1) >> 1) + input_cursor_rem / input_width + k_h - base / input_width >= valid_height && k_h_padding_end > 0);
              for(int k_w = 0; k_w < kernel_width; k_w++){
                  int width_padding_start = -(out_w == 0) & ((zero_max(input_padding_left-k_w) + 1) >> 1);
                  int width_padding_end = -((output_cur + vlmax) / output_width != out_h) & ((zero_max(k_w + ((output_width-1) << 1) - (input_padding_left + input_width-1)) + 1) >> 1);
                  int input_offset_with_pad = stride_width * width_padding_start - (input_padding_left - k_w);
                  int input_offset_with_pad_cond = -(k_w < input_padding_left);
                  int input_cur_offset = (input_offset_with_pad_cond & input_offset_with_pad) + \
                                          (~input_offset_with_pad_cond & (k_w - input_padding_left));
                  // for remainder
                  int width_padding_start_rem = -(remainder > 0) & ((zero_max(input_padding_left-k_w) + 1) >> 1);
                  input_offset_with_pad = stride_width * width_padding_start_rem - (input_padding_left - k_w);
                  int input_cur_offset_rem = (input_offset_with_pad_cond & input_offset_with_pad) + \
                                          (~input_offset_with_pad_cond & (k_w - input_padding_left));
                  uint32_t* in_ptr_now = in_ptr + input_cur_offset;
                  uint32_t* in_ptr_remainder = in_ptr_rem + input_cur_offset_rem;
                  for(size_t in_ch = 0; in_ch < group_input_channels; in_ch++){
                      out_ptr += width_padding_start;
                      int vl = vlmax-width_padding_end-width_padding_start-remainder;
                      vuint32m4_t v_w0 = __riscv_vlse32_v_u32m4 (in_ptr_now, stride_width << 2, vl);
                      __riscv_vse32_v_u32m4(out_ptr, v_w0, vl);
                      out_ptr += vl + width_padding_end;
                      // for remainder
                      out_ptr += width_padding_start_rem;
                      vl = zero_max(remainder - width_padding_start_rem);
                      // std::cout << "vl for remainder = " << vl << "\n";
                      v_w0 = __riscv_vlse32_v_u32m4 (in_ptr_remainder, stride_width << 2, ~padded & vl);
                      __riscv_vse32_v_u32m4(out_ptr, v_w0, ~padded & vl);
                      out_ptr += vl;
                      in_ptr_now += input_size;
                      in_ptr_remainder += input_size;
                  }
              }
              in_ptr += input_width;
              in_ptr_rem += input_width;
          }
          output_padding_top_stride -= vlmax;
          output_cur += vlmax;
          input_cursor = base + input_width + ((input_width << 1) * (output_cur / output_width - ((input_padding_top + 1) >> 1))) + ((output_cur % output_width) << 1);
      }
      while(output_cur < output_size){
          int out_h = output_cur / output_width;
          int out_w = output_cur % output_width;
          input_cursor = base + input_width + ((input_width << 1) * (out_h - ((input_padding_top + 1) >> 1))) + (out_w << 1);
          int input_cursor_rem = input_cursor - (out_w << 1) + (input_width << 1);
          in_ptr = input + input_cursor;
          in_ptr_rem = input + input_cursor_rem;
          remainder = -(output_width - out_w < vlmax) & (vlmax - output_width + out_w);
          for(int k_h = 0; k_h < kernel_height-k_h_padding_end; k_h++){
              for(int k_w = 0; k_w < kernel_width; k_w++){
                  int width_padding_start = -(out_w == 0) & ((zero_max(input_padding_left-k_w) + 1) >> 1);
                  int width_padding_end = -((output_cur + vlmax) / output_width != out_h) & ((zero_max(k_w + ((output_width-1) << 1) - (input_padding_left + input_width-1)) + 1) >> 1);
                  int input_offset_with_pad = stride_width * width_padding_start - (input_padding_left - k_w);
                  int input_offset_with_pad_cond = -(k_w < input_padding_left);
                  int input_cur_offset = (input_offset_with_pad_cond & input_offset_with_pad) + \
                                          (~input_offset_with_pad_cond & (k_w - input_padding_left));
                  int width_padding_start_rem = -(remainder > 0) & ((zero_max(input_padding_left-k_w) + 1) >> 1);
                  input_offset_with_pad = stride_width * width_padding_start_rem - (input_padding_left - k_w);
                  int input_cur_offset_rem = (input_offset_with_pad_cond & input_offset_with_pad) + \
                                          (~input_offset_with_pad_cond & (k_w - input_padding_left));
                  uint32_t* in_ptr_now = in_ptr + input_cur_offset;
                  uint32_t* in_ptr_remainder = in_ptr_rem + input_cur_offset_rem;
                  int padded_rem = -(remainder > 0 && k_h < input_padding_top - ((((output_cur + vlmax) % (output_size)) / output_width) << 1));
                  int exceed_boundary = -(output_cur + vlmax - remainder >= batch_size * output_size);
                  for(size_t in_ch = 0; in_ch < group_input_channels; in_ch++){
                      out_ptr += width_padding_start;
                      int vl = vlmax-width_padding_end-width_padding_start-remainder;
                      vuint32m4_t v_w0 = __riscv_vlse32_v_u32m4 (in_ptr_now, stride_width << 2, vl);
                      __riscv_vse32_v_u32m4(out_ptr, v_w0, vl);
                      out_ptr += vl + width_padding_end;
                      out_ptr += width_padding_start_rem;
                      vl = zero_max(remainder - width_padding_start_rem);
                      // std::cout << "vl for remainder = " << vl << "\n";
                      v_w0 = __riscv_vlse32_v_u32m4 (in_ptr_remainder, stride_width << 2, (~padded_rem & ~exceed_boundary) & vl);
                      __riscv_vse32_v_u32m4(out_ptr, v_w0, (~padded_rem & ~exceed_boundary) & vl);
                      out_ptr += vl;
                      in_ptr_now += input_size;
                      in_ptr_remainder += input_size;
                  }
              }
              in_ptr += output_width << 1;
              in_ptr_rem += -(remainder > 0) & input_width;
          }
          out_ptr += k_h_padding_end * vlmax * group_input_channels * kernel_width;
          output_cur += vlmax;
      }
      int last_part_in_this_batch = (output_size*(batch + 1)) % vlmax;
      int finished_part_in_next_batch = -(last_part_in_this_batch > 0) & (vlmax - last_part_in_this_batch);
      output_cur = finished_part_in_next_batch;
      output_padding_top_stride = zero_max(output_padding_top_stride_tmp - finished_part_in_next_batch);
  }
}


void xnn_x32_packa_in_T_gemm_im2col_s1_d1_1x4v(uint32_t batch_size, const size_t input_height, const size_t input_width, size_t group_input_channels, \
  const int output_height, const int output_width,
  const size_t kernel_height, const size_t kernel_width, const size_t stride_height, const size_t stride_width, \
  const int dilation_height, const int dilation_width, const int input_padding_top,const int input_padding_left, \
  uint32_t* input, uint32_t* output){
  const size_t output_size = output_height*output_width;
  const size_t input_size = input_height*input_width*batch_size;
  uint32_t* in_ptr = input;
  uint32_t* out_ptr = output;
  const int nr = __riscv_vsetvlmax_e32m4();
  const int vlmax = __riscv_vsetvlmax_e32m1();
  int valid_height = input_padding_top + input_height;
  int valid_width = input_padding_left + input_width;
  int height_padding_start = input_padding_top;
  int height_padding_end = zero_max(output_height-valid_height);
  int output_padding_top_stride = height_padding_start * output_width;
  int output_padding_down_stride = height_padding_end * output_width;
  int width_padding_start;
  int width_padding_end;
  int input_stride;
  int remainder = 0;
  int input_cursor = 0;
  int out_h, out_w, batch;
  int im2col_cur = 0;
  int output_cur = 0;

  for(batch = 0; batch < batch_size; batch++){
  // top_pad
  /*
      im2col_cur < nr * ceil(output_width / nr)
  --> im2col_cur < nr * ((output_width + nr - 1) / nr)
  */ 
      while(output_cur < output_width * ((output_width + nr - 1) / nr)){
          out_h = output_cur / output_width;
          out_w = output_cur % output_width;
          remainder = output_padding_top_stride;
          uint32_t* in_ptr_now = in_ptr + input_cursor;
          for(int k_h = 0; k_h < kernel_height; k_h++){
              int moved_input_ptr_step = 0;
              int is_partial_stride_padded_part = -(output_padding_top_stride && zero_max(input_padding_top-k_h));
              for(int k_w = 0; k_w < kernel_width; k_w++){
                  int output_padding_left = zero_max(input_padding_left-k_w);
                  int output_padding_right = zero_max(k_w + output_width-valid_width);
                  int remainder_padding_left = output_padding_left & (-(remainder > 0 && out_w == 0));
                  int remainder_padding_right = output_padding_right & (-(remainder > 0));
                  int vl;
                  for(int in_ch = 0; in_ch < group_input_channels; in_ch++){
                      out_ptr += remainder_padding_left;
                      vl = remainder \
                          - (is_partial_stride_padded_part & (output_padding_top_stride - remainder_padding_left - remainder_padding_right)) \
                          - remainder_padding_right - remainder_padding_left;
                      vuint32m1_t v_w0 = __riscv_vle32_v_u32m1(in_ptr_now, vl);
                      __riscv_vse32_v_u32m1(out_ptr, v_w0, vl);
                      out_ptr += remainder - remainder_padding_left; // including remainder_padding_right
                      int is_added = vl + ((output_padding_left + remainder_padding_right) & -(vl > 0));
                      in_ptr_now = in_ptr_now + is_added;
                      for(int cur_vl = 0; cur_vl < nr - remainder; cur_vl += output_width){
                          out_ptr += output_padding_left;
                          int is_whole_stride = -(nr - remainder - cur_vl >= output_width);
                          int segment_padding_right = (output_padding_right & is_whole_stride);
                          vl = (is_whole_stride & (output_width - output_padding_left - segment_padding_right)) + \
                              (~is_whole_stride & nr - remainder - cur_vl - output_padding_left);
                          v_w0 = __riscv_vle32_v_u32m1(in_ptr_now, vl);
                          __riscv_vse32_v_u32m1(out_ptr, v_w0, vl);
                          in_ptr_now += vl + ((output_padding_left + segment_padding_right) & is_whole_stride);
                          out_ptr += vl + segment_padding_right;
                          // + vl - (output_width - output_padding_left - remainder_padding_right) + output_width = vl + output_padding_left
                          is_added += vl + ((output_padding_left + segment_padding_right) & is_whole_stride);
                      }
                      in_ptr_now += input_size - is_added;
                  }
                  in_ptr_now = in_ptr_now - input_size*group_input_channels + 1 - (output_padding_left & (-(out_w == 0) || is_partial_stride_padded_part));
                  moved_input_ptr_step += (1 - (output_padding_left & (-(out_w == 0) || is_partial_stride_padded_part)));
              }
              in_ptr_now = in_ptr_now - moved_input_ptr_step - (is_partial_stride_padded_part & (output_padding_top_stride + (input_padding_left & -(out_w != 0)))) + input_width;
              output_padding_top_stride -= (is_partial_stride_padded_part & output_padding_top_stride);
          }
          int input_offset = input_width * ((nr - remainder) / input_width) + ((nr - remainder) % input_width - input_padding_left);
          input_cursor = input_cursor + input_offset;
          im2col_cur += nr;
          output_cur += nr;
      }
      // middle part
      while((output_cur + nr) / output_width < output_height - input_padding_top){
          out_h = output_cur / output_width;
          out_w = output_cur % output_width;
          remainder = output_width - ((nr - remainder) % output_width);
          int k_h = 0, k_w = 0;
          int input_offset = 0;
          uint32_t* in_ptr_now = in_ptr + input_cursor;
          int is_in_left_part = -((out_w == 0) || (remainder > 0));
          int is_in_right_part1 = -((out_h != (output_cur + nr) / output_width));
          int is_in_right_part2 = -((output_cur + nr) % output_width > 0);
          for(k_h = 0; k_h < kernel_height; k_h++){
              int moved_input_ptr_step = 0;
              for(k_w = 0; k_w < kernel_width; k_w++){
                  int output_padding_left = (zero_max(input_padding_left-k_w));
                  int remainder_padding_left = output_padding_left & (-(remainder > 0 && out_w == 0));
                  int remainder_padding_right = (-(remainder > 0) & zero_max(k_w + output_width-valid_width));
                  int output_padding_right = zero_max(k_w + output_width-valid_width);
                  int vl;
                  for(int in_ch = 0; in_ch < group_input_channels; in_ch++){
                      out_ptr += remainder_padding_left;
                      vl = zero_max(remainder - remainder_padding_right - remainder_padding_left);
                      vuint32m1_t v_w0 = __riscv_vle32_v_u32m1(in_ptr_now, vl);
                      __riscv_vse32_v_u32m1(out_ptr, v_w0, vl);
                      // remainder with padding
                      out_ptr = out_ptr + remainder - remainder_padding_left;
                      // in_ptr_now + vl - (output_width - output_padding_left - remainder_padding_right + input_width)
                      // two segments
                      int is_added = vl + ((output_padding_left + remainder_padding_right) & -(remainder > 0));
                      in_ptr_now = in_ptr_now + is_added;
                      // std::cout << "vl for remainder = " << vl << "\n";
                      for(int cur_vl = 0; cur_vl < nr - remainder; cur_vl += output_width){
                          out_ptr += output_padding_left;
                          int is_whole_stride = -(nr - remainder - cur_vl >= output_width);
                          int segment_padding_right = (output_padding_right & is_whole_stride);
                          // std::cout << "vl = " << vl << "\n";
                          vl = (is_whole_stride & (output_width - output_padding_left - segment_padding_right)) + \
                              (~is_whole_stride & nr - remainder - cur_vl - output_padding_left);
                          v_w0 = __riscv_vle32_v_u32m1(in_ptr_now, vl);
                          __riscv_vse32_v_u32m1(out_ptr, v_w0, vl);
                          in_ptr_now += vl + ((output_padding_left + segment_padding_right) & is_whole_stride);
                          out_ptr += vl + segment_padding_right;
                          // + vl - (output_width - output_padding_left - remainder_padding_right) + output_width = vl + output_padding_left
                          is_added += vl + ((output_padding_left + segment_padding_right) & is_whole_stride);
                      }
                      in_ptr_now = in_ptr_now - is_added + input_size;
                  }
                  in_ptr_now = in_ptr_now - input_size*group_input_channels + 1 - (output_padding_left & (-(out_w == 0)));
                  moved_input_ptr_step += (1 - (output_padding_left & (-(out_w == 0))));
              }
              in_ptr_now = in_ptr_now - moved_input_ptr_step + input_width;
          }
          int cond = -(input_cursor % input_width == 0);
          input_offset = ((~cond & (nr + (~is_in_right_part2 & input_padding_left))) | \
                          (cond & (nr - remainder - input_padding_left + (output_width * (remainder / output_width)))));
          input_cursor = input_cursor + input_offset;
          im2col_cur += nr;
          output_cur += nr;
      }
      // end part
      while(output_cur < output_size){
          out_h = output_cur / output_width;
          out_w = output_cur % output_width;
          int is_next_batch = -(im2col_cur + vlmax > output_size*(batch+1) && batch + 1 != batch_size);
          const int r = (output_cur + nr) % output_width;
          remainder = output_width - ((nr - remainder) % output_width);
          int k_h = 0, k_w = 0;
          int input_offset = 0;
          uint32_t* in_ptr_now = in_ptr + input_cursor;
          int is_in_left_part = -((out_w == 0) || (remainder > 0));
          int is_in_right_part1 = -((out_h != (output_cur + nr) / output_width));
          int is_in_right_part2 = -((output_cur + nr) % output_width > 0);
          int is_whole_stride_padded = ~is_next_batch & -(out_h >= output_height - input_padding_top);            int last_part_in_this_batch = output_size % vlmax;
          for(k_h = 0; k_h < kernel_height - (is_whole_stride_padded & input_padding_top); k_h++){
              int moved_input_ptr_step = 0;
              for(k_w = 0; k_w < kernel_width; k_w++){
                  int output_padding_left = zero_max(input_padding_left-k_w);
                  int output_padding_right = zero_max(k_w + output_width-valid_width);
                  int remainder_padding_left = output_padding_left & (-(remainder > 0 && out_w == 0));
                  int remainder_padding_right = (-(remainder > 0) & zero_max(k_w + output_width-valid_width));
                  int vl;
                  for(int in_ch = 0; in_ch < group_input_channels; in_ch++){
                      int is_padded_remainder = -((zero_max(k_h - input_padding_top)) & -(output_size - output_width * input_padding_top <= output_cur));
                      out_ptr += remainder_padding_left;
                      vl = zero_max(remainder - remainder_padding_right - remainder_padding_left);
                      vuint32m1_t v_w0 = __riscv_vle32_v_u32m1(in_ptr_now, ~is_padded_remainder & vl);
                      __riscv_vse32_v_u32m1(out_ptr, v_w0, ~is_padded_remainder & vl);
                      // remainder with padding
                      out_ptr = out_ptr + remainder - remainder_padding_left;
                      // in_ptr_now + vl - (output_width - output_padding_left - remainder_padding_right + input_width)
                      // two segments
                      int is_added = vl + ((output_padding_left + remainder_padding_right) & -(remainder > 0));
                      in_ptr_now = in_ptr_now + is_added;
                      for(int cur_vl = 0; cur_vl < nr - remainder; cur_vl += output_width){
                          out_ptr += output_padding_left;
                          int is_whole_stride = -(nr - remainder - cur_vl >= output_width);
                          int segment_padding_right = (output_padding_right & is_whole_stride);
                          int current_output_offset = output_cur + remainder + cur_vl;
                          // std::cout << "vl = " << vl << "\n";
                          vl = (is_whole_stride & (output_width - output_padding_left - segment_padding_right)) + \
                              (~is_whole_stride & nr - remainder - cur_vl - output_padding_left);
                          int is_exceed_boarder = -(current_output_offset >= output_size);
                          int is_exceed_batch_output_size = -(batch + 1 >= batch_size) & is_exceed_boarder;
                          int is_padded_end = -(zero_max(k_h - input_padding_top) & -(output_size - output_width * input_padding_top <= current_output_offset && current_output_offset < output_size));
                          int is_next_batch_padding_top = -(zero_max(input_padding_top-k_h) & -(batch + 1 != batch_size && (current_output_offset % output_size < output_width*input_padding_top)) & is_exceed_boarder);
                          v_w0 = __riscv_vle32_v_u32m1(in_ptr_now, (~is_padded_end & ~is_exceed_batch_output_size & ~is_next_batch_padding_top) & vl);
                          __riscv_vse32_v_u32m1(out_ptr, v_w0, (~is_padded_end & ~is_exceed_batch_output_size & ~is_next_batch_padding_top) & vl);
                          in_ptr_now += vl + ((output_padding_left + segment_padding_right) & is_whole_stride);
                          out_ptr += vl + segment_padding_right;
                          // + vl - (output_width - output_padding_left - remainder_padding_right) + output_width = vl + output_padding_left
                          is_added += vl + ((output_padding_left + segment_padding_right) & is_whole_stride);
                      }
                      in_ptr_now = in_ptr_now - is_added + input_size;
                  }
                  // at the start of each stride, we need to take padding_left into consideration
                  in_ptr_now = in_ptr_now - input_size*group_input_channels + 1 - (output_padding_left & (-(out_w == 0)));
                  moved_input_ptr_step += (1 - (output_padding_left & (-(out_w == 0))));
              }
              // std::cout << "moved_input_ptr_step = " << moved_input_ptr_step << "\n";
              in_ptr_now = in_ptr_now - moved_input_ptr_step + input_width;
          }
          int cond = -(input_cursor % input_width == 0);
          input_offset = ((~cond & (nr + (~is_in_right_part2 & input_padding_left))) | \
                              (cond & nr));
          input_cursor = input_cursor + input_offset;
          out_ptr += (is_whole_stride_padded & min(vlmax, output_width) * group_input_channels * kernel_width);
          im2col_cur += nr;
          output_cur += nr;
      }
      int last_part_in_this_batch = (output_size*(batch + 1)) % nr;
      int finished_part_in_next_batch = -(last_part_in_this_batch > 0) & (nr - last_part_in_this_batch);
      output_cur = finished_part_in_next_batch;
      output_padding_top_stride = zero_max(height_padding_start * output_width - finished_part_in_next_batch);
  }
}


void xnn_x32_packa_in_T_gemm_im2col_s1_d1_2x4v(uint32_t batch_size, const size_t input_height, const size_t input_width, size_t group_input_channels, \
  const int output_height, const int output_width,
  const size_t kernel_height, const size_t kernel_width, const size_t stride_height, const size_t stride_width, \
  const int dilation_height, const int dilation_width, const int input_padding_top,const int input_padding_left, \
  uint32_t* input, uint32_t* output){
  const size_t output_size = output_height*output_width;
  const size_t input_size = input_height*input_width*batch_size;
  uint32_t* in_ptr = input;
  uint32_t* out_ptr = output;
  const int nr = __riscv_vsetvlmax_e32m4();
  const int vlmax = __riscv_vsetvlmax_e32m2();
  int valid_height = input_padding_top + input_height;
  int valid_width = input_padding_left + input_width;
  int height_padding_start = input_padding_top;
  int height_padding_end = zero_max(output_height-valid_height);
  int output_padding_top_stride = height_padding_start * output_width;
  int output_padding_down_stride = height_padding_end * output_width;
  int width_padding_start;
  int width_padding_end;
  int input_stride;
  int remainder = 0;
  int input_cursor = 0;
  int out_h, out_w, batch;
  int im2col_cur = 0;
  int output_cur = 0;

  for(batch = 0; batch < batch_size; batch++){
  // top_pad
  /*
      im2col_cur < nr * ceil(output_width / nr)
  --> im2col_cur < nr * ((output_width + nr - 1) / nr)
  */ 
      while(output_cur < output_width * ((output_width + nr - 1) / nr)){
          out_h = output_cur / output_width;
          out_w = output_cur % output_width;
          remainder = output_padding_top_stride;
          uint32_t* in_ptr_now = in_ptr + input_cursor;
          for(int k_h = 0; k_h < kernel_height; k_h++){
              int moved_input_ptr_step = 0;
              int is_partial_stride_padded_part = -(output_padding_top_stride && zero_max(input_padding_top-k_h));
              for(int k_w = 0; k_w < kernel_width; k_w++){
                  int output_padding_left = zero_max(input_padding_left-k_w);
                  int output_padding_right = zero_max(k_w + output_width-valid_width);
                  int remainder_padding_left = output_padding_left & (-(remainder > 0 && out_w == 0));
                  int remainder_padding_right = output_padding_right & (-(remainder > 0));
                  int vl;
                  for(int in_ch = 0; in_ch < group_input_channels; in_ch++){
                      out_ptr += remainder_padding_left;
                      vl = remainder \
                          - (is_partial_stride_padded_part & (output_padding_top_stride - remainder_padding_left - remainder_padding_right)) \
                          - remainder_padding_right - remainder_padding_left;
                      vuint32m2_t v_w0 = __riscv_vle32_v_u32m2(in_ptr_now, vl);
                      __riscv_vse32_v_u32m2(out_ptr, v_w0, vl);
                      out_ptr += remainder - remainder_padding_left; // including remainder_padding_right
                      int is_added = vl + ((output_padding_left + remainder_padding_right) & -(vl > 0));
                      in_ptr_now = in_ptr_now + is_added;
                      for(int cur_vl = 0; cur_vl < nr - remainder; cur_vl += output_width){
                          out_ptr += output_padding_left;
                          int is_whole_stride = -(nr - remainder - cur_vl >= output_width);
                          int segment_padding_right = (output_padding_right & is_whole_stride);
                          vl = (is_whole_stride & (output_width - output_padding_left - segment_padding_right)) + \
                              (~is_whole_stride & nr - remainder - cur_vl - output_padding_left);
                          v_w0 = __riscv_vle32_v_u32m2(in_ptr_now, vl);
                          __riscv_vse32_v_u32m2(out_ptr, v_w0, vl);
                          in_ptr_now += vl + ((output_padding_left + segment_padding_right) & is_whole_stride);
                          out_ptr += vl + segment_padding_right;
                          // + vl - (output_width - output_padding_left - remainder_padding_right) + output_width = vl + output_padding_left
                          is_added += vl + ((output_padding_left + segment_padding_right) & is_whole_stride);
                      }
                      in_ptr_now += input_size - is_added;
                  }
                  in_ptr_now = in_ptr_now - input_size*group_input_channels + 1 - (output_padding_left & (-(out_w == 0) || is_partial_stride_padded_part));
                  moved_input_ptr_step += (1 - (output_padding_left & (-(out_w == 0) || is_partial_stride_padded_part)));
              }
              in_ptr_now = in_ptr_now - moved_input_ptr_step - (is_partial_stride_padded_part & (output_padding_top_stride + (input_padding_left & -(out_w != 0)))) + input_width;
              output_padding_top_stride -= (is_partial_stride_padded_part & output_padding_top_stride);
          }
          int input_offset = input_width * ((nr - remainder) / input_width) + ((nr - remainder) % input_width - input_padding_left);
          input_cursor = input_cursor + input_offset;
          im2col_cur += nr;
          output_cur += nr;
      }
      // middle part
      while((output_cur + nr) / output_width < output_height - input_padding_top){
          out_h = output_cur / output_width;
          out_w = output_cur % output_width;
          remainder = output_width - ((nr - remainder) % output_width);
          int k_h = 0, k_w = 0;
          int input_offset = 0;
          uint32_t* in_ptr_now = in_ptr + input_cursor;
          int is_in_left_part = -((out_w == 0) || (remainder > 0));
          int is_in_right_part1 = -((out_h != (output_cur + nr) / output_width));
          int is_in_right_part2 = -((output_cur + nr) % output_width > 0);
          for(k_h = 0; k_h < kernel_height; k_h++){
              int moved_input_ptr_step = 0;
              for(k_w = 0; k_w < kernel_width; k_w++){
                  int output_padding_left = (zero_max(input_padding_left-k_w));
                  int remainder_padding_left = output_padding_left & (-(remainder > 0 && out_w == 0));
                  int remainder_padding_right = (-(remainder > 0) & zero_max(k_w + output_width-valid_width));
                  int output_padding_right = zero_max(k_w + output_width-valid_width);
                  int vl;
                  for(int in_ch = 0; in_ch < group_input_channels; in_ch++){
                      out_ptr += remainder_padding_left;
                      vl = zero_max(remainder - remainder_padding_right - remainder_padding_left);
                      vuint32m2_t v_w0 = __riscv_vle32_v_u32m2(in_ptr_now, vl);
                      __riscv_vse32_v_u32m2(out_ptr, v_w0, vl);
                      // remainder with padding
                      out_ptr = out_ptr + remainder - remainder_padding_left;
                      // in_ptr_now + vl - (output_width - output_padding_left - remainder_padding_right + input_width)
                      // two segments
                      int is_added = vl + ((output_padding_left + remainder_padding_right) & -(remainder > 0));
                      in_ptr_now = in_ptr_now + is_added;
                      // std::cout << "vl for remainder = " << vl << "\n";
                      for(int cur_vl = 0; cur_vl < nr - remainder; cur_vl += output_width){
                          out_ptr += output_padding_left;
                          int is_whole_stride = -(nr - remainder - cur_vl >= output_width);
                          int segment_padding_right = (output_padding_right & is_whole_stride);
                          // std::cout << "vl = " << vl << "\n";
                          vl = (is_whole_stride & (output_width - output_padding_left - segment_padding_right)) + \
                              (~is_whole_stride & nr - remainder - cur_vl - output_padding_left);
                          v_w0 = __riscv_vle32_v_u32m2(in_ptr_now, vl);
                          __riscv_vse32_v_u32m2(out_ptr, v_w0, vl);
                          in_ptr_now += vl + ((output_padding_left + segment_padding_right) & is_whole_stride);
                          out_ptr += vl + segment_padding_right;
                          // + vl - (output_width - output_padding_left - remainder_padding_right) + output_width = vl + output_padding_left
                          is_added += vl + ((output_padding_left + segment_padding_right) & is_whole_stride);
                      }
                      in_ptr_now = in_ptr_now - is_added + input_size;
                  }
                  in_ptr_now = in_ptr_now - input_size*group_input_channels + 1 - (output_padding_left & (-(out_w == 0)));
                  moved_input_ptr_step += (1 - (output_padding_left & (-(out_w == 0))));
              }
              in_ptr_now = in_ptr_now - moved_input_ptr_step + input_width;
          }
          int cond = -(input_cursor % input_width == 0);
          input_offset = ((~cond & (nr + (~is_in_right_part2 & input_padding_left))) | \
                          (cond & (nr - remainder - input_padding_left + (output_width * (remainder / output_width)))));
          input_cursor = input_cursor + input_offset;
          im2col_cur += nr;
          output_cur += nr;
      }
      // end part
      while(output_cur < output_size){
          out_h = output_cur / output_width;
          out_w = output_cur % output_width;
          int is_next_batch = -(im2col_cur + vlmax > output_size*(batch+1) && batch + 1 != batch_size);
          const int r = (output_cur + nr) % output_width;
          remainder = output_width - ((nr - remainder) % output_width);
          int k_h = 0, k_w = 0;
          int input_offset = 0;
          uint32_t* in_ptr_now = in_ptr + input_cursor;
          int is_in_left_part = -((out_w == 0) || (remainder > 0));
          int is_in_right_part1 = -((out_h != (output_cur + nr) / output_width));
          int is_in_right_part2 = -((output_cur + nr) % output_width > 0);
          int is_whole_stride_padded = ~is_next_batch & -(out_h >= output_height - input_padding_top);            int last_part_in_this_batch = output_size % vlmax;
          for(k_h = 0; k_h < kernel_height - (is_whole_stride_padded & input_padding_top); k_h++){
              int moved_input_ptr_step = 0;
              for(k_w = 0; k_w < kernel_width; k_w++){
                  int output_padding_left = zero_max(input_padding_left-k_w);
                  int output_padding_right = zero_max(k_w + output_width-valid_width);
                  int remainder_padding_left = output_padding_left & (-(remainder > 0 && out_w == 0));
                  int remainder_padding_right = (-(remainder > 0) & zero_max(k_w + output_width-valid_width));
                  int vl;
                  for(int in_ch = 0; in_ch < group_input_channels; in_ch++){
                      int is_padded_remainder = -((zero_max(k_h - input_padding_top)) & -(output_size - output_width * input_padding_top <= output_cur));
                      out_ptr += remainder_padding_left;
                      vl = zero_max(remainder - remainder_padding_right - remainder_padding_left);
                      vuint32m2_t v_w0 = __riscv_vle32_v_u32m2(in_ptr_now, ~is_padded_remainder & vl);
                      __riscv_vse32_v_u32m2(out_ptr, v_w0, ~is_padded_remainder & vl);
                      // remainder with padding
                      out_ptr = out_ptr + remainder - remainder_padding_left;
                      // in_ptr_now + vl - (output_width - output_padding_left - remainder_padding_right + input_width)
                      // two segments
                      int is_added = vl + ((output_padding_left + remainder_padding_right) & -(remainder > 0));
                      in_ptr_now = in_ptr_now + is_added;
                      for(int cur_vl = 0; cur_vl < nr - remainder; cur_vl += output_width){
                          out_ptr += output_padding_left;
                          int is_whole_stride = -(nr - remainder - cur_vl >= output_width);
                          int segment_padding_right = (output_padding_right & is_whole_stride);
                          int current_output_offset = output_cur + remainder + cur_vl;
                          // std::cout << "vl = " << vl << "\n";
                          vl = (is_whole_stride & (output_width - output_padding_left - segment_padding_right)) + \
                              (~is_whole_stride & nr - remainder - cur_vl - output_padding_left);
                          int is_exceed_boarder = -(current_output_offset >= output_size);
                          int is_exceed_batch_output_size = -(batch + 1 >= batch_size) & is_exceed_boarder;
                          int is_padded_end = -(zero_max(k_h - input_padding_top) & -(output_size - output_width * input_padding_top <= current_output_offset && current_output_offset < output_size));
                          int is_next_batch_padding_top = -(zero_max(input_padding_top-k_h) & -(batch + 1 != batch_size && (current_output_offset % output_size < output_width*input_padding_top)) & is_exceed_boarder);
                          v_w0 = __riscv_vle32_v_u32m2(in_ptr_now, (~is_padded_end & ~is_exceed_batch_output_size & ~is_next_batch_padding_top) & vl);
                          __riscv_vse32_v_u32m2(out_ptr, v_w0, (~is_padded_end & ~is_exceed_batch_output_size & ~is_next_batch_padding_top) & vl);
                          in_ptr_now += vl + ((output_padding_left + segment_padding_right) & is_whole_stride);
                          out_ptr += vl + segment_padding_right;
                          // + vl - (output_width - output_padding_left - remainder_padding_right) + output_width = vl + output_padding_left
                          is_added += vl + ((output_padding_left + segment_padding_right) & is_whole_stride);
                      }
                      in_ptr_now = in_ptr_now - is_added + input_size;
                  }
                  // at the start of each stride, we need to take padding_left into consideration
                  in_ptr_now = in_ptr_now - input_size*group_input_channels + 1 - (output_padding_left & (-(out_w == 0)));
                  moved_input_ptr_step += (1 - (output_padding_left & (-(out_w == 0))));
              }
              // std::cout << "moved_input_ptr_step = " << moved_input_ptr_step << "\n";
              in_ptr_now = in_ptr_now - moved_input_ptr_step + input_width;
          }
          int cond = -(input_cursor % input_width == 0);
          input_offset = ((~cond & (nr + (~is_in_right_part2 & input_padding_left))) | \
                              ((cond) & (nr - remainder - input_padding_left)));
          input_cursor = input_cursor + input_offset;
          out_ptr += (is_whole_stride_padded & min(vlmax, output_width) * group_input_channels * kernel_width);
          // std::cout << "input_cursor = " << input_cursor << "\n";
          im2col_cur += nr;
          output_cur += nr;
      }
      int last_part_in_this_batch = (output_size*(batch + 1)) % nr;
      int finished_part_in_next_batch = -(last_part_in_this_batch > 0) & (nr - last_part_in_this_batch);
      output_cur = finished_part_in_next_batch;
      output_padding_top_stride = zero_max(height_padding_start * output_width - finished_part_in_next_batch);
  }
}


void xnn_x32_packa_in_T_gemm_im2col_s1_d1_4x4v(uint32_t batch_size, const size_t input_height, const size_t input_width, size_t group_input_channels, \
  const int output_height, const int output_width,
  const size_t kernel_height, const size_t kernel_width, const size_t stride_height, const size_t stride_width, \
  const int dilation_height, const int dilation_width, const int input_padding_top,const int input_padding_left, \
  uint32_t* input, uint32_t* output){
  const size_t output_size = output_height*output_width;
  const size_t input_size = input_height*input_width*batch_size;
  uint32_t* in_ptr = input;
  uint32_t* out_ptr = output;
  const int vlmax = __riscv_vsetvlmax_e32m4();
  int valid_height = input_padding_top + input_height;
  int valid_width = input_padding_left + input_width;
  int height_padding_start = input_padding_top;
  int height_padding_end = zero_max(output_height-valid_height);
  int output_padding_top_stride = height_padding_start * output_width;
  int output_padding_down_stride = height_padding_end * output_width;
  int width_padding_start;
  int width_padding_end;
  int input_stride;
  int remainder = 0;
  int input_cursor = 0;
  int out_h, out_w, batch;
  int im2col_cur = 0;
  int batch_cur = 0;
  int output_cur = 0;
  batch = im2col_cur / output_size;
  // top_pad
  /*
      output_cur < vlmax * ceil(output_width / vlmax)
  --> output_cur < vlmax * ((output_width + vlmax - 1) / vlmax)
  */ 
  for(batch = 0; batch < batch_size; batch ++){
      while(output_cur < vlmax * ((output_width + vlmax - 1) / vlmax)){
          out_h = output_cur / output_width;
          out_w = output_cur % output_width;
          int is_whole_stride_padded_part = -(output_padding_top_stride/vlmax >= 1);
          remainder = ~is_whole_stride_padded_part & output_padding_top_stride;
          uint32_t* in_ptr_now = in_ptr + input_cursor;
          out_ptr += (is_whole_stride_padded_part & (min(vlmax, output_width) * group_input_channels * kernel_width));
          output_padding_top_stride -= (is_whole_stride_padded_part & vlmax);
          for(int k_h = is_whole_stride_padded_part & input_padding_top; k_h < kernel_height; k_h++){
              int moved_input_ptr_step = 0;
              int is_partial_stride_padded_part = -(output_padding_top_stride && zero_max(input_padding_top-k_h));
              for(int k_w = 0; k_w < kernel_width; k_w++){
                  int output_padding_left = zero_max(input_padding_left-k_w);
                  int remainder_padding_left = output_padding_left & (-(remainder > 0 && out_w == 0));
                  int remainder_padding_right = (-(remainder > 0) & zero_max(k_w + output_width-valid_width));
                  int vl;
                  for(int in_ch = 0; in_ch < group_input_channels; in_ch++){
                      // |---- pad_top ---- |  --> output_padding_top_stride 
                      // |- real -| -pad_r -|  --> remainder_padding_right(for k_w == 2 when kernel_width = 3)
                      out_ptr += output_padding_left & -(remainder > 0 && out_w == 0);
                      vl = remainder \
                          - (is_partial_stride_padded_part & (output_padding_top_stride - remainder_padding_left - remainder_padding_right)) \
                          - remainder_padding_right - remainder_padding_left;
                      // std::cout << "output_padding_left = " << output_padding_left << ", vl = " << vl << "\n";
                      vuint32m4_t v_w0 = __riscv_vle32_v_u32m4(in_ptr_now, vl);
                      __riscv_vse32_v_u32m4(out_ptr, v_w0, vl);
                      out_ptr += remainder - remainder_padding_left; // including remainder_padding_right
                      out_ptr += output_padding_left;
                      int is_added = vl + ((output_padding_left + remainder_padding_right) & -(vl > 0));
                      in_ptr_now = in_ptr_now + is_added;
                      // std::cout << "is_added = " << is_added << "\n";
                      vl = vlmax - remainder - output_padding_left;
                      v_w0 = __riscv_vle32_v_u32m4(in_ptr_now, vl);
                      __riscv_vse32_v_u32m4(out_ptr, v_w0, vl);
                      out_ptr += vl;
                      in_ptr_now += input_size - is_added;
                  }
                  in_ptr_now = in_ptr_now - input_size*group_input_channels + 1 - (output_padding_left & (-(out_w == 0) || is_partial_stride_padded_part));
                  moved_input_ptr_step += (1 - (output_padding_left & (-(out_w == 0) || is_partial_stride_padded_part)));
                  // std::cout << "moved_input_ptr_step = " << moved_input_ptr_step << "\n";
              }
              in_ptr_now = in_ptr_now - moved_input_ptr_step - (is_partial_stride_padded_part & (output_padding_top_stride + (input_padding_left & -(out_w != 0)))) + input_width;
              output_padding_top_stride -= (is_partial_stride_padded_part & output_padding_top_stride);
          }
          // todo : input_offset, end height padding
          int input_offset = -(output_padding_top_stride == 0) & (vlmax - remainder - input_padding_left);
          input_cursor = input_cursor + input_offset;
          im2col_cur += vlmax;
          output_cur += vlmax;
      }
      // middle part
      while((output_cur + vlmax) / output_width < output_height - input_padding_top){
          out_h = output_cur / output_width;
          out_w = output_cur % output_width;
          // -(out_h < input_padding_top) --> for the condition with padding
          // 0xFFFFFFFF if cond=1, else 0x00000000
          const int r = (output_cur + vlmax) % output_width;
          remainder = ((-(out_h != (output_cur + vlmax) / output_width)) & \
                  (((-(r != 0)) & (vlmax - r)) + ((-(r == 0)) & zero_max(vlmax - output_width))));
          int k_h = 0, k_w = 0;
          int input_offset = 0;
          uint32_t* in_ptr_now = in_ptr + input_cursor;
          // std::cout << "output_cur = " << output_cur << ", r = " << r << ", remainder = " << remainder << "\n";
          // || -- pad -- | -- input ----- ||                         --> pad before second load/store
          // || -- remainder ------------- || -- pad -- | - input -|| --> pad before second load/store
          int is_in_left_part = -((out_w == 0) || (remainder > 0));
          // || --- input ---   |-- pad -- ||                         --> pad after second load/store
          // || -- remainder -- |-- pad -- || ------ input ------- || --> pad before second load/store
          int is_in_right_part1 = -((out_h != (output_cur + vlmax) / output_width));
          int is_in_right_part2 = -((output_cur + vlmax) % output_width > 0);
          for(k_h = 0; k_h < kernel_height; k_h++){
              int moved_input_ptr_step = 0;
              for(k_w = 0; k_w < kernel_width; k_w++){
                  int output_padding_left = (is_in_left_part & zero_max(input_padding_left-k_w));
                  int remainder_padding_left = output_padding_left & (-(remainder > 0 && out_w == 0));
                  int remainder_padding_right = (-(remainder > 0) & zero_max(k_w + output_width-valid_width));
                  int output_padding_right = (-(remainder == 0 || (remainder > 0 && ~is_in_right_part2)) & is_in_right_part1 & zero_max(k_w + output_width-valid_width));
                  int vl;
                  // std::cout << "remainder_padding_right = " << output_padding_right << "\n";
                  for(int in_ch = 0; in_ch < group_input_channels; in_ch++){
                      out_ptr += remainder_padding_left;
                      vl = zero_max(remainder - remainder_padding_right - remainder_padding_left);
                      vuint32m4_t v_w0 = __riscv_vle32_v_u32m4(in_ptr_now, vl);
                      __riscv_vse32_v_u32m4(out_ptr, v_w0, vl);
                      // remainder with padding
                      out_ptr = out_ptr + remainder - remainder_padding_left + output_padding_left;
                      // in_ptr_now + vl - (output_width - output_padding_left - remainder_padding_right + input_width)
                      // two segments
                      int is_added = vl + ((output_padding_left + remainder_padding_right) & -(remainder > 0));
                      in_ptr_now = in_ptr_now + is_added;
                      // std::cout << "vl for remainder = " << vl << "\n";
                      vl = vlmax - remainder - output_padding_left - output_padding_right;
                      // std::cout << "vl = " << vl << "\n";
                      v_w0 = __riscv_vle32_v_u32m4(in_ptr_now, vl);
                      __riscv_vse32_v_u32m4(out_ptr, v_w0, vl);

                      out_ptr += vl + output_padding_right;
                      in_ptr_now = in_ptr_now - is_added + input_size;
                  }
                  // at the start of each stride, we need to take padding_left into consideration
                  in_ptr_now = in_ptr_now - input_size*group_input_channels + 1 - (output_padding_left & (-(out_w == 0)));
                  moved_input_ptr_step += (1 - (output_padding_left & (-(out_w == 0))));
              }
              // std::cout << "moved_input_ptr_step = " << moved_input_ptr_step << "\n";
              in_ptr_now = in_ptr_now - moved_input_ptr_step + input_width;
          }
          // || -- pad -- | -- input ----- ||
          //      --> + len(input)
          // || -- remainder ------------- || -- pad -- | - input -|| 
          //      --> + remainder - (output_width - input_padding_left) + input_width + (vlmax - remainder - input_padding_left) \
                      = remainder - output_width + input_padding_left + input_width + vlmax - remainder - input_padding_left \
                      = vlmax
          // || ----------- input -------- ||
          //      --> + vlmax
          // || --- pad( == remainder) --- || -- pad -- | - input -||
          int cond = -(input_cursor % input_width == 0);
          // todo : input_offset, end height padding
          input_offset = ((~cond & (vlmax + (~is_in_right_part2 & input_padding_left))) | \
                          (cond & (vlmax - remainder - input_padding_left + (output_width * (remainder / output_width)))));
          input_cursor = input_cursor + input_offset;
          // std::cout << "input_offset = " << input_offset << "\n";
          im2col_cur += vlmax;
          output_cur += vlmax;
      }

      // end part
      while(output_cur < output_size){
          out_h = output_cur / output_width;
          out_w = output_cur % output_width;
          int is_next_batch = -(im2col_cur + vlmax > output_size*(batch+1) && batch + 1 != batch_size);
          // std::cout << "is_next_batch = " << (is_next_batch & 1) << "\n";
          const int r = (output_cur + vlmax) % output_width;
          remainder = ((-(out_h != (output_cur + vlmax) / output_width)) & \
                  (((-(r != 0)) & (vlmax - r)) + ((-(r == 0)) & zero_max(vlmax - output_width))));
          int k_h = 0, k_w = 0;
          int input_offset = 0;
          uint32_t* in_ptr_now = in_ptr + input_cursor;
          int is_in_left_part = -((out_w == 0) || (remainder > 0));
          int is_in_right_part1 = -(vlmax - remainder >= output_width || remainder == 0); 
          int is_in_right_part2 = -((output_cur + vlmax) % output_width > 0);
          int is_whole_stride_padded = ~is_next_batch & -(out_h >= output_height - input_padding_top);
          int last_part_in_this_batch = output_size % vlmax;
          for(k_h = 0; k_h < kernel_height - (is_whole_stride_padded & input_padding_top); k_h++){
              int moved_input_ptr_step = 0;
              for(k_w = 0; k_w < kernel_width; k_w++){
                  int output_padding_left = is_in_left_part & zero_max(input_padding_left-k_w);
                  int output_padding_right = is_in_right_part1 & zero_max(k_w + output_width-valid_width);
                  int remainder_padding_left = output_padding_left & (-(remainder > 0 && out_w == 0));
                  int remainder_padding_right = zero_max(k_w + output_width-valid_width) & (-(remainder > 0 && out_h != (output_cur + vlmax) / output_width));
                  int vl;
                  // std::cout << "k_h = " << k_h << ", k_w = " << k_w << ", output_padding_left = " << output_padding_left << "\n";
                  // std::cout << "output_padding_right = " << output_padding_right << "\n";
                  for(int in_ch = 0; in_ch < group_input_channels; in_ch++){
                      int is_padded_remainder = -(
                          (zero_max(k_h - input_padding_top)) && out_h >= output_height - input_padding_top
                      );
                      vl = zero_max(remainder - remainder_padding_right);
                      vuint32m4_t v_w0 = __riscv_vle32_v_u32m4(in_ptr_now, ~is_padded_remainder & vl);
                      __riscv_vse32_v_u32m4(out_ptr, v_w0, ~is_padded_remainder & vl);
                      // remainder with padding
                      out_ptr = out_ptr + remainder + output_padding_left;
                      // in_ptr_now + vl - (output_width - output_padding_left - output_padding_right + input_width)
                      // two segments
                      int is_added = vl + ((output_padding_left + remainder_padding_right) & -(remainder > 0));
                      in_ptr_now = in_ptr_now + is_added;
                      int is_padded = -(
                                      ((zero_max(k_h - input_padding_top)) & -(((output_cur + vlmax) / output_width + 1 > output_height - input_padding_top))) \
                                      || ((output_cur + remainder >= output_size) && (batch + 1 >= batch_size)) \
                                      || (zero_max(input_padding_top-k_h) & -(output_cur + remainder >= output_size && batch + 1 != batch_size))
                                      );
                      int is_in_next_batch = -(output_cur + remainder >= output_size && batch + 1 != batch_size);
                      vl = vlmax - remainder - output_padding_left - output_padding_right;
                      v_w0 = __riscv_vle32_v_u32m4(in_ptr_now, (~is_padded | is_padded_remainder) & vl);
                      __riscv_vse32_v_u32m4(out_ptr, v_w0, (~is_padded | is_padded_remainder) & vl);

                      out_ptr += (vl + output_padding_right);
                      in_ptr_now = in_ptr_now - is_added + input_size;
                  }
                  // at the start of each stride, we need to take padding_left into consideration
                  in_ptr_now = in_ptr_now - input_size*group_input_channels + 1 - (output_padding_left & (-(out_w == 0)));
                  moved_input_ptr_step += (1 - (output_padding_left & (-(out_w == 0))));
              }
              // std::cout << "moved_input_ptr_step = " << moved_input_ptr_step << "\n";
              in_ptr_now = in_ptr_now - moved_input_ptr_step + input_width;
          }
          int cond = -(input_cursor % input_width == 0);
          input_offset = ((~cond & (vlmax + (~is_in_right_part2 & input_padding_left))) | \
                              ((cond) & (vlmax - remainder - input_padding_left)));
          input_cursor = input_cursor + input_offset;
          out_ptr += (is_whole_stride_padded & min(vlmax, output_width) * group_input_channels * kernel_width);
          im2col_cur += vlmax;
          output_cur += vlmax;
      }
      int last_part_in_this_batch = (output_size*(batch + 1)) % vlmax;
      int finished_part_in_next_batch = -(last_part_in_this_batch > 0) & (vlmax - last_part_in_this_batch);
      output_cur = finished_part_in_next_batch;
      output_padding_top_stride = zero_max(height_padding_start * output_width - finished_part_in_next_batch);
      input_cursor = output_size*(batch + 1);// + ((vlmax - remainder) % output_width);// output_size = output_width*output_height = input_width*input_height
      // std::cout << "output_padding_top_stride = " << output_padding_top_stride << ", input_cursor = " << input_cursor << "\n";
  }
}

void xnn_x32_packa_in_T_gemm_im2col_s1_d1_4v(uint32_t batch_size, const size_t input_height, const size_t input_width, size_t group_input_channels, \
  const int output_height, const int output_width,
  const size_t kernel_height, const size_t kernel_width, const size_t stride_height, const size_t stride_width, \
  const int dilation_height, const int dilation_width, const int input_padding_top,const int input_padding_left, \
  uint32_t* input, uint32_t* output){
  const size_t output_size = output_height*output_width;
  const size_t input_size = input_height*input_width*batch_size;
  uint32_t* in_ptr = input;
  uint32_t* out_ptr = output;
  const int vlmax = __riscv_vsetvlmax_e32m4();
  int valid_height = input_padding_top + input_height;
  int valid_width = input_padding_left + input_width;
  int height_padding_start = input_padding_top;
  int height_padding_end = zero_max(output_height-valid_height);
  int output_padding_top_stride = height_padding_start * output_width;
  int output_padding_down_stride = height_padding_end * output_width;
  int width_padding_start;
  int width_padding_end;
  int input_stride;
  int remainder = 0;
  int input_cursor = 0;
  int out_h, out_w, batch;
  int im2col_cur = 0;
  int batch_cur = 0;
  int output_cur = 0;
  batch = im2col_cur / output_size;
  // top_pad
  /*
      output_cur < vlmax * ceil(output_width / vlmax)
  --> output_cur < vlmax * ((output_width + vlmax - 1) / vlmax)
  */ 
  for(batch = 0; batch < batch_size; batch ++){
      while(output_cur < vlmax * ((output_width + vlmax - 1) / vlmax)){
          out_h = output_cur / output_width;
          out_w = output_cur % output_width;
          int is_whole_stride_padded_part = -(output_padding_top_stride/vlmax >= 1);
          remainder = ~is_whole_stride_padded_part & output_padding_top_stride;
          uint32_t* in_ptr_now = in_ptr + input_cursor;
          out_ptr += (is_whole_stride_padded_part & (min(vlmax, output_width) * group_input_channels * kernel_width));
          output_padding_top_stride -= (is_whole_stride_padded_part & vlmax);
          for(int k_h = is_whole_stride_padded_part & input_padding_top; k_h < kernel_height; k_h++){
              int moved_input_ptr_step = 0;
              int is_partial_stride_padded_part = -(output_padding_top_stride && zero_max(input_padding_top-k_h));
              for(int k_w = 0; k_w < kernel_width; k_w++){
                  int output_padding_left = zero_max(input_padding_left-k_w);
                  int remainder_padding_left = output_padding_left & (-(remainder > 0 && out_w == 0));
                  int remainder_padding_right = (-(remainder > 0) & zero_max(k_w + output_width-valid_width));
                  int vl;
                  for(int in_ch = 0; in_ch < group_input_channels; in_ch++){
                      out_ptr += output_padding_left & -(remainder > 0 && out_w == 0);
                      vl = remainder \
                          - (is_partial_stride_padded_part & (output_padding_top_stride - remainder_padding_left - remainder_padding_right)) \
                          - remainder_padding_right - remainder_padding_left;
                      memcpy(out_ptr, in_ptr_now, vl << 2);
                      out_ptr += remainder - remainder_padding_left; // including remainder_padding_right
                      out_ptr += output_padding_left;
                      int is_added = vl + ((output_padding_left + remainder_padding_right) & -(vl > 0));
                      in_ptr_now = in_ptr_now + is_added;
                      vl = vlmax - remainder - output_padding_left;
                      memcpy(out_ptr, in_ptr_now, vl << 2);
                      out_ptr += vl;
                      in_ptr_now += input_size - is_added;
                  }
                  in_ptr_now = in_ptr_now - input_size*group_input_channels + 1 - (output_padding_left & (-(out_w == 0) || is_partial_stride_padded_part));
                  moved_input_ptr_step += (1 - (output_padding_left & (-(out_w == 0) || is_partial_stride_padded_part)));
              }
              in_ptr_now = in_ptr_now - moved_input_ptr_step - (is_partial_stride_padded_part & (output_padding_top_stride + (input_padding_left & -(out_w != 0)))) + input_width;
              output_padding_top_stride -= (is_partial_stride_padded_part & output_padding_top_stride);
          }
          // todo : input_offset, end height padding
          int input_offset = -(output_padding_top_stride == 0) & (vlmax - remainder - input_padding_left);
          input_cursor = input_cursor + input_offset;
          im2col_cur += vlmax;
          output_cur += vlmax;
      }
      // middle part
      while((output_cur + vlmax) / output_width < output_height - input_padding_top){
          out_h = output_cur / output_width;
          out_w = output_cur % output_width;
          // -(out_h < input_padding_top) --> for the condition with padding
          // 0xFFFFFFFF if cond=1, else 0x00000000
          const int r = (output_cur + vlmax) % output_width;
          remainder = ((-(out_h != (output_cur + vlmax) / output_width)) & \
                  (((-(r != 0)) & (vlmax - r)) + ((-(r == 0)) & zero_max(vlmax - output_width))));
          int k_h = 0, k_w = 0;
          int input_offset = 0;
          uint32_t* in_ptr_now = in_ptr + input_cursor;
          int is_in_left_part = -((out_w == 0) || (remainder > 0));
          int is_in_right_part1 = -((out_h != (output_cur + vlmax) / output_width));
          int is_in_right_part2 = -((output_cur + vlmax) % output_width > 0);
          for(k_h = 0; k_h < kernel_height; k_h++){
              int moved_input_ptr_step = 0;
              for(k_w = 0; k_w < kernel_width; k_w++){
                  int output_padding_left = (is_in_left_part & zero_max(input_padding_left-k_w));
                  int remainder_padding_left = output_padding_left & (-(remainder > 0 && out_w == 0));
                  int remainder_padding_right = (-(remainder > 0) & zero_max(k_w + output_width-valid_width));
                  int output_padding_right = (-(remainder == 0 || (remainder > 0 && ~is_in_right_part2)) & is_in_right_part1 & zero_max(k_w + output_width-valid_width));
                  int vl;
                  for(int in_ch = 0; in_ch < group_input_channels; in_ch++){
                      out_ptr += remainder_padding_left;
                      vl = zero_max(remainder - remainder_padding_right - remainder_padding_left);
                      memcpy(out_ptr, in_ptr_now, vl << 2);
                      // remainder with padding
                      out_ptr = out_ptr + remainder - remainder_padding_left + output_padding_left;
                      // in_ptr_now + vl - (output_width - output_padding_left - remainder_padding_right + input_width)
                      // two segments
                      int is_added = vl + ((output_padding_left + remainder_padding_right) & -(remainder > 0));
                      in_ptr_now = in_ptr_now + is_added;
                      vl = vlmax - remainder - output_padding_left - output_padding_right;
                      memcpy(out_ptr, in_ptr_now, vl << 2);

                      out_ptr += vl + output_padding_right;
                      in_ptr_now = in_ptr_now - is_added + input_size;
                  }
                  // at the start of each stride, we need to take padding_left into consideration
                  in_ptr_now = in_ptr_now - input_size*group_input_channels + 1 - (output_padding_left & (-(out_w == 0)));
                  moved_input_ptr_step += (1 - (output_padding_left & (-(out_w == 0))));
              }
              in_ptr_now = in_ptr_now - moved_input_ptr_step + input_width;
          }
          int cond = -(input_cursor % input_width == 0);
          input_offset = ((~cond & (vlmax + (~is_in_right_part2 & input_padding_left))) | \
                          (cond & (vlmax - remainder - input_padding_left + (output_width * (remainder / output_width)))));
          input_cursor = input_cursor + input_offset;
          im2col_cur += vlmax;
          output_cur += vlmax;
      }

      // end part
      while(output_cur < output_size){
          out_h = output_cur / output_width;
          out_w = output_cur % output_width;
          int is_next_batch = -(im2col_cur + vlmax > output_size*(batch+1) && batch + 1 != batch_size);
          const int r = (output_cur + vlmax) % output_width;
          remainder = ((-(out_h != (output_cur + vlmax) / output_width)) & \
                  (((-(r != 0)) & (vlmax - r)) + ((-(r == 0)) & zero_max(vlmax - output_width))));
          int k_h = 0, k_w = 0;
          int input_offset = 0;
          uint32_t* in_ptr_now = in_ptr + input_cursor;
          int is_in_left_part = -((out_w == 0) || (remainder > 0));
          int is_in_right_part1 = -(vlmax - remainder >= output_width || remainder == 0); 
          int is_in_right_part2 = -((output_cur + vlmax) % output_width > 0);
          int is_whole_stride_padded = ~is_next_batch & -(out_h >= output_height - input_padding_top);
          int last_part_in_this_batch = output_size % vlmax;
          for(k_h = 0; k_h < kernel_height - (is_whole_stride_padded & input_padding_top); k_h++){
              int moved_input_ptr_step = 0;
              for(k_w = 0; k_w < kernel_width; k_w++){
                  int output_padding_left = is_in_left_part & zero_max(input_padding_left-k_w);
                  int output_padding_right = is_in_right_part1 & zero_max(k_w + output_width-valid_width);
                  int remainder_padding_left = output_padding_left & (-(remainder > 0 && out_w == 0));
                  int remainder_padding_right = zero_max(k_w + output_width-valid_width) & (-(remainder > 0 && out_h != (output_cur + vlmax) / output_width));
                  int vl;
                  for(int in_ch = 0; in_ch < group_input_channels; in_ch++){
                      int is_padded_remainder = -(
                          (zero_max(k_h - input_padding_top)) && out_h >= output_height - input_padding_top
                      );
                      vl = zero_max(remainder - remainder_padding_right);
                      memcpy(out_ptr, in_ptr_now, (~is_padded_remainder & vl) << 2);
                      // remainder with padding
                      out_ptr = out_ptr + remainder + output_padding_left;
                      // in_ptr_now + vl - (output_width - output_padding_left - output_padding_right + input_width)
                      // two segments
                      int is_added = vl + ((output_padding_left + remainder_padding_right) & -(remainder > 0));
                      in_ptr_now = in_ptr_now + is_added;
                      int is_padded = -(
                                      ((zero_max(k_h - input_padding_top)) & -(((output_cur + vlmax) / output_width + 1 > output_height - input_padding_top))) \
                                      || ((output_cur + remainder >= output_size) && (batch + 1 >= batch_size)) \
                                      || (zero_max(input_padding_top-k_h) & -(output_cur + remainder >= output_size && batch + 1 != batch_size))
                                      );
                      int is_in_next_batch = -(output_cur + remainder >= output_size && batch + 1 != batch_size);
                      vl = vlmax - remainder - output_padding_left - output_padding_right;
                      memcpy(out_ptr, in_ptr_now, ((~is_padded | is_padded_remainder) & vl) << 2);
                      out_ptr += (vl + output_padding_right);
                      in_ptr_now = in_ptr_now - is_added + input_size;
                  }
                  // at the start of each stride, we need to take padding_left into consideration
                  in_ptr_now = in_ptr_now - input_size*group_input_channels + 1 - (output_padding_left & (-(out_w == 0)));
                  moved_input_ptr_step += (1 - (output_padding_left & (-(out_w == 0))));
              }
              // std::cout << "moved_input_ptr_step = " << moved_input_ptr_step << "\n";
              in_ptr_now = in_ptr_now - moved_input_ptr_step + input_width;
          }
          int cond = -(input_cursor % input_width == 0);
          input_offset = ((~cond & (vlmax + (~is_in_right_part2 & input_padding_left))) | \
                              ((cond) & (vlmax - remainder - input_padding_left)));
          input_cursor = input_cursor + input_offset;
          out_ptr += (is_whole_stride_padded & min(vlmax, output_width) * group_input_channels * kernel_width);
          im2col_cur += vlmax;
          output_cur += vlmax;
      }
      int last_part_in_this_batch = (output_size*(batch + 1)) % vlmax;
      int finished_part_in_next_batch = -(last_part_in_this_batch > 0) & (vlmax - last_part_in_this_batch);
      output_cur = finished_part_in_next_batch;
      output_padding_top_stride = zero_max(height_padding_start * output_width - finished_part_in_next_batch);
      input_cursor = output_size*(batch + 1);// + ((vlmax - remainder) % output_width);// output_size = output_width*output_height = input_width*input_height
      // std::cout << "output_padding_top_stride = " << output_padding_top_stride << ", input_cursor = " << input_cursor << "\n";
  }
}

void xnn_x32_packa_in_T_gemm_im2col_s2_d1_x8v(uint32_t batch_size, const size_t input_height, const size_t input_width, size_t group_input_channels, \
  const int output_height, const int output_width,
  const size_t kernel_height, const size_t kernel_width, const size_t stride_height, const size_t stride_width, \
  const int dilation_height, const int dilation_width, const int input_padding_top,const int input_padding_left, \
  uint32_t* input, uint32_t* output){
  const size_t output_size = output_height*output_width;
  const size_t input_size = batch_size * input_height*input_width;
  uint32_t* in_ptr = input;
  uint32_t* in_ptr_rem = input;
  uint32_t* out_ptr = output;
  const int vlmax = __riscv_vsetvlmax_e32m8();
  int output_cur = 0;
  int input_cursor = 0;
  int valid_height = input_padding_top + input_height - 1;
  int last_stride = kernel_height - 1 + (output_height - 1)*2;
  int k_h_padding_end = last_stride - valid_height;
  int remainder = 0;
  for(size_t batch = 0; batch < batch_size; batch++){
      int base = batch * input_height*input_width;
      int output_padding_top_stride = ((input_padding_top + 1) >> 1) * output_width;
      int output_padding_down_stride = output_padding_top_stride;
      while(output_padding_top_stride > 0){
          int out_h = output_cur / output_width;
          int out_w = output_cur % output_width;
          input_cursor = base + (out_w << 1);
          remainder = -(output_width - out_w < vlmax) & (vlmax - output_width + out_w);
          // replace `* stride` to `<< 1` by the fact that stride = 2
          int padded_k_h = zero_max(input_padding_top - (out_h << 1) - (-(remainder > 0) & ((out_h + 1) << 1)));
          // std::cout << "remainder = " << remainder << ", padded_k_h = " << padded_k_h << "\n";
          in_ptr = input + input_cursor;
          in_ptr_rem = input + base + ((output_cur + remainder) / output_width) * output_width;
          out_ptr += vlmax * group_input_channels * kernel_width * padded_k_h;
          for(int k_h = padded_k_h; k_h < kernel_height; k_h++){
              int padded = -(remainder > 0 && k_h < input_padding_top - (out_h << 1));
              for(int k_w = 0; k_w < kernel_width; k_w++){
                  int width_padding_start = -(out_w == 0) & ((zero_max(input_padding_left-k_w) + 1) >> 1);
                  int width_padding_end = -((output_cur + vlmax) / output_width != out_h) & ((zero_max(k_w + ((output_width-1) << 1) - (input_padding_left + input_width-1)) + 1) >> 1);
                  int input_offset_with_pad = stride_width * width_padding_start - (input_padding_left - k_w);
                  int input_offset_with_pad_cond = -(k_w < input_padding_left);
                  int input_cur_offset = (input_offset_with_pad_cond & input_offset_with_pad) + \
                                          (~input_offset_with_pad_cond & (k_w - input_padding_left));
                  // for remainder
                  int width_padding_start_rem = -(remainder > 0) & ((zero_max(input_padding_left-k_w) + 1) >> 1);
                  input_offset_with_pad = stride_width * width_padding_start_rem - (input_padding_left - k_w);
                  int input_cur_offset_rem = (input_offset_with_pad_cond & input_offset_with_pad) + \
                                          (~input_offset_with_pad_cond & (k_w - input_padding_left));
                  uint32_t* in_ptr_now = in_ptr + input_cur_offset;
                  uint32_t* in_ptr_remainder = in_ptr_rem + input_cur_offset_rem;
                  for(size_t in_ch = 0; in_ch < group_input_channels; in_ch++){
                      out_ptr += width_padding_start;
                      int vl = vlmax-width_padding_end-width_padding_start-remainder;
                      // std::cout << "vl = " << vl << "\n";
                      vuint32m8_t v_w0 = __riscv_vlse32_v_u32m8 (in_ptr_now, stride_width << 2, ~padded & vl);
                      __riscv_vse32_v_u32m8(out_ptr, v_w0, ~padded & vl);
                      out_ptr += vl + width_padding_end;
                      // for remainder
                      out_ptr += width_padding_start_rem;
                      vl = zero_max(remainder - width_padding_start_rem);
                      // std::cout << "vl for remainder = " << vl << "\n";
                      v_w0 = __riscv_vlse32_v_u32m8 (in_ptr_remainder, stride_width << 2, vl);
                      __riscv_vse32_v_u32m8(out_ptr, v_w0, vl);
                      out_ptr += vl;
                      in_ptr_now += input_size;
                      in_ptr_remainder += input_size;
                  }
              }
              in_ptr += ~padded & (output_width << 1);
              in_ptr_rem += -(remainder > 0) & (output_width << 1);
              // std::cout << "moved offset = " << (~padded & (output_width << 1)) << "\n";
          }
          output_padding_top_stride -= vlmax;
          output_cur += vlmax;
      }
      input_cursor = base + input_width + (remainder << 1);
      while(((input_padding_top + 1) >> 1) + input_cursor / input_width + kernel_height - base / input_width <= valid_height){
          int out_h = output_cur / output_width;
          int out_w = output_cur % output_width;
          int input_cursor_rem = base + input_width + ((input_width << 1) * (out_h - 1));
          in_ptr = input + input_cursor;
          in_ptr_rem = input + input_cursor_rem;
          remainder = -(output_width - out_w < vlmax) & (vlmax - output_width + out_w);
          for(int k_h = 0; k_h < kernel_height; k_h++){
              int padded = -(remainder > 0 && ((input_padding_top + 1) >> 1) + input_cursor_rem / input_width + k_h - base / input_width >= valid_height);
              for(int k_w = 0; k_w < kernel_width; k_w++){
                  int width_padding_start = -(out_w == 0) & ((zero_max(input_padding_left-k_w) + 1) >> 1);
                  int width_padding_end = -((output_cur + vlmax) / output_width != out_h) & ((zero_max(k_w + ((output_width-1) << 1) - (input_padding_left + input_width-1)) + 1) >> 1);
                  int input_offset_with_pad = stride_width * width_padding_start - (input_padding_left - k_w);
                  int input_offset_with_pad_cond = -(k_w < input_padding_left);
                  int input_cur_offset = (input_offset_with_pad_cond & input_offset_with_pad) + \
                                          (~input_offset_with_pad_cond & (k_w - input_padding_left));
                  // for remainder
                  int width_padding_start_rem = -(remainder > 0) & ((zero_max(input_padding_left-k_w) + 1) >> 1);
                  input_offset_with_pad = stride_width * width_padding_start_rem - (input_padding_left - k_w);
                  int input_cur_offset_rem = (input_offset_with_pad_cond & input_offset_with_pad) + \
                                          (~input_offset_with_pad_cond & (k_w - input_padding_left));
                  uint32_t* in_ptr_now = in_ptr + input_cur_offset;
                  uint32_t* in_ptr_remainder = in_ptr_rem + input_cur_offset_rem;
                  for(size_t in_ch = 0; in_ch < group_input_channels; in_ch++){
                      out_ptr += width_padding_start;
                      int vl = vlmax-width_padding_end-width_padding_start-remainder;
                      vuint32m8_t v_w0 = __riscv_vlse32_v_u32m8 (in_ptr_now, stride_width << 2, vl);
                      __riscv_vse32_v_u32m8(out_ptr, v_w0, vl);
                      out_ptr += vl + width_padding_end;
                      // for remainder
                      out_ptr += width_padding_start_rem;
                      vl = zero_max(remainder - width_padding_start_rem);
                      // std::cout << "vl for remainder = " << vl << "\n";
                      v_w0 = __riscv_vlse32_v_u32m8 (in_ptr_remainder, stride_width << 2, ~padded & vl);
                      __riscv_vse32_v_u32m8(out_ptr, v_w0, ~padded & vl);
                      out_ptr += vl;
                      in_ptr_now += input_size;
                      in_ptr_remainder += input_size;
                  }
              }
              in_ptr += output_width << 1;
              in_ptr_rem += output_width << 1;
          }
          output_padding_top_stride -= vlmax;
          output_cur += vlmax;
          input_cursor = base + input_width + ((input_width << 1) * (output_cur / output_width - 2)) + ((output_cur % output_width) << 1);
      }
      while(output_cur < output_size){
          int out_h = output_cur / output_width;
          int out_w = output_cur % output_width;
          input_cursor = base + input_width + ((input_width << 1) * (out_h - 2)) + (out_w << 1);
          int input_cursor_rem = base + input_width + ((input_width << 1) * (out_h - 1));
          in_ptr = input + input_cursor;
          remainder = -(output_width - out_w < vlmax) & (vlmax - output_width + out_w);
          for(int k_h = 0; k_h < kernel_height-k_h_padding_end; k_h++){
              for(int k_w = 0; k_w < kernel_width; k_w++){
                  int width_padding_start = -(out_w == 0) & ((zero_max(input_padding_left-k_w) + 1) >> 1);
                  int width_padding_end = -((output_cur + vlmax) / output_width != out_h) & ((zero_max(k_w + ((output_width-1) << 1) - (input_padding_left + input_width-1)) + 1) >> 1);
                  int input_offset_with_pad = stride_width * width_padding_start - (input_padding_left - k_w);
                  int input_offset_with_pad_cond = -(k_w < input_padding_left);
                  int input_cur_offset = (input_offset_with_pad_cond & input_offset_with_pad) + \
                                          (~input_offset_with_pad_cond & (k_w - input_padding_left));
                  uint32_t* in_ptr_now = in_ptr + input_cur_offset;
                  for(size_t in_ch = 0; in_ch < group_input_channels; in_ch++){
                      out_ptr += width_padding_start;
                      int vl = vlmax-width_padding_end-width_padding_start-remainder;
                      vuint32m8_t v_w0 = __riscv_vlse32_v_u32m8 (in_ptr_now, stride_width << 2, vl);
                      __riscv_vse32_v_u32m8(out_ptr, v_w0, vl);
                      out_ptr += vl + width_padding_end;
                      in_ptr_now += input_size;
                  }
              }
              in_ptr += output_width << 1;
          }
          out_ptr += k_h_padding_end * vlmax * group_input_channels * kernel_width;
          output_padding_top_stride -= vlmax;
          output_cur += vlmax;
      }
      output_cur = 0;
  }
}

void xnn_x32_packa_in_T_gemm_im2col_s1_d1_1x8v(uint32_t batch_size, const size_t input_height, const size_t input_width, size_t group_input_channels, \
  const int output_height, const int output_width,
  const size_t kernel_height, const size_t kernel_width, const size_t stride_height, const size_t stride_width, \
  const int dilation_height, const int dilation_width, const int input_padding_top,const int input_padding_left, \
  uint32_t* input, uint32_t* output){
  const size_t output_size = output_height*output_width;
  const size_t input_size = input_height*input_width*batch_size;
  uint32_t* in_ptr = input;
  uint32_t* out_ptr = output;
  const int nr = __riscv_vsetvlmax_e32m8();
  const int vlmax = __riscv_vsetvlmax_e32m1();
  int valid_height = input_padding_top + input_height;
  int valid_width = input_padding_left + input_width;
  int height_padding_start = input_padding_top;
  int height_padding_end = zero_max(output_height-valid_height);
  int output_padding_top_stride = height_padding_start * output_width;
  int output_padding_down_stride = height_padding_end * output_width;
  int width_padding_start;
  int width_padding_end;
  int input_stride;
  int remainder = 0;
  int input_cursor = 0;
  int out_h, out_w, batch;
  int im2col_cur = 0;
  int output_cur = 0;

  for(batch = 0; batch < batch_size; batch++){
  // top_pad
  /*
      im2col_cur < nr * ceil(output_width / nr)
  --> im2col_cur < nr * ((output_width + nr - 1) / nr)
  */ 
      while(output_padding_top_stride){
          out_h = output_cur / output_width;
          out_w = output_cur % output_width;
          remainder = output_padding_top_stride;
          uint32_t* in_ptr_now = in_ptr + input_cursor;
          for(int k_h = 0; k_h < kernel_height; k_h++){
              int moved_input_ptr_step = 0;
              int is_partial_stride_padded_part = -(output_padding_top_stride && zero_max(input_padding_top-k_h));
              for(int k_w = 0; k_w < kernel_width; k_w++){
                  int output_padding_left = zero_max(input_padding_left-k_w);
                  int output_padding_right = zero_max(k_w + output_width-valid_width);
                  int remainder_padding_left = output_padding_left & (-(remainder > 0 && out_w == 0));
                  int remainder_padding_right = output_padding_right & (-(remainder > 0));
                  int vl;
                  for(int in_ch = 0; in_ch < group_input_channels; in_ch++){
                      out_ptr += remainder_padding_left;
                      vl = remainder \
                          - (is_partial_stride_padded_part & (output_padding_top_stride - remainder_padding_left - remainder_padding_right)) \
                          - remainder_padding_right - remainder_padding_left;
                      vuint32m1_t v_w0 = __riscv_vle32_v_u32m1(in_ptr_now, vl);
                      __riscv_vse32_v_u32m1(out_ptr, v_w0, vl);
                      out_ptr += remainder - remainder_padding_left; // including remainder_padding_right
                      int is_added = vl + ((output_padding_left + remainder_padding_right) & -(vl > 0));
                      in_ptr_now = in_ptr_now + is_added;
                      for(int cur_vl = 0; cur_vl < nr - remainder; cur_vl += output_width){
                          out_ptr += output_padding_left;
                          int is_whole_stride = -(nr - remainder - cur_vl >= output_width);
                          int segment_padding_right = (output_padding_right & is_whole_stride);
                          int current_output_offset = output_cur + remainder + cur_vl;
                          vl = (is_whole_stride & (output_width - output_padding_left - segment_padding_right)) + \
                              (~is_whole_stride & nr - remainder - cur_vl - output_padding_left);
                          int is_exceed_boarder = -(current_output_offset >= output_size);
                          int is_exceed_batch_output_size = -(batch + 1 >= batch_size) & is_exceed_boarder;
                          int is_padded_end = -(zero_max(k_h - input_padding_top) & -(output_size - output_width * input_padding_top <= current_output_offset && current_output_offset < output_size));
                          int is_next_batch_padding_top = -(zero_max(input_padding_top-k_h) & -(batch + 1 != batch_size && (current_output_offset % output_size < output_width*input_padding_top)) & is_exceed_boarder);
                          v_w0 = __riscv_vle32_v_u32m1(in_ptr_now, (~is_padded_end & ~is_exceed_batch_output_size & ~is_next_batch_padding_top) & vl);
                          __riscv_vse32_v_u32m1(out_ptr, v_w0, (~is_padded_end & ~is_exceed_batch_output_size & ~is_next_batch_padding_top) & vl);
                          in_ptr_now += vl + ((output_padding_left + segment_padding_right) & is_whole_stride);
                          out_ptr += vl + segment_padding_right;
                          // + vl - (output_width - output_padding_left - remainder_padding_right) + output_width = vl + output_padding_left
                          is_added += vl + ((output_padding_left + segment_padding_right) & is_whole_stride);
                      }
                      in_ptr_now += input_size - is_added;
                  }
                  in_ptr_now = in_ptr_now - input_size*group_input_channels + 1 - (output_padding_left & (-(out_w == 0) || is_partial_stride_padded_part));
                  moved_input_ptr_step += (1 - (output_padding_left & (-(out_w == 0) || is_partial_stride_padded_part)));
              }
              in_ptr_now = in_ptr_now - moved_input_ptr_step - (is_partial_stride_padded_part & (output_padding_top_stride + (input_padding_left & -(out_w != 0)))) + input_width;
              output_padding_top_stride -= (is_partial_stride_padded_part & output_padding_top_stride);
          }
          int input_offset = input_width * ((nr - remainder) / input_width) + ((nr - remainder) % input_width - input_padding_left);
          input_cursor = input_cursor + input_offset;
          im2col_cur += nr;
          output_cur += nr;
      }
      // middle part
      while((output_cur + nr) / output_width < output_height - input_padding_top){
          out_h = output_cur / output_width;
          out_w = output_cur % output_width;
          remainder = output_width - ((nr - remainder) % output_width);
          int k_h = 0, k_w = 0;
          int input_offset = 0;
          uint32_t* in_ptr_now = in_ptr + input_cursor;
          int is_in_left_part = -((out_w == 0) || (remainder > 0));
          int is_in_right_part1 = -((out_h != (output_cur + nr) / output_width));
          int is_in_right_part2 = -((output_cur + nr) % output_width > 0);
          for(k_h = 0; k_h < kernel_height; k_h++){
              int moved_input_ptr_step = 0;
              for(k_w = 0; k_w < kernel_width; k_w++){
                  int output_padding_left = (zero_max(input_padding_left-k_w));
                  int remainder_padding_left = output_padding_left & (-(remainder > 0 && out_w == 0));
                  int remainder_padding_right = (-(remainder > 0) & zero_max(k_w + output_width-valid_width));
                  int output_padding_right = zero_max(k_w + output_width-valid_width);
                  int vl;
                  for(int in_ch = 0; in_ch < group_input_channels; in_ch++){
                      out_ptr += remainder_padding_left;
                      vl = zero_max(remainder - remainder_padding_right - remainder_padding_left);
                      vuint32m1_t v_w0 = __riscv_vle32_v_u32m1(in_ptr_now, vl);
                      __riscv_vse32_v_u32m1(out_ptr, v_w0, vl);
                      // remainder with padding
                      out_ptr = out_ptr + remainder - remainder_padding_left;
                      // in_ptr_now + vl - (output_width - output_padding_left - remainder_padding_right + input_width)
                      // two segments
                      int is_added = vl + ((output_padding_left + remainder_padding_right) & -(remainder > 0));
                      in_ptr_now = in_ptr_now + is_added;
                      // std::cout << "vl for remainder = " << vl << "\n";
                      for(int cur_vl = 0; cur_vl < nr - remainder; cur_vl += output_width){
                          out_ptr += output_padding_left;
                          int is_whole_stride = -(nr - remainder - cur_vl >= output_width);
                          int segment_padding_right = (output_padding_right & is_whole_stride);
                          // std::cout << "vl = " << vl << "\n";
                          vl = (is_whole_stride & (output_width - output_padding_left - segment_padding_right)) + \
                              (~is_whole_stride & nr - remainder - cur_vl - output_padding_left);
                          v_w0 = __riscv_vle32_v_u32m1(in_ptr_now, vl);
                          __riscv_vse32_v_u32m1(out_ptr, v_w0, vl);
                          in_ptr_now += vl + ((output_padding_left + segment_padding_right) & is_whole_stride);
                          out_ptr += vl + segment_padding_right;
                          // + vl - (output_width - output_padding_left - remainder_padding_right) + output_width = vl + output_padding_left
                          is_added += vl + ((output_padding_left + segment_padding_right) & is_whole_stride);
                      }
                      in_ptr_now = in_ptr_now - is_added + input_size;
                  }
                  in_ptr_now = in_ptr_now - input_size*group_input_channels + 1 - (output_padding_left & (-(out_w == 0)));
                  moved_input_ptr_step += (1 - (output_padding_left & (-(out_w == 0))));
              }
              in_ptr_now = in_ptr_now - moved_input_ptr_step + input_width;
          }
          int cond = -(input_cursor % input_width == 0);
          input_offset = ((~cond & (nr + (~is_in_right_part2 & input_padding_left))) | \
                          (cond & (nr - remainder - input_padding_left + (output_width * ((remainder + (-((remainder + input_padding_left) % output_width == 0) & 1)) / output_width)))));
          input_cursor = input_cursor + input_offset;
          im2col_cur += nr;
          output_cur += nr;
      }
      // end part
      while(output_cur < output_size){
          out_h = output_cur / output_width;
          out_w = output_cur % output_width;
          int is_next_batch = -(im2col_cur + vlmax > output_size*(batch+1) && batch + 1 != batch_size);
          const int r = (output_cur + nr) % output_width;
          remainder = output_width - ((nr - remainder) % output_width);
          int k_h = 0, k_w = 0;
          int input_offset = 0;
          uint32_t* in_ptr_now = in_ptr + input_cursor;
          int is_in_left_part = -((out_w == 0) || (remainder > 0));
          int is_in_right_part1 = -((out_h != (output_cur + nr) / output_width));
          int is_in_right_part2 = -((output_cur + nr) % output_width > 0);
          int is_whole_stride_padded = ~is_next_batch & -(out_h >= output_height - input_padding_top);
          int last_part_in_this_batch = output_size % vlmax;
          for(k_h = 0; k_h < kernel_height - (is_whole_stride_padded & input_padding_top); k_h++){
              int moved_input_ptr_step = 0;
              for(k_w = 0; k_w < kernel_width; k_w++){
                  int output_padding_left = zero_max(input_padding_left-k_w);
                  int output_padding_right = zero_max(k_w + output_width-valid_width);
                  int remainder_padding_left = output_padding_left & (-(remainder > 0 && out_w == 0));
                  int remainder_padding_right = (-(remainder > 0) & zero_max(k_w + output_width-valid_width));
                  int vl;
                  for(int in_ch = 0; in_ch < group_input_channels; in_ch++){
                      int is_padded_remainder = -((zero_max(k_h - input_padding_top)) & -(output_size - output_width * input_padding_top <= output_cur));
                      out_ptr += remainder_padding_left;
                      vl = zero_max(remainder - remainder_padding_right - remainder_padding_left);
                      vuint32m1_t v_w0 = __riscv_vle32_v_u32m1(in_ptr_now, ~is_padded_remainder & vl);
                      __riscv_vse32_v_u32m1(out_ptr, v_w0, ~is_padded_remainder & vl);
                      // remainder with padding
                      out_ptr = out_ptr + remainder - remainder_padding_left;
                      // in_ptr_now + vl - (output_width - output_padding_left - remainder_padding_right + input_width)
                      // two segments
                      int is_added = vl + ((output_padding_left + remainder_padding_right) & -(remainder > 0));
                      in_ptr_now = in_ptr_now + is_added;
                      int current_output_offset = output_cur + remainder;
                      for(int cur_vl = 0; cur_vl < nr - remainder; cur_vl += output_width){
                          out_ptr += output_padding_left;
                          int is_whole_stride = -(nr - remainder - cur_vl >= output_width);
                          int segment_padding_right = (output_padding_right & is_whole_stride);
                          vl = (is_whole_stride & (output_width - output_padding_left - segment_padding_right)) + \
                              (~is_whole_stride & nr - remainder - cur_vl - output_padding_left);
                          int is_exceed_boarder = -(current_output_offset >= output_size);
                          int is_exceed_batch_output_size = -(im2col_cur + remainder + cur_vl >= input_size);
                          int is_padded_end = -(zero_max(k_h - input_padding_top) & -(output_size - output_width * input_padding_top <= current_output_offset && current_output_offset < output_size));
                          int is_next_batch_padding_top = -(zero_max(input_padding_top-k_h) & -(batch + 1 != batch_size && (current_output_offset % output_size < output_width*input_padding_top)) & is_exceed_boarder);
                          v_w0 = __riscv_vle32_v_u32m1(in_ptr_now, (~is_padded_end & ~is_exceed_batch_output_size & ~is_next_batch_padding_top) & vl);
                          __riscv_vse32_v_u32m1(out_ptr, v_w0, (~is_padded_end & ~is_exceed_batch_output_size & ~is_next_batch_padding_top) & vl);
                          in_ptr_now += vl + ((output_padding_left + segment_padding_right) & is_whole_stride);
                          out_ptr += vl + segment_padding_right;
                          // + vl - (output_width - output_padding_left - remainder_padding_right) + output_width = vl + output_padding_left
                          is_added += vl + ((output_padding_left + segment_padding_right) & is_whole_stride);
                          current_output_offset = current_output_offset + output_width - (is_exceed_boarder & output_size);
                      }
                      in_ptr_now = in_ptr_now - is_added + input_size;
                  }
                  // at the start of each stride, we need to take padding_left into consideration
                  in_ptr_now = in_ptr_now - input_size*group_input_channels + 1 - (output_padding_left & (-(out_w == 0)));
                  moved_input_ptr_step += (1 - (output_padding_left & (-(out_w == 0))));
              }
              // std::cout << "moved_input_ptr_step = " << moved_input_ptr_step << "\n";
              in_ptr_now = in_ptr_now - moved_input_ptr_step + input_width;
          }
          int cond = -(input_cursor % input_width == 0);
          input_offset = ((~cond & (nr + (~is_in_right_part2 & input_padding_left))) | \
                              (cond & nr));
          input_cursor = input_cursor + input_offset;
          out_ptr += (is_whole_stride_padded & min(vlmax, output_width) * group_input_channels * kernel_width);
          im2col_cur += nr;
          output_cur += nr;
      }
      int last_part_in_this_batch = (output_size*(batch + 1)) % nr;
      int finished_part_in_next_batch = -(last_part_in_this_batch > 0) & (nr - last_part_in_this_batch);
      output_cur = finished_part_in_next_batch;
      output_padding_top_stride = zero_max(height_padding_start * output_width - finished_part_in_next_batch);
  }
}

void xnn_x32_packa_in_T_gemm_im2col_s1_d1_2x8v(uint32_t batch_size, const size_t input_height, const size_t input_width, size_t group_input_channels, \
  const int output_height, const int output_width,
  const size_t kernel_height, const size_t kernel_width, const size_t stride_height, const size_t stride_width, \
  const int dilation_height, const int dilation_width, const int input_padding_top,const int input_padding_left, \
  uint32_t* input, uint32_t* output){
  const size_t output_size = output_height*output_width;
  const size_t input_size = input_height*input_width*batch_size;
  uint32_t* in_ptr = input;
  uint32_t* out_ptr = output;
  const int nr = __riscv_vsetvlmax_e32m8();
  const int vlmax = __riscv_vsetvlmax_e32m2();
  int valid_height = input_padding_top + input_height;
  int valid_width = input_padding_left + input_width;
  int height_padding_start = input_padding_top;
  int height_padding_end = zero_max(output_height-valid_height);
  int output_padding_top_stride = height_padding_start * output_width;
  int output_padding_down_stride = height_padding_end * output_width;
  int width_padding_start;
  int width_padding_end;
  int input_stride;
  int remainder = 0;
  int input_cursor = 0;
  int out_h, out_w, batch;
  int im2col_cur = 0;
  int output_cur = 0;

  for(batch = 0; batch < batch_size; batch++){
  // top_pad
  /*
      im2col_cur < nr * ceil(output_width / nr)
  --> im2col_cur < nr * ((output_width + nr - 1) / nr)
  */ 
      while(output_cur < output_width * ((output_width + nr - 1) / nr)){
          out_h = output_cur / output_width;
          out_w = output_cur % output_width;
          remainder = output_padding_top_stride;
          uint32_t* in_ptr_now = in_ptr + input_cursor;
          for(int k_h = 0; k_h < kernel_height; k_h++){
              int moved_input_ptr_step = 0;
              int is_partial_stride_padded_part = -(output_padding_top_stride && zero_max(input_padding_top-k_h));
              for(int k_w = 0; k_w < kernel_width; k_w++){
                  int output_padding_left = zero_max(input_padding_left-k_w);
                  int output_padding_right = zero_max(k_w + output_width-valid_width);
                  int remainder_padding_left = output_padding_left & (-(remainder > 0 && out_w == 0));
                  int remainder_padding_right = output_padding_right & (-(remainder > 0));
                  int vl;
                  for(int in_ch = 0; in_ch < group_input_channels; in_ch++){
                      out_ptr += remainder_padding_left;
                      vl = remainder \
                          - (is_partial_stride_padded_part & (output_padding_top_stride - remainder_padding_left - remainder_padding_right)) \
                          - remainder_padding_right - remainder_padding_left;
                      vuint32m2_t v_w0 = __riscv_vle32_v_u32m2(in_ptr_now, vl);
                      __riscv_vse32_v_u32m2(out_ptr, v_w0, vl);
                      out_ptr += remainder - remainder_padding_left; // including remainder_padding_right
                      int is_added = vl + ((output_padding_left + remainder_padding_right) & -(vl > 0));
                      in_ptr_now = in_ptr_now + is_added;
                      for(int cur_vl = 0; cur_vl < nr - remainder; cur_vl += output_width){
                          out_ptr += output_padding_left;
                          int is_whole_stride = -(nr - remainder - cur_vl >= output_width);
                          int segment_padding_right = (output_padding_right & is_whole_stride);
                          vl = (is_whole_stride & (output_width - output_padding_left - segment_padding_right)) + \
                              (~is_whole_stride & nr - remainder - cur_vl - output_padding_left);
                          v_w0 = __riscv_vle32_v_u32m2(in_ptr_now, vl);
                          __riscv_vse32_v_u32m2(out_ptr, v_w0, vl);
                          in_ptr_now += vl + ((output_padding_left + segment_padding_right) & is_whole_stride);
                          out_ptr += vl + segment_padding_right;
                          // + vl - (output_width - output_padding_left - remainder_padding_right) + output_width = vl + output_padding_left
                          is_added += vl + ((output_padding_left + segment_padding_right) & is_whole_stride);
                      }
                      in_ptr_now += input_size - is_added;
                  }
                  in_ptr_now = in_ptr_now - input_size*group_input_channels + 1 - (output_padding_left & (-(out_w == 0) || is_partial_stride_padded_part));
                  moved_input_ptr_step += (1 - (output_padding_left & (-(out_w == 0) || is_partial_stride_padded_part)));
              }
              in_ptr_now = in_ptr_now - moved_input_ptr_step - (is_partial_stride_padded_part & (output_padding_top_stride + (input_padding_left & -(out_w != 0)))) + input_width;
              output_padding_top_stride -= (is_partial_stride_padded_part & output_padding_top_stride);
          }
          int input_offset = input_width * ((nr - remainder) / input_width) + ((nr - remainder) % input_width - input_padding_left);
          input_cursor = input_cursor + input_offset;
          im2col_cur += nr;
          output_cur += nr;
      }
      // middle part
      while((output_cur + nr) / output_width < output_height - input_padding_top){
          out_h = output_cur / output_width;
          out_w = output_cur % output_width;
          remainder = output_width - ((nr - remainder) % output_width);
          int k_h = 0, k_w = 0;
          int input_offset = 0;
          uint32_t* in_ptr_now = in_ptr + input_cursor;
          int is_in_left_part = -((out_w == 0) || (remainder > 0));
          int is_in_right_part1 = -((out_h != (output_cur + nr) / output_width));
          int is_in_right_part2 = -((output_cur + nr) % output_width > 0);
          for(k_h = 0; k_h < kernel_height; k_h++){
              int moved_input_ptr_step = 0;
              for(k_w = 0; k_w < kernel_width; k_w++){
                  int output_padding_left = (zero_max(input_padding_left-k_w));
                  int remainder_padding_left = output_padding_left & (-(remainder > 0 && out_w == 0));
                  int remainder_padding_right = (-(remainder > 0) & zero_max(k_w + output_width-valid_width));
                  int output_padding_right = zero_max(k_w + output_width-valid_width);
                  int vl;
                  for(int in_ch = 0; in_ch < group_input_channels; in_ch++){
                      out_ptr += remainder_padding_left;
                      vl = zero_max(remainder - remainder_padding_right - remainder_padding_left);
                      vuint32m2_t v_w0 = __riscv_vle32_v_u32m2(in_ptr_now, vl);
                      __riscv_vse32_v_u32m2(out_ptr, v_w0, vl);
                      // remainder with padding
                      out_ptr = out_ptr + remainder - remainder_padding_left;
                      // in_ptr_now + vl - (output_width - output_padding_left - remainder_padding_right + input_width)
                      // two segments
                      int is_added = vl + ((output_padding_left + remainder_padding_right) & -(remainder > 0));
                      in_ptr_now = in_ptr_now + is_added;
                      // std::cout << "vl for remainder = " << vl << "\n";
                      for(int cur_vl = 0; cur_vl < nr - remainder; cur_vl += output_width){
                          out_ptr += output_padding_left;
                          int is_whole_stride = -(nr - remainder - cur_vl >= output_width);
                          int segment_padding_right = (output_padding_right & is_whole_stride);
                          // std::cout << "vl = " << vl << "\n";
                          vl = (is_whole_stride & (output_width - output_padding_left - segment_padding_right)) + \
                              (~is_whole_stride & nr - remainder - cur_vl - output_padding_left);
                          v_w0 = __riscv_vle32_v_u32m2(in_ptr_now, vl);
                          __riscv_vse32_v_u32m2(out_ptr, v_w0, vl);
                          in_ptr_now += vl + ((output_padding_left + segment_padding_right) & is_whole_stride);
                          out_ptr += vl + segment_padding_right;
                          // + vl - (output_width - output_padding_left - remainder_padding_right) + output_width = vl + output_padding_left
                          is_added += vl + ((output_padding_left + segment_padding_right) & is_whole_stride);
                      }
                      in_ptr_now = in_ptr_now - is_added + input_size;
                  }
                  in_ptr_now = in_ptr_now - input_size*group_input_channels + 1 - (output_padding_left & (-(out_w == 0)));
                  moved_input_ptr_step += (1 - (output_padding_left & (-(out_w == 0))));
              }
              in_ptr_now = in_ptr_now - moved_input_ptr_step + input_width;
          }
          int cond = -(input_cursor % input_width == 0);
          input_offset = ((~cond & (nr + (~is_in_right_part2 & input_padding_left))) | \
                          (cond & (nr - remainder - input_padding_left + (output_width * (remainder / output_width)))));
          input_cursor = input_cursor + input_offset;
          im2col_cur += nr;
          output_cur += nr;
      }
      // end part
      while(output_cur < output_size){
          out_h = output_cur / output_width;
          out_w = output_cur % output_width;
          int is_next_batch = -(im2col_cur + vlmax > output_size*(batch+1) && batch + 1 != batch_size);
          const int r = (output_cur + nr) % output_width;
          remainder = output_width - ((nr - remainder) % output_width);
          int k_h = 0, k_w = 0;
          int input_offset = 0;
          uint32_t* in_ptr_now = in_ptr + input_cursor;
          int is_in_left_part = -((out_w == 0) || (remainder > 0));
          int is_in_right_part1 = -((out_h != (output_cur + nr) / output_width));
          int is_in_right_part2 = -((output_cur + nr) % output_width > 0);
          int is_whole_stride_padded = ~is_next_batch & -(out_h >= output_height - input_padding_top);            int last_part_in_this_batch = output_size % vlmax;
          for(k_h = 0; k_h < kernel_height - (is_whole_stride_padded & input_padding_top); k_h++){
              int moved_input_ptr_step = 0;
              for(k_w = 0; k_w < kernel_width; k_w++){
                  int output_padding_left = zero_max(input_padding_left-k_w);
                  int output_padding_right = zero_max(k_w + output_width-valid_width);
                  int remainder_padding_left = output_padding_left & (-(remainder > 0 && out_w == 0));
                  int remainder_padding_right = (-(remainder > 0) & zero_max(k_w + output_width-valid_width));
                  int vl;
                  for(int in_ch = 0; in_ch < group_input_channels; in_ch++){
                      int is_padded_remainder = -((zero_max(k_h - input_padding_top)) & -(output_size - output_width * input_padding_top <= output_cur));
                      out_ptr += remainder_padding_left;
                      vl = zero_max(remainder - remainder_padding_right - remainder_padding_left);
                      vuint32m2_t v_w0 = __riscv_vle32_v_u32m2(in_ptr_now, ~is_padded_remainder & vl);
                      __riscv_vse32_v_u32m2(out_ptr, v_w0, ~is_padded_remainder & vl);
                      // remainder with padding
                      out_ptr = out_ptr + remainder - remainder_padding_left;
                      // in_ptr_now + vl - (output_width - output_padding_left - remainder_padding_right + input_width)
                      // two segments
                      int is_added = vl + ((output_padding_left + remainder_padding_right) & -(remainder > 0));
                      in_ptr_now = in_ptr_now + is_added;
                      for(int cur_vl = 0; cur_vl < nr - remainder; cur_vl += output_width){
                          out_ptr += output_padding_left;
                          int is_whole_stride = -(nr - remainder - cur_vl >= output_width);
                          int segment_padding_right = (output_padding_right & is_whole_stride);
                          int current_output_offset = output_cur + remainder + cur_vl;
                          // std::cout << "vl = " << vl << "\n";
                          vl = (is_whole_stride & (output_width - output_padding_left - segment_padding_right)) + \
                              (~is_whole_stride & nr - remainder - cur_vl - output_padding_left);
                          int is_exceed_boarder = -(current_output_offset >= output_size);
                          int is_exceed_batch_output_size = -(batch + 1 >= batch_size) & is_exceed_boarder;
                          int is_padded_end = -(zero_max(k_h - input_padding_top) & -(output_size - output_width * input_padding_top <= current_output_offset && current_output_offset < output_size));
                          int is_next_batch_padding_top = -(zero_max(input_padding_top-k_h) & -(batch + 1 != batch_size && (current_output_offset % output_size < output_width*input_padding_top)) & is_exceed_boarder);
                          v_w0 = __riscv_vle32_v_u32m2(in_ptr_now, (~is_padded_end & ~is_exceed_batch_output_size & ~is_next_batch_padding_top) & vl);
                          __riscv_vse32_v_u32m2(out_ptr, v_w0, (~is_padded_end & ~is_exceed_batch_output_size & ~is_next_batch_padding_top) & vl);
                          in_ptr_now += vl + ((output_padding_left + segment_padding_right) & is_whole_stride);
                          out_ptr += vl + segment_padding_right;
                          // + vl - (output_width - output_padding_left - remainder_padding_right) + output_width = vl + output_padding_left
                          is_added += vl + ((output_padding_left + segment_padding_right) & is_whole_stride);
                      }
                      in_ptr_now = in_ptr_now - is_added + input_size;
                  }
                  // at the start of each stride, we need to take padding_left into consideration
                  in_ptr_now = in_ptr_now - input_size*group_input_channels + 1 - (output_padding_left & (-(out_w == 0)));
                  moved_input_ptr_step += (1 - (output_padding_left & (-(out_w == 0))));
              }
              // std::cout << "moved_input_ptr_step = " << moved_input_ptr_step << "\n";
              in_ptr_now = in_ptr_now - moved_input_ptr_step + input_width;
          }
          int cond = -(input_cursor % input_width == 0);
          input_offset = ((~cond & (nr + (~is_in_right_part2 & input_padding_left))) | \
                              ((cond) & (nr - remainder - input_padding_left)));
          input_cursor = input_cursor + input_offset;
          out_ptr += (is_whole_stride_padded & min(vlmax, output_width) * group_input_channels * kernel_width);
          // std::cout << "input_cursor = " << input_cursor << "\n";
          im2col_cur += nr;
          output_cur += nr;
      }
      int last_part_in_this_batch = (output_size*(batch + 1)) % nr;
      int finished_part_in_next_batch = -(last_part_in_this_batch > 0) & (nr - last_part_in_this_batch);
      output_cur = finished_part_in_next_batch;
      output_padding_top_stride = zero_max(height_padding_start * output_width - finished_part_in_next_batch);
  }
}

void xnn_x32_packa_in_T_gemm_im2col_s1_d1_4x8v(uint32_t batch_size, const size_t input_height, const size_t input_width, size_t group_input_channels, \
  const int output_height, const int output_width,
  const size_t kernel_height, const size_t kernel_width, const size_t stride_height, const size_t stride_width, \
  const int dilation_height, const int dilation_width, const int input_padding_top,const int input_padding_left, \
  uint32_t* input, uint32_t* output){
  const size_t output_size = output_height*output_width;
  const size_t input_size = input_height*input_width*batch_size;
  uint32_t* in_ptr = input;
  uint32_t* out_ptr = output;
  const int nr = __riscv_vsetvlmax_e32m8();
  const int vlmax = __riscv_vsetvlmax_e32m4();
  int valid_height = input_padding_top + input_height;
  int valid_width = input_padding_left + input_width;
  int height_padding_start = input_padding_top;
  int height_padding_end = zero_max(output_height-valid_height);
  int output_padding_top_stride = height_padding_start * output_width;
  int output_padding_down_stride = height_padding_end * output_width;
  int width_padding_start;
  int width_padding_end;
  int input_stride;
  int remainder = 0;
  int input_cursor = 0;
  int out_h, out_w, batch;
  int im2col_cur = 0;
  int output_cur = 0;

  for(batch = 0; batch < batch_size; batch++){
  // top_pad
  /*
      im2col_cur < nr * ceil(output_width / nr)
  --> im2col_cur < nr * ((output_width + nr - 1) / nr)
  */ 
      while(output_padding_top_stride){
          out_h = output_cur / output_width;
          out_w = output_cur % output_width;
          remainder = output_padding_top_stride;
          uint32_t* in_ptr_now = in_ptr + input_cursor;
          for(int k_h = 0; k_h < kernel_height; k_h++){
              int moved_input_ptr_step = 0;
              int is_partial_stride_padded_part = -(output_padding_top_stride && zero_max(input_padding_top-k_h));
              for(int k_w = 0; k_w < kernel_width; k_w++){
                  int output_padding_left = zero_max(input_padding_left-k_w);
                  int output_padding_right = zero_max(k_w + output_width-valid_width);
                  int remainder_padding_left = output_padding_left & (-(remainder > 0 && out_w == 0));
                  int remainder_padding_right = output_padding_right & (-(remainder > 0));
                  int vl;
                  for(int in_ch = 0; in_ch < group_input_channels; in_ch++){
                      out_ptr += remainder_padding_left;
                      vl = remainder \
                          - (is_partial_stride_padded_part & (output_padding_top_stride - remainder_padding_left - remainder_padding_right)) \
                          - remainder_padding_right - remainder_padding_left;
                      vuint32m4_t v_w0 = __riscv_vle32_v_u32m4(in_ptr_now, vl);
                      __riscv_vse32_v_u32m4(out_ptr, v_w0, vl);
                      out_ptr += remainder - remainder_padding_left; // including remainder_padding_right
                      int is_added = vl + ((output_padding_left + remainder_padding_right) & -(vl > 0));
                      in_ptr_now = in_ptr_now + is_added;
                      for(int cur_vl = 0; cur_vl < nr - remainder; cur_vl += output_width){
                          out_ptr += output_padding_left;
                          int is_whole_stride = -(nr - remainder - cur_vl >= output_width);
                          int segment_padding_right = (output_padding_right & is_whole_stride);
                          vl = (is_whole_stride & (output_width - output_padding_left - segment_padding_right)) + \
                              (~is_whole_stride & nr - remainder - cur_vl - output_padding_left);
                          v_w0 = __riscv_vle32_v_u32m4(in_ptr_now, vl);
                          __riscv_vse32_v_u32m4(out_ptr, v_w0, vl);
                          in_ptr_now += vl + ((output_padding_left + segment_padding_right) & is_whole_stride);
                          out_ptr += vl + segment_padding_right;
                          // + vl - (output_width - output_padding_left - remainder_padding_right) + output_width = vl + output_padding_left
                          is_added += vl + ((output_padding_left + segment_padding_right) & is_whole_stride);
                      }
                      in_ptr_now += input_size - is_added;
                  }
                  in_ptr_now = in_ptr_now - input_size*group_input_channels + 1 - (output_padding_left & (-(out_w == 0) || is_partial_stride_padded_part));
                  moved_input_ptr_step += (1 - (output_padding_left & (-(out_w == 0) || is_partial_stride_padded_part)));
              }
              in_ptr_now = in_ptr_now - moved_input_ptr_step - (is_partial_stride_padded_part & (output_padding_top_stride + (input_padding_left & -(out_w != 0)))) + input_width;
              output_padding_top_stride -= (is_partial_stride_padded_part & output_padding_top_stride);
          }
          int input_offset = input_width * ((nr - remainder) / input_width) + ((nr - remainder) % input_width - input_padding_left);
          input_cursor = input_cursor + input_offset;
          im2col_cur += nr;
          output_cur += nr;
      }
      // middle part
      while((output_cur + nr) / output_width < output_height - input_padding_top){
          out_h = output_cur / output_width;
          out_w = output_cur % output_width;
          remainder = output_width - ((nr - remainder) % output_width);
          int k_h = 0, k_w = 0;
          int input_offset = 0;
          uint32_t* in_ptr_now = in_ptr + input_cursor;
          int is_in_left_part = -((out_w == 0) || (remainder > 0));
          int is_in_right_part1 = -((out_h != (output_cur + nr) / output_width));
          int is_in_right_part2 = -((output_cur + nr) % output_width > 0);
          for(k_h = 0; k_h < kernel_height; k_h++){
              int moved_input_ptr_step = 0;
              for(k_w = 0; k_w < kernel_width; k_w++){
                  int output_padding_left = (zero_max(input_padding_left-k_w));
                  int remainder_padding_left = output_padding_left & (-(remainder > 0 && out_w == 0));
                  int remainder_padding_right = (-(remainder > 0) & zero_max(k_w + output_width-valid_width));
                  int output_padding_right = zero_max(k_w + output_width-valid_width);
                  int vl;
                  for(int in_ch = 0; in_ch < group_input_channels; in_ch++){
                      out_ptr += remainder_padding_left;
                      vl = zero_max(remainder - remainder_padding_right - remainder_padding_left);
                      vuint32m4_t v_w0 = __riscv_vle32_v_u32m4(in_ptr_now, vl);
                      __riscv_vse32_v_u32m4(out_ptr, v_w0, vl);
                      // remainder with padding
                      out_ptr = out_ptr + remainder - remainder_padding_left;
                      // in_ptr_now + vl - (output_width - output_padding_left - remainder_padding_right + input_width)
                      // two segments
                      int is_added = vl + ((output_padding_left + remainder_padding_right) & -(remainder > 0));
                      in_ptr_now = in_ptr_now + is_added;
                      // std::cout << "vl for remainder = " << vl << "\n";
                      for(int cur_vl = 0; cur_vl < nr - remainder; cur_vl += output_width){
                          out_ptr += output_padding_left;
                          int is_whole_stride = -(nr - remainder - cur_vl >= output_width);
                          int segment_padding_right = (output_padding_right & is_whole_stride);
                          // std::cout << "vl = " << vl << "\n";
                          vl = (is_whole_stride & (output_width - output_padding_left - segment_padding_right)) + \
                              (~is_whole_stride & nr - remainder - cur_vl - output_padding_left);
                          v_w0 = __riscv_vle32_v_u32m4(in_ptr_now, vl);
                          __riscv_vse32_v_u32m4(out_ptr, v_w0, vl);
                          in_ptr_now += vl + ((output_padding_left + segment_padding_right) & is_whole_stride);
                          out_ptr += vl + segment_padding_right;
                          // + vl - (output_width - output_padding_left - remainder_padding_right) + output_width = vl + output_padding_left
                          is_added += vl + ((output_padding_left + segment_padding_right) & is_whole_stride);
                      }
                      in_ptr_now = in_ptr_now - is_added + input_size;
                  }
                  in_ptr_now = in_ptr_now - input_size*group_input_channels + 1 - (output_padding_left & (-(out_w == 0)));
                  moved_input_ptr_step += (1 - (output_padding_left & (-(out_w == 0))));
              }
              in_ptr_now = in_ptr_now - moved_input_ptr_step + input_width;
          }
          int cond = -(input_cursor % input_width == 0);
          input_offset = ((~cond & (nr + (~is_in_right_part2 & input_padding_left))) | \
                          (cond & (nr - remainder - input_padding_left + (output_width * (remainder / output_width)))));
          input_cursor = input_cursor + input_offset;
          im2col_cur += nr;
          output_cur += nr;
      }
      // end part
      while(output_cur < output_size){
          out_h = output_cur / output_width;
          out_w = output_cur % output_width;
          int is_next_batch = -(im2col_cur + vlmax > output_size*(batch+1) && batch + 1 != batch_size);
          const int r = (output_cur + nr) % output_width;
          remainder = output_width - ((nr - remainder) % output_width);
          int k_h = 0, k_w = 0;
          int input_offset = 0;
          uint32_t* in_ptr_now = in_ptr + input_cursor;
          int is_in_left_part = -((out_w == 0) || (remainder > 0));
          int is_in_right_part1 = -((out_h != (output_cur + nr) / output_width));
          int is_in_right_part2 = -((output_cur + nr) % output_width > 0);
          int is_whole_stride_padded = ~is_next_batch & -(out_h >= output_height - input_padding_top);            int last_part_in_this_batch = output_size % vlmax;
          for(k_h = 0; k_h < kernel_height - (is_whole_stride_padded & input_padding_top); k_h++){
              int moved_input_ptr_step = 0;
              for(k_w = 0; k_w < kernel_width; k_w++){
                  int output_padding_left = zero_max(input_padding_left-k_w);
                  int output_padding_right = zero_max(k_w + output_width-valid_width);
                  int remainder_padding_left = output_padding_left & (-(remainder > 0 && out_w == 0));
                  int remainder_padding_right = (-(remainder > 0) & zero_max(k_w + output_width-valid_width));
                  int vl;
                  for(int in_ch = 0; in_ch < group_input_channels; in_ch++){
                      int is_padded_remainder = -((zero_max(k_h - input_padding_top)) & -(output_size - output_width * input_padding_top <= output_cur));
                      out_ptr += remainder_padding_left;
                      vl = zero_max(remainder - remainder_padding_right - remainder_padding_left);
                      vuint32m4_t v_w0 = __riscv_vle32_v_u32m4(in_ptr_now, ~is_padded_remainder & vl);
                      __riscv_vse32_v_u32m4(out_ptr, v_w0, ~is_padded_remainder & vl);
                      // remainder with padding
                      out_ptr = out_ptr + remainder - remainder_padding_left;
                      // in_ptr_now + vl - (output_width - output_padding_left - remainder_padding_right + input_width)
                      // two segments
                      int is_added = vl + ((output_padding_left + remainder_padding_right) & -(remainder > 0));
                      in_ptr_now = in_ptr_now + is_added;
                      for(int cur_vl = 0; cur_vl < nr - remainder; cur_vl += output_width){
                          out_ptr += output_padding_left;
                          int is_whole_stride = -(nr - remainder - cur_vl >= output_width);
                          int segment_padding_right = (output_padding_right & is_whole_stride);
                          int current_output_offset = output_cur + remainder + cur_vl;
                          // std::cout << "vl = " << vl << "\n";
                          vl = (is_whole_stride & (output_width - output_padding_left - segment_padding_right)) + \
                              (~is_whole_stride & nr - remainder - cur_vl - output_padding_left);
                          int is_exceed_boarder = -(current_output_offset >= output_size);
                          int is_exceed_batch_output_size = -(batch + 1 >= batch_size) & is_exceed_boarder;
                          int is_padded_end = -(zero_max(k_h - input_padding_top) & -(output_size - output_width * input_padding_top <= current_output_offset && current_output_offset < output_size));
                          int is_next_batch_padding_top = -(zero_max(input_padding_top-k_h) & -(batch + 1 != batch_size && (current_output_offset % output_size < output_width*input_padding_top)) & is_exceed_boarder);
                          v_w0 = __riscv_vle32_v_u32m4(in_ptr_now, (~is_padded_end & ~is_exceed_batch_output_size & ~is_next_batch_padding_top) & vl);
                          __riscv_vse32_v_u32m4(out_ptr, v_w0, (~is_padded_end & ~is_exceed_batch_output_size & ~is_next_batch_padding_top) & vl);
                          in_ptr_now += vl + ((output_padding_left + segment_padding_right) & is_whole_stride);
                          out_ptr += vl + segment_padding_right;
                          // + vl - (output_width - output_padding_left - remainder_padding_right) + output_width = vl + output_padding_left
                          is_added += vl + ((output_padding_left + segment_padding_right) & is_whole_stride);
                      }
                      in_ptr_now = in_ptr_now - is_added + input_size;
                  }
                  // at the start of each stride, we need to take padding_left into consideration
                  in_ptr_now = in_ptr_now - input_size*group_input_channels + 1 - (output_padding_left & (-(out_w == 0)));
                  moved_input_ptr_step += (1 - (output_padding_left & (-(out_w == 0))));
              }
              // std::cout << "moved_input_ptr_step = " << moved_input_ptr_step << "\n";
              in_ptr_now = in_ptr_now - moved_input_ptr_step + input_width;
          }
          int cond = -(input_cursor % input_width == 0);
          input_offset = ((~cond & (nr + (~is_in_right_part2 & input_padding_left))) | \
                              ((cond) & (nr - remainder - input_padding_left)));
          input_cursor = input_cursor + input_offset;
          out_ptr += (is_whole_stride_padded & min(vlmax, output_width) * group_input_channels * kernel_width);
          // std::cout << "input_cursor = " << input_cursor << "\n";
          im2col_cur += nr;
          output_cur += nr;
      }
      int last_part_in_this_batch = (output_size*(batch + 1)) % nr;
      int finished_part_in_next_batch = -(last_part_in_this_batch > 0) & (nr - last_part_in_this_batch);
      output_cur = finished_part_in_next_batch;
      output_padding_top_stride = zero_max(height_padding_start * output_width - finished_part_in_next_batch);
      input_cursor = output_size * (batch + 1) + (-(output_padding_top_stride == 0) & ((output_cur / output_width - 1) * output_width + (output_cur % output_width - input_padding_left)));
  }
}

void xnn_x32_packa_in_T_gemm_im2col_s1_d1_8x8v(uint32_t batch_size, const size_t input_height, const size_t input_width, size_t group_input_channels, \
  const int output_height, const int output_width,
  const size_t kernel_height, const size_t kernel_width, const size_t stride_height, const size_t stride_width, \
  const int dilation_height, const int dilation_width, const int input_padding_top,const int input_padding_left, \
  uint32_t* input, uint32_t* output){
  const size_t output_size = output_height*output_width;
  const size_t input_size = input_height*input_width*batch_size;
  uint32_t* in_ptr = input;
  uint32_t* out_ptr = output;
  const int nr = __riscv_vsetvlmax_e32m8();
  const int vlmax = __riscv_vsetvlmax_e32m8();
  int valid_height = input_padding_top + input_height;
  int valid_width = input_padding_left + input_width;
  int height_padding_start = input_padding_top;
  int height_padding_end = zero_max(output_height-valid_height);
  int output_padding_top_stride = height_padding_start * output_width;
  int output_padding_down_stride = height_padding_end * output_width;
  int width_padding_start;
  int width_padding_end;
  int input_stride;
  int remainder = 0;
  int input_cursor = 0;
  int out_h, out_w, batch;
  int im2col_cur = 0;
  int output_cur = 0;

  for(batch = 0; batch < batch_size; batch++){
  // top_pad
  /*
      im2col_cur < nr * ceil(output_width / nr)
  --> im2col_cur < nr * ((output_width + nr - 1) / nr)
  */ 
      while(output_padding_top_stride){
          out_h = output_cur / output_width;
          out_w = output_cur % output_width;
          remainder = output_padding_top_stride;
          uint32_t* in_ptr_now = in_ptr + input_cursor;
          for(int k_h = 0; k_h < kernel_height; k_h++){
              int moved_input_ptr_step = 0;
              int is_partial_stride_padded_part = -(output_padding_top_stride && zero_max(input_padding_top-k_h));
              for(int k_w = 0; k_w < kernel_width; k_w++){
                  int output_padding_left = zero_max(input_padding_left-k_w);
                  int output_padding_right = zero_max(k_w + output_width-valid_width);
                  int remainder_padding_left = output_padding_left & (-(remainder > 0 && out_w == 0));
                  int remainder_padding_right = output_padding_right & (-(remainder > 0));
                  int vl;
                  for(int in_ch = 0; in_ch < group_input_channels; in_ch++){
                      out_ptr += remainder_padding_left;
                      vl = remainder \
                          - (is_partial_stride_padded_part & (output_padding_top_stride - remainder_padding_left - remainder_padding_right)) \
                          - remainder_padding_right - remainder_padding_left;
                      vuint32m8_t v_w0 = __riscv_vle32_v_u32m8(in_ptr_now, vl);
                      __riscv_vse32_v_u32m8(out_ptr, v_w0, vl);
                      out_ptr += remainder - remainder_padding_left; // including remainder_padding_right
                      int is_added = vl + ((output_padding_left + remainder_padding_right) & -(vl > 0));
                      in_ptr_now = in_ptr_now + is_added;
                      for(int cur_vl = 0; cur_vl < nr - remainder; cur_vl += output_width){
                          out_ptr += output_padding_left;
                          int is_whole_stride = -(nr - remainder - cur_vl >= output_width);
                          int segment_padding_right = (output_padding_right & is_whole_stride);
                          vl = (is_whole_stride & (output_width - output_padding_left - segment_padding_right)) + \
                              (~is_whole_stride & nr - remainder - cur_vl - output_padding_left);
                          v_w0 = __riscv_vle32_v_u32m8(in_ptr_now, vl);
                          __riscv_vse32_v_u32m8(out_ptr, v_w0, vl);
                          in_ptr_now += vl + ((output_padding_left + segment_padding_right) & is_whole_stride);
                          out_ptr += vl + segment_padding_right;
                          // + vl - (output_width - output_padding_left - remainder_padding_right) + output_width = vl + output_padding_left
                          is_added += vl + ((output_padding_left + segment_padding_right) & is_whole_stride);
                      }
                      in_ptr_now += input_size - is_added;
                  }
                  in_ptr_now = in_ptr_now - input_size*group_input_channels + 1 - (output_padding_left & (-(out_w == 0) || is_partial_stride_padded_part));
                  moved_input_ptr_step += (1 - (output_padding_left & (-(out_w == 0) || is_partial_stride_padded_part)));
              }
              in_ptr_now = in_ptr_now - moved_input_ptr_step - (is_partial_stride_padded_part & (output_padding_top_stride + (input_padding_left & -(out_w != 0)))) + input_width;
              output_padding_top_stride -= (is_partial_stride_padded_part & output_padding_top_stride);
          }
          int input_offset = input_width * ((nr - remainder) / input_width) + ((nr - remainder) % input_width - input_padding_left);
          input_cursor = input_cursor + input_offset;
          im2col_cur += nr;
          output_cur += nr;
      }
      // middle part
      while((output_cur + nr) / output_width < output_height - input_padding_top){
          out_h = output_cur / output_width;
          out_w = output_cur % output_width;
          remainder = output_width - ((nr - remainder) % output_width);
          int k_h = 0, k_w = 0;
          int input_offset = 0;
          uint32_t* in_ptr_now = in_ptr + input_cursor;
          int is_in_left_part = -((out_w == 0) || (remainder > 0));
          int is_in_right_part1 = -((out_h != (output_cur + nr) / output_width));
          int is_in_right_part2 = -((output_cur + nr) % output_width > 0);
          for(k_h = 0; k_h < kernel_height; k_h++){
              int moved_input_ptr_step = 0;
              for(k_w = 0; k_w < kernel_width; k_w++){
                  int output_padding_left = (zero_max(input_padding_left-k_w));
                  int remainder_padding_left = output_padding_left & (-(remainder > 0 && out_w == 0));
                  int remainder_padding_right = (-(remainder > 0) & zero_max(k_w + output_width-valid_width));
                  int output_padding_right = zero_max(k_w + output_width-valid_width);
                  int vl;
                  for(int in_ch = 0; in_ch < group_input_channels; in_ch++){
                      out_ptr += remainder_padding_left;
                      vl = zero_max(remainder - remainder_padding_right - remainder_padding_left);
                      vuint32m8_t v_w0 = __riscv_vle32_v_u32m8(in_ptr_now, vl);
                      __riscv_vse32_v_u32m8(out_ptr, v_w0, vl);
                      // remainder with padding
                      out_ptr = out_ptr + remainder - remainder_padding_left;
                      // in_ptr_now + vl - (output_width - output_padding_left - remainder_padding_right + input_width)
                      // two segments
                      int is_added = vl + ((output_padding_left + remainder_padding_right) & -(remainder > 0));
                      in_ptr_now = in_ptr_now + is_added;
                      // std::cout << "vl for remainder = " << vl << "\n";
                      for(int cur_vl = 0; cur_vl < nr - remainder; cur_vl += output_width){
                          out_ptr += output_padding_left;
                          int is_whole_stride = -(nr - remainder - cur_vl >= output_width);
                          int segment_padding_right = (output_padding_right & is_whole_stride);
                          // std::cout << "vl = " << vl << "\n";
                          vl = (is_whole_stride & (output_width - output_padding_left - segment_padding_right)) + \
                              (~is_whole_stride & nr - remainder - cur_vl - output_padding_left);
                          v_w0 = __riscv_vle32_v_u32m8(in_ptr_now, vl);
                          __riscv_vse32_v_u32m8(out_ptr, v_w0, vl);
                          in_ptr_now += vl + ((output_padding_left + segment_padding_right) & is_whole_stride);
                          out_ptr += vl + segment_padding_right;
                          // + vl - (output_width - output_padding_left - remainder_padding_right) + output_width = vl + output_padding_left
                          is_added += vl + ((output_padding_left + segment_padding_right) & is_whole_stride);
                      }
                      in_ptr_now = in_ptr_now - is_added + input_size;
                  }
                  in_ptr_now = in_ptr_now - input_size*group_input_channels + 1 - (output_padding_left & (-(out_w == 0)));
                  moved_input_ptr_step += (1 - (output_padding_left & (-(out_w == 0))));
              }
              in_ptr_now = in_ptr_now - moved_input_ptr_step + input_width;
          }
          int cond = -(input_cursor % input_width == 0);
          input_offset = ((~cond & (nr + (~is_in_right_part2 & input_padding_left))) | \
                          (cond & (nr - remainder - input_padding_left + (output_width * (remainder / output_width)))));
          input_cursor = input_cursor + input_offset;
          im2col_cur += nr;
          output_cur += nr;
      }
      // end part
      while(output_cur < output_size){
          out_h = output_cur / output_width;
          out_w = output_cur % output_width;
          int is_next_batch = -(im2col_cur + vlmax > output_size*(batch+1) && batch + 1 != batch_size);
          const int r = (output_cur + nr) % output_width;
          remainder = output_width - ((nr - remainder) % output_width);
          int k_h = 0, k_w = 0;
          int input_offset = 0;
          uint32_t* in_ptr_now = in_ptr + input_cursor;
          int is_in_left_part = -((out_w == 0) || (remainder > 0));
          int is_in_right_part1 = -((out_h != (output_cur + nr) / output_width));
          int is_in_right_part2 = -((output_cur + nr) % output_width > 0);
          int is_whole_stride_padded = ~is_next_batch & -(out_h >= output_height - input_padding_top);            int last_part_in_this_batch = output_size % vlmax;
          for(k_h = 0; k_h < kernel_height - (is_whole_stride_padded & input_padding_top); k_h++){
              int moved_input_ptr_step = 0;
              for(k_w = 0; k_w < kernel_width; k_w++){
                  int output_padding_left = zero_max(input_padding_left-k_w);
                  int output_padding_right = zero_max(k_w + output_width-valid_width);
                  int remainder_padding_left = output_padding_left & (-(remainder > 0 && out_w == 0));
                  int remainder_padding_right = (-(remainder > 0) & zero_max(k_w + output_width-valid_width));
                  int vl;
                  for(int in_ch = 0; in_ch < group_input_channels; in_ch++){
                      int is_padded_remainder = -((zero_max(k_h - input_padding_top)) & -(output_size - output_width * input_padding_top <= output_cur));
                      out_ptr += remainder_padding_left;
                      vl = zero_max(remainder - remainder_padding_right - remainder_padding_left);
                      vuint32m8_t v_w0 = __riscv_vle32_v_u32m8(in_ptr_now, ~is_padded_remainder & vl);
                      __riscv_vse32_v_u32m8(out_ptr, v_w0, ~is_padded_remainder & vl);
                      // remainder with padding
                      out_ptr = out_ptr + remainder - remainder_padding_left;
                      // in_ptr_now + vl - (output_width - output_padding_left - remainder_padding_right + input_width)
                      // two segments
                      int is_added = vl + ((output_padding_left + remainder_padding_right) & -(remainder > 0));
                      in_ptr_now = in_ptr_now + is_added;
                      for(int cur_vl = 0; cur_vl < nr - remainder; cur_vl += output_width){
                          out_ptr += output_padding_left;
                          int is_whole_stride = -(nr - remainder - cur_vl >= output_width);
                          int segment_padding_right = (output_padding_right & is_whole_stride);
                          int current_output_offset = output_cur + remainder + cur_vl;
                          // std::cout << "vl = " << vl << "\n";
                          vl = (is_whole_stride & (output_width - output_padding_left - segment_padding_right)) + \
                              (~is_whole_stride & nr - remainder - cur_vl - output_padding_left);
                          int is_exceed_boarder = -(current_output_offset >= output_size);
                          int is_exceed_batch_output_size = -(batch + 1 >= batch_size) & is_exceed_boarder;
                          int is_padded_end = -(zero_max(k_h - input_padding_top) & -(output_size - output_width * input_padding_top <= current_output_offset && current_output_offset < output_size));
                          int is_next_batch_padding_top = -(zero_max(input_padding_top-k_h) & -(batch + 1 != batch_size && (current_output_offset % output_size < output_width*input_padding_top)) & is_exceed_boarder);
                          v_w0 = __riscv_vle32_v_u32m8(in_ptr_now, (~is_padded_end & ~is_exceed_batch_output_size & ~is_next_batch_padding_top) & vl);
                          __riscv_vse32_v_u32m8(out_ptr, v_w0, (~is_padded_end & ~is_exceed_batch_output_size & ~is_next_batch_padding_top) & vl);
                          in_ptr_now += vl + ((output_padding_left + segment_padding_right) & is_whole_stride);
                          out_ptr += vl + segment_padding_right;
                          // + vl - (output_width - output_padding_left - remainder_padding_right) + output_width = vl + output_padding_left
                          is_added += vl + ((output_padding_left + segment_padding_right) & is_whole_stride);
                      }
                      in_ptr_now = in_ptr_now - is_added + input_size;
                  }
                  // at the start of each stride, we need to take padding_left into consideration
                  in_ptr_now = in_ptr_now - input_size*group_input_channels + 1 - (output_padding_left & (-(out_w == 0)));
                  moved_input_ptr_step += (1 - (output_padding_left & (-(out_w == 0))));
              }
              // std::cout << "moved_input_ptr_step = " << moved_input_ptr_step << "\n";
              in_ptr_now = in_ptr_now - moved_input_ptr_step + input_width;
          }
          int cond = -(input_cursor % input_width == 0);
          input_offset = ((~cond & (nr + (~is_in_right_part2 & input_padding_left))) | \
                              ((cond) & (nr - remainder - input_padding_left)));
          input_cursor = input_cursor + input_offset;
          out_ptr += (is_whole_stride_padded & min(vlmax, output_width) * group_input_channels * kernel_width);
          // std::cout << "input_cursor = " << input_cursor << "\n";
          im2col_cur += nr;
          output_cur += nr;
      }
      int last_part_in_this_batch = (output_size*(batch + 1)) % nr;
      int finished_part_in_next_batch = -(last_part_in_this_batch > 0) & (nr - last_part_in_this_batch);
      output_cur = finished_part_in_next_batch;
      output_padding_top_stride = zero_max(height_padding_start * output_width - finished_part_in_next_batch);
      input_cursor = output_size * (batch + 1) + (-(output_padding_top_stride == 0) & ((output_cur / output_width - 1) * output_width + (output_cur % output_width - input_padding_left)));
  }
}