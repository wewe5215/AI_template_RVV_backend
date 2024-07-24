#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <stdexcept>
// #include "../../../XNNPACK/include/xnnpack.h"
// #include <riscv_vector.h>
void transposed_conv2d_bias_relu_3 (
    void* in_ptr,
    void* weight_ptr,
    void* out_ptr,
    void* bias_ptr,
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
  int64_t HO = (HI - 1) * SH - 2 * PH + KHEff;
  int64_t WO = (WI - 1) * SW - 2 * PW + KWEff;
  *out_batch = NO;
  *out_h = HO;
  *out_w = WO;
  *out_ch = CO;
  float* in_ptr_tmp = static_cast<float*>(in_ptr);
  float* weight_ptr_tmp = static_cast<float*>(weight_ptr);
  float* out_ptr_tmp = static_cast<float*>(out_ptr);
  float* bias_ptr_tmp = static_cast<float*>(bias_ptr);
    memset(out_ptr_tmp, 0.0f, NO * HO * WO * CO * sizeof(float));
    int n_ = HI * WI * CI;
    int h_ = WI * CI;
    int co_ = KH * KW * CI;
    int kh_ = KW * CI;
    int on_ = HO * WO * CO;
    int oh_ = WO * CO;
    // Transposed Convolution
    for (int n = 0; n < NI; n++) { // batch
        for (int c = 0; c < CI; c++) { // input_channel
            for (int h = 0; h < HI; h++) { // input_height
                for (int w = 0; w < WI; w++) { // input_width
                    for (int kh = 0; kh < KH; kh++) { // kernel_height
                        for (int kw = 0; kw < KW; kw++) { // kernel_width
                            int oh = h * SH - PH + kh * DH;
                            int ow = w * SW - PW + kw * DW;
                            if (oh >= 0 && oh < HO && ow >= 0 && ow < WO) {
                                for (int co = 0; co < CO; co++) {
                                    // vfloat32m1_t in0 = __riscv_vle32_v_f32m1(weight_ptr + co * (co_) + kh * (kh_) + kw * CI + c, 8);
                                    int in_idx = n * (n_) + h * (h_) + w * CI + c;
                                    int weight_idx = co * (co_) + kh * (kh_) + kw * CI + c;
                                    int out_idx = n * (on_) + oh * (oh_) + ow * CO + co;
                                    out_ptr_tmp[out_idx] += in_ptr_tmp[in_idx] * weight_ptr_tmp[weight_idx];
                                    if(n == 0)
                                        printf("in_idx = %d, weight_idx = %d, out_idx = %d\n", in_idx, weight_idx, out_idx);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Add bias and apply ReLU
    for (int n = 0; n < NO; n++) {
        for (int h = 0; h < HO; h++) {
            for (int w = 0; w < WO; w++) {
                for (int c = 0; c < CO; c++) {
                    int out_idx = n * (HO * WO * CO) + h * (WO * CO) + w * CO + c;
                    out_ptr_tmp[out_idx] += bias_ptr_tmp[c];
                    if (out_ptr_tmp[out_idx] < 0) {
                        out_ptr_tmp[out_idx] = 0;
                    }
                }
            }
        }
    }
}

// int main() {
//     // Define tensor shapes
//     int64_t batch = 4;
//     int64_t in_ch = 32;
//     int64_t in_h = 14;
//     int64_t in_w = 14;
//     int64_t out_ch = 32;
//     int64_t kernel_h = 2;
//     int64_t kernel_w = 2;
//     int64_t out_batch;
//     int64_t out_h;
//     int64_t out_w;
//     int strideh = 2;
//     int stridew = 2;
//     int dilationh = 1;
//     int dilationw = 1;
//     int padh = 0;
//     int padw = 0;

//     // Calculate output dimensions
//     int64_t KHEff = (kernel_h - 1) * dilationh + 1;
//     int64_t KWEff = (kernel_w - 1) * dilationw + 1;
//     out_batch = batch;
//     out_h = (in_h - 1) * strideh - 2 * padh + KHEff;
//     out_w = (in_w - 1) * stridew - 2 * padw + KWEff;

//     // Allocate memory for input, weight, bias, and output
//     float* input = (float*)malloc(batch * in_h * in_w * in_ch * sizeof(float));
//     float* weight = (float*)malloc(out_ch * kernel_h * kernel_w * in_ch * sizeof(float));
//     float* bias = (float*)malloc(out_ch * sizeof(float));
//     float* output = (float*)malloc(batch * out_h * out_w * out_ch * sizeof(float));

//     // Initialize input, weight, and bias with some values (for example purposes)
//     for (int i = 0; i < batch * in_h * in_w * in_ch; ++i) {
//         input[i] = 1.0f; // Example value
//     }
//     for (int i = 0; i < out_ch * kernel_h * kernel_w * in_ch; ++i) {
//         weight[i] = 1.0f; // Example value
//     }
//     for (int i = 0; i < out_ch; ++i) {
//         bias[i] = 0.5f; // Example value
//     }

//     // Call the function
//     transposed_conv2d_bias_relu_3(input, weight, output, bias, nullptr, &batch, &out_ch, &in_ch, &kernel_h, &kernel_w, &in_h, &in_w, &out_batch, &out_h, &out_w, strideh, dilationh, padh, stridew, dilationw, padw);

//     // Example to print part of the output tensor
//     for (int n = 0; n < batch; ++n) {
//         for (int h = 0; h < out_h; ++h) {
//             for (int w = 0; w < out_w; ++w) {
//                 for (int c = 0; c < out_ch; ++c) {
//                     int out_idx = n * (out_h * out_w * out_ch) + h * (out_w * out_ch) + w * out_ch + c;
//                     printf("output[%d][%d][%d][%d] = %f\n", n, h, w, c, output[out_idx]);
//                     if (h == 1 && w == 1 && c == 1) break; // Limit the output for readability
//                 }
//                 if (h == 1 && w == 1) break; // Limit the output for readability
//             }
//             if (h == 1) break; // Limit the output for readability
//         }
//         if (n == 0) break; // Limit the output for readability
//     }

//     // Free allocated memory
//     free(input);
//     free(weight);
//     free(bias);
//     free(output);

//     return 0;
// }
