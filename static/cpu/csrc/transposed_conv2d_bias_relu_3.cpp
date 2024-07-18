
#include <cstdio>
#include <stdexcept>
#include <iostream>

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
    std::cout << "testing transposed_conv2d_bias_relu_3\n";
    return;
}