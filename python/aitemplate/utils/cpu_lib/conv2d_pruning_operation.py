#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
import enum
from copy import deepcopy
from dataclasses import dataclass
from enum import auto
from typing import List

import jinja2

# import library
from aitemplate.utils.cpu_lib import library
from aitemplate.utils.cpu_lib.conv2d_template import(
    template, 
    template_depthwise, 
    code_snippet, 
    binary_func_minmax_flag_op, 
    binary_func_flag_op, 
    transpose_func, 
    template_with_pruning,
    microkernel_lambda_func,
    microkernel_computation,
    template_choose_merge_im2col_packing_setting_x1v,
    template_choose_merge_im2col_packing_setting_x2v,
    template_choose_merge_im2col_packing_setting_x4v,
    template_choose_merge_im2col_packing_setting_x8v,
)
from aitemplate.utils.cpu_lib.conv2d_common import Conv2DSpecialization, Conv2DSpecializationTag, \
                        BIAS_KINDS, DEPTHWISE_KINDS, RELU_KINDS, RELU6_KINDS, BINARY_OP_KIND, BINARY_FLAG_OP_KIND, \
                        NHWC_KINDS, CNHW_KINDS, TRANSPOSE_AFTER_CONV_KINDS, PRUNING_KINDS



@dataclass
class Conv2D_Pruning_Operation:
    operation_kind: library.Conv2dPruningKind
    extra_kind: library.TensorOperation
    A: library.TensorDesc
    B: library.TensorDesc
    C: library.TensorDesc
    a_elem_op: library.TensorOperation
    b_elem_op: library.TensorOperation
    epilogue_functor: library.TensorOperation
    conv2d_specialization: Conv2DSpecialization
    LMUL: int
    tile_size: int

    def __str__(self) -> str:
        io_name = "{conv2d_kind}_{conv2d_specialization}_{a_dtype}_{tile_size}x{LMUL}v".format(
            conv2d_kind=library.Conv2dPruningKindNames[self.operation_kind],
            conv2d_specialization=self.conv2d_specialization.value,
            a_dtype=library.DataTypeNames[self.A.element],
            b_dtype=library.DataTypeNames[self.B.element],
            c_dtype=library.DataTypeNames[self.C.element],
            tile_size = self.tile_size,
            LMUL = self.LMUL,
        )
        return "{io_name}".format(
            io_name=io_name
        )

    def accumulator_type(self):
        return library.DataType.f32
    def emit(self) -> str:
        is_cnhw = self.conv2d_specialization in CNHW_KINDS
        if is_cnhw == False:
            raise RuntimeError("sparse convolution only supports cnhw layout")
        if (library.LayoutTag[self.A.layout] != "cnhw" or library.LayoutTag[self.B.layout] != "cnhw" or library.LayoutTag[self.C.layout] != "cnhw"):
            raise RuntimeError(
                f"data layout is not supported, with data layout of A is {library.LayoutTag[self.A.layout]}, B is {library.LayoutTag[self.B.layout]}, C is {library.LayoutTag[self.C.layout]}"
            )
        if (self.A.element != library.DataType.f32 or self.B.element != library.DataType.f32 or self.C.element != library.DataType.f32):
            raise RuntimeError(
                f"data type is not supported, with data type of A is {library.LayoutTag[self.A.element]}, B is {library.LayoutTag[self.B.element]}, C is {library.LayoutTag[self.C.element]}"
            )
        def generate_binary_op(operation_kind, operation_type, element, template_kind):
            return template_kind.render(
                indent="  ",
                is_relu = (self.operation_kind in RELU_KINDS),
                is_relu6 = (self.operation_kind in RELU6_KINDS),
                operation = library.TensorOperationTag[operation_type],
                DataType = library.DataTypeNames[element],
                DataName = library.DataTypeTag[element],
            )
        def generate_binary_func_flag_op(operation_kind, operation_type, element, template_kind):
            return template_kind.render(
                indent="  ",
                operation = library.TensorOperationTag[operation_type],
                DataType = library.DataTypeNames[element],
                DataName = library.DataTypeTag[element],
            )
        def generate_tensorOP(operation_kind, operation_type, element):
            code_gen = ""
            if operation_type == library.TensorOperation.PassThrough:
                code_gen = ""
            # add, mul, div, sub without quantization
            elif (operation_type in BINARY_OP_KIND) :
                code_gen = generate_binary_op(operation_kind, operation_type, element, binary_func_minmax_flag_op)
            elif (operation_type == library.TensorOperation.Copysign and library.DataTypeNames[element] == 'f32') or \
                 (operation_type in BINARY_FLAG_OP_KIND):
                code_gen = generate_binary_func_flag_op(operation_kind, operation_type, element, binary_func_flag_op)
            elif (operation_type == library.TensorOperation.Transpose):
                if library.DataTypeNames[element] == 'f32':
                    DataType = "x32"
                elif library.DataTypeNames[element] == 'f16':
                    DataType = "x16"
                code_gen = transpose_func.render(
                indent="  ",
                operation = library.TensorOperationTag[operation_type],
                DataType = DataType,
                DataName = library.DataTypeTag[element],
            )
            return code_gen
        is_bias      = self.operation_kind in BIAS_KINDS
        if self.operation_kind in PRUNING_KINDS:
            if (self.LMUL, self.tile_size) == (1, 7) or (self.LMUL, self.tile_size) == (2, 8) or (self.LMUL, self.tile_size) == (4, 7) or (self.LMUL, self.tile_size) == (8, 3):
                conv2d = template_with_pruning.render(
                    indent="  ",
                    name=self.__str__(),
                    DataName = library.DataTypeTag[self.A.element],
                    is_relu = (self.operation_kind in RELU_KINDS),
                    is_relu6 = (self.operation_kind in RELU6_KINDS),
                    ADType=library.DataTypeTag[self.A.element],
                    BDType=library.DataTypeTag[self.B.element],
                    CDType=library.DataTypeTag[self.C.element],
                    AccDType=library.DataTypeTag[library.DataType.f32],
                    CShuffleDType=library.DataTypeTag[self.C.element],
                    # epilogue_functor=library.TensorOperationTag[self.epilogue_functor],
                    Conv2DSpecialization=Conv2DSpecializationTag[self.conv2d_specialization],
                    LMUL=self.LMUL
                )
                extra_kind_code = generate_tensorOP(self.operation_kind, self.extra_kind, self.A.element)
                program = code_snippet.render(
                    is_bias = is_bias,
                    DataName = library.DataTypeTag[self.A.element],
                    conv2d = conv2d,
                    extra_kind = extra_kind_code,
                )
            else:
                _TEMPLATES_BY_LMUL = {
                    1: template_choose_merge_im2col_packing_setting_x1v,
                    2: template_choose_merge_im2col_packing_setting_x2v,
                    4: template_choose_merge_im2col_packing_setting_x4v,
                    8: template_choose_merge_im2col_packing_setting_x8v,
                }
                _LOG_NR_LMUL = {
                    1: "const int log_nr = 3;",
                    2: "const int log_nr = 4;",
                    4: "const int log_nr = 5;",
                    8: "const int log_nr = 6;",
                }
                try:
                    merge_im2col_packing = _TEMPLATES_BY_LMUL[self.LMUL].render(indent="  ")
                except KeyError:                             # optional, but nice to have
                    raise ValueError(f"Unsupported LMUL value: {self.LMUL}")
                if self.operation_kind in RELU_KINDS:
                    ACTIVATION = "RELU"
                    output_min = "0"
                    output_max = "std::numeric_limits<float>::infinity()"
                elif self.operation_kind in RELU6_KINDS:
                    ACTIVATION = "MINMAX"
                    output_min = "0"
                    output_max = "6"
                else:
                    ACTIVATION = "LINEAR"
                    output_min = "-std::numeric_limits<float>::infinity()"
                    output_max = "std::numeric_limits<float>::infinity()"
                log_nr = _LOG_NR_LMUL[self.LMUL]
                microkernel_func = microkernel_lambda_func.render(
                    ACTIVATION = ACTIVATION,
                    MR = self.tile_size,
                    LMUL = self.LMUL,
                    log_nr = log_nr
                )
                extra_kind_code = generate_tensorOP(self.operation_kind, self.extra_kind, self.A.element)
                program = microkernel_computation.render(
                    MR = self.tile_size,
                    LMUL = self.LMUL,
                    microkernel_lambda_func = microkernel_func,
                    merge_im2col_packing = merge_im2col_packing,
                    indent="  ",
                    extra_kind_code=extra_kind_code,
                    output_min=output_min,
                    output_max=output_max,
                )
        else:
            raise RuntimeError("only operation with pruning is supported")
        return program


if __name__ == "__main__":
    A = library.TensorDesc(library.DataType.f32, library.LayoutType.CNHW)
    B = library.TensorDesc(library.DataType.f32, library.LayoutType.CNHW)
    C = library.TensorDesc(library.DataType.f32, library.LayoutType.CNHW)
    Conv2DOp = Conv2D_Pruning_Operation(
        operation_kind=library.Conv2dPruningKind.Conv2dPruningBiasAdd,
        extra_kind=library.TensorOperation.PassThrough,
        A=A,
        B=B,
        C=C,
        a_elem_op=library.TensorOperation.PassThrough,
        b_elem_op=library.TensorOperation.PassThrough,
        epilogue_functor=library.TensorOperation.PassThrough,
        conv2d_specialization=Conv2DSpecialization.ConvCNHWF32, # must match the dataType of A, B, C
        LMUL=2,
        tile_size=7
    )
    output = jinja2.Template(
        """
#include <cstdio>
#include <stdexcept>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>
#include <random>
#include "xnnpack.h"
#include "logging.h"

#include <functional>
#include <random>
#include <cstddef> // For size_t
#include <cstring>
{{indent}}void conv2d_cnhw_pruning_bias_45 (
{{indent}}    void* in_ptr,
{{indent}}    void* weight_ptr,
{{indent}}    void* out_ptr,

{{indent}}    void* bias_ptr,
{{indent}}    void* weight_indice_ptr,

{{indent}}    uint8_t* workspace,
{{indent}}    int64_t* batch,
{{indent}}    int64_t* out_ch,
{{indent}}    int64_t* in_ch,
{{indent}}    int64_t* kernel_h,
{{indent}}    int64_t* kernel_w,
{{indent}}    int64_t* in_h,
{{indent}}    int64_t* in_w,
{{indent}}    int64_t* out_batch,
{{indent}}    int64_t* out_h,
{{indent}}    int64_t* out_w,
{{indent}}    int strideh,
{{indent}}    int dilationh,
{{indent}}    int padh,
{{indent}}    int stridew,
{{indent}}    int dilationw,
{{indent}}    int padw,
{{indent}}    float pruning_ratio,
{{indent}}    pthreadpool* pthreadpool_
{{indent}}    ){

  
{{indent}}  int64_t NI = *batch;
{{indent}}  int64_t HI = *in_h;
{{indent}}  int64_t WI = *in_w;
{{indent}}  int64_t CI = *in_ch;
{{indent}}  int64_t CO = *out_ch;
{{indent}}  int64_t KH = *kernel_h;
{{indent}}  int64_t KW = *kernel_w;
{{indent}}  int64_t SH = strideh;
{{indent}}  int64_t SW = stridew;
{{indent}}  int64_t DH = dilationh;
{{indent}}  int64_t DW = dilationw;
{{indent}}  int64_t PH = padh;
{{indent}}  int64_t PW = padw;
{{indent}}  int64_t KHEff = (KH - 1) * DH + 1;
{{indent}}  int64_t KWEff = (KW - 1) * DW + 1;
{{indent}}  int64_t NO = NI;
{{indent}}  int64_t HO = (HI + PH + PH - KHEff) / SH + 1;
{{indent}}  int64_t WO = (WI + PW + PW - KWEff) / SW + 1;

{{indent}}  int i32_batch = *batch;
{{indent}}  int i32_in_h = *in_h;
{{indent}}  int i32_in_w = *in_w;
{{indent}}  int i32_in_ch = *in_ch;
{{indent}}  int i32_out_ch = *out_ch;
{{indent}}  int i32_kernel_h = *kernel_h;
{{indent}}  int i32_kernel_w = *kernel_w;
{{indent}}  int i32_out_batch = *out_batch;
{{indent}}  int i32_out_h = *out_h;
{{indent}}  int i32_out_w = *out_w;
{{code}}
}
"""
    )
    # print(str(Conv2DOp))
    output_code = output.render(
        indent = "  ",
        code = Conv2DOp.emit()
    )
    print(output_code)
