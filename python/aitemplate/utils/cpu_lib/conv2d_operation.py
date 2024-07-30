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


class Conv2DSpecialization(enum.Enum):
    # ConvNchwF32 = auto()
    # ConvNchwF16 = auto()
    ConvNhwcF32 = auto()
    ConvNhwcF16 = auto()
    ConvNhwcQd8F32Qc8w = auto()
    ConvNhwcQd8F16Qc8w = auto()
    ConvNhwcQc8 = auto()
    ConvNhwcQs8 = auto()
    ConvNhwcQu8 = auto()


Conv2DSpecializationTag = {
    # the code format for ConvNchw is different from the one of ConvNhwc
    # TODO : handle it independently
    # Conv2DSpecialization.ConvNchwF32: "convolution2d_nchw_f32",
    # Conv2DSpecialization.ConvNchwF16: "convolution2d_nchw_f16",
    Conv2DSpecialization.ConvNhwcF32: "convolution2d_nhwc_f32",
    Conv2DSpecialization.ConvNhwcF16: "convolution2d_nhwc_f16",
    Conv2DSpecialization.ConvNhwcQd8F32Qc8w: "convolution2d_nhwc_qd8_f32_qc8w",
    Conv2DSpecialization.ConvNhwcQd8F16Qc8w: "convolution2d_nhwc_qd8_f16_qc8w",
    Conv2DSpecialization.ConvNhwcQc8: "convolution2d_nhwc_qc8",
    Conv2DSpecialization.ConvNhwcQs8: "convolution2d_nhwc_qs8",
    Conv2DSpecialization.ConvNhwcQu8: "convolution2d_nhwc_qu8",
}

@dataclass
class Conv2DOperation:
    operation_kind: library.Conv2dKind
    extra_kind: library.TensorOperation
    A: library.TensorDesc
    B: library.TensorDesc
    C: library.TensorDesc
    a_elem_op: library.TensorOperation
    b_elem_op: library.TensorOperation
    epilogue_functor: library.TensorOperation
    conv2d_specialization: Conv2DSpecialization

    def __str__(self) -> str:
        io_name = "{conv2d_kind}_{conv2d_specialization}_{a_dtype}_{b_dtype}_{c_dtype}".format(
            conv2d_kind=library.Conv2dKindNames[self.operation_kind],
            conv2d_specialization=self.conv2d_specialization.value,
            a_dtype=library.DataTypeNames[self.A.element],
            b_dtype=library.DataTypeNames[self.B.element],
            c_dtype=library.DataTypeNames[self.C.element],
        )
        return "{io_name}".format(
            io_name=io_name
        )

    def accumulator_type(self):
        return library.DataType.f32

    def emit(self) -> str:
        template = jinja2.Template(
            """
{{indent}}//{{name}}
{{indent}}xnn_operator_t op_conv = nullptr;
{{indent}}const xnn_status status = xnn_create_{{Conv2DSpecialization}}(
{{indent}}  PH, PW, PH, PW, i32_kernel_h, i32_kernel_w,
{{indent}}  SH, SW, DH, DW, 1, CI,
{{indent}}  CO, 1 * CI, 1 * CO, (float*)(weight_ptr), (float*)(bias_ptr),
{{indent}}  -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
{{indent}}  /*flags=*/0, nullptr, nullptr, &op_conv);
{{indent}}std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op_conv(op_conv, xnn_delete_operator);
{{indent}}CHECK_EQ(status, xnn_status_success);
{{indent}}CHECK_NE(op_conv, nullptr);
{{indent}}size_t workspace_size = SIZE_MAX;
{{indent}}size_t workspace_alignment = SIZE_MAX;
{{indent}}CHECK_EQ(
{{indent}}  xnn_reshape_{{Conv2DSpecialization}}(
{{indent}}    op_conv, i32_batch, i32_in_h, i32_in_w,
{{indent}}    &workspace_size, &workspace_alignment,
{{indent}}    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
{{indent}}    /*threadpool=*/nullptr), xnn_status_success);
{{indent}}CHECK_EQ(workspace_size, 0);
{{indent}}CHECK_EQ(workspace_alignment, 1);
{{indent}}CHECK_EQ(xnn_setup_{{Conv2DSpecialization}}(
{{indent}}    op_conv, 
{{indent}}    /*workspace=*/nullptr, 
{{indent}}    (float*)(in_ptr), 
{{indent}}    (float*)(out_ptr)), xnn_status_success);
{{indent}}CHECK_EQ(xnn_run_operator(op_conv, /*threadpool=*/nullptr), xnn_status_success);
            """
        )
        # 
        binary_func_minmax_flag_op = jinja2.Template(
            """
{{indent}}xnn_operator_t binary_func_minmax_flag_op = nullptr;
{{indent}}CHECK_EQ(xnn_status_success, xnn_create_{{epilogue_functor}}_{{DataType}}(0, std::numeric_limits<{{DataName}}>::infinity(), 0, &binary_func_minmax_flag_op));
{{indent}}std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_binary_func_minmax_flag_op(binary_func_minmax_flag_op, xnn_delete_operator);
{{indent}}const size_t a_shape[] = { (size_t)i32_out_batch, (size_t)i32_out_h, (size_t)i32_out_w, (size_t)i32_out_ch};
{{indent}}const size_t b_shape[] = { (size_t)i32_out_batch, (size_t)i32_out_h, (size_t)i32_out_w, (size_t)i32_out_ch};
{{indent}}CHECK_EQ(
{{indent}}xnn_status_success, xnn_reshape_{{epilogue_functor}}_{{DataType}}(
{{indent}}                        binary_func_minmax_flag_op, 4, a_shape, 4, b_shape,
{{indent}}                        /*threadpool=*/nullptr));
{{indent}}CHECK_EQ(
{{indent}}  xnn_status_success, xnn_setup_{{epilogue_functor}}_{{DataType}}(binary_func_minmax_flag_op, ({{DataName}}*)(res_ptr), ({{DataName}}*)(out_ptr), ({{DataName}}*)(out_ptr)));
{{indent}}CHECK_EQ(xnn_status_success, xnn_run_operator(binary_func_minmax_flag_op, /*threadpool=*/nullptr));
            """
        )
        code_snippet = jinja2.Template(
            """
{{conv2d}}

{{epilogue_functor}}
            """
        )
        conv2d = template.render(
            indent="  ",
            name=self.__str__(),
            ADType=library.DataTypeTag[self.A.element],
            BDType=library.DataTypeTag[self.B.element],
            CDType=library.DataTypeTag[self.C.element],
            AccDType=library.DataTypeTag[library.DataType.f32],
            CShuffleDType=library.DataTypeTag[self.C.element],
            epilogue_functor=library.TensorOperationTag[self.epilogue_functor],
            Conv2DSpecialization=Conv2DSpecializationTag[self.conv2d_specialization],
        )
        functor = ""
        # add, mul, div, sub without quantization
        if self.epilogue_functor == library.TensorOperation.Add or \
            self.epilogue_functor == library.TensorOperation.Mul or \
            self.epilogue_functor == library.TensorOperation.Div or \
            self.epilogue_functor == library.TensorOperation.Sub :
            functor = binary_func_minmax_flag_op.render(
                indent="  ",
                epilogue_functor = library.TensorOperationTag[self.epilogue_functor],
                DataType = library.DataTypeNames[self.A.element],
                DataName = library.DataTypeTag[self.A.element],
            )
        program = code_snippet.render(
            conv2d = conv2d,
            epilogue_functor = functor
        )
        return program


if __name__ == "__main__":
    A = library.TensorDesc(library.DataType.f32, library.LayoutType.NHWC)
    B = library.TensorDesc(library.DataType.f32, library.LayoutType.NHWC)
    C = library.TensorDesc(library.DataType.f32, library.LayoutType.NHWC)
    Conv2DOp = Conv2DOperation(
        operation_kind=library.Conv2dKind.Conv2d,
        extra_kind=library.TensorOperation.PassThrough,
        A=A,
        B=B,
        C=C,
        a_elem_op=library.TensorOperation.PassThrough,
        b_elem_op=library.TensorOperation.PassThrough,
        epilogue_functor=library.TensorOperation.Add,
        conv2d_specialization=Conv2DSpecialization.ConvNhwcF32, # must match the dataType of A, B, C
    )
    print(str(Conv2DOp))
    print(Conv2DOp.emit())
