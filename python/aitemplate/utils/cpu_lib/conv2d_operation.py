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
from aitemplate.utils.cpu_lib.conv2d_nhwc_template import template, template_depthwise, code_snippet, binary_func_minmax_flag_op, binary_func_flag_op
from aitemplate.utils.cpu_lib.conv2d_common import Conv2DSpecialization, Conv2DSpecializationTag, \
                        BIAS_KINDS, DEPTHWISE_KINDS, RELU_KINDS, RELU6_KINDS, BINARY_OP_KIND, BINARY_FLAG_OP_KIND, \
                        NHWC_KINDS, CNHW_KINDS

# TODO : revise min/max for relu


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
        is_nhwc = self.conv2d_specialization in NHWC_KINDS
        is_cnhw = self.conv2d_specialization in CNHW_KINDS
        if (is_nhwc and (library.LayoutTag[self.A.layout] != "nhwc" or library.LayoutTag[self.B.layout] != "nhwc" or library.LayoutTag[self.C.layout] != "nhwc")) or \
            (is_cnhw and (library.LayoutTag[self.A.layout] != "cnhw" or library.LayoutTag[self.B.layout] != "cnhw" or library.LayoutTag[self.C.layout] != "cnhw")):
            raise RuntimeError(
                f"data type mismatch, with data type of A is {library.LayoutTag[self.A.layout]}, B is {library.LayoutTag[self.B.layout]}, C is {library.LayoutTag[self.C.layout]} and is_nhwc = {is_nhwc}, is_cnhw = {is_cnhw}"
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
            return code_gen
        def generate_conv2d(template_kind):
            return template_kind.render(
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
                epilogue_functor=library.TensorOperationTag[self.epilogue_functor],
                Conv2DSpecialization=Conv2DSpecializationTag[self.conv2d_specialization],
            )
        # `is_bias` is handled in `code_snippet template`, if !is_bias, create a dummy bias
        is_bias      = self.operation_kind in BIAS_KINDS
        is_depthwise = self.operation_kind in DEPTHWISE_KINDS
        if is_depthwise:
            conv2d = generate_conv2d(template_depthwise)
        else:
            conv2d = generate_conv2d(template)
        extra_kind_code = generate_tensorOP(self.operation_kind, self.extra_kind, self.A.element)
        program = code_snippet.render(
            is_bias = is_bias,
            DataName = library.DataTypeTag[self.A.element],
            conv2d = conv2d,
            extra_kind = extra_kind_code,
        )
        return program


if __name__ == "__main__":
    A = library.TensorDesc(library.DataType.f32, library.LayoutType.NHWC)
    B = library.TensorDesc(library.DataType.f32, library.LayoutType.NHWC)
    C = library.TensorDesc(library.DataType.f32, library.LayoutType.NHWC)
    Conv2DOp = Conv2DOperation(
        operation_kind=library.Conv2dKind.Conv2d,
        extra_kind=library.TensorOperation.Add,
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
