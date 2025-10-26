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
from aitemplate.utils.cpu_lib import library
from aitemplate.utils.cpu_lib.conv2d_common import BINARY_OP_KIND, RELU_KINDS, RELU6_KINDS
# import library
# TODO : flag should be enabled for transposed weight
template = jinja2.Template(
"""
{{indent}}//{{name}}
{{indent}}xnn_operator_t gemm_op = nullptr;
{{indent}}const xnn_status status = xnn_create_{{GemmSpecialization}}(
{{indent}}    K, N, K, N, 
{{indent}}    ({{DataName}}*)(b_ptr), ({{DataName}}*)(bias_ptr), 
{{indent}}    -std::numeric_limits<{{DataName}}>::infinity(), std::numeric_limits<{{DataName}}>::infinity(),
{{indent}}    /*flags=*/0, nullptr, nullptr, &gemm_op);
{{indent}}  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op(gemm_op, xnn_delete_operator);
{{indent}}  CHECK_EQ(xnn_status_success, status);
{{indent}}  CHECK_NE(nullptr, gemm_op);
{{indent}}  CHECK_EQ(xnn_status_success, xnn_reshape_{{GemmSpecialization}}(gemm_op, M, /*threadpool=*/pthreadpool_));
{{indent}}  CHECK_EQ(xnn_status_success, xnn_setup_{{GemmSpecialization}}(gemm_op, ({{DataName}}*)(a_ptr), ({{DataName}}*)(c_ptr)));
{{indent}}  CHECK_EQ(xnn_status_success, xnn_run_operator(gemm_op, /*threadpool=*/pthreadpool_));

"""
)

binary_func_minmax_flag_op = jinja2.Template(
"""
{{indent}}xnn_operator_t binary_func_minmax_flag_op = nullptr;
{% if is_relu %}
{{indent}}CHECK_EQ(xnn_status_success, xnn_create_{{operation}}_{{DataType}}(0, std::numeric_limits<{{DataName}}>::infinity(), 0, &binary_func_minmax_flag_op));
{% elif is_relu6 %}
{{indent}}CHECK_EQ(xnn_status_success, xnn_create_{{operation}}_{{DataType}}(0, 6, 0, &binary_func_minmax_flag_op));
{% else %}
{{indent}}CHECK_EQ(xnn_status_success, xnn_create_{{operation}}_{{DataType}}(-std::numeric_limits<{{DataName}}>::infinity(), std::numeric_limits<{{DataName}}>::infinity(), 0, &binary_func_minmax_flag_op));
{% endif %}
{{indent}}std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_binary_func_minmax_flag_op(binary_func_minmax_flag_op, xnn_delete_operator);
{{indent}}const size_t a_shape[] = { (size_t)M, (size_t)N};
{{indent}}const size_t b_shape[] = { (size_t)M, (size_t)N};
{{indent}}CHECK_EQ(
{{indent}}xnn_status_success, xnn_reshape_{{operation}}_{{DataType}}(
{{indent}}                        binary_func_minmax_flag_op, 2, a_shape, 2, b_shape,
{{indent}}                        /*threadpool=*/pthreadpool_));
{{indent}}CHECK_EQ(
{{indent}}  xnn_status_success, xnn_setup_{{operation}}_{{DataType}}(binary_func_minmax_flag_op, ({{DataName}}*)(d0_ptr), ({{DataName}}*)(c_ptr), ({{DataName}}*)(c_ptr)));
{{indent}}CHECK_EQ(xnn_status_success, xnn_run_operator(binary_func_minmax_flag_op, /*threadpool=*/pthreadpool_));
"""
)

code_snippet = jinja2.Template(
"""
{% if not is_bias %}
{{indent}}void* bias_ptr = ({{DataName}}*)malloc(N * sizeof({{DataName}}));
{{indent}}std::memset(bias_ptr, 0, N * sizeof({{DataName}}));
{% endif %}
{{gemm}}
{{extra_kind}}
{% if not is_bias %}
{{indent}}free(bias_ptr);
{% endif %}
"""
        )
class GemmSpecialization(enum.Enum):
    # fully-connected-nc.c
    GemmRRR_f16 = auto()
    GemmRRR_f32 = auto()
    GemmRRR_qd8_f16_qc4w = auto()
    GemmRRR_qd8_f32_qc4w = auto()
    GemmRRR_qd8_f16_qc8w = auto()
    GemmRRR_qd8_f32_qc8w = auto()
    GemmRRR_qp8_f32_qc4w = auto()
    GemmRCR_f16 = auto()
    GemmRCR_f32 = auto()
    GemmRCR_qd8_f16_qc4w = auto()
    GemmRCR_qd8_f32_qc4w = auto()
    GemmRCR_qd8_f16_qc8w = auto()
    GemmRCR_qd8_f32_qc8w = auto()
    GemmRCR_qp8_f32_qc4w = auto()


GemmSpecializationTag = {
    GemmSpecialization.GemmRRR_f16: "fully_connected_nc_f16",
    GemmSpecialization.GemmRRR_f32: "fully_connected_nc_f32",
    GemmSpecialization.GemmRRR_qd8_f16_qc4w: "fully_connected_nc_qd8_f16_qc4w",
    GemmSpecialization.GemmRRR_qd8_f32_qc4w: "fully_connected_nc_qd8_f32_qc4w",
    GemmSpecialization.GemmRRR_qd8_f16_qc8w: "fully_connected_nc_qd8_f16_qc8w",
    GemmSpecialization.GemmRRR_qd8_f32_qc8w: "fully_connected_nc_qd8_f32_qc8w",
    GemmSpecialization.GemmRRR_qp8_f32_qc4w: "fully_connected_nc_qp8_f32_qc4w",
    GemmSpecialization.GemmRCR_f16: "fully_connected_nc_f16",
    GemmSpecialization.GemmRCR_f32: "fully_connected_nc_f32",
    GemmSpecialization.GemmRCR_qd8_f16_qc4w: "fully_connected_nc_qd8_f16_qc4w",
    GemmSpecialization.GemmRCR_qd8_f32_qc4w: "fully_connected_nc_qd8_f32_qc4w",
    GemmSpecialization.GemmRCR_qd8_f16_qc8w: "fully_connected_nc_qd8_f16_qc8w",
    GemmSpecialization.GemmRCR_qd8_f32_qc8w: "fully_connected_nc_qd8_f32_qc8w",
    GemmSpecialization.GemmRCR_qp8_f32_qc4w: "fully_connected_nc_qp8_f32_qc4w",
}



@dataclass
class GemmOperation:
    operation_kind: library.OperationKind
    extra_kind: library.TensorOperation
    A: library.TensorDesc
    B: library.TensorDesc
    C: library.TensorDesc
    a_elem_op: library.TensorOperation
    b_elem_op: library.TensorOperation
    epilogue_functor: library.TensorOperation
    gemm_specialization: GemmSpecialization

    def __str__(self) -> str:
        io_name = "{gemm_kind}_{gemm_specialization}_{a_dtype}_{b_dtype}_{c_dtype}_{a_layout}_{b_layout}_{c_layout}".format(
            gemm_kind=library.GemmKindNames[self.operation_kind],
            gemm_specialization=self.gemm_specialization.value,
            a_dtype=library.DataTypeNames[self.A.element],
            b_dtype=library.DataTypeNames[self.B.element],
            c_dtype=library.DataTypeNames[self.C.element],
            a_layout=library.LayoutTag[self.A.layout],
            b_layout=library.LayoutTag[self.B.layout],
            c_layout=library.LayoutTag[self.C.layout],
        )

        return io_name

    def accumulator_type(self):
        return library.DataType.f32

    def emit(self) -> str:
        def generate_binary_op(operation_kind, operation_type, element, template_kind):
            return template_kind.render(
                indent="  ",
                is_relu = (self.operation_kind in RELU_KINDS),
                is_relu6 = (self.operation_kind in RELU6_KINDS),
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
            return code_gen
        gemm = template.render(
            name=self.__str__(),
            gemm_kind=library.GemmKindNames[self.operation_kind],
            DataName = library.DataTypeTag[self.A.element],
            GemmSpecialization=GemmSpecializationTag[self.gemm_specialization],
            epilogue_func=library.TensorOperationTag[self.epilogue_functor],
        )
        is_bias = False
        if self.operation_kind == library.GemmKind.GemmBias or self.operation_kind == library.GemmKind.GemmBiasAdd:
          is_bias = True
        extra_kind_code = generate_tensorOP(self.operation_kind, self.extra_kind, self.A.element)
        program = code_snippet.render(
            is_bias = is_bias,
            DataName = library.DataTypeTag[self.A.element],
            gemm = gemm,
            extra_kind=extra_kind_code
        )
        return program


if __name__ == "__main__":
    A = library.TensorDesc(library.DataType.f32, library.LayoutType.RowMajor)
    B = library.TensorDesc(library.DataType.f32, library.LayoutType.ColumnMajor)
    C = library.TensorDesc(library.DataType.f32, library.LayoutType.RowMajor)
    GemmOp = GemmOperation(
        operation_kind=library.GemmKind.GemmBias,
        extra_kind=library.TensorOperation.PassThrough,
        A=A,
        B=B,
        C=C,
        a_elem_op=library.TensorOperation.PassThrough,
        b_elem_op=library.TensorOperation.PassThrough,
        epilogue_functor=library.TensorOperation.PassThrough,
        gemm_specialization=GemmSpecialization.GemmRCR_f32,
    )
    print(str(GemmOp))
    print(GemmOp.emit())
