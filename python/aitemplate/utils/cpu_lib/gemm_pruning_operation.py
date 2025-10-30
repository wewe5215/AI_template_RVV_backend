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
from aitemplate.utils.cpu_lib.conv2d_common import BINARY_OP_KIND
from aitemplate.utils.cpu_lib.conv2d_template import(
    microkernel_lambda_func,
)
from aitemplate.utils.cpu_lib.gemm_operation import (
    GemmSpecialization, 
    BIAS_KINDS, 
    RELU_KINDS,
    binary_func_minmax_flag_op,
    gelu_op
)
PRUNING_KINDS = {
    library.GemmPruningKind.GemmPruning,
    library.GemmPruningKind.GemmPruningBias,
    library.GemmPruningKind.GemmPruningBiasAdd,
    library.GemmPruningKind.GemmPruningBiasAddRelu,
    library.GemmPruningKind.GemmPruningBiasGelu,
}

microkernel_computation = jinja2.Template(
    """
{% if not is_bias %}
{{indent}}void* bias_ptr = ({{DataName}}*)malloc(N * sizeof({{DataName}}));
{{indent}}std::memset(bias_ptr, 0, N * sizeof({{DataName}}));
{% endif %}
{{indent}}const size_t num_threads = std::thread::hardware_concurrency();
{{indent}}uint32_t nr = __riscv_vsetvlmax_e32m{{LMUL}}();
{{indent}}uint32_t* input_packed_ptr = (uint32_t*)malloc(K * round_up(M, nr) << 2);
{{indent}}float* c_ptr_tmp = (float*)malloc(M * N << 2);
{{indent}}xnn_x32_pack_transpose_ukernel_x{{LMUL}}v__rvv_u8(
{{indent}}{{indent}}/*g=*/1, M, K,
{{indent}}{{indent}}nr, 1, 1, 
{{indent}}{{indent}}reinterpret_cast<uint32_t*>(a_ptr), 
{{indent}}{{indent}}/*scale=*/nullptr, 
{{indent}}{{indent}}reinterpret_cast<uint32_t*>(input_packed_ptr), 
{{indent}}{{indent}}/*extra_bytes=*/0, 
{{indent}}{{indent}}/*params=*/nullptr
{{indent}});
{{indent}}{{microkernel_lambda_func}}
{{indent}}const size_t im2col_row_cnt = K;
{{indent}}const size_t pruned_group_input_channels = (size_t)(im2col_row_cnt * (1.0f - pruning_ratio));
{{indent}}union xnn_f32_minmax_params gemm_params;
{{indent}}xnn_init_f32_minmax_scalar_params(&gemm_params, {{output_min}}, {{output_max}});
{{indent}}struct function_context context = (struct function_context){
{{indent}}{{indent}}.input = (float*)(a_ptr),
{{indent}}{{indent}}.bias = (float*)(bias_ptr),
{{indent}}{{indent}}.pruned_weight = (float*)(b_ptr),
{{indent}}{{indent}}.output = (float*)(c_ptr_tmp),
{{indent}}{{indent}}.input_channel = static_cast<size_t>(K),
{{indent}}{{indent}}.output_channel = static_cast<size_t>(N),
{{indent}}{{indent}}.mr = {{MR}},
{{indent}}{{indent}}.nr = nr,
{{indent}}{{indent}}.im2col_packing = input_packed_ptr,
{{indent}}{{indent}}.indice = (uint16_t*)(weight_indice_ptr),
{{indent}}{{indent}}.microkernel = (f32_gemm_input_T_N_M_pruning)xnn_f32_gemm_ukernel_{{MR}}x{{LMUL}}v_columnwise_pruned__rvv,
{{indent}}{{indent}}.a_stride  = (im2col_row_cnt << 2),
{{indent}}{{indent}}.cm_stride = static_cast<size_t>(M << 2),
{{indent}}{{indent}}.cn_stride = (nr << 2),
{{indent}}{{indent}}.k_scaled = (pruned_group_input_channels << 2),
{{indent}}{{indent}}.w_stride = (pruned_group_input_channels << 2),// bias + transposed weight[out_ch][in_ch]
{{indent}}{{indent}}.params = static_cast<void*>(&gemm_params),
{{indent}}};
{{indent}}const size_t num_other_tiles = 1 * divide_round_up(N, {{MR}});
{{indent}}const size_t target_tiles_per_thread = 5;
{{indent}}const size_t max_nc = divide_round_up(N * num_other_tiles, num_threads * target_tiles_per_thread);
{{indent}}size_t nc = M;
{{indent}}if (max_nc < nc) {
{{indent}}  nc = min(nc, divide_round_up(nc, max_nc * nr) * nr);
{{indent}}}
{{indent}}pthreadpool_parallelize_2d_tile_2d(
{{indent}}    pthreadpool_,
{{indent}}    (pthreadpool_task_2d_tile_2d_t)conv2d_columnwise_pruning_vector,
{{indent}}    (void*) ((uintptr_t) &context),
{{indent}}    N, M,
{{indent}}    {{MR}}, nc,
{{indent}}    0x00000001);
xnn_operator_t transpose_op = nullptr;
std::vector<size_t> shape = { (size_t)N, (size_t)M};
std::vector<size_t> perm = {1, 0};
CHECK_EQ(xnn_status_success, xnn_create_transpose_nd_x32(0, &transpose_op));
CHECK_NE(nullptr, transpose_op);
CHECK_EQ(
xnn_status_success, xnn_reshape_transpose_nd_x32(
transpose_op, shape.size(), shape.data(), perm.data(), pthreadpool_));
CHECK_EQ(
xnn_status_success, xnn_setup_transpose_nd_x32(transpose_op, c_ptr_tmp, c_ptr));
CHECK_EQ(xnn_status_success, xnn_run_operator(transpose_op, /*threadpool=*/pthreadpool_));
{{extra_kind_code}}
{{indent}}free(input_packed_ptr);
{{indent}}free(c_ptr_tmp);
{% if not is_bias %}
{{indent}}free(bias_ptr);
{% endif %}
""")
@dataclass
class Gemm_Pruning_Operation:
    operation_kind: library.OperationKind
    extra_kind: library.TensorOperation
    A: library.TensorDesc
    B: library.TensorDesc
    C: library.TensorDesc
    a_elem_op: library.TensorOperation
    b_elem_op: library.TensorOperation
    epilogue_functor: library.TensorOperation
    gemm_specialization: GemmSpecialization
    LMUL: int
    tile_size: int

    def __str__(self) -> str:
        io_name = "{gemm_kind}_{gemm_specialization}_{a_dtype}_{tile_size}x{LMUL}v".format(
            gemm_kind=library.GemmPruningKindNames[self.operation_kind],
            gemm_specialization=self.gemm_specialization.value,
            a_dtype=library.DataTypeNames[self.A.element],
            b_dtype=library.DataTypeNames[self.B.element],
            c_dtype=library.DataTypeNames[self.C.element],
            a_layout=library.LayoutTag[self.A.layout],
            b_layout=library.LayoutTag[self.B.layout],
            c_layout=library.LayoutTag[self.C.layout],
            tile_size = self.tile_size,
            LMUL = self.LMUL,
        )
        return "{io_name}".format(
            io_name=io_name
        )

    def accumulator_type(self):
        return library.DataType.f32
    def emit(self) -> str:
        def generate_binary_op(operation_kind, operation_type, element, template_kind):
            return template_kind.render(
                indent="  ",
                is_relu = (self.operation_kind in RELU_KINDS),
                is_relu6 = False,
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
        is_bias  = self.operation_kind in BIAS_KINDS
        if self.operation_kind in PRUNING_KINDS:
            _LOG_NR_LMUL = {
                1: "const int log_nr = 3;",
                2: "const int log_nr = 4;",
                4: "const int log_nr = 5;",
                8: "const int log_nr = 6;",
            }
            if self.operation_kind in RELU_KINDS:
                ACTIVATION = "RELU"
                output_min = "0"
                output_max = "std::numeric_limits<float>::infinity()"
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
            activation_code = gelu_op.render(
            DataName = library.DataTypeTag[self.A.element])
            if self.operation_kind == library.GemmKind.GemmBiasGelu:
                extra_kind_code += activation_code
            program = microkernel_computation.render(
                MR = self.tile_size,
                LMUL = self.LMUL,
                microkernel_lambda_func = microkernel_func,
                indent="  ",
                extra_kind_code=extra_kind_code,
                output_min=output_min,
                output_max=output_max,
                is_bias=is_bias,
                DataName = library.DataTypeTag[self.A.element],
            )
        else:
            raise RuntimeError("only operation with pruning is supported")
        return program


if __name__ == "__main__":
    A = library.TensorDesc(library.DataType.f32, library.LayoutType.CNHW)
    B = library.TensorDesc(library.DataType.f32, library.LayoutType.CNHW)
    C = library.TensorDesc(library.DataType.f32, library.LayoutType.CNHW)
    GemmOp = Gemm_Pruning_Operation(
        operation_kind=library.GemmKind.GemmBias,
        extra_kind=library.TensorOperation.PassThrough,
        A=A,
        B=B,
        C=C,
        a_elem_op=library.TensorOperation.PassThrough,
        b_elem_op=library.TensorOperation.PassThrough,
        epilogue_functor=library.TensorOperation.PassThrough,
        gemm_specialization=GemmSpecialization.GemmRCR_f32,
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
void gemm_rcr_bias_2 (
    void* a_ptr,
    void* b_ptr,
    void* bias_ptr,
    void* c_ptr,
    int64_t* a_dim0,
    int64_t* a_dim1,
    int64_t* b_dim0,
    int64_t* b_dim1,
    int64_t* c_dim0,
    int64_t* c_dim1,
    pthreadpool* pthreadpool_
){

{{code}}
}
"""
    )
    # print(str(GemmOp))
    output_code = output.render(
        indent = "  ",
        code = GemmOp.emit()
    )
    print(output_code)
