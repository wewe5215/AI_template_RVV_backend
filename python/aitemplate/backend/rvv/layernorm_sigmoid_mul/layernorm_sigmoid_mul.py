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
"""
LayerNorm_Sigmoid_Mul codegen for rvv.
"""

import os
from typing import Any, Dict

import jinja2

from aitemplate.backend import registry
from aitemplate.backend.backend_spec import RVVSpec
from aitemplate.backend.common import tensor_accessor_codegen
from aitemplate.backend.rvv.layernorm_sigmoid_mul import layernorm_common
from aitemplate.backend.target import Target

# pylint: disable=C0301

FUNC_TEMPLATE = jinja2.Template(
    """
#include "logging.h"
#include "xnnpack.h"
#include <assert.h>
#include <cmath>
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
{% if is_remote_compile %}
#include "rvv_utils.h"
#include <riscv_vector.h>
{% endif %}
#include <thread>
#include <pthreadpool.h>
template<typename T>
void layer_norm(
    T* input,
    T* output,
    const T* gamma,
    const T* beta,
    int m,
    int hidden_size,
    T epsilon = (T)1e-5,
    pthreadpool* pthreadpool_ = nullptr
) {
    for (int b = 0; b < m; ++b) {
        T* in_ptr  = input  + (size_t)b * hidden_size;
        T* out_ptr = output + (size_t)b * hidden_size;

        T mean = (T)0;
        for (int i = 0; i < hidden_size; ++i) {
            mean += in_ptr[i];
        }
        mean /= hidden_size;

        T var = (T)0;
        for (int i = 0; i < hidden_size; ++i) {
            T diff = in_ptr[i] - mean;
            var += diff * diff;
        }
        var /= hidden_size;

        T inv_std = (T)1.0 / std::sqrt(var + epsilon);
        for (int i = 0; i < hidden_size; ++i) {
            T normalized = (in_ptr[i] - mean) * inv_std;
            out_ptr[i] = normalized * gamma[i] + beta[i];
        }
    }
}
{% if is_remote_compile %}
template<>
void layer_norm<float>(
    float* input,
    float* output,
    const float* gamma,
    const float* beta,
    int m,
    int hidden_size,
    float epsilon,
    pthreadpool* pthreadpool_
) {
    uint32_t nr = __riscv_vsetvlmax_e32m8();
    uint32_t nr_byte = nr << 2;
    std::vector<float> kernel_packed(hidden_size * round_up(m, nr));
    std::vector<float> output_T(m * hidden_size);
    xnn_x32_pack_transpose_ukernel_x8v__rvv_u8(
        /*g=*/1, m, hidden_size,
        nr, 1, 1,
        reinterpret_cast<uint32_t*>(input),
        /*scale=*/nullptr,
        reinterpret_cast<uint32_t*>(kernel_packed.data()),
        /*extra_bytes=*/0,
        /*params=*/nullptr
    );
    float* zero = (float*)malloc(nr_byte);
    memset(zero, 0, nr_byte);
    float* input_cur = kernel_packed.data();
    float* output_T_cur = output_T.data();
    size_t input_increment = 3 * nr;
    size_t nc = m;
    do {
        size_t vl = nr;
        if UNLIKELY(nc < nr) {
            vl = __riscv_vsetvl_e32m8(nc);
        }
        nc -= vl;
        float* i0 = input_cur;
        {%- for M in range(1, 3) %}
        float* i{{ M }} = (float*) ((uintptr_t) i{{ M-1 }} + nr_byte);
        {%- endfor %}
        vfloat32m8_t acc_f32v = __riscv_vfmv_v_f_f32m8(0.f, nr);
        for (int r = hidden_size; r > 0; r -= 3) {
            if UNPREDICTABLE(r < 2) {
                i1 = zero;
            }
            if UNPREDICTABLE(r <= 2) {
                i2 = zero;
            }
            {%- for M in range(0, 3) %}
            vfloat32m8_t in{{ M }}_f32v = __riscv_vle32_v_f32m8(i{{ M }}, vl);
            acc_f32v = __riscv_vfadd_vv_f32m8(acc_f32v, in{{ M }}_f32v, vl);
            {%- endfor %}
            {%- for M in range(0, 3) %}
            i{{M}} += input_increment;
            {%- endfor %}
        }
        vfloat32m8_t mean = __riscv_vfdiv_vf_f32m8(acc_f32v, (float)hidden_size, vl);
        vfloat32m8_t var = __riscv_vfmv_v_f_f32m8(0.f, vl);
        vfloat32m8_t diff;
        i0 = input_cur;
        for (int i = 0; i < hidden_size; ++i) {
            vfloat32m8_t in0_f32v = __riscv_vle32_v_f32m8(i0, vl);
            diff = __riscv_vfsub_vv_f32m8(in0_f32v, mean, vl);
            var = __riscv_vfadd_vv_f32m8(var, __riscv_vfmul_vv_f32m8(diff, diff, vl), vl);
            i0 += nr;
        }
        var = __riscv_vfdiv_vf_f32m8(var, (float)hidden_size, vl);
        vfloat32m8_t one = __riscv_vfmv_v_f_f32m8(1.f, vl);
        vfloat32m8_t inv_std = __riscv_vfdiv_vv_f32m8(one, __riscv_vfsqrt_v_f32m8(__riscv_vfadd_vf_f32m8(var, epsilon, vl), vl), vl);
        i0 = input_cur;
        for (int i = 0; i < hidden_size; ++i) {
            vfloat32m8_t in0_f32v = __riscv_vle32_v_f32m8(i0, vl);
            vfloat32m8_t normalized = __riscv_vfmul_vv_f32m8(__riscv_vfsub_vv_f32m8(in0_f32v, mean, vl), inv_std, vl);
            vfloat32m8_t out = __riscv_vfadd_vf_f32m8(__riscv_vfmul_vf_f32m8(normalized, gamma[i], vl), beta[i], vl);
            __riscv_vse32_v_f32m8(output_T_cur + i * m, out, vl);
            i0 += nr;
        }
        input_cur = (float*) ((uintptr_t) input_cur + hidden_size * nr_byte);
        output_T_cur = (float*) ((uintptr_t) output_T_cur + nr_byte);
    } while (nc != 0);
    free(zero);
    xnn_operator_t transpose_op = nullptr;
    std::vector<size_t> shape = { (size_t)hidden_size, (size_t)m};
    std::vector<size_t> perm = {1, 0};
    CHECK_EQ(xnn_status_success, xnn_create_transpose_nd_x32(0, &transpose_op));
    CHECK_NE(nullptr, transpose_op);
    CHECK_EQ(
    xnn_status_success, xnn_reshape_transpose_nd_x32(
    transpose_op, shape.size(), shape.data(), perm.data(), pthreadpool_));
    CHECK_EQ(
    xnn_status_success, xnn_setup_transpose_nd_x32(transpose_op, output_T.data(), output));
    CHECK_EQ(xnn_status_success, xnn_run_operator(transpose_op, /*threadpool=*/pthreadpool_));
}
{% endif %}
{{func_signature}}
{
    float* in = static_cast<float*>(input);
    float* out = static_cast<float*>(output);
    layer_norm<float>(
      in, out,
      gamma, beta,
      m, n, eps, pthreadpool_
    );

    return;
}
    """
)

FUNC_SIGNATURE = jinja2.Template(
    """
void {{func_name}}(void* output,
                   void* input,
                   const float* gamma,
                   const float* beta,
                   int m,
                   int n,
                   const float eps,
                   pthreadpool* pthreadpool_)
    """
)

FUNC_DECL = jinja2.Template(
    """
    {{func_signature}};
    """
)

FUNC_CALL_TEMPLATE = jinja2.Template(
    """
{{indent}}{
{{indent}}  {{m_n_shape_func}}
{{indent}}  {{func_name}}(
{{indent}}     {{output}}, {{input}}, static_cast<const float*>({{gamma}}), static_cast<const float*>({{beta}}),
{{indent}}     {{m}}, {{n}}, {{eps}}, threadpool_.get()
{{indent}}  );
{{indent}}}
    """
)


@registry.reg("rvv.layernorm.gen_function")
def layernorm_gen_function(func_attrs: Dict[str, Any]) -> str:
    backend_spec = RVVSpec()
    from aitemplate.compiler.compiler import IS_REMOTE_COMPILE
    return FUNC_TEMPLATE.render(
        func_signature=FUNC_SIGNATURE.render(func_name=func_attrs["name"]),
        fuse_sigmoid_mul=False,
        is_remote_compile=IS_REMOTE_COMPILE,
    )


@registry.reg("rvv.layernorm_sigmoid_mul.gen_function")
def layernorm_sigmoid_mul_gen_function(func_attrs: Dict[str, Any]) -> str:
    return FUNC_TEMPLATE.render(
        func_signature=FUNC_SIGNATURE.render(func_name=func_attrs["name"]),
        fuse_sigmoid_mul=True,
    )


@registry.reg("rvv.layernorm.func_decl")
@registry.reg("rvv.layernorm_sigmoid_mul.func_decl")
def layernorm_sigmoid_mul_gen_function_decl(func_attrs: Dict[str, Any]):
    return FUNC_DECL.render(
        func_signature=FUNC_SIGNATURE.render(func_name=func_attrs["name"]).strip()
    )


@registry.reg("rvv.layernorm.func_call")
@registry.reg("rvv.layernorm_sigmoid_mul.func_call")
def layernorm_sigmoid_mul_gen_function_call(func_attrs, indent="  "):
    output_name = ""
    assert len(func_attrs["outputs"]) == 1
    assert 1 <= len(
        func_attrs["inputs"]
    ), "expected at least 1 inputs but got {}".format(len(func_attrs["inputs"]))

    output_name = func_attrs["outputs"][0]._attrs["name"]
    (input_name, gamma_name, beta_name) = layernorm_common.get_input_names(func_attrs)

    input_accessor = func_attrs["input_accessors"][0]
    shapes = input_accessor.original_shapes
    norm_ndim = len(func_attrs["normalized_shape"])
    m_name = "M"

    m_shape_func = layernorm_common.generate_m_shape_func(
        shapes,
        norm_ndim,
        m_name,
        indent + "    ",
    )

    n_name = "N"
    n_shape_func = layernorm_common.generate_n_shape_func(
        shapes,
        norm_ndim,
        n_name,
        indent + "    ",
    )

    m_n_shape_func = f"{m_shape_func}\n{n_shape_func}"
    eps = func_attrs["eps"]

    return FUNC_CALL_TEMPLATE.render(
        func_name=func_attrs["name"],
        m_n_shape_func=m_n_shape_func,
        output=output_name,
        input=input_name,
        gamma=gamma_name,
        beta=beta_name,
        m=m_name,
        n=n_name,
        eps=eps,
        indent=indent + "  ",
    )
