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
#include <assert.h>
#include <cmath>
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
template<typename T>
void layer_norm(
    T* input,
    T* output,
    const T* gamma,
    const T* beta,
    int m,
    int hidden_size,
    T epsilon = (T)1e-5
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

{{func_signature}}
{
    float* in = static_cast<float*>(input);
    float* out = static_cast<float*>(output);
    layer_norm<float>(
      in, out,
      gamma, beta,
      m, n, eps
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
                   const float eps)
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
{{indent}}     {{m}}, {{n}}, {{eps}}
{{indent}}  );
{{indent}}}
    """
)


@registry.reg("rvv.layernorm.gen_function")
def layernorm_gen_function(func_attrs: Dict[str, Any]) -> str:
    backend_spec = RVVSpec()

    return FUNC_TEMPLATE.render(
        func_signature=FUNC_SIGNATURE.render(func_name=func_attrs["name"]),
        fuse_sigmoid_mul=False,
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
