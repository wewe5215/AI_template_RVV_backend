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
bert_embeddings kernel codegen for RVV.
"""

from typing import Any, Dict

import jinja2

from aitemplate.backend import registry
from aitemplate.backend.backend_spec import RVVSpec

# pylint: disable=C0301

FUNC_TEMPLATE = jinja2.Template(
    """
#include <vector>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>

void layer_norm(float *x, int length, const float *gamma, const float *beta, float eps) {
    float mean = 0.0f;
    for (int i = 0; i < length; ++i) {
        mean += x[i];
    }
    mean /= length;
    float var = 0.0f;
    for (int i = 0; i < length; ++i) {
        float d = x[i] - mean;
        var += d * d;
    }
    var /= length;
    float inv_std = 1.0f / std::sqrt(var + eps);
    for (int i = 0; i < length; ++i) {
        x[i] = (x[i] - mean) * inv_std * gamma[i] + beta[i];
    }
}

{{func_signature}}
{
    float* out = reinterpret_cast<float*>(output);
    float* word_embeddings_f32 = reinterpret_cast<float*>(word_embeddings);
    float* token_type_embeddings_f32 = reinterpret_cast<float*>(token_type_embeddings);
    float* position_embeddings_f32 = reinterpret_cast<float*>(position_embeddings);
    const float* gamma_f32 = reinterpret_cast<const float*>(gamma);
    const float* beta_f32 = reinterpret_cast<const float*>(beta);
    for (int i = 0; i < indices_num; ++i) {
        int input_id   = input_ids[i];
        int token_type_id = token_type_ids[i];
        int position_id   = position_ids[i];
        assert(input_id  >= 0 && input_id < vocab_size);
        assert(token_type_id >= 0 && token_type_id < type_vocab_size);
        assert(position_id >= 0 && position_id < max_position_embeddings);
        const float* word_emb_vec = word_embeddings_f32 + input_id * embedding_dim;
        const float* token_emb_vec = token_type_embeddings_f32 + token_type_id * embedding_dim;
        const float* pos_emb_vec = position_embeddings_f32 + position_id * embedding_dim;
        float* out_ptr = out + i * embedding_dim;
        // sum embeddings
        for (int j = 0; j < embedding_dim; ++j) {
            out_ptr[j] = word_emb_vec[j] + token_emb_vec[j] + pos_emb_vec[j];
        }
        // layer norm
        layer_norm(out_ptr, embedding_dim, gamma_f32, beta_f32, eps);
        // (dropout omitted)
    }
}

"""
)

FUNC_SIGNATURE = jinja2.Template(
    """
void {{func_name}}(void* output,
                   {{index_type}}* input_ids,
                   {{index_type}}* token_type_ids,
                   {{index_type}}* position_ids,
                   void* word_embeddings,
                   void* token_type_embeddings,
                   void* position_embeddings,
                   void* gamma,
                   void* beta,
                   const int64_t indices_num,
                   const int64_t embedding_dim,
                   const int64_t vocab_size,
                   const int64_t type_vocab_size,
                   const int64_t max_position_embeddings,
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
{{indent}}  {{calculate_indices_num}}
{{indent}}  {{func_name}}(
{{indent}}            {{output}},
{{indent}}            {{input_ids}},
{{indent}}            {{token_type_ids}},
{{indent}}            {{position_ids}},
{{indent}}            {{word_embeddings}},
{{indent}}            {{token_type_embeddings}},
{{indent}}            {{position_embeddings}},
{{indent}}            {{gamma}},
{{indent}}            {{beta}},
{{indent}}            {{indices_num}},
{{indent}}            {{embedding_dim}},
{{indent}}            {{vocab_size}},
{{indent}}            {{type_vocab_size}},
{{indent}}            {{max_position_embeddings}},
{{indent}}            {{eps}}
{{indent}} );

{{indent}}}
    """
)

INDICES_NUM_TEMPLATE = jinja2.Template(
    """
  int64_t indices_num = 1;
  {% for dim_name in dim_names %}
  indices_num *= {{dim_name}};
  {% endfor %}
  """
)


def python_int_dtype_to_c_dtype(dtype):
    if dtype == "int64":
        return "int64_t"
    if dtype in ["int", "int32"]:
        return "int32_t"
    return dtype


@registry.reg("rvv.bert_embeddings.gen_function")
def bert_embeddings_gen_function(func_attrs: Dict[str, Any]) -> str:
    backend_spec = RVVSpec()
    elem_input_type = backend_spec.dtype_to_backend_type(
        func_attrs["inputs"][3]._attrs["dtype"]
    )
    dtype = python_int_dtype_to_c_dtype(func_attrs["inputs"][0]._attrs["dtype"])
    return FUNC_TEMPLATE.render(
        index_type=dtype,
        elem_input_type=elem_input_type,
        func_signature=FUNC_SIGNATURE.render(
            func_name=func_attrs["name"],
            index_type=dtype,
        ).strip(),
    )


@registry.reg("rvv.bert_embeddings.func_decl")
def bert_embeddings_gen_function_decl(func_attrs: Dict[str, Any]) -> str:
    dtype = python_int_dtype_to_c_dtype(func_attrs["inputs"][0]._attrs["dtype"])
    return FUNC_DECL.render(
        func_signature=FUNC_SIGNATURE.render(
            func_name=func_attrs["name"],
            index_type=dtype,
        ).strip()
    )


FUNC_CALL_INT64_PARAM_TEMPLATE = jinja2.Template("reinterpret_cast<int64_t*>({{name}})")
FUNC_CALL_INT32_PARAM_TEMPLATE = jinja2.Template("reinterpret_cast<int32_t*>({{name}})")


def get_int_param_template(tensor):
    name = tensor._attrs["name"]
    dtype = tensor._attrs["dtype"]
    if dtype == "int64":
        return FUNC_CALL_INT64_PARAM_TEMPLATE.render(name=name)
    elif dtype in ("int", "int32"):
        return FUNC_CALL_INT32_PARAM_TEMPLATE.render(name=name)
    else:
        raise NotImplementedError(f"Unsupported dtype: {dtype}")


@registry.reg("rvv.bert_embeddings.func_call")
def bert_embeddings_gen_function_call(func_attrs: Dict[str, Any], indent="  ") -> str:
    (
        input_ids,
        token_type_ids,
        position_ids,
        word_embeddings,
        token_type_embeddings,
        position_embeddings,
        gamma,
        beta,
    ) = func_attrs["inputs"]

    indices_dims = [shape._attrs["name"] for shape in input_ids.shape()]
    indices_num_str = INDICES_NUM_TEMPLATE.render(
        dim_names=indices_dims,
    )
    embedding_dim = word_embeddings._size(-1).value()
    vocab_size = word_embeddings._size(0).value()
    type_vocab_size = token_type_embeddings._size(0).value()
    max_position_embeddings = position_embeddings._size(0).value()

    eps = func_attrs["eps"]
    output_str = func_attrs["outputs"][0]._attrs["name"]

    input_ids_str = get_int_param_template(input_ids)
    token_type_ids_str = get_int_param_template(token_type_ids)
    position_ids_str = get_int_param_template(position_ids)

    word_embeddings_str = word_embeddings._attrs["name"]
    token_type_embeddings_str = token_type_embeddings._attrs["name"]
    position_embeddings_str = position_embeddings._attrs["name"]

    gamma_str = gamma._attrs["name"]
    beta_str = beta._attrs["name"]

    return FUNC_CALL_TEMPLATE.render(
        func_name=func_attrs["name"],
        calculate_indices_num=indices_num_str,
        output=output_str,
        input_ids=input_ids_str,
        token_type_ids=token_type_ids_str,
        position_ids=position_ids_str,
        word_embeddings=word_embeddings_str,
        token_type_embeddings=token_type_embeddings_str,
        position_embeddings=position_embeddings_str,
        gamma=gamma_str,
        beta=beta_str,
        indices_num="indices_num",
        embedding_dim=embedding_dim,
        vocab_size=vocab_size,
        type_vocab_size=type_vocab_size,
        max_position_embeddings=max_position_embeddings,
        eps=eps,
        indent=indent,
    )
