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
attention kernel codegen for rvv
"""
from typing import Any, Dict

import jinja2

from aitemplate.backend import registry

# pylint: disable=C0301

FUNC_CALL_INT32_PARAM_TEMPLATE = jinja2.Template("reinterpret_cast<int*>({{name}})")

FUNC_CALL_FP32_PARAM_TEMPLATE = jinja2.Template("reinterpret_cast<float*>({{name}})")

FUNC_TEMPLATE = jinja2.Template(
    """
#include <cmath>
#include <vector>
#include <iostream>
#include <limits>

{{func_signature}}
{
    // Interpret pointers as floats
    const float* qkv_f = reinterpret_cast<const float*>(qkv);
    float* output_f       = reinterpret_cast<float*>(output);
    // We assume layout:
    //   For each batch b in [0, batch_size):
    //     For each time step t in [0, seq_len):  // assuming fixed seq_len for simplicity
    //        One “row” of qkv contains: Q_h0 [head_size], Q_h1 [head_size], … Q_h(num_heads-1)[head_size],
    //                                   K_h0 […], K_h1 […], …,
    //                                   V_h0 […], V_h1 […], …
    //        The row stride (in floats) = 3 * num_heads * head_size
    //
    //   Output for each (b, t) has num_heads * head_size floats.
    //
    int qkv_row_stride = 3 * num_heads * head_size;   // number of floats per time-step row for qkv
    int out_row_stride =    num_heads * head_size;   // number of floats per time-step row for output

    // For each batch
    for (int b = 0; b < batch_size; ++b) {
      // For each query position t
      for (int t = 0; t < seq_len; ++t) {
        const float* row_base = qkv_f + (b * seq_len + t) * qkv_row_stride;

        // For each head h
        for (int h = 0; h < num_heads; ++h) {
          const float* Q_h = row_base + (0 * num_heads * head_size) + h * head_size;
          const float* K_h = row_base + (1 * num_heads * head_size) + h * head_size;
          const float* V_h = row_base + (2 * num_heads * head_size) + h * head_size;

          // We will compute scores for this head: size = seq_len (for all source positions s)
          std::vector<float> scores(seq_len);
          float scale = softmax_scale;  // e.g., 1.0 / sqrt(head_size)

          // For each source position s
          for (int s = 0; s < seq_len; ++s) {
            const float* K_h_s = (reinterpret_cast<const float*>(qkv_f) + (b * seq_len + s) * qkv_row_stride
                                 + (1 * num_heads * head_size) + h * head_size);
            // dot product Q_h · K_h_s
            float dot = 0.0f;
            for (int d = 0; d < head_size; ++d) {
              dot += Q_h[d] * K_h_s[d];
            }
            float sc = dot * scale;

            // Causal mask (if enabled): if s > t, then mask (i.e., invalidate attention from query t to key s)
            if (is_causal && s > t) {
              sc = -std::numeric_limits<float>::infinity();
            }

            // (If you have padding/mask via cu_seqlens etc, you'd apply it here.)
            scores[s] = sc;
          }

          // Compute softmax over scores
          float max_score = scores[0];
          for (float v : scores) if (v > max_score) max_score = v;
          float sum_exp = 0.0f;
          for (int s = 0; s < seq_len; ++s) {
            if (scores[s] == -std::numeric_limits<float>::infinity()) {
              scores[s] = 0.0f;
            } else {
              scores[s] = std::exp(scores[s] - max_score);
              sum_exp += scores[s];
            }
          }
          for (int s = 0; s < seq_len; ++s) {
            scores[s] /= sum_exp;
          }

          // Weighted sum of values V_h
          float* out_h = output_f + (b * seq_len + t) * out_row_stride + h * head_size;
          for (int d = 0; d < head_size; ++d) {
            float acc = 0.0f;
            for (int s = 0; s < seq_len; ++s) {
              const float* V_h_s = (reinterpret_cast<const float*>(qkv_f) + (b * seq_len + s) * qkv_row_stride
                                   + (2 * num_heads * head_size) + h * head_size);
              acc += scores[s] * V_h_s[d];
            }
            out_h[d] = acc;
          }

          // Optionally: store log-sum-exp of that head for softmax_lse[t, h] if used
          if (softmax_lse) {
            softmax_lse[(b * seq_len + t) * num_heads + h] = max_score + std::log(sum_exp);
          }

          // (We skip dropout and o_tmp in this scalar version for simplicity.)
        }
      }
    }
}
    """
)


FUNC_SIGNATURE = jinja2.Template(
    """
void {{func_name}}(void* output,
                   const void* qkv,
                   const int* cu_seqlens,
                   float* softmax_lse,
                   float* o_tmp,
                   int batch_size,
                   int seq_len,
                   int num_heads,
                   int head_size,
                   float p_dropout,
                   float softmax_scale,
                   bool is_causal,
                   bool loop)
    """
)

FUNC_DECL = jinja2.Template(
    """
    {{func_signature}};
    """
)

FUNC_CALL_TEMPLATE = jinja2.Template(
    """
{{indent}}{{func_name}}(
{{indent}}   {{output}}, {{qkv}}, {{cu_seqlens}},
{{indent}}    {{softmax_lse}}, {{o_tmp}},
{{indent}}    {{batch_size}},
{{indent}}    {{seq_len}},
{{indent}}    {{num_heads}},
{{indent}}    {{head_size}},
{{indent}}    {{p_dropout}},
{{indent}}    {{softmax_scale}},
{{indent}}    {{is_causal}}, {{loop}}
{{indent}});
    """
)


@registry.reg("rvv.flash_attention.gen_function")
def flash_attention_gen_function(func_attrs: Dict[str, Any]) -> str:
    """the function for generating attention kernel"""
    return FUNC_TEMPLATE.render(
        func_signature=FUNC_SIGNATURE.render(func_name=func_attrs["name"]),
    )


@registry.reg("rvv.flash_attention.func_decl")
def flash_attention_gen_function_decl(func_attrs: Dict[str, Any]):
    return FUNC_DECL.render(
        func_signature=FUNC_SIGNATURE.render(func_name=func_attrs["name"]).strip()
    )


@registry.reg("rvv.flash_attention.func_call")
def flash_attention_gen_function_call(func_attrs, indent="  "):
    """the function for generating a function call for attention"""
    output_name = ""
    assert len(func_attrs["outputs"]) == 1
    assert len(func_attrs["inputs"]) == 2

    output_name = func_attrs["outputs"][0]._attrs["name"]

    qkv_name = func_attrs["inputs"][0]._attrs["name"]

    seqlens_name = FUNC_CALL_INT32_PARAM_TEMPLATE.render(
        name=func_attrs["inputs"][1]._attrs["name"]
    )

    x = func_attrs["inputs"][0]

    batch_size = func_attrs["batch_size"]
    seq_len = func_attrs["seq_len"]

    num_heads = x._attrs["shape"][2]._attrs["values"][0]
    head_size = x._attrs["shape"][3]._attrs["values"][0]
    p_dropout = func_attrs["dropout"]
    is_causal = func_attrs["causal"]
    softmax_scale = head_size ** (-0.5)

    return FUNC_CALL_TEMPLATE.render(
        func_name=func_attrs["name"],
        output=output_name,
        qkv=qkv_name,
        cu_seqlens=seqlens_name,
        softmax_lse="reinterpret_cast<float*>(global_workspace_)",
        o_tmp="reinterpret_cast<float*>(global_workspace_ + {} * sizeof(float))".format(
            batch_size * num_heads * seq_len
        ),
        batch_size=batch_size,
        seq_len=seq_len,
        num_heads=num_heads,
        head_size=head_size,
        p_dropout=p_dropout,
        softmax_scale=softmax_scale,
        is_causal="true" if is_causal else "false",
        loop="true" if seq_len > 256 else "false",
        indent=indent,
    )
