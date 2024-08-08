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
RVV codegen for nhwc3to4 op
"""
import jinja2

from ... import registry
from ...backend_spec import RVVSpec

# pylint: disable=C0301,W0613,W0612

FUNC_DECL_TEMPLATE = jinja2.Template(
    """
void {{func_name}}(
  void*,
  void*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*
);
"""
)

FUNC_CALL_TEMPLATE = jinja2.Template(
    """
{{indent}}{{func_name}}(
{{indent}}    {{in_ptr}},
{{indent}}    {{out_ptr}},
{{indent}}    {{p_batch}},
{{indent}}    {{p_in_h}},
{{indent}}    {{p_in_w}},
{{indent}}    {{p_out_batch}},
{{indent}}    {{p_out_h}},
{{indent}}    {{p_out_w}}
{{indent}});
"""
)


EXEC_TEMPLATE = jinja2.Template(
    """
{{indent}}memset(out_ptr, 0, NO * HO * WO * 4 * sizeof(float));
{{indent}}float* X = static_cast<float*>(in_ptr);
{{indent}}float* Y = static_cast<float*>(out_ptr);
{{indent}}for (int b = 0; b < NI; ++b) {
{{indent}}    for (int i = 0; i < HI; ++i) {
{{indent}}        #pragma unroll
{{indent}}        for (int j = 0; j < WI; ++j) {
{{indent}}            int idx_in = ((b * HI + i) * WI + j) * 3;
{{indent}}            int idx_out = ((b * HI + i) * WI + j) * 4;
{{indent}}            float32x4_t x_vals = vld1q_f32(&X[idx_in]);
{{indent}}            vst1q_lane_f32(&Y[idx_out], x_vals, 0);
{{indent}}            vst1q_lane_f32(&Y[idx_out + 1], x_vals, 1);
{{indent}}            vst1q_lane_f32(&Y[idx_out + 2], x_vals, 2);
{{indent}}        }
{{indent}}    }
{{indent}}}
{{indent}}return;
"""
)

SRC_TEMPLATE = jinja2.Template(
    """

#include <arm_neon.h>
#include <cstring> // for memset
#include <iostream>
void {{function_name}} (
    void* in_ptr,
    void* out_ptr,
    int64_t* batch,
    int64_t* in_h,
    int64_t* in_w,
    int64_t* out_batch,
    int64_t* out_h,
    int64_t* out_w
) {
  {{shape_function}}
  {{exec_paths}}
}

"""
)


@registry.reg("rvv.nhwc3to4.gen_function")
def gen_function(func_attrs, template_path, shape_eval_template, shape_save_template):
    """

    Parameters
    ----------
    func_attrs : [type]
        [description]
    template_path : [type]
        [description]
    shape_eval_template : [type]
        [description]
    shape_save_template : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    func_name = func_attrs["name"]
    backend_spec = RVVSpec()
    elem_input_type = backend_spec.dtype_to_backend_type(
        func_attrs["inputs"][0]._attrs["dtype"]
    )
    shape_eval_func = shape_eval_template.render(
        indent="  ",
        dtype="int64_t ",
        x_dim0="*batch",
        x_dim1="*in_h",
        x_dim2="*in_w",
    )
    shape_save_func = shape_save_template.render(
        indent="  ",
        y_dim0="*out_batch",
        y_dim1="*out_h",
        y_dim2="*out_w",
    )
    shape_func = shape_eval_func + shape_save_func
    exec_paths = EXEC_TEMPLATE.render(elem_input_type=elem_input_type)
    return SRC_TEMPLATE.render(
        function_name=func_name,
        elem_input_type=elem_input_type,
        shape_function=shape_func,
        exec_paths=exec_paths,
    )


@registry.reg("rvv.nhwc3to4.func_decl")
def gen_function_decl(func_attrs):
    func_name = func_attrs["name"]
    return FUNC_DECL_TEMPLATE.render(func_name=func_name)


@registry.reg("rvv.nhwc3to4.func_call")
def gen_function_call(func_attrs, indent="  "):
    x = func_attrs["inputs"][0]
    xshape = x._attrs["shape"]
    y = func_attrs["outputs"][0]
    yshape = y._attrs["shape"]
    return FUNC_CALL_TEMPLATE.render(
        func_name=func_attrs["name"],
        in_ptr=x._attrs["name"],
        out_ptr=y._attrs["name"],
        p_batch="&" + xshape[0]._attrs["name"],
        p_in_h="&" + xshape[1]._attrs["name"],
        p_in_w="&" + xshape[2]._attrs["name"],
        p_out_batch="&" + yshape[0]._attrs["name"],
        p_out_h="&" + yshape[1]._attrs["name"],
        p_out_w="&" + yshape[2]._attrs["name"],
        indent=indent,
    )
