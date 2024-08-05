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
RVV codegen functions for pool2d.
"""
from hashlib import sha1

import jinja2
import re
import logging
from aitemplate.backend.backend_spec import RVVSpec
# pylint: disable=C0103,C0301,W0613,W0612
_LOGGER = logging.getLogger(__name__)
# TODO : add f16 implementation later
EXEC_TEMPLATE_AVG = jinja2.Template(
    """
{{indent}}xnn_operator_t op_avg = nullptr;
{{indent}}const xnn_status status = xnn_create_average_pooling2d_nhwc_f32(
{{indent}}  PH, PW, PH, PW, KH, KW, SH, SW, 
{{indent}}  -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(), 
{{indent}}  /*flags=*/0, &op_avg);

{{indent}}CHECK_EQ(xnn_status_success, status);
{{indent}}CHECK_NE(nullptr, op_avg);
{{indent}}std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op(op_avg, xnn_delete_operator);

{{indent}}size_t w_size = 0;
{{indent}}size_t w_align = 0;
{{indent}}CHECK_EQ(
{{indent}}  xnn_status_success,
{{indent}}  xnn_reshape_average_pooling2d_nhwc_f32(
{{indent}}    op_avg, NI, HI, WI,
{{indent}}    CI, /*input_pixel_stride=*/CI, /*output_pixel_stride=*/CO,
{{indent}}    &w_size, &w_align,
{{indent}}    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
{{indent}}    /*threadpool=*/nullptr));
{{indent}}CHECK_LE(w_align, 16);
{{indent}}std::vector<char> workspace_vector(w_size + w_align + 16);
{{indent}}void* maybe_aligned_workspace = workspace_vector.data();
{{indent}}void* aligned_workspace = \
    (void*)((intptr_t)maybe_aligned_workspace + w_align - (intptr_t)maybe_aligned_workspace % w_align);
{{indent}}CHECK_EQ(xnn_status_success, xnn_setup_average_pooling2d_nhwc_f32(op_avg, aligned_workspace, (float*)(in_ptr), (float*)(out_ptr)));
{{indent}}CHECK_EQ(xnn_status_success, xnn_run_operator(op_avg, /*threadpool=*/nullptr));
"""
)

EXEC_TEMPLATE_MAX = jinja2.Template(
    """
{{indent}}xnn_operator_t op_max = nullptr;
{{indent}}const xnn_status status = xnn_create_max_pooling2d_nhwc_f32(
{{indent}}  PH, PW, PH, PW, KH, KW, SH, SW,
{{indent}}  1, 1, -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(), /*flags=*/0, &op_max);
{{indent}}std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op(op_max, xnn_delete_operator);
{{indent}}CHECK_EQ(xnn_status_success, status);
{{indent}}CHECK_NE(nullptr, op_max);
{{indent}}CHECK_EQ(
{{indent}}  xnn_status_success, xnn_reshape_max_pooling2d_nhwc_f32(
{{indent}}                        op_max, NI, HI, WI, CI, /*input_pixel_stride=*/CI,
{{indent}}                        /*output_pixel_stride=*/CO, /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
{{indent}}                        /*threadpool=*/nullptr));
{{indent}}CHECK_EQ(xnn_status_success, xnn_setup_max_pooling2d_nhwc_f32(op_max, (float*)(in_ptr), (float*)(out_ptr)));
{{indent}}CHECK_EQ(xnn_status_success, xnn_run_operator(op_max, /*threadpool=*/nullptr));
"""
)

SRC_TEMPLATE = jinja2.Template(
    """
#include <cstdio>
#include <stdexcept>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>
#include "xnnpack.h"
#include "logging.h"
#include <functional>
#include <random>
#include <stdint.h>
#include <cstddef> // For size_t
#include <cstring> // For memcpy


void {{function_name}}(
    void* in_ptr,
    void* out_ptr,
    int64_t* batch,
    int64_t* in_ch,
    int64_t* in_h,
    int64_t* in_w,
    int64_t* out_batch,
    int64_t* out_h,
    int64_t* out_w,
    int64_t kernel_h,
    int64_t kernel_w,
    int64_t stride,
    int64_t pad
    ) {
  {{shape_function}}
  {% if is_first_op %}
    const xnn_status status_init = xnn_initialize(nullptr);
  {% endif %}
  {{exec_paths}}
  return;
  throw std::runtime_error(
      "Unsupported workload for this conv2d specialization."
  );
}
"""
)


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
  int64_t*,
  int64_t*,
  int64_t,
  int64_t,
  int64_t,
  int64_t
);
"""
)

FUNC_CALL_TEMPLATE = jinja2.Template(
    """
{{indent}}{{func_name}}(
{{indent}}    {{in_ptr}},
{{indent}}    {{out_ptr}},
{{indent}}    {{p_batch}},
{{indent}}    {{p_in_ch}},
{{indent}}    {{p_in_h}},
{{indent}}    {{p_in_w}},
{{indent}}    {{p_out_batch}},
{{indent}}    {{p_out_h}},
{{indent}}    {{p_out_w}},
{{indent}}    {{kernel_h}},
{{indent}}    {{kernel_w}},
{{indent}}    {{stride}},
{{indent}}    {{pad}}
{{indent}});
"""
)


def gen_function(
    func_attrs,
    exec_cond_template,
    shape_eval_template,
    shape_save_template,
):
    """
    Parameters
    ----------
    func_attrs : Dict
        Stores the operation attributes.
    exec_cond_template : jinja2.Template
        Generates if statement to execute kernel.
    shape_eval_template : jinja2.Template
        Generates shape calculation.
        The template is passed from compiler/ops/pool.
    shape_save_template : jinja2.Template
        Generates output dimensions.
        The template is passed from compiler/ops/pool.

    Returns
    -------
    str
        The rendered template of generated function body.

    Raises
    ------
    NotImplementedError
        An error is raised if op_type is not max or average pooling.
    """
    import cpu_lib

    spec = RVVSpec()
    dtype = spec.dtype_to_backend_type(func_attrs["inputs"][0]._attrs["dtype"])
    op_type = func_attrs["op"]
    func_name = func_attrs["name"]
    exec_path = func_attrs["exec_path"]
    exec_paths = ""
    if "max" in op_type:
        exec_paths = EXEC_TEMPLATE_MAX.render(indent="    ", DataName=dtype)
    elif "avg" in op_type:
        exec_paths = EXEC_TEMPLATE_AVG.render(indent="    ", DataName=dtype)
    else:
        raise NotImplementedError
    shape_eval_func = shape_eval_template.render(
        indent="  ",
        dtype="int64_t ",
        x_dim0="*batch",
        x_dim1="*in_h",
        x_dim2="*in_w",
        x_dim3="*in_ch",
        kernel_h="kernel_h",
        kernel_w="kernel_w",
        stride="stride",
        pad="pad",
        div="/",
    )
    shape_save_func = shape_save_template.render(
        indent="  ",
        y_dim0="*out_batch",
        y_dim1="*out_h",
        y_dim2="*out_w",
        y_dim3="*in_ch",
    )
    shape_func = shape_eval_func + shape_save_func
    match = re.search(r'(\d+)$', func_name)
    _LOGGER.info(f"match.group(1) = {match.group(1)}")
    return SRC_TEMPLATE.render(
        function_name=func_name,
        shape_function=shape_func,
        is_first_op = (match.group(1) == '0'),
        exec_paths=exec_paths,
    )


def gen_function_decl(func_name):
    return FUNC_DECL_TEMPLATE.render(func_name=func_name)


def gen_function_call(func_attrs, indent="  "):
    """
    Parameters
    ----------
    func_attrs : Dict
        Stores the operation attributes.
    indent : str, optional
        Indent for codegen, target dependent e.g. C++, python, etc., by default "  ".

    Returns
    -------
    str
        The rendered template of generated function call.
    """
    x = func_attrs["inputs"][0]
    xshape = x._attrs["shape"]
    y = func_attrs["outputs"][0]
    yshape = y._attrs["shape"]
    return FUNC_CALL_TEMPLATE.render(
        func_name=func_attrs["name"],
        in_ptr=x._attrs["name"],
        out_ptr=y._attrs["name"],
        p_batch="&" + xshape[0]._attrs["name"],
        p_in_ch="&" + xshape[3]._attrs["name"],
        p_in_h="&" + xshape[1]._attrs["name"],
        p_in_w="&" + xshape[2]._attrs["name"],
        p_out_batch="&" + yshape[0]._attrs["name"],
        p_out_h="&" + yshape[1]._attrs["name"],
        p_out_w="&" + yshape[2]._attrs["name"],
        kernel_h=func_attrs["kernel_size"],
        kernel_w=func_attrs["kernel_size"],
        stride=func_attrs["stride"],
        pad=func_attrs["pad"],
        indent=indent,
    )
