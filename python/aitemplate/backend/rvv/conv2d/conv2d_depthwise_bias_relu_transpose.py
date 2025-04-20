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
Codegen for conv2d_depthwise_bias_relu_transpose.
"""

from aitemplate.backend import registry
from aitemplate.backend.rvv.conv2d import common

# pylint: disable=C0103,C0415,W0613,C0301


@registry.reg("rvv.conv2d_depthwise_bias_relu_transpose.config")
def conv2d_depthwise_config(func_attrs, dtype="float16"):
    """Populates conv2d_depthwise configs into 'op_instance' field."""
    import cpu_lib
    op_kind = cpu_lib.library.Conv2dKind.Conv2dDepthwiseBiasReluTranspose
    extra_kind = cpu_lib.library.TensorOperation.Transpose
    # if dtype == "float32": --> TODO: uncomment later
    Layout = cpu_lib.library.LayoutType.NHWC
    func_attrs["op_instance"] = common.extract_config(
        dtype = dtype,
        op_kind = op_kind,
        extra_kind = extra_kind,
        Layout = Layout)


@registry.reg("rvv.conv2d_depthwise_bias_relu_transpose.gen_profiler")
def gen_profiler(func_attrs, workdir, profiler_filename, shape_template):
    """Codegen for conv2d_depthwise_bias_relu_transpose profiler."""
    return common.gen_profiler(
        func_attrs=func_attrs,
        workdir=workdir,
        profiler_filename=profiler_filename,
        shape_template=shape_template,
        is_bias=True,
    )


@registry.reg("rvv.conv2d_depthwise_bias_relu_transpose.gen_function")
def gen_function(
    func_attrs,
    exec_cond_template,
    shape_eval_template,
    shape_save_template,
):
    """Codegen for conv2d_depthwise_bias_relu_transpose function."""
    return common.gen_function(
        func_attrs=func_attrs,
        exec_cond_template=exec_cond_template,
        shape_eval_template=shape_eval_template,
        shape_save_template=shape_save_template,
        is_bias=True,
    )


@registry.reg("rvv.conv2d_depthwise_bias_relu_transpose.func_decl")
def conv2d_depthwise_gen_function_decl(func_attrs):
    """Codegen for conv2d_depthwise_bias_relu_transpose function declaration."""
    return common.gen_function_decl(
        func_attrs=func_attrs,
        is_bias=True,
    )


@registry.reg("rvv.conv2d_depthwise_bias_relu_transpose.func_call")
def conv2d_depthwise_gen_function_call(func_attrs, indent="  "):
    """Codegen for conv2d_depthwise_bias_relu_transpose function call."""
    return common.gen_function_call(
        func_attrs=func_attrs,
        indent=indent,
        is_bias=True,
    )


@registry.reg("rvv.conv2d_depthwise_bias_relu_transpose.filter")
def conv2d_depthwise_bias_function_filter(cfg, func_attrs, x_shape):
    """Generates function filter.

    Parameters
    ----------
    cfg: str
        The filename generated for profiler.
    func_attrs : Dict
        Stores the operation attributes.
    x_shape:
        Input shapes.

    Returns
    -------
    bool
        If input cfg should be filtered.
    """
    return True
