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
Codegen for conv2d_depthwise.
"""
from collections import OrderedDict

from aitemplate.backend import registry

from aitemplate.backend.backend_spec import RVVSpec
from aitemplate.backend.rvv.conv2d_cnhw_pruning import common
from aitemplate.backend.target import Target


def emit_instance(op):
    """Emits cutlass instance."""
    import cpu_lib

    op_def = op.emit()
    return op_def


@registry.reg("rvv.conv2d_cnhw_pruning_depthwise.config")
def conv2d_depthwise_config(func_attrs, dtype="float16"):
    """Populates conv2d_depthwise cutlass configs into 'op_instance' field."""
    import cpu_lib
    op_kind = cpu_lib.library.Conv2dKind.Conv2dDepthwise
    extra_kind = cpu_lib.library.TensorOperation.PassThrough
    # if dtype == "float32": --> TODO: uncomment later
    Layout = cpu_lib.library.LayoutType.CNHW
    func_attrs["op_instance"] = common.extract_config(
        dtype = dtype,
        op_kind = op_kind,
        extra_kind = extra_kind,
        Layout = Layout)


@registry.reg("rvv.conv2d_cnhw_pruning_depthwise.gen_profiler")
def gen_profiler(
    func_attrs,
    workdir,
    profiler_filename,
    shape_template,
):
    """Codegen for conv2d profiler."""
    return common.gen_profiler(
        func_attrs=func_attrs,
        workdir=workdir,
        profiler_filename=profiler_filename,
        shape_template=shape_template,
    )


@registry.reg("rvv.conv2d_cnhw_pruning_depthwise.gen_function")
def gen_function(
    func_attrs,
    exec_cond_template,
    shape_eval_template,
    shape_save_template,
):
    """Codegen for conv2d function."""
    import cpu_lib
    op_kind = cpu_lib.library.Conv2dKind.Conv2dDepthwise
    extra_kind = cpu_lib.library.TensorOperation.PassThrough
    # if dtype == "float32": --> TODO: uncomment later
    Layout = cpu_lib.library.LayoutType.CNHW
    op_instance = common.extract_config(
        dtype = func_attrs["inputs"][0]._attrs["dtype"],
        op_kind = op_kind,
        extra_kind = extra_kind,
        Layout = Layout)
    return common.gen_function(
        func_attrs=func_attrs,
        exec_cond_template=exec_cond_template,
        shape_eval_template=shape_eval_template,
        shape_save_template=shape_save_template,
        op_instance=op_instance,
    )


@registry.reg("rvv.conv2d_cnhw_pruning_depthwise.func_decl")
def conv2d_depthwise_gen_function_decl(func_attrs):
    """Codegen for conv2d_depthwise function declaration."""
    return common.gen_function_decl(
        func_attrs=func_attrs,
    )


@registry.reg("rvv.conv2d_cnhw_pruning_depthwise.func_call")
def conv2d_depthwise_gen_function_call(func_attrs, indent="  "):
    """Codegen for conv2d_depthwise function call."""
    return common.gen_function_call(
        func_attrs=func_attrs,
        indent=indent,
    )


@registry.reg("rvv.conv2d_cnhw_pruning_depthwise.filter")
def conv2d_depthwise_function_filter(cfg, func_attrs, x_shape):
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
