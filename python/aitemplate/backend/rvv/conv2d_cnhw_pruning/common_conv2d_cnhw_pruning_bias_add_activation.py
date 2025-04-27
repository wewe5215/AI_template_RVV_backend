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
common functions for conv2d bias act residual add
"""

from aitemplate.backend.rvv.conv2d_cnhw_pruning import common

# pylint: disable=C0301,C0103

EXTRA_HEADER = """
#include <functional>
#include <random>
#include <cstddef> // For size_t
#include <cstring> // For memcpy
"""


def extract_config(
    func_attrs,
    dtype="float32",
    activation_op_name="Identity",
    binary_op_name="Plus",
    unary_op_name="Identity",
):
    import cpu_lib
    if unary_op_name == "ReLu":
        op_kind = cpu_lib.library.Conv2dKind.Conv2dPruningBiasAddRelu
    elif unary_op_name == "Identity":
        op_kind = cpu_lib.library.Conv2dKind.Conv2dPruningBiasAdd
    extra_kind = cpu_lib.library.TensorOperation.Add
    # if dtype == "float32": --> TODO: uncomment later
    Layout = cpu_lib.library.LayoutType.CNHW
    return common.extract_config(
        dtype = dtype,
        op_kind = op_kind,
        extra_kind = extra_kind,
        Layout = Layout)



def gen_profiler(
    func_attrs,
    workdir,
    profiler_filename,
    shape_template,
):
    return common.gen_profiler(
        func_attrs=func_attrs,
        workdir=workdir,
        profiler_filename=profiler_filename,
        shape_template=shape_template,
        is_bias_add=True,
        extra_header=EXTRA_HEADER,
    )


def gen_function(
    func_attrs,
    op_instance,
    exec_cond_template,
    shape_eval_template,
    shape_save_template,
):
    return common.gen_function(
        func_attrs=func_attrs,
        exec_cond_template=exec_cond_template,
        shape_eval_template=shape_eval_template,
        shape_save_template=shape_save_template,
        op_instance=op_instance,
        is_bias_add=True,
        extra_header=EXTRA_HEADER,
    )


def gen_function_decl(
    func_attrs,
):
    return common.gen_function_decl(
        func_attrs=func_attrs,
        is_bias_add=True,
    )


def gen_function_call(
    func_attrs,
    indent="  ",
):
    return common.gen_function_call(
        func_attrs=func_attrs,
        indent=indent,
        is_bias_add=True,
    )
