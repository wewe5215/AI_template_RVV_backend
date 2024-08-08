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
GEMM Specialization for
C = GeMM(A, B) + bias
where A[RowMajor][M, K], B[ColMajor][N, K], bias[RowMajor][N]
"""
import jinja2

from aitemplate.backend import registry

from aitemplate.backend.backend_spec import RVVSpec
from aitemplate.backend.rvv.gemm_universal import common, common_bias, gemm_rcr
from aitemplate.backend.rvv.gemm_universal.layout import RCR

# pylint: disable=C0103,C0415,W0613,C0301,R1705,R1703


EXTRA_CODE = jinja2.Template(
    """
using elem_input_type = {{elem_input_type}};
using elem_output_type = {{elem_output_type}};
"""
)

@registry.reg("rvv.gemm_rcr_bias.config")
def gemm_rcr_config(func_attrs, dtype="float16"):
    func_attrs["op_instance"] = common.extract_config(
        dtype=func_attrs["inputs"][0].dtype(),
        layout=RCR
    )



@registry.reg("rvv.gemm_rcr_bias.gen_profiler")
def gen_profiler(func_attrs, workdir, profiler_filename, dim_info_dict):
    backend_spec = RVVSpec()
    elem_input_type = backend_spec.dtype_to_lib_type(
        func_attrs["inputs"][0]._attrs["dtype"]
    )
    elem_output_type = backend_spec.dtype_to_lib_type(
        func_attrs["outputs"][0]._attrs["dtype"]
    )
    extra_code = EXTRA_CODE.render(
        elem_input_type=elem_input_type,
        elem_output_type=elem_output_type,
    )
    return gemm_rcr.common_gen_profiler(
        func_attrs=func_attrs,
        workdir=workdir,
        profiler_filename=profiler_filename,
        dim_info_dict=dim_info_dict,
        src_template=common_bias.SRC_TEMPLATE,
        bias_ptr_arg="memory_pool->RequestTensorByIdx(3)",
        extra_code=extra_code,
    )


@registry.reg("rvv.gemm_rcr_bias.gen_function")
def gen_function(
    func_attrs,
    exec_cond_template,
    dim_info_dict,
):
    input_addr_calculator = gemm_rcr.get_input_addr_calculator(func_attrs)
    input_ndims = len(func_attrs["input_accessors"][0].original_shapes)
    weight_ndims = len(func_attrs["input_accessors"][1].original_shapes)
    output_ndims = len(func_attrs["output_accessors"][0].original_shapes)
    backend_spec = RVVSpec()
    elem_input_type = backend_spec.dtype_to_lib_type(
        func_attrs["inputs"][0]._attrs["dtype"]
    )
    elem_output_type = backend_spec.dtype_to_lib_type(
        func_attrs["outputs"][0]._attrs["dtype"]
    )
    extra_code = EXTRA_CODE.render(
        elem_input_type=elem_input_type,
        elem_output_type=elem_output_type,
    )
    return common.gen_function(
        func_attrs=func_attrs,
        src_template=common_bias.SRC_TEMPLATE,
        exec_cond_template=exec_cond_template,
        problem_args="",
        input_ndims=input_ndims,
        weight_ndims=weight_ndims,
        output_ndims=output_ndims,
        dim_info_dict=dim_info_dict,
        support_split_k=True,
        input_addr_calculator=input_addr_calculator,
        output_addr_calculator=common.OUTPUT_ADDR_CALCULATOR.render(
            stride_dim="N", output_accessor=func_attrs["output_accessors"][0]
        ),
        extra_code=extra_code,
    )


@registry.reg("rvv.gemm_rcr_bias.func_decl")
def gen_function_decl(func_attrs):
    func_name = func_attrs["name"]
    input_ndims = len(func_attrs["input_accessors"][0].original_shapes)
    weight_ndims = len(func_attrs["input_accessors"][1].original_shapes)
    return common_bias.FUNC_DECL_TEMPLATE.render(
        func_name=func_name,
        input_ndims=input_ndims,
        weight_ndims=weight_ndims,
        support_split_k=True,
    )


@registry.reg("rvv.gemm_rcr_bias.func_call")
def gen_function_call(func_attrs, indent="  "):
    bias = func_attrs["inputs"][2]
    return common.gen_function_call(
        func_attrs, indent, bias_ptr_arg=bias._attrs["name"]
    )


@registry.reg("rvv.gemm_rcr_bias.filter")
def function_filter(cfg, func_attrs, ab_alignment):
    """Generates function filter.

    Parameters
    ----------
    cfg: str
        The filename generated for profiler.
    func_attrs : Dict
        Stores the operation attributes.
    ab_alignment:
        Input alignments.

    Returns
    -------
    bool
        If input cfg should be filtered.
    """
    return common.function_filter(cfg, func_attrs, ab_alignment)
