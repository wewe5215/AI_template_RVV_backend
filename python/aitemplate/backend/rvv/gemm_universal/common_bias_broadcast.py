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
C = UnaryOp2(BinaryOp2(BinaryOp1(UnaryOp1(GeMM(A, B) + bias), D1), D2)),
"""

import re
from functools import partial

import jinja2

from aitemplate.backend.backend_spec import RVVSpec
from aitemplate.backend.common import gemm_common

from aitemplate.backend.rvv.gemm_universal import common, gemm_rcr
from aitemplate.backend.target import Target

# pylint: disable=C0103,C0415,W0613,C0301,R1705,R1703

EXTRA_CODE = jinja2.Template(
    """"""
)


SRC_TEMPLATE = jinja2.Template(
    """
#include <iostream>
#include <memory>
#include <random>
#include <vector>
#include "xnnpack.h"
#include "logging.h"



void {{function_name}} (
    void* a_ptr,
    void* b_ptr,
    void* bias_ptr,
    void* d0_ptr,
{% if has_d1 %}
    void* d1_ptr,
{% endif %}
    void* c_ptr,
{% for idx in range(input_ndims) %}
    int64_t* a_dim{{idx}},
{% endfor %}
{% for idx in range(weight_ndims) %}
    int64_t* b_dim{{idx}},
{% endfor %}
{% for idx in range(input_ndims) %}
    int64_t* c_dim{{idx}},
{% endfor %}
    pthreadpool* pthreadpool_
  ) {
  {{shape_eval}}
  {{input_addr_calculator}}
  {{output_addr_calculator}}
  {{extra_shape}}
  {{input_output_checks}}

  if (!bias_ptr) {
    throw std::runtime_error("bias is null!");
  }
  if (!d0_ptr) {
    throw std::runtime_error("d0_ptr is null!");
  }
{% if has_d1 %}
  if (!d1_ptr) {
    throw std::runtime_error("d1_ptr is null!");
  }
{% endif %}

  {{exec_paths}}
  return;
  throw std::runtime_error(
      "Unsupported workload for this {{function_name}} specialization."
  );
}
""",
    trim_blocks=True,
    lstrip_blocks=True,
)

# For function declaration codegen.
FUNC_DECL_TEMPLATE = jinja2.Template(
    """
void {{func_name}}(
  void*,
  void*,
  void*,
  void*,
{% if has_d1 %}
  void*,
{% endif %}
  void*,
{% for idx in range(input_ndims) %}
  int64_t*,
{% endfor %}
{% for idx in range(weight_ndims) %}
  int64_t*,
{% endfor %}
{% for idx in range(input_ndims) %}
  int64_t*,
{% endfor %}
  pthreadpool*
);
"""
)


# For function call codegen.
FUNC_CALL_TEMPLATE = jinja2.Template(
    """
{{indent}}{
{{indent}}{{local_dim_defs}}
{{indent}}{{func_name}}(
{{indent}}    {{a_ptr}},
{{indent}}    {{b_ptr}},
{{indent}}    {{bias_ptr}},
{{indent}}    {{d0_ptr}},
{% if has_d1 %}
{{indent}}    {{d1_ptr}},
{% endif %}
{{indent}}    {{c_ptr}},
{% for dim in adims %}
{{indent}}    {{dim}},
{% endfor %}
{% for dim in bdims %}
{{indent}}    {{dim}},
{% endfor %}
{% for dim in cdims %}
{{indent}}    {{dim}},
{% endfor %}
{{indent}}    threadpool_.get()
{{indent}});
{{indent}}}
"""
)

ARGS_PARSER_TEMPLATE = jinja2.Template(
    """
  int64_t M = std::atoi(argv[1]);
  int64_t N = std::atoi(argv[2]);
  int64_t K = std::atoi(argv[3]);
  {{layout.args_parser}}
"""
)

TENSOR_DECL_TEMPLATE = jinja2.Template(
    """
  int64_t a_ptr_sz = a_dim0 * a_dim1;
  int64_t b_ptr_sz = b_dim0 * b_dim1;
  int64_t c_ptr_sz = c_dim0 * c_dim1;
  // The value 1 is used to force ptr_max_sz to be non-zero
  int64_t ptr_max_sz = std::max<int64_t>({1, a_ptr_sz, b_ptr_sz, c_ptr_sz});
  size_t one_copy_sz = a_ptr_sz + b_ptr_sz + c_ptr_sz + c_dim1 + c_ptr_sz;
{% if has_d1 %}
  one_copy_sz += c_ptr_sz;
{%endif%}

  memory_pool->AllocateTensor(a_ptr_sz);  // a_ptr: index 0
  memory_pool->AllocateTensor(b_ptr_sz);  // b_ptr: index 1
  memory_pool->AllocateTensor(c_ptr_sz, /*is_output*/true);  // c_ptr: index 2
  memory_pool->AllocateTensor(c_dim1);  // bias_ptr: index 3
  memory_pool->AllocateTensor(c_ptr_sz);  // d0 ptr: index 4
{% if has_d1 %}
  memory_pool->AllocateTensor(c_ptr_sz);  // d1 ptr: index 5
{% endif %}
"""
)
def gen_profiler(
    func_attrs,
    workdir,
    profiler_filename,
    dim_info_dict,
    layout,
    unary_op1,
    binary_op1,
    binary_op2,
    unary_op2,
):
    op_type = func_attrs["op"]
    op_instance = func_attrs["op_instance"]
    backend_spec = RVVSpec()
    elem_input_type = backend_spec.dtype_to_lib_type(
        func_attrs["inputs"][0]._attrs["dtype"]
    )
    elem_output_type = backend_spec.dtype_to_lib_type(
        func_attrs["outputs"][0]._attrs["dtype"]
    )
    elem_type = backend_spec.dtype_to_backend_type(
        func_attrs["inputs"][0]._attrs["dtype"]
    )
    has_d1 = common.has_d1(func_attrs)

    ndims = 2
    adims = ["&a_dim" + str(i) for i in range(ndims)]
    bdims = ["&b_dim" + str(i) for i in range(ndims)]
    cdims = ["&c_dim" + str(i) for i in range(ndims)]
    shape_func = gemm_common.gen_shape_eval_code(
        indent=2, dtype="int64_t", dim_info_dict=dim_info_dict, is_ptr=True
    )

    instance_name_base = "GemmInstance"
    
    input_output_checks = common.INPUT_OUTPUT_CHECKS_TEMPLATE.render(
        input_ndims=ndims,
        weight_ndims=ndims,
        output_ndims=ndims,
    )

    function_name = "gemm"
    instances = []
    benchmark_instances = []
    for instance_idx, (op_name, op) in enumerate(op_instance.items()):
        config = common.emit_instance(op)
        instance_name = f"{instance_name_base}_{instance_idx}"
        gemm_op = f"gemm_op_{instance_idx}"
        benchmark_instance = common.BENCHMARK_INSTANCE_TEMPLATE.render(
            indent="  ",
            instance_name=instance_name,
            gemm_op=gemm_op,
            gemm_op_name=op_name,
            func_name=f"benchmark_{function_name}",
            adims=adims,
            bdims=bdims,
            cdims=cdims,
        )
        instances.append(config)
        benchmark_instances.append(benchmark_instance)
    exec_program = common.EXEC_TEMPLATE.render(
        indent="  ",
        instance=instance_name_base,
        instances="\n".join(instances),
    )
    op_func = SRC_TEMPLATE.render(
        is_profiler=True,
        function_name=function_name,
        elem_input_type=elem_input_type,
        elem_output_type=elem_output_type,
        input_ndims=ndims,
        weight_ndims=ndims,
        shape_eval=shape_func,
        input_output_checks=input_output_checks,
        exec_paths=exec_program,
        output_addr_calculator=common.DEFAULT_OUTPUT_ADDR_CALCULATOR.render(
            stride_dim="N"
        ),
        has_d1=has_d1,
    )
    benchmark_adims = ["a_dim" + str(i) for i in range(ndims)]
    benchmark_bdims = ["b_dim" + str(i) for i in range(ndims)]
    benchmark_cdims = ["c_dim" + str(i) for i in range(ndims)]
    func_call = FUNC_CALL_TEMPLATE.render(
        is_profiler=True,
        func_name="gemm",
        a_ptr="memory_pool->RequestTensorByIdx(0)",
        b_ptr="memory_pool->RequestTensorByIdx(1)",
        c_ptr="memory_pool->RequestTensorByIdx(2)",
        d0_ptr="memory_pool->RequestTensorByIdx(4)",
        d1_ptr="memory_pool->RequestTensorByIdx(5)",
        bias_ptr="memory_pool->RequestTensorByIdx(3)",
        adims=benchmark_adims,
        bdims=benchmark_bdims,
        cdims=benchmark_cdims,
        has_d1=has_d1,
    )
    code = common.PROFILER_TEMPLATE.render(
        op_func=op_func,
        has_bias=True,
        has_d=True,
        has_d1=has_d1,
        args_parse=ARGS_PARSER_TEMPLATE.render(
            layout=layout
        ),
        function_name=function_name,
        input_ndims=ndims,
        weight_ndims=ndims,
        output_ndims=ndims,
        func_call=func_call,
        name=instance_name_base,
        tensor_decl=TENSOR_DECL_TEMPLATE.render(has_d1=has_d1),
        benchmark_instances="\n".join(benchmark_instances),
        elem_type=elem_type,
    )
    # FIXME: remove file_pairs once we have make -j ready for building
    # an entire graph
    file_pairs = []
    common.add_profiler(file_pairs, workdir, op_type, profiler_filename, code)
    # build
    return common.build_profiler(file_pairs)

def gen_function_decl(func_attrs):
    input_ndims = len(func_attrs["input_accessors"][0].original_shapes)
    weight_ndims = len(func_attrs["input_accessors"][1].original_shapes)
    return FUNC_DECL_TEMPLATE.render(
        func_name=func_attrs["name"],
        input_ndims=input_ndims,
        weight_ndims=weight_ndims,
        has_d1=common.has_d1(func_attrs),
    )


def gen_function_call(func_attrs, indent="  "):
    has_d1 = common.has_d1(func_attrs)
    if has_d1:
        (a, b, bias, d0, d1) = func_attrs["inputs"]
    else:
        (a, b, bias, d0) = func_attrs["inputs"]
        d1 = None
    c = func_attrs["outputs"][0]
    # overwrite the global defs if we have input TensorAccessor
    local_dim_defs = common.gen_local_dim_defs(func_attrs, indent=indent)
    adims = [
        "&" + dim._attrs["name"]
        for dim in func_attrs["input_accessors"][0].original_shapes
    ]
    bdims = [
        "&" + dim._attrs["name"]
        for dim in func_attrs["input_accessors"][1].original_shapes
    ]
    cdims = [
        "&" + dim._attrs["name"]
        for dim in func_attrs["output_accessors"][0].original_shapes
    ]
    return FUNC_CALL_TEMPLATE.render(
        local_dim_defs=local_dim_defs,
        func_name=func_attrs["name"],
        a_ptr=a._attrs["name"],
        b_ptr=b._attrs["name"],
        bias_ptr=bias._attrs["name"],
        d0_ptr=d0._attrs["name"],
        d1_ptr=d1._attrs["name"] if has_d1 else "",
        c_ptr=c._attrs["name"],
        split_k=func_attrs["split_k"],
        adims=adims,
        bdims=bdims,
        cdims=cdims,
        indent=indent,
        has_d1=has_d1,
    )
