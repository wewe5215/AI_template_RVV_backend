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
Common codegen functions for gemm.
"""

import os
import random
import re
from collections import OrderedDict
from hashlib import sha1
from typing import Any, Dict, List, Tuple

import jinja2
import logging
from aitemplate.backend.backend_spec import RVVSpec
from aitemplate.backend.rvv.gemm_universal.layout import RCR
from aitemplate.backend.common import gemm_common, tensor_accessor_codegen
from aitemplate.backend.target import Target

from aitemplate.compiler.base import IntImm
from aitemplate.utils import alignment

# pylint: disable=C0301,C0415,R1705
_LOGGER = logging.getLogger(__name__)

INPUT_ADDR_CALCULATOR = jinja2.Template(
    """
  int64_t input_a_batch_stride = {{input_a_batch_stride_dim}};
  int64_t input_a_stride = {{input_a_stride_dim}};
  int64_t input_a_offset = {{input_a_offset_val}}; // default to 0
  int64_t input_b_batch_stride = {{input_b_batch_stride_dim}};
  int64_t input_b_stride = {{input_b_stride_dim}};
  int64_t input_b_offset = {{input_b_offset_val}}; // default to 0
    """
)


# These should be only used for 2D gemm
# For templates for bmm, see bmm_common
OUTPUT_ADDR_CALCULATOR = jinja2.Template(
    """
  {% if not output_accessor.is_from_strided_tensor %}
  int64_t output_stride = {{stride_dim}};
  int64_t output_offset = 0;
  {% else %}
  int64_t output_stride = {{output_accessor.actual_total_elements_from_stride_dim}};
  int64_t output_offset = {{output_accessor.offset}};
  {% endif %}
    """
)

DEFAULT_OUTPUT_ADDR_CALCULATOR = jinja2.Template(
    """
  int64_t output_stride = {{stride_dim}};
  int64_t output_offset = 0;
    """
)

DIM_DEFS_TEMPLATE = jinja2.Template(
    """
{% for dim, value in dims.items() %}
{{indent}}int64_t {{dim}} = {{value}};
{% endfor %}
"""
)


INPUT_OUTPUT_CHECKS_TEMPLATE = jinja2.Template(
    """
  int64_t a_size = 1;
{% for idx in range(input_ndims) %}
    a_size *= *a_dim{{idx}};
{% endfor %}
  if (a_size != 0 && !a_ptr) {
    throw std::runtime_error("input a is null!");
  }

  int64_t b_size = 1;
{% for idx in range(weight_ndims) %}
    b_size *= *b_dim{{idx}};
{% endfor %}
  if (b_size != 0 && !b_ptr) {
    throw std::runtime_error("input b is null!");
  }

  int64_t c_size = 1;
{% for idx in range(output_ndims) %}
    c_size *= *c_dim{{idx}};
{% endfor %}
  if (c_size != 0) {
    if (!c_ptr) {
      throw std::runtime_error("input c is null!");
    }
  } else {
    // output is empty and safe to return
    return;
  }

  // One of the input tensor are empty
  if (a_size == 0 || b_size == 0) {
    return;
  }
"""
)

INSTANCE_TEMPLATE = jinja2.Template(
    """
{{config}}
"""
)


SRC_TEMPLATE = jinja2.Template(
    """
#include <iostream>
#include <memory>
#include <random>
#include <vector>
#include <cstdio>
#include <stdexcept>
#include <cstdlib>
#include <string>
#include <pthreadpool.h>
#include "xnnpack.h"
#include "logging.h"
{{extra_code}}


{{instances}}


void {{function_name}} (
    void* a_ptr,
    void* b_ptr,
{% if has_d %}
    void* d_ptr,
{% endif %}
    void* c_ptr,
{% for idx in range(input_ndims) %}
    int64_t* a_dim{{idx}},
{% endfor %}
{% for idx in range(weight_ndims) %}
    int64_t* b_dim{{idx}},
{% endfor %}
{% for idx in range(output_ndims) %}
    {% if idx == output_ndims - 1 %}
    int64_t* c_dim{{idx}}
    {% else %}
    int64_t* c_dim{{idx}},
    {% endif %}
{% endfor %}
  ) {
  {{shape_eval}}
  {{input_addr_calculator}}
  {{output_addr_calculator}}
  {{extra_shape}}
  {{input_output_checks}}
  {% if is_first_op %}
    const xnn_status status_init = xnn_initialize(nullptr);
  {% endif %}
  {{exec_paths}}
  {% for idx in range(input_ndims) %}
      std::cout << "input_ndims{{idx}}: " << *a_dim{{idx}} << std::endl;
  {% endfor %}
  {% for idx in range(weight_ndims) %}
      std::cout << "weight_ndims{{idx}}: " << *b_dim{{idx}} << std::endl;
  {% endfor %}
  {% for idx in range(output_ndims) %}
      std::cout << "output_ndims{{idx}}: " << *c_dim{{idx}} << std::endl;
  {% endfor %}
  throw std::runtime_error(
      "Unsupported workload for this {{function_name}} specialization."
  );
}
""",
    trim_blocks=True,
    lstrip_blocks=True,
)


EXEC_TEMPLATE = jinja2.Template(
    """

//{{instance}}
{{indent}}return;
"""
)


FUNC_DECL_TEMPLATE = jinja2.Template(
    """
void {{func_name}}(
  void*,
  void*,
  void*,
  uint8_t*,
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


FUNC_CALL_TEMPLATE = jinja2.Template(
    """
{{indent}}{
{{indent}}{{local_dim_defs}}
{{indent}}{{func_name}}(
{{indent}}    {{a_ptr}},
{{indent}}    {{b_ptr}},
{% if has_bias %}
{{indent}}    {{bias_ptr}},
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
{{indent}}    threadpool_.get());
{{indent}}}
"""
)


BENCHMARK_INSTANCE_TEMPLATE = jinja2.Template(
    """
{{indent}}{
{{indent}}
{{indent}}int ret = 0;
{{indent}}try {
{{indent}}ret = {{func_name}}(
{{indent}}    memory_pool.get(),
{% for dim in adims %}
{{indent}}    {{dim}},
{% endfor %}
{% for dim in bdims %}
{{indent}}    {{dim}},
{% endfor %}
{% for dim in cdims %}
{{indent}}    {{dim}}{% if not loop.last %},{% endif %}
{% endfor %}
{{indent}});
{{indent}}} catch (...) {}
{{indent}}if (ret != 0)
{{indent}}  return ret;
{{indent}}
{{indent}}}
"""
)


TENSOR_DECL_TEMPLATE = jinja2.Template(
    """
  int64_t a_ptr_sz = a_dim0 * a_dim1;
  int64_t b_ptr_sz = b_dim0 * b_dim1;
  int64_t c_ptr_sz = c_dim0 * c_dim1;

  memory_pool->AllocateTensor(a_ptr_sz);  // a_ptr: index 0
  memory_pool->AllocateTensor(b_ptr_sz);  // b_ptr: index 1
  memory_pool->AllocateTensor(c_ptr_sz, /*is_output*/true);  // c_ptr: index 2

{% if has_bias %}
  memory_pool->AllocateTensor(c_dim1);  // bias_ptr: index 3
{% endif %}
"""
)


# TODO Merge all alignment into single profiler
PROFILER_TEMPLATE = jinja2.Template(
    """
#include <sstream>
#include <iostream>
#include <memory>
#include <random>
#include <vector>
#include <sstream>
#include <memory>
#include <ctime>
#include <cstdlib>
#include <stdexcept>
#include <cstring>
#include <thread>
size_t GLOBAL_WORKSPACE_SIZE = 0;
template <typename DType>
struct ProfilerMemoryPool {
  ProfilerMemoryPool() {
    std::random_device rd;
    gen = std::mt19937(rd());
    uniform_dist = std::uniform_int_distribution<int64_t>(1, 48964896);
    ptrs.reserve(512);
  }
  ~ProfilerMemoryPool() {
    for (void* ptr : ptrs) {
      free(ptr);
    }
  }

  int AllocateTensor(int64_t size, bool is_output = false) {
    if (size <= 0) {
      throw std::invalid_argument("Size must be positive");
    }
    DType* ptr = (DType*)malloc(size * sizeof(DType));
    if (!ptr) {
      throw std::bad_alloc();
    }

    if(is_output){
      std::memset(ptr, 0, size * sizeof(DType));
    }
    else{
      for (int64_t i = 0; i < size; ++i) {
        ptr[i] = static_cast<DType>(uniform_dist(gen));
      }
    }
    ptrs.push_back(reinterpret_cast<void*>(ptr));
    return ptrs.size() - 1;
  }

  DType* RequestTensorByIdx(int idx) {
    if (idx < 0 || idx >= ptrs.size()) {
      throw std::out_of_range("Index out of range");
    }
    return reinterpret_cast<DType*>(ptrs.at(idx));
  }

  std::vector<void*> ptrs;
  std::mt19937 gen;
  std::uniform_int_distribution<int64_t> uniform_dist;
};
{{op_func}}



int benchmark_{{function_name}} (
{% if is_group_gemm %}
    void **ptr_A,
    void **ptr_B,
    void **ptr_C,
    {% if has_bias %}
    void **ptr_bias,
    {% endif %}
    int64_t* lda,
    int64_t* ldb,
    int64_t* ldc,
    {% if has_bias %}
    int64_t* ldd,
    {% endif %}
    int occupancy
{% else %}
    ProfilerMemoryPool<{{elem_type}}>* memory_pool,
{% for idx in range(input_ndims) %}
    int64_t* a_dim{{idx}},
{% endfor %}
{% for idx in range(weight_ndims) %}
    int64_t* b_dim{{idx}},
{% endfor %}
{% for idx in range(output_ndims) %}
    int64_t* c_dim{{idx}}{% if not loop.last %},{% endif %}
{% endfor %}
{% endif %}
  ) {
  size_t num_threads = std::thread::hardware_concurrency();
  std::unique_ptr<pthreadpool, decltype(&pthreadpool_destroy)> threadpool_(
      pthreadpool_create(num_threads), pthreadpool_destroy);
  // warmup
  for (int i = 0; i < 5; ++i) {
    {{func_call}}
  }
  struct timespec start, end;
  clock_gettime(CLOCK_MONOTONIC, &start);
  for (int i = 0; i < 10; ++i) {
    {{func_call}}
  }
  clock_gettime(CLOCK_MONOTONIC, &end);
  float runtime_ms = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e6;
  // TODO: output workspace
  if (runtime_ms < 0.00001) {
      throw std::runtime_error(
      "OOB in xnnpack."
    );
  }
  std::cout << "OP:" << "gemm_rcr" << ",";
  std::cout << "TIME:" << runtime_ms << ",";
  std::cout << "WS:" << GLOBAL_WORKSPACE_SIZE << std::endl;
  return 0;
}


int main(int argc, char** argv) {
  auto memory_pool = std::make_unique<ProfilerMemoryPool<{{elem_type}}>>();
  {{args_parse}}

  {{tensor_decl}}

  {{benchmark_instances}}
  return 0;
}
"""
)


def has_d(func_attrs):
    if "has_d" in func_attrs:
        return func_attrs["has_d"]
    else:
        return False


def has_d1(func_attrs):
    return func_attrs.get("num_sources", 0) >= 2

def universal_gemm_instance(
    op_def: str,
    func_attrs: Dict[str, Any],
    for_profiler: bool
) -> str:
    return ""
# TODO : take reference on conv2d/common.py emit_instance and fetch the desired instance
def emit_instance(
    op,
    func_attrs=None,
):
    import cpu_lib
    op_def = op.emit()
    return op_def


def extract_config(
    dtype,
    layout=RCR,
    op_kind=None,
    extra_kind=None,
    Layout=None
):
    import cpu_lib
    spec = RVVSpec()
    lib_dtype = spec.dtype_to_lib_type(dtype)
    gemm_ops = OrderedDict()
    extract_ops = list(Target.current()._operators[op_kind][extra_kind][Layout].items())
    for key, value in extract_ops:
        for op in value:
            if layout == RCR:
                if cpu_lib.library.LayoutTag[op.A.layout] == "row" and \
                    cpu_lib.library.LayoutTag[op.B.layout] == "column" and \
                    cpu_lib.library.LayoutTag[op.C.layout] == "row" and \
                    lib_dtype == cpu_lib.library.DataTypeNames[op.A.element]:
                    gemm_ops[key] = value[0]
    _LOGGER.info(f"gemm_ops = {gemm_ops}, value =  {value}")
    return gemm_ops

def gen_function(
    func_attrs,
    src_template,
    exec_cond_template,
    problem_args,
    input_ndims,
    weight_ndims,
    output_ndims,
    dim_info_dict,
    f_instance_convertor=universal_gemm_instance,
    emit_kernel=False,
    support_split_k=False,
    input_addr_calculator="",
    output_addr_calculator="",
    extra_code="",
):
    backend_spec = RVVSpec()
    elem_input_type = backend_spec.dtype_to_lib_type(
        func_attrs["inputs"][0]._attrs["dtype"]
    )
    elem_output_type = backend_spec.dtype_to_lib_type(
        func_attrs["outputs"][0]._attrs["dtype"]
    )
    func_name = func_attrs["name"]
    exec_path = func_attrs["exec_path"]
    op_instance = func_attrs["op_instance"]
    _LOGGER.info(f"exec_path = {exec_path}, op_instance = {op_instance}")
    for exec_item in exec_path.values():
        fname = "f" + sha1(exec_item.exec_cond.encode()).hexdigest()
        algo = exec_item.algo
        op_key = next(iter(op_instance.keys()))
        config = emit_instance(
            op_instance[op_key],
            func_attrs=func_attrs,
        )
        exec_paths = ""
        exec_inst = exec_cond_template.render(
            indent="  ",
            cond=exec_item.exec_cond,
            program=config,
        )
        exec_paths += exec_inst
    shape_eval_func = gemm_common.gen_shape_eval_code(
        indent=1, dtype="int64_t", dim_info_dict=dim_info_dict, is_ptr=True
    )
    input_output_checks = INPUT_OUTPUT_CHECKS_TEMPLATE.render(
        input_ndims=input_ndims,
        weight_ndims=weight_ndims,
        output_ndims=output_ndims,
    )
    match = re.search(r'(\d+)$', func_name)
    return src_template.render(
        instances="",
        function_name=func_name,
        dtype="float",
        is_first_op = (match.group(1) == '0'),
        shape_eval=shape_eval_func,
        input_addr_calculator=input_addr_calculator,
        output_addr_calculator=output_addr_calculator,
        input_output_checks=input_output_checks,
        exec_paths=exec_paths,
        input_ndims=input_ndims,
        weight_ndims=weight_ndims,
        output_ndims=output_ndims,
        has_d=has_d(func_attrs),
        has_d1=has_d1(func_attrs),
        extra_code=extra_code,
        elem_input_type=elem_input_type,
        elem_output_type=elem_output_type,
    )


def build_profiler(file_pairs):
    target = Target.current()
    if target.disable_profiler_codegen():
        file_pairs = []
    elif target.use_dummy_profiling_results():
        # if it is circle CI only random build 2 profilers
        random.shuffle(file_pairs)
        file_pairs = file_pairs[:2]
    return file_pairs


def add_profiler(file_pairs, workdir, op_type, output_name, code):
    prefix = os.path.join(workdir, "profiler", op_type)
    if not os.path.exists(prefix):
        os.makedirs(prefix)

    obj_path = os.path.join(prefix, output_name)
    if os.path.exists(obj_path):
        return
    # add logging.h to file_pairs
    file_pairs.extend(Target.current().copy_headers_and_csrc_to_workdir(prefix))
    if isinstance(code, dict):
        # multi-source profiler
        src_paths = []
        for src_name, src_code in code.items():
            # create each source file separately
            src_path = os.path.join(prefix, src_name + ".cpp")
            with open(src_path, "w") as f:
                f.write(src_code)
            src_paths.append(src_path)
        # add multiple src paths to file_pairs
        file_pairs.append((src_paths, obj_path))
    else:
        # single-source profiler
        src_path = os.path.join(prefix, output_name + ".cpp")
        with open(src_path, "w") as f:
            f.write(code)
        # add single src path to file_pairs
        file_pairs.append((src_path, obj_path))

def gen_profiler(
    func_attrs,
    workdir,
    profiler_filename,
    dim_info_dict,
    src_template,
    args_parser_template,
    support_split_k=False,
    output_addr_calculator="",
    bias_ptr_arg=None,
    extra_code="",
):
    import cpu_lib

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
    ndims = 2
    adims = ["&a_dim" + str(i) for i in range(ndims)]
    bdims = ["&b_dim" + str(i) for i in range(ndims)]
    cdims = ["&c_dim" + str(i) for i in range(ndims)]
    shape_func = gemm_common.gen_shape_eval_code(
        indent=2, dtype="int64_t", dim_info_dict=dim_info_dict, is_ptr=True
    )

    has_bias = bias_ptr_arg is not None
    instance_name_base = "GemmInstance"
    exec_program = ""
    input_output_checks = INPUT_OUTPUT_CHECKS_TEMPLATE.render(
        input_ndims=ndims,
        weight_ndims=ndims,
        output_ndims=ndims,
    )

    function_name = "gemm"
    instances = []
    benchmark_instances = []
    for instance_idx, (op_name, op) in enumerate(op_instance.items()):
        config = emit_instance(op)
        gemm_op = f"gemm_op_{instance_idx}"
        benchmark_instance = BENCHMARK_INSTANCE_TEMPLATE.render(
            indent="  ",
            gemm_op=gemm_op,
            gemm_op_name=op_name,
            func_name=f"benchmark_{function_name}",
            adims=adims,
            bdims=bdims,
            cdims=cdims,
        )
        instances.append(config)
        benchmark_instances.append(benchmark_instance)
    # TODO: Render args_parse by caller.
    args_parse = (
        args_parser_template
        if isinstance(args_parser_template, str)
        else args_parser_template.render()
    )
    op_func = src_template.render(
        is_profiler=True,
        function_name=function_name,
        input_ndims=ndims,
        weight_ndims=ndims,
        output_ndims=ndims,
        shape_eval=shape_func,
        input_output_checks=input_output_checks,
        is_first_op = True,
        exec_paths="\n".join(instances),
        output_addr_calculator=output_addr_calculator,
        extra_code=extra_code,
    )
    benchmark_adims = ["a_dim" + str(i) for i in range(ndims)]
    benchmark_bdims = ["b_dim" + str(i) for i in range(ndims)]
    benchmark_cdims = ["c_dim" + str(i) for i in range(ndims)]
    func_call = FUNC_CALL_TEMPLATE.render(
        is_profiler=True,
        func_name=function_name,
        a_ptr="memory_pool->RequestTensorByIdx(0)",
        b_ptr="memory_pool->RequestTensorByIdx(1)",
        has_bias=has_bias,
        bias_ptr=bias_ptr_arg,
        c_ptr="memory_pool->RequestTensorByIdx(2)",
        adims=benchmark_adims,
        bdims=benchmark_bdims,
        cdims=benchmark_cdims,
    )
    tensor_decl = TENSOR_DECL_TEMPLATE.render(
        elem_input_type=elem_input_type,
        elem_output_type=elem_output_type,
        has_bias=has_bias,
    )
    code = PROFILER_TEMPLATE.render(
        op_func=op_func,
        has_bias=has_bias,
        has_d=has_d(func_attrs),
        args_parse=args_parse,
        function_name=function_name,
        input_ndims=ndims,
        weight_ndims=ndims,
        output_ndims=ndims,
        func_call=func_call,
        name=instance_name_base,
        tensor_decl=tensor_decl,
        benchmark_instances="\n".join(benchmark_instances),
        elem_type=elem_type,
    )
    # FIXME: remove file_pairs once we have make -j ready for building
    # an entire graph
    file_pairs = []
    add_profiler(file_pairs, workdir, op_type, profiler_filename, code)
    # build
    return build_profiler(file_pairs)


def gen_local_dim_defs(func_attrs, indent="  "):
    """
    used together with input TensorAccessor to access a strided input
    """
    if "input_accessors" not in func_attrs:
        return ""

    dims = {}
    for input_idx, input_accessor in enumerate(func_attrs["input_accessors"]):
        if not input_accessor.is_from_strided_tensor:
            continue
        original_shape = input_accessor.original_shapes
        for idx, dim in enumerate(original_shape):
            # skip dynamic dims
            if isinstance(dim, IntImm):
                input_shape = func_attrs["inputs"][input_idx]._attrs["shape"]
                if idx < len(input_shape):
                    name = input_shape[idx]._attrs["name"]
                    if name in dims:
                        assert dims[name] == dim.value(), "bmm inputs shape mismatch"
                    else:
                        dims[name] = dim.value()
    return DIM_DEFS_TEMPLATE.render(dims=dims, indent=indent)


def gen_function_call(func_attrs, indent="  ", bias_ptr_arg=None):
    a = func_attrs["inputs"][0]
    ashapes = func_attrs["input_accessors"][0].original_shapes
    b = func_attrs["inputs"][1]
    bshapes = func_attrs["input_accessors"][1].original_shapes
    c = func_attrs["outputs"][0]
    cshapes = func_attrs["output_accessors"][0].original_shapes
    has_bias = bias_ptr_arg is not None
    # overwrite the global defs if we have input TensorAccessor
    local_dim_defs = gen_local_dim_defs(func_attrs, indent=indent)
    adims = ["&" + dim._attrs["name"] for dim in ashapes]
    bdims = ["&" + dim._attrs["name"] for dim in bshapes]
    cdims = ["&" + dim._attrs["name"] for dim in cshapes]
    return FUNC_CALL_TEMPLATE.render(
        local_dim_defs=local_dim_defs,
        func_name=func_attrs["name"],
        a_ptr=a._attrs["name"],
        b_ptr=b._attrs["name"],
        has_bias=has_bias,
        bias_ptr=bias_ptr_arg,
        c_ptr=c._attrs["name"],
        adims=adims,
        bdims=bdims,
        cdims=cdims,
        indent=indent,
    )

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
    return True
