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

from aitemplate.backend.backend_spec import RVVSpec

from aitemplate.backend.common import gemm_common, tensor_accessor_codegen
from aitemplate.backend.target import Target

from aitemplate.compiler.base import IntImm
from aitemplate.utils import alignment

# pylint: disable=C0301,C0415,R1705


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

{{indent}}{{instance}}
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
{% if support_split_k %}
  int,
{% endif %}
{% for idx in range(input_ndims) %}
  int64_t*,
{% endfor %}
{% for idx in range(weight_ndims) %}
  int64_t*,
{% endfor %}
{% for idx in range(input_ndims) %}
  int64_t*,
{% endfor %}
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
{{indent}}    global_workspace_,
{{indent}}    {{split_k}},
{% for dim in adims %}
{{indent}}    {{dim}},
{% endfor %}
{% for dim in bdims %}
{{indent}}    {{dim}},
{% endfor %}
{% for dim in cdims %}
{{indent}}    {{dim}},
{% endfor %}
{{indent}}    stream
{{indent}});
{{indent}}}
"""
)


BENCHMARK_INSTANCE_TEMPLATE = jinja2.Template(
    """
{{indent}}{
{{indent}}
{{indent}}{{instance_name}} {{gemm_op}};
{{indent}}const char *gemm_op_name = "{{gemm_op_name}}";
{{indent}}int ret = 0;
{{indent}}try {
{{indent}}ret = {{func_name}}(
{{indent}}    {{gemm_op}},
{{indent}}    gemm_op_name,
{{indent}}    memory_pool.get(),
{{indent}}    global_workspace_,
{% if support_split_k %}
{{indent}}    {{split_k}},
{% endif %}
{% for dim in adims %}
{{indent}}    {{dim}},
{% endfor %}
{% for dim in bdims %}
{{indent}}    {{dim}},
{% endfor %}
{% for dim in cdims %}
{{indent}}    {{dim}},
{% endfor %}
{{indent}}    stream
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

  // The value 1 is used to force ptr_max_sz to be non-zero
  int64_t ptr_max_sz = std::max<int64_t>({1, a_ptr_sz, b_ptr_sz, c_ptr_sz});

  size_t one_copy_sz = a_ptr_sz + b_ptr_sz + c_ptr_sz;
{% if has_bias %}
  one_copy_sz += c_dim1;
{%endif%}
  int64_t mem_pool_sz = memory_pool->ComputeMemPoolSize(one_copy_sz, ptr_max_sz, device_properties.l2CacheSize);

  memory_pool->AllocateTensor(a_ptr_sz, mem_pool_sz);  // a_ptr: index 0
  memory_pool->AllocateTensor(b_ptr_sz, mem_pool_sz);  // b_ptr: index 1
  memory_pool->AllocateTensor(c_ptr_sz, mem_pool_sz, /*is_output*/true);  // c_ptr: index 2

{% if has_bias %}
  memory_pool->AllocateTensor(c_dim1, mem_pool_sz);  // bias_ptr: index 3
{% endif %}
"""
)


# TODO Merge all alignment into single profiler
PROFILER_TEMPLATE = jinja2.Template(
    """
size_t GLOBAL_WORKSPACE_SIZE = 0;

#include <sstream>

{{op_func}}

template <typename DType>
struct ProfilerMemoryPool;

template <typename GemmInstance>
int benchmark_{{function_name}} (
{% if is_group_gemm %}
    int sharedMemPerMultiprocessor,
    int multiProcessorCount,
    uint8_t* global_workspace_,
    int problem_count,
    cutlass::gemm::GemmCoord* problem_sizes_device,
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
    int occupancy,
    cudaStream_t stream

{% else %}
    ProfilerMemoryPool<{{elem_type}}>* memory_pool,
    uint8_t* global_workspace_,
{% if support_split_k %}
    int split_k,
{% endif %}
{% for idx in range(input_ndims) %}
    int64_t* a_dim{{idx}},
{% endfor %}
{% for idx in range(weight_ndims) %}
    int64_t* b_dim{{idx}},
{% endfor %}
{% for idx in range(output_ndims) %}
    int64_t* c_dim{{idx}},
{% endfor %}
    cudaStream_t stream
{% endif %}
  ) {
  // warmup
  for (int i = 0; i < 5; ++i) {
    {{func_call}}
  }
  cudaEvent_t events[2];
  for (auto & event : events) {
    cudaEventCreate(&event);
  }
  cudaEventRecord(events[0], stream);
  for (int i = 0; i < 10; ++i) {
    {{func_call}}
  }
  cudaEventRecord(events[1], stream);
  cudaEventSynchronize(events[1]);
  float runtime_ms = 0;
  cudaEventElapsedTime(&runtime_ms, events[0], events[1]);
  for (auto event : events) {
    (void)cudaEventDestroy(event);
  }
  // TODO: output workspace
  if (runtime_ms < 0.00001) {
      throw std::runtime_error(
      "OOB in cutlass."
    );
  }
  std::cout << "TIME:" << runtime_ms << ",";
  std::cout << "WS:" << GLOBAL_WORKSPACE_SIZE << std::endl;
  return 0;
}

template <typename DType>
struct ProfilerMemoryPool {
  ProfilerMemoryPool() : shared_input_tensor(false) {
    std::random_device rd;
    gen = std::mt19937(rd());
    uniform_dist = std::uniform_int_distribution<int64_t>(1, 48964896);
    offsets.reserve(512);
    strides.reserve(512);
    copies.reserve(512);
    ptrs.reserve(512);
    blobs.reserve(512);
  }
  ~ProfilerMemoryPool() {}

  int64_t ComputeMemPoolSize(size_t one_copy_sz, size_t ptr_max_sz, size_t l2_cache_bytes) {
    int times_covers_l2_cache = (int)std::ceil(l2_cache_bytes / sizeof(DType) / ptr_max_sz);
    int64_t mem_pool_sz = std::max(2, std::min(512, times_covers_l2_cache));
    size_t free_global_mem = 0;
    size_t total_global_mem = 0;
    cudaError_t cuda_error = cudaMemGetInfo(&free_global_mem, &total_global_mem);
    if (cuda_error != cudaSuccess) {
      auto error_msg = std::string("Failed to invoke cudaMemGetInfo: ") +
          cudaGetErrorName(cuda_error) + ", at " + __FILE__;
      throw std::runtime_error(error_msg);
    }
    size_t single_copy_nbytes = one_copy_sz * sizeof(DType);
    while (mem_pool_sz > 0) {
      size_t nbytes = single_copy_nbytes * mem_pool_sz;
      if (nbytes < free_global_mem) {
        break;
      }
      mem_pool_sz--;
    }

    if (mem_pool_sz <= 1) {
      size_t minimal_required_nbytes = ptr_max_sz * sizeof(DType);
      if (minimal_required_nbytes > free_global_mem) {
        // We absolutely run out of memory
        auto error_msg = std::string("no enough GPU memory: requested ") +
            std::to_string(minimal_required_nbytes) + ", available: " +
            std::to_string(free_global_mem) + ", ptr_max_sz: " +
            std::to_string(ptr_max_sz) + ", at " + __FILE__;
        throw std::runtime_error(error_msg);
      } else {
        // Let's try to allocate a single blob that is large enough to hold
        // all input tensors. Note that this is still an approximation, because
        // we may still hit cudaErrorMemoryAllocation error while allocating
        // memory for the output. We will rely on cudaMalloc to throw out
        // an exception in such a case.
        shared_input_tensor = true;
        AllocateGaussianTensor(ptr_max_sz);
      }
      return 1;
    }
    return mem_pool_sz;
  }

  DType* AllocateGaussianTensor(int64_t size) {
    size_t length = size * sizeof(DType);
    blobs.emplace_back(length);
    DType* ptr = reinterpret_cast<DType*>(blobs.back().get());

    uint64_t seed = uniform_dist(gen);
    double mean = 0.f;
    double std = 1.f;

    cutlass::reference::device::BlockFillRandomGaussian(ptr, size, seed, mean,
                                                        std);

    return ptr;
  }

  int AllocateTensor(int64_t size, int64_t copy, bool is_output = false) {
    offsets.push_back(0);
    strides.push_back(size);
    copies.push_back(copy);
    DType *ptr;
    if (!is_output && shared_input_tensor) {
      ptr = reinterpret_cast<DType*>(blobs.back().get());
    } else {
      ptr = AllocateGaussianTensor(size * copy);
    }
    ptrs.push_back(reinterpret_cast<void*>(ptr));
    return ptrs.size() - 1;
  }

  DType* RequestTensorByIdx(int idx) {
    auto copy = copies.at(idx);
    auto offset = offsets.at(idx);
    auto stride = strides.at(idx);
    DType* ptr = reinterpret_cast<DType*>(ptrs.at(idx));
    ptr += offset;
    offset += stride;
    if (offset == copy * stride) {
        offset = 0;
    }
    offsets[idx] = offset;
    return ptr;
  }

  std::vector<int64_t> offsets;
  std::vector<int64_t> strides;
  std::vector<int64_t> copies;
  std::vector<void*> ptrs;
  std::vector<cutlass::DeviceAllocation<uint8_t> > blobs;
  std::mt19937 gen;
  std::uniform_int_distribution<int64_t> uniform_dist;
  // make a shared blob to hold all inputs in cases we do not have
  // enough GPU memory
  bool shared_input_tensor;
};


int main(int argc, char** argv) {
  auto memory_pool = std::make_unique<ProfilerMemoryPool<{{elem_type}}>>();
  {{args_parse}}

  uint8_t* global_workspace_ = nullptr;

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


def get_gemm_instance_template_params(
    op_def: str,
    kernel_config: Tuple[str, int, int] = ("cutlass::gemm::device::Gemm", 21, 3),
) -> List[str]:
    """
    For a given op_def string generated by cutlass's gemm emiter, parse and
    return the gemm instance's template parameters.
    kernel_config is a tuple used for finding kernel params. The first element
    of kernel_config is the kernel kind, the second is the expected number
    of params, and the third is the index offset of alignment values in the
    full op_def string.
    """
    kernel_kind, expected_num_params, _ = kernel_config
    params = re.findall(rf"{kernel_kind}<([\s\S]+)>;", op_def)
    assert len(params) == 1
    param = params[0]
    gemm_universal_params = param.strip().split("\n")
    gemm_universal_params = [param.strip(",") for param in gemm_universal_params]
    assert len(gemm_universal_params) == expected_num_params, (
        f"expected len(gemm_universal_params) to be {expected_num_params}, but got "
        f"{len(gemm_universal_params)}, {gemm_universal_params=}"
    )
    return gemm_universal_params


def get_tensor_accessor_alignments(func_attrs):
    """Infer the A, B, and epilogue alignments from the respective TAs."""
    input_accessors = func_attrs["input_accessors"]
    a_alignment = tensor_accessor_codegen.find_max_alignment_for_accessor(
        input_accessors[0]
    )
    b_alignment = tensor_accessor_codegen.find_max_alignment_for_accessor(
        input_accessors[1]
    )
    output_accessor = func_attrs["output_accessors"][0]
    epilogue_alignment = tensor_accessor_codegen.find_max_alignment_for_accessor(
        output_accessor
    )

    # if the last dim is dynamic, force align=1
    if not isinstance(output_accessor.original_shapes[-1], IntImm):
        epilogue_alignment = 1

    return a_alignment, b_alignment, epilogue_alignment


def update_alignments_in_gemm_instance(
    op_def: str,
    func_attrs: Dict[str, Any],
    for_profiler: bool,
    kernel_config: Tuple[str, int, int] = ("cutlass::gemm::device::Gemm", 21, 3),
) -> str:
    """
    update kAlignmentA, kAlignmentB, and epilogue_alignment in op_def,
    which is a gemm instance emitted by the gemm instance emitter of cutlass.
    kernel_config is a tuple used for finding kernel params. The first element
    of kernel_config is the kernel kind, the second is the expected number
    of params, and the third is the index offset of alignment values in the
    full op_def string.
    """
    if for_profiler:
        return op_def

    a_alignment, b_alignment, epilogue_alignment = get_tensor_accessor_alignments(
        func_attrs
    )

    gemm_params = get_gemm_instance_template_params(op_def, kernel_config)
    epilogue_align_idx = 11
    a_align_idx = 17
    b_align_idx = 18
    a_curr_align = gemm_params[a_align_idx].strip()
    b_curr_align = gemm_params[b_align_idx].strip()
    epilogue_curr_align = gemm_params[epilogue_align_idx].strip()
    a_alignment = min(a_alignment, int(a_curr_align))
    b_alignment = min(b_alignment, int(b_curr_align))
    epilogue_alignment = min(epilogue_alignment, int(epilogue_curr_align))
    instance_lines = op_def.split("\n")
    # a_align_idx + idx_offset in the full instance string
    idx_offset = kernel_config[2]

    def _replace_align(align_idx, curr_align, alignment):
        curr_align_line = instance_lines[align_idx + idx_offset]
        assert curr_align == curr_align_line.strip(
            " ,"
        ), f"expected {curr_align=} equal to {curr_align_line=}"
        instance_lines[align_idx + idx_offset] = curr_align_line.replace(
            curr_align, str(alignment)
        )

    _replace_align(a_align_idx, a_curr_align, a_alignment)
    _replace_align(b_align_idx, b_curr_align, b_alignment)
    _replace_align(epilogue_align_idx, epilogue_curr_align, epilogue_alignment)
    return "\n".join(instance_lines)


def universal_gemm_instance(
    op_def: str,
    func_attrs: Dict[str, Any],
    for_profiler: bool
) -> str:
    return ""
# TODO : take reference on conv2d/common.py emit_instance and fetch the desired instance
def emit_instance(
    op,
    for_profiler,
    emit_kernel=False,
    func_attrs=None,
):
    import cpu_lib

    return ""


def extract_config(
    f_proc_op,
    f_kernel_name="",
    include_cutlass_3x_ops=False,
):
    import cutlass_lib

    op_kind = cutlass_lib.library.OperationKind.Gemm
    gemm_kinds = {cutlass_lib.library.GemmKind.Universal}
    if include_cutlass_3x_ops:
        gemm_kinds.add(cutlass_lib.library.GemmKind.Universal3x)
    gemm_ops = OrderedDict()
    extract_ops = list(Target.current()._operators[op_kind].items())

    for _, value in extract_ops:
        op = value[0]
        if op.gemm_kind in gemm_kinds:
            ret = f_proc_op(op)
            if len(ret) > 0:
                for op_inst in ret:
                    key = f_kernel_name(op_inst)
                    gemm_ops[key] = op_inst
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
    problem_args_cutlass_3x="",
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
    inst_def_flag = set()
    instances = {}
    instance_decl = ""
    for exec_item in exec_path.values():
        fname = "f" + sha1(exec_item.exec_cond.encode()).hexdigest()
        algo = exec_item.algo
        if algo not in inst_def_flag:
            config = emit_instance(
                op_instance[algo],
                for_profiler=False,
                emit_kernel=emit_kernel,
                func_attrs=func_attrs,
            )
            inst_def_flag.add(algo)
        else:
            config = ""
        inst = INSTANCE_TEMPLATE.render(
            config=config,
            name=fname
        )
        instances[exec_item.exec_cond] = inst
        instance_decl += inst
    shape_eval_func = gemm_common.gen_shape_eval_code(
        indent=1, dtype="int64_t", dim_info_dict=dim_info_dict, is_ptr=True
    )

    exec_paths = ""
    for exec_cond in instances:
        fname = "f" + sha1(exec_cond.encode()).hexdigest()
        program = EXEC_TEMPLATE.render(
            indent="    ",
            instance=fname,
            support_split_k=support_split_k,
        )
        exec_inst = exec_cond_template.render(
            indent="  ",
            cond=exec_cond,
            program=program,
        )
        exec_paths += exec_inst
    input_output_checks = INPUT_OUTPUT_CHECKS_TEMPLATE.render(
        input_ndims=input_ndims,
        weight_ndims=weight_ndims,
        output_ndims=output_ndims,
    )
    return src_template.render(
        instances=instance_decl,
        function_name=func_name,
        dtype="float",
        shape_eval=shape_eval_func,
        input_addr_calculator=input_addr_calculator,
        output_addr_calculator=output_addr_calculator,
        input_output_checks=input_output_checks,
        exec_paths=exec_paths,
        input_ndims=input_ndims,
        weight_ndims=weight_ndims,
        output_ndims=output_ndims,
        support_split_k=support_split_k,
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
    problem_args_template,
    args_parser_template,
    support_split_k=False,
    output_addr_calculator="",
    bias_ptr_arg=None,
    extra_code="",
    problem_args_template_cutlass_3x=None,
):
    import cutlass_lib

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
    exec_program = EXEC_TEMPLATE.render(
        indent="  ",
        instance=instance_name_base,
        is_profiler=True,
        support_split_k=support_split_k,
    )
    input_output_checks = INPUT_OUTPUT_CHECKS_TEMPLATE.render(
        input_ndims=ndims,
        weight_ndims=ndims,
        output_ndims=ndims,
    )

    function_name = "gemm"
    instances = []
    benchmark_instances = []
    for instance_idx, (op_name, op) in enumerate(op_instance.items()):
        config = emit_instance(op, for_profiler=True)
        instance_name = f"{instance_name_base}_{instance_idx}"
        gemm_op = f"gemm_op_{instance_idx}"
        benchmark_instance = BENCHMARK_INSTANCE_TEMPLATE.render(
            indent="  ",
            instance_name=instance_name,
            gemm_op=gemm_op,
            gemm_op_name=op_name,
            func_name=f"benchmark_{function_name}",
            support_split_k=support_split_k,
            split_k="split_k",
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
        instances="\n".join(instances),
        function_name=function_name,
        input_ndims=ndims,
        weight_ndims=ndims,
        output_ndims=ndims,
        shape_eval=shape_func,
        input_output_checks=input_output_checks,
        exec_paths=exec_program,
        output_addr_calculator=output_addr_calculator,
        support_split_k=support_split_k,
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
        split_k="split_k",
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
        support_split_k=support_split_k,
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
        split_k=func_attrs["split_k"],
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
