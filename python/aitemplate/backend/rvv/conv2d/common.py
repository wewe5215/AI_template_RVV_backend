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
common template for conv2d
"""
import re
import os
from collections import OrderedDict
from hashlib import sha1
from typing import List

import jinja2
import logging
from aitemplate.backend.backend_spec import RVVSpec
from aitemplate.backend.rvv.gemm_universal.common import add_profiler, build_profiler
from aitemplate.backend.target import Target

from aitemplate.utils import alignment

_LOGGER = logging.getLogger(__name__)

INSTANCE_TEMPLATE = jinja2.Template(
    """
{{config}}
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
{{extra_header}}

{{functions}}
"""
)

FUNCTION_TEMPLATE = jinja2.Template(
    """
void {{function_name}} (
    void* in_ptr,
    void* weight_ptr,
    void* out_ptr,
{% if is_bias %}
    void* bias_ptr,
{% elif is_bias_add %}
    void* bias_ptr,
    void* res_ptr,
{% endif %}
    uint8_t* workspace,
    int64_t* batch,
    int64_t* out_ch,
    int64_t* in_ch,
    int64_t* kernel_h,
    int64_t* kernel_w,
    int64_t* in_h,
    int64_t* in_w,
    int64_t* out_batch,
    int64_t* out_h,
    int64_t* out_w,
    int strideh,
    int dilationh,
    int padh,
    int stridew,
    int dilationw,
    int padw,
    pthreadpool* pthreadpool_
  ) {

  {{shape_function}}

  int i32_batch = *batch;
  int i32_in_h = *in_h;
  int i32_in_w = *in_w;
  int i32_in_ch = *in_ch;
  int i32_out_ch = *out_ch;
  int i32_kernel_h = *kernel_h;
  int i32_kernel_w = *kernel_w;
  int i32_out_batch = *out_batch;
  int i32_out_h = *out_h;
  int i32_out_w = *out_w;
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

BENCHMARK_INSTANCE_TEMPLATE = jinja2.Template(
    """
{{indent}}{
{{indent}}  int ret = 0;
{{indent}}  try {
{{indent}}    ret = {{func_name}}(
{{indent}}      &runtime,
{{indent}}      &workspace_size,
{{indent}}      {{ni}},
{{indent}}      {{hi}},
{{indent}}      {{wi}},
{{indent}}      {{ci}},
{{indent}}      {{co}},
{{indent}}      {{kh}},
{{indent}}      {{kw}},
{{indent}}      {{no}},
{{indent}}      {{ho}},
{{indent}}      {{wo}},
{{indent}}      {{strideh}},
{{indent}}      {{dilationh}},
{{indent}}      {{padh}},
{{indent}}      {{stridew}},
{{indent}}      {{dilationw}},
{{indent}}      {{padw}},
{{indent}}      global_workspace_
{{indent}}    );
{{indent}}  } catch (...) {
{{indent}}    runtime = 0;
{{indent}}    workspace_size = 0;
{{indent}}  }
{{indent}}  if (ret != 0)
{{indent}}    return ret;
{{indent}}  std::cout << "OP:{{conv_op_name}},"
{{indent}}            << "TIME:" << runtime << ","
{{indent}}            << "WS:" << workspace_size << std::endl;
{{indent}}}
"""
)

BENCHMARK_DECL_TEMPLATE = jinja2.Template(
    """
int benchmark_{{function_name}} (
  float*,
  size_t*,
  int64_t,
  int64_t,
  int64_t,
  int64_t,
  int64_t,
  int64_t,
  int64_t,
  int64_t,
  int64_t,
  int64_t,
  int,
  int,
  int,
  int,
  int,
  int,
  uint8_t*
);
"""
)

BENCHMARK_TEMPLATE = jinja2.Template(
    """
using Ptr = std::unique_ptr<void, std::function<void(void*)>>;
inline Ptr RAII_DeviceMalloc(
    size_t num_bytes) {
  auto* output = malloc(num_bytes);
  if (!output) {
    throw std::bad_alloc();
  }
  auto deleter = [](void* ptr) { free(ptr); };
  return Ptr(output, deleter);
}
std::mt19937 rnd_generator(1234);
std::uniform_real_distribution<> dist(-10, 10);
template<typename T>
void fill_random(T* data, size_t size) {
  for (size_t i = 0; i < size; ++i) {
    data[i] = static_cast<T>(dist(rnd_generator));
  }
}
int benchmark_{{function_name}} (
  float* runtime,
  size_t* workspace_size,
  int64_t NI,
  int64_t HI,
  int64_t WI,
  int64_t CI,
  int64_t CO,
  int64_t KH,
  int64_t KW,
  int64_t NO,
  int64_t HO,
  int64_t WO,
  int strideh,
  int dilationh,
  int padh,
  int stridew,
  int dilationw,
  int padw,
  uint8_t* global_workspace_
) {
  Ptr in_data = RAII_DeviceMalloc(NI*HI*WI*CI*2);
  Ptr weight_data = RAII_DeviceMalloc(CO*KH*KW*CI*2);
{% if is_bias %}
  Ptr bias_data = RAII_DeviceMalloc(CO*2);
{% elif is_bias_add %}
  Ptr bias_data = RAII_DeviceMalloc(CO*2);
  Ptr res_data = RAII_DeviceMalloc(NO*HO*WO*CO*2);
{% endif %}
  Ptr out_data = RAII_DeviceMalloc(NO*HO*WO*CO*2);
{% if is_f16 %}
  auto* input = static_cast<__fp16*>(in_data.get());
  auto* weight = static_cast<__fp16*>(weight_data.get());
  auto* output = static_cast<__fp16*>(out_data.get());
    {% if is_bias %}
  auto* bias = static_cast<__fp16*>(bias_data.get());
    {% elif is_bias_add %}
  auto* bias = static_cast<__fp16*>(bias_data.get());
  auto* res = static_cast<__fp16*>(res_data.get());
    {% endif %}
{% elif is_f32 %}
  auto* input = static_cast<float*>(in_data.get());
  auto* weight = static_cast<float*>(weight_data.get());
  auto* output = static_cast<float*>(out_data.get());
    {% if is_bias %}
  auto* bias = static_cast<float*>(bias_data.get());
    {% elif is_bias_add %}
  auto* bias = static_cast<float*>(bias_data.get());
  auto* res = static_cast<float*>(res_data.get());
    {% endif %}
{% endif %}
  fill_random(input, NI * HI * WI * CI);
  fill_random(weight, CO * KH * KW * CI);
  std::memset(output, 0, NO*HO*WO*CO*2);
    {% if is_bias %}
  fill_random(bias, CO);
    {% elif is_bias_add %}
  fill_random(bias, CO);
  fill_random(res, NO*HO*WO*CO);
    {% endif %}
  size_t num_threads = std::thread::hardware_concurrency();
  std::unique_ptr<pthreadpool, decltype(&pthreadpool_destroy)> threadpool_(
      pthreadpool_create(num_threads), pthreadpool_destroy);
  // warmup
{{func_call}}
  struct timespec start, end;
  clock_gettime(CLOCK_MONOTONIC, &start);
  for (int i = 0; i < 5; ++i) {
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
  *runtime = runtime_ms;
  *workspace_size = GLOBAL_WORKSPACE_SIZE_{{instance_name}};
  return 0;
}
"""
)

PROFILER_BENCHMARK_TEMPLATE = jinja2.Template(
"""
#include <cstdio>
#include <stdexcept>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>
#include <random>
#include "xnnpack.h"
#include "logging.h"
#include <pthreadpool.h>
#include <thread>
{{extra_header}}
static size_t GLOBAL_WORKSPACE_SIZE_{{instance_name}} = 0;
{{functions}}

{{benchmark}}
"""
)

PROFILER_MAIN_TEMPLATE = jinja2.Template(
    """
#include <iostream>
#include <string>
#include <time.h>
#include "xnnpack.h"
{{benchmark_decls}}

int main(int argc, char** argv) {
  int64_t batch = std::stoi(argv[1]);
  int64_t in_h = std::stoi(argv[2]);
  int64_t in_w = std::stoi(argv[3]);
  int64_t in_ch = std::stoi(argv[4]);
  int64_t kernel_h = std::stoi(argv[5]);
  int64_t kernel_w = std::stoi(argv[6]);
  int64_t out_ch = std::stoi(argv[7]);
  int strideh = std::stoi(argv[8]);
  int padh = std::stoi(argv[9]);
  int dilationh = std::stoi(argv[10]);
  int stridew = std::stoi(argv[11]);
  int padw = std::stoi(argv[12]);
  int dilationw = std::stoi(argv[13]);

{{shape_func}}
  float runtime = 0;
  size_t workspace_size = 0;
  uint8_t* global_workspace_ = nullptr;

{{benchmark_instances}}

  return 0;
}
"""
)

FUNC_DECL_TEMPLATE = jinja2.Template(
    """
void {{func_name}}(
  void*,
  void*,
  void*,
{% if is_bias %}
  void*,
{% elif is_bias_add %}
  void*,
  void*,
{% endif %}
  uint8_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int,
  int,
  int,
  int,
  int,
  int,
  pthreadpool*
);
"""
)

FUNC_CALL_TEMPLATE = jinja2.Template(
    """
{{indent}}{{func_name}}(
{{indent}}    {{in_ptr}},
{{indent}}    {{weight_ptr}},
{{indent}}    {{out_ptr}},
{% if is_bias %}
{{indent}}    {{bias_ptr}},
{% elif is_bias_add %}
{{indent}}    {{bias_ptr}},
{{indent}}    {{res_ptr}},
{% endif %}
{{indent}}    global_workspace_,
{{indent}}    {{p_batch}},
{{indent}}    {{p_out_ch}},
{{indent}}    {{p_in_ch}},
{{indent}}    {{p_kernel_h}},
{{indent}}    {{p_kernel_w}},
{{indent}}    {{p_in_h}},
{{indent}}    {{p_in_w}},
{{indent}}    {{p_out_batch}},
{{indent}}    {{p_out_h}},
{{indent}}    {{p_out_w}},
{{indent}}    {{strideh}},
{{indent}}    {{dilationh}},
{{indent}}    {{padh}},
{{indent}}    {{stridew}},
{{indent}}    {{dilationw}},
{{indent}}    {{padw}},
{{indent}}    threadpool_.get()
{{indent}});
"""
)

def emit_instance(op):
    """Emits instance."""
    import cpu_lib  # noqa: F401

    op_def = op.emit()
    return op_def

def extract_config(
    dtype="float16",
    op_kind=None,
    extra_kind=None,
    conv2d_specialization=None
):
    """Extracts config for conv kernels."""
    import copy
    import cpu_lib

    spec = RVVSpec()
    lib_dtype = spec.dtype_to_lib_type(dtype)
    conv2d_ops = OrderedDict()
    _LOGGER.debug(f"_operators =  {Target.current()._operators}")
    extract_ops = list(Target.current()._operators[op_kind][extra_kind].items())
    for key, value in extract_ops:
        for op in value:
            if lib_dtype == cpu_lib.library.DataTypeNames[op.A.element]:
                    conv2d_ops[key] = value[0]
    _LOGGER.debug(f"conv2d_ops = {conv2d_ops}, value =  {value}")
    return conv2d_ops


def gen_profiler(
    func_attrs,
    workdir,
    profiler_filename,
    shape_template,
    f_emit_instance="",
    is_bias=False,
    is_bias_add=False,
    is_transpose=False,
    is_depthwise=False,
    extra_header="",
    instance_name_base="DeviceConvFwdInstance",
):
    """Generate profiler sources."""
    op_type = func_attrs["op"]
    op_instance = func_attrs["op_instance"]

    backend_spec = RVVSpec()
    dtype = backend_spec.dtype_to_lib_type(func_attrs["inputs"][0]._attrs["dtype"])

    func_call_extra_args = {}
    if is_bias:
        func_call_extra_args = {
            "bias_ptr": "(void*)bias",
        }
    elif is_bias_add:
        func_call_extra_args = {
            "bias_ptr": "(void*)bias",
            "res_ptr": "(void*)res",
        }
    benchmark_decls = []
    benchmark_instances = []
    profiler_benchmarks = {}

    for instance_idx, (op_name, op) in enumerate(op_instance.items()):
        instance_name = f"{instance_name_base}_{instance_idx}"
        function_name = f"{op_type}_{op_name}"

        exec_program = emit_instance(op)
        shape_func = shape_template.render(
            indent="  ",
            dtype="int64_t ",
            div="/",
            x_dim0="*batch",
            x_dim1="*in_h",
            x_dim2="*in_w",
            x_dim3="*in_ch",
            w_dim0="*out_ch",
            w_dim1="*kernel_h",
            w_dim2="*kernel_w",
            strideh="strideh",
            dilateh="dilationh",
            padh="padh",
            stridew="stridew",
            dilatew="dilationw",
            padw="padw",
        )
        function = FUNCTION_TEMPLATE.render(
            is_bias=is_bias,
            is_bias_add=is_bias_add,
            is_transpose=is_transpose,
            is_depthwise=is_depthwise,
            function_name=function_name,
            shape_function=shape_func,
            is_first_op = True,
            exec_paths=exec_program,
        )

        func_call = FUNC_CALL_TEMPLATE.render(
            indent="  ",
            is_bias=is_bias,
            is_bias_add=is_bias_add,
            func_name=function_name,
            in_ptr="(void*)input",
            weight_ptr="(void*)weight",
            out_ptr="(void*)output",
            **func_call_extra_args,
            p_batch="&NI",
            p_out_ch="&CO",
            p_in_ch="&CI",
            p_kernel_h="&KH",
            p_kernel_w="&KW",
            p_in_h="&HI",
            p_in_w="&WI",
            p_out_batch="&NO",
            p_out_h="&HO",
            p_out_w="&WO",
            strideh="strideh",
            dilationh="dilationh",
            padh="padh",
            stridew="stridew",
            dilationw="dilationw",
            padw="padw",
        )
        benchmark = BENCHMARK_TEMPLATE.render(
            is_bias=is_bias,
            is_bias_add=is_bias_add,
            is_f16=(dtype == "f16"),
            is_f32=(dtype == "f32"),
            instance_name_base=instance_name_base,
            function_name=function_name,
            func_call=func_call,
            instance_name=instance_name,
        )

        profiler_benchmarks[function_name] = PROFILER_BENCHMARK_TEMPLATE.render(
            extra_header=extra_header,
            functions=function,
            benchmark=benchmark,
            instance_name=instance_name,
        )

        benchmark_instance = BENCHMARK_INSTANCE_TEMPLATE.render(
            indent="  ",
            conv_op_name=op_name,
            func_name=f"benchmark_{function_name}",
            ni="NI",
            hi="HI",
            wi="WI",
            ci="CI",
            co="CO",
            kh="KH",
            kw="KW",
            no="NO",
            ho="HO",
            wo="WO",
            strideh="SH",
            dilationh="DH",
            padh="PH",
            stridew="SW",
            dilationw="DW",
            padw="PW",
        )
        benchmark_instances.append(benchmark_instance)

        benchmark_decl = BENCHMARK_DECL_TEMPLATE.render(
            function_name=function_name,
        )
        benchmark_decls.append(benchmark_decl)

    shape_func = shape_template.render(
        indent="  ",
        dtype="int64_t ",
        div="/",
        x_dim0="batch",
        x_dim1="in_h",
        x_dim2="in_w",
        x_dim3="in_ch",
        w_dim0="out_ch",
        w_dim1="kernel_h",
        w_dim2="kernel_w",
        strideh="strideh",
        dilateh="dilationh",
        padh="padh",
        stridew="stridew",
        dilatew="dilationw",
        padw="padw",
    )
    profiler_main_code = PROFILER_MAIN_TEMPLATE.render(
        shape_func=shape_func,
        benchmark_decls="\n".join(benchmark_decls),
        benchmark_instances="\n".join(benchmark_instances),
    )

    code = {profiler_filename: profiler_main_code}
    for benchmark_filename, benchmark_code in profiler_benchmarks.items():
        code[benchmark_filename] = benchmark_code

    # FIXME: remove file_pairs once we have make -j ready for building
    # an entire graph
    file_pairs = []
    add_profiler(file_pairs, workdir, op_type, profiler_filename, code)
    # build
    return build_profiler(file_pairs)



def gen_function(
    func_attrs,
    exec_cond_template,
    shape_eval_template,
    shape_save_template,
    f_emit_instance="",
    is_bias=False,
    is_bias_add=False,
    is_transpose=False,
    is_depthwise=False,
    extra_header="",
):
    """Function definition codegen."""
    func_name = func_attrs["name"]
    exec_path = func_attrs["exec_path"]
    op_instance = func_attrs["op_instance"]

    backend_spec = RVVSpec()
    dtype = backend_spec.dtype_to_lib_type(func_attrs["inputs"][0]._attrs["dtype"])
    shape_eval_func = shape_eval_template.render(
        indent="  ",
        dtype="int64_t ",
        x_dim0="*batch",
        x_dim1="*in_h",
        x_dim2="*in_w",
        x_dim3="*in_ch",
        w_dim0="*out_ch",
        w_dim1="*kernel_h",
        w_dim2="*kernel_w",
        strideh="strideh",
        dilateh="dilationh",
        padh="padh",
        stridew="stridew",
        dilatew="dilationw",
        padw="padw",
        div="/",
    )
    shape_save_func = shape_save_template.render(
        indent="  ",
        y_dim0="*out_batch",
        y_dim1="*out_h",
        y_dim2="*out_w",
        y_dim3="*out_ch",
    )
    shape_func = shape_eval_func + shape_save_func
    program = ""
    for instance_idx, (op_name, op) in enumerate(op_instance.items()):
        program += emit_instance(op)
    match = re.search(r'(\d+)$', func_name)
    function = FUNCTION_TEMPLATE.render(
        is_bias=is_bias,
        is_bias_add=is_bias_add,
        is_transpose=is_transpose,
        is_depthwise=is_depthwise,
        function_name=func_name,
        is_first_op = (match.group(1) == '0' or match.group(1) == '1' or match.group(1) == '2'),
        shape_function=shape_func,
        exec_paths=program,
    )

    return SRC_TEMPLATE.render(
        extra_header=extra_header,
        functions=function,
    )


def gen_function_decl(
    func_attrs,
    is_bias=False,
    is_bias_add=False,
):
    func_name = func_attrs["name"]

    return FUNC_DECL_TEMPLATE.render(
        is_bias=is_bias,
        is_bias_add=is_bias_add,
        func_name=func_name,
    )


def gen_function_call(
    func_attrs,
    indent="  ",
    is_bias=False,
    is_bias_add=False,
    is_transpose=False,
):
    x = func_attrs["inputs"][0]
    xshape = x._attrs["shape"]
    w = func_attrs["inputs"][1]
    wshape = w._attrs["shape"]
    y = func_attrs["outputs"][0]
    yshape = y._attrs["shape"]

    func_call_extra_args = {}
    if is_bias:
        b = func_attrs["inputs"][2]
        func_call_extra_args = {
            "bias_ptr": b._attrs["name"],
        }
    elif is_bias_add:
        b = func_attrs["inputs"][2]
        r = func_attrs["inputs"][3]
        func_call_extra_args = {
            "bias_ptr": b._attrs["name"],
            "res_ptr": r._attrs["name"],
        }

    out_ch = wshape[-1]._attrs["name"] if is_transpose else wshape[0]._attrs["name"]
    return FUNC_CALL_TEMPLATE.render(
        is_bias=is_bias,
        is_bias_add=is_bias_add,
        func_name=func_attrs["name"],
        in_ptr=x._attrs["name"],
        weight_ptr=w._attrs["name"],
        out_ptr=y._attrs["name"],
        **func_call_extra_args,
        p_batch="&" + xshape[0]._attrs["name"],
        p_out_ch="&" + out_ch,
        p_in_ch="&" + xshape[3]._attrs["name"],
        p_kernel_h="&" + wshape[1]._attrs["name"],
        p_kernel_w="&" + wshape[2]._attrs["name"],
        p_in_h="&" + xshape[1]._attrs["name"],
        p_in_w="&" + xshape[2]._attrs["name"],
        p_out_batch="&" + yshape[0]._attrs["name"],
        p_out_h="&" + yshape[1]._attrs["name"],
        p_out_w="&" + yshape[2]._attrs["name"],
        strideh=(
            func_attrs["stride"]
            if isinstance(func_attrs["stride"], int)
            else func_attrs["stride"][0]
        ),
        dilationh=(
            func_attrs["dilate"]
            if isinstance(func_attrs["dilate"], int)
            else func_attrs["dilate"][0]
        ),
        padh=(
            func_attrs["pad"]
            if isinstance(func_attrs["pad"], int)
            else func_attrs["pad"][0]
        ),
        stridew=(
            func_attrs["stride"]
            if isinstance(func_attrs["stride"], int)
            else func_attrs["stride"][1]
        ),
        dilationw=(
            func_attrs["dilate"]
            if isinstance(func_attrs["dilate"], int)
            else func_attrs["dilate"][1]
        ),
        padw=(
            func_attrs["pad"]
            if isinstance(func_attrs["pad"], int)
            else func_attrs["pad"][1]
        ),
        indent=indent,
    )


def _cal_align_ab(x_shape: List[int], dtype="float16") -> int:
    """Returns input alignment."""
    k = x_shape[3]  # CI
    return alignment.find_max_alignment(k, dtype)


def function_filter(
    cfg,
    func_attrs,
    x_shape,
):
    """Generates function filter.

    Parameters
    ----------
    cfg: str
        The filename generated for profiler.
    func_attrs : Dict
        Stores the operation attributes.
    offset: Int
        Offset of split(cfg,"_") to get conv2d specialization

    Returns
    -------
    bool
        If input cfg should be filtered.
    """
    from cpu_lib.conv2d_operation import Conv2DSpecialization
    # _LOGGER.info(f"func_attrs = {func_attrs}, x_shape =  {x_shape}")
    return True
