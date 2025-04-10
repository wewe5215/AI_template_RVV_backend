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
Elementwise codegen for RVV.
"""
import jinja2
import os
from typing import Any, Dict
from dataclasses import dataclass
from aitemplate.backend import registry
from aitemplate.backend.backend_spec import RVVSpec, BackendSpec
from aitemplate.backend.common import elementwise_common
from aitemplate.backend.target import Target
from aitemplate.backend.common import tensor_accessor_codegen
from aitemplate.backend.target import Target
from typing import Any, Dict, List, Optional, Tuple
from aitemplate.compiler.base import IntImm, IntVar, JaggedIntVar, Operator, Tensor
from aitemplate.compiler.tensor_accessor import TensorAccessor
from aitemplate.utils import alignment as alignment_utils, shape_utils
backend_spec = RVVSpec()
import math

FUNC_DECL_INPUT_PARAM_TEMPLATE = jinja2.Template("const void* input{{idx}}")
FUNC_DECL_OUTPUT_PARAM_TEMPLATE = jinja2.Template("void* output{{idx}}")
INPUT_INIT_TEMPLATE = jinja2.Template(
    "const {{read_t}}* input{{idx}}_tmp = reinterpret_cast<const {{read_t}}*>(input{{idx}})"
)
OUTPUT_INIT_TEMPLATE = jinja2.Template(
    "{{read_t}}* output{{idx}}_tmp = reinterpret_cast<{{read_t}}*>(output{{idx}})"
)
KERNEL_TMP_INPUT_TEMPLATE = jinja2.Template("input{{idx}}_tmp[s]")
KERNEL_TMP_INPUT_TEMPLATE_FOR_INPUT = jinja2.Template("input{{idx}}_tmp[idx]")
KERNEL_TMP_OUTPUT_TEMPLATE = jinja2.Template("output{{idx}}_tmp[idx]")
FUNC_TEMPLATE = jinja2.Template(
    """
{{head}}

#include "jagged.h"

namespace {

{{constant}}

{{custom_libs}}


}  // namespace
#include <cmath>
void {{func_name}}({{output_params}}, {{input_params}}, {{dynamic_dims_decl}} {{offsets_decl}} {{index_type}} N, {{index_type}} H, {{index_type}} W, {{index_type}} C) {
    if (C == 0 || N == 0 || H == 0 || W == 0) {
      return;
    }
    {{output_reinterpret}};
    {{input_reinterpret}};
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < H; ++j) {
            for (int k = 0; k < W; ++k) {
                for (int s = 0; s < C; ++s) {
                    // Compute the linear index for the input/output tensor element
                    int idx = ((i * H + j) * W + k) * C + s;
                    {{fused_funcs}}
                }
            }
        }
    }
}
    """
)

FUNC_DECL_TEMPLATE = jinja2.Template(
    """
void {{func_name}}({{output_params}}, {{input_params}}, {{dynamic_dims}} {{offsets}} {{index_type}} N, {{index_type}} H, {{index_type}} W, {{index_type}} C);
    """
)

FUNC_CALL_TEMPLATE = jinja2.Template(
    """
{{indent}}{
    {{indent}}{{func_name}}({{output_params}}, {{input_params}}, {{dynamic_dims}} {{offsets}} {{int_var_decl_str}});
{{indent}}}
    """
)


@dataclass
class ElementwiseMetaData:
    func_name: str
    op_t: str
    args: List[Tensor]
    outputs: List[Tensor]


@dataclass
class FusedElementwiseMetaData:
    # Input / output Tensors and TensorAccessors.
    inputs: List[Tensor]
    outputs: List[Tensor]
    input_accessors: List[TensorAccessor]
    output_accessors: List[TensorAccessor]

    # Original input / output Tensors before graph transformation.
    # Kept here for elementwise -> fused elementwise Tensor mapping.
    original_inputs: List[Tensor]
    original_outputs: List[Tensor]

    # Holds the largest read type for the fused kernel.
    # This is equivalent to write_t in the current implementation.
    # This is used to determine N_ELEMENTS_PER_THREAD.
    max_read_t: str

    # Holds the read_t for each input of the fused kernel.
    # Note: read_types is only used for a small optimization for last_dim input broadcasting.
    # General mixed read_types are not supported (which requires multiple get_strided_inputs calls).
    read_types: List[str]

    op_t: str
    data_t: str

    input_broadcast_sizes: List[List[IntVar]]
    dynamic_dims: List[IntVar]
    sub_funcs: List[ElementwiseMetaData]

    # this flag specifies if the jagged and mixed inputs need
    # separate indexing logic within the generated kernel code.
    # this typically happens when the shape of at least one of
    # the dense inputs overlaps with one or more jagged dimensions
    # of the jagged inputs (all jagged inputs are assume to have
    # the same rank and JaggedIntVar / jagged dimensions).
    mixed_jagged_dense_indexing: bool = False

    # this attribute is relevant only when mixed_jagged_dense_indexing
    # is True. it specifies the smallest rectangular volume that fits
    # all inputs (jagged and dense) and outputs (jagged): i.e., the maximum
    # rectangular volume that the jagged output Tensor can fit in.
    # the output_volume list, therefore, can't contain a JaggedIntVar, as
    # the latter in the jagged output Tensor shape is "expanded" to the
    # list with `batch_dim` followed by an IntImm for each jagged dim.
    output_volume: Optional[List[IntVar]] = None

    # this attribute is relevant only when mixed_jagged_dense_indexing
    # is True. wether the jagged index space implementation (as opposed
    # to the dense index space implementation) should be use to compute
    # the dense_idx and jagged_idx separately in the mixed jagged /
    # dense indexing cases. the dense space indexing runs over the
    # (dense) output volume and computes jagged_idx from dense_idx.
    # the jagged space indexing runs over the jagged output shape
    # and computes the dense_inx from jagged_idx (with binary search).
    use_jagged_space_indexing: bool = False

    # whether all intermediate computations should be performed in float32
    use_fp32_acc: bool = False
    # the float32 type of the used back-end
    float32_t: str = "float"


def gen_function_single_thread(
    fused_func_metadata,
    input_names,
    output_names,
    type_converter,
) -> str:
    """Per thread elementwise function codegen."""
    tensor_to_expr: Dict[Tensor, str] = {}
    float32_t = fused_func_metadata.float32_t
    body = ""

    for tensor, name in zip(fused_func_metadata.original_inputs, input_names):
        if fused_func_metadata.use_fp32_acc and fused_func_metadata.op_t != float32_t:
            input_converter = type_converter.get(fused_func_metadata.op_t).get(
                float32_t
            )
            name = "{}({})".format(input_converter, name)
        tensor_to_expr[tensor] = name

    tmp_output_idx: int = 0
    for func_metadata in fused_func_metadata.sub_funcs:
        params: List[str] = []
        input_converter = None
        output_converter = None
        func_op_t = func_metadata.op_t

        # intermediate input / output converters are not
        # required when doing all computation in float32
        if not fused_func_metadata.use_fp32_acc:
            if func_op_t != fused_func_metadata.op_t:
                input_converter = type_converter.get(fused_func_metadata.op_t).get(
                    func_op_t
                )
                output_converter = type_converter.get(func_op_t).get(
                    fused_func_metadata.op_t
                )
                assert (
                    input_converter is not None
                ), "Unsupported convertion from {} to {}".format(
                    fused_func_metadata.op_t, func_op_t
                )
                assert (
                    output_converter is not None
                ), "Unsupported convertion from {} to {}".format(
                    func_op_t, fused_func_metadata.op_t
                )

        for arg in func_metadata.args:
            if arg in tensor_to_expr:
                param = tensor_to_expr[arg]
                params.append(
                    "{}({})".format(input_converter, param)
                    if input_converter is not None
                    else param
                )
            elif arg.is_a_const_num():
                arg_str = ""
                if math.isinf(arg._attrs["value"]):
                    arg_str = "CUDART_INF_F"
                else:
                    arg_str = str(arg._attrs["value"])
                if func_op_t[-1] == "2":
                    params.append(
                        "{}({},{})".format(
                            func_op_t,
                            arg_str,
                            arg_str,
                        )
                    )
                else:
                    params.append("{}({})".format(func_op_t, arg_str))
            else:
                raise RuntimeError(
                    "Cannot generate expression for node {}, ops: {}".format(
                        arg, func_metadata
                    )
                )
        assert (
            len(func_metadata.outputs) == 1
        ), "Operator has more than 1 output! Operator: {}".format(func_metadata)

        output = func_metadata.outputs[0]
        func_def = "{}({})".format(func_metadata.func_name, ",".join(params))
        func_def = (
            "{}({})".format(output_converter, func_def)
            if output_converter is not None
            else func_def
        )
        if len(output._attrs["dst_ops"]) > 1:
            name = "tmp_" + (str)(tmp_output_idx)
            tmp_output_idx += 1
            temp_t = (
                float32_t
                if fused_func_metadata.use_fp32_acc
                else fused_func_metadata.op_t
            )
            body += "{} {} = {};\n".format(temp_t, name, func_def)
            tensor_to_expr[output] = name
        else:
            tensor_to_expr[output] = func_def

    for tensor, name in zip(fused_func_metadata.original_outputs, output_names):
        if tensor not in tensor_to_expr:
            raise RuntimeError(
                "Cannot generate expression for node {}, outputs: {}".format(
                    tensor, fused_func_metadata.original_outputs
                )
            )
        expr = tensor_to_expr[tensor]
        if fused_func_metadata.use_fp32_acc and fused_func_metadata.op_t != float32_t:
            output_converter = type_converter.get(float32_t).get(
                fused_func_metadata.op_t
            )
            expr = "{}({})".format(output_converter, expr)
        body += "{} = {};\n".format(name, expr)

    return body


def _get_sub_func_metadata(
    ops: List[Operator],
    data_t: str,
    op_t: str,
    backend_spec: BackendSpec,
    float32_t: str,
) -> Tuple[List[ElementwiseMetaData], str]:
    use_fp32_acc = Target.current()._kwargs.get("elementwise_use_fp32_acc", False)
    if use_fp32_acc:
        # vectorized op types are not allowed when all
        # intermediate computation is done in float32
        op_t = data_t
        # only float functions must be used
        candidate_op_types = [float32_t]
    else:
        candidate_op_types = backend_spec.get_candidate_op_types(op_t)
        func_enums = []
        for op in ops:
            func_enum = op._attrs["func"]
            func_enums.append(func_enum)
            funcs = backend_spec.func_enum_to_func_name.get(func_enum)
            if funcs is None:
                raise NotImplementedError("Func {} is not supported!".format(func_enum))
            for candidate_op_t in candidate_op_types:
                func_name = funcs.get(candidate_op_t)
                if func_name is not None:
                    candidate_op_types = backend_spec.get_candidate_op_types(
                        candidate_op_t
                    )
                    break
        if len(candidate_op_types) == 0:
            raise RuntimeError(
                "Cannot find a common backend data type! candidate_op_types: {}, op_t: {}.".format(
                    candidate_op_types, op_t
                )
            )
        if op_t in set(candidate_op_types):
            op_t = candidate_op_types[0]
        else:
            op_t = data_t
            candidate_op_types = backend_spec.get_candidate_op_types(op_t)

    sub_func_metadata = []
    for op in ops:
        func_enum = op._attrs["func"]
        funcs = backend_spec.func_enum_to_func_name.get(func_enum)
        func_name = None
        func_op_t = None
        for candidate_op_t in candidate_op_types:
            func_name = funcs.get(candidate_op_t)
            if func_name is not None:
                func_op_t = candidate_op_t
                break
        if func_name is None:
            raise NotImplementedError(
                "Unsupported func {} and op type {}!".format(func_enum, op_t)
            )
        sub_func_metadata.append(
            ElementwiseMetaData(
                func_name, func_op_t, op._attrs["args"], op._attrs["outputs"]
            )
        )

    return sub_func_metadata, op_t, use_fp32_acc


def _is_jagged_shape(shape: List[IntVar]) -> bool:
    """Whether the given shape is a shape of a jagged Tensor."""
    return len(shape) > 0 and isinstance(shape[0], JaggedIntVar)


def _get_input_alignments(
    input_accessors: List[TensorAccessor],
    input_broadcast_sizes: List[Optional[List[IntVar]]],
    max_num_rightmost_dims_considered_for_alignments: int,
    output_shape: List[IntVar],
    dtype: str,
    global_max_alignment: int,
) -> List[int]:
    # Broadcasts need to be handled carefully.
    # We have a hacky optimization for last-dim broadcasting:
    # The element is read once, and broadcasted multiple times.
    # However, we don't support reading more than 1 element for broadcasting.
    #
    # Consider following cases:
    # X1[2, 1, 1]
    # X2[1, 1, 2]
    # X3[1, 2, 1]
    # We do not support global_max_alignment 8 (reading two X1, X2, X3 per thread).
    # We only support global_max_alignment 2, so that we make sure each thread
    # reads at most 1 element for broadcasting.

    # Update global_max_alignment based on broadcasting rules,
    # and find max_alignments for each input.
    alignments = [None] * len(input_broadcast_sizes)
    for i, input_broadcast_size in enumerate(input_broadcast_sizes):
        if input_broadcast_size is not None:
            prev_is_broadcast = None
            for j in range(max_num_rightmost_dims_considered_for_alignments):
                is_broadcast = input_broadcast_size[-j - 1] != output_shape[-j - 1]
                if (
                    not is_broadcast
                    and input_broadcast_size[-j - 1] == IntImm(1)
                    and prev_is_broadcast is None
                ):
                    # Skip last-dim 1s if the output shape is the same.
                    is_broadcast = None
                if prev_is_broadcast is None:
                    prev_is_broadcast = is_broadcast
                    if is_broadcast:
                        # Update alignment for last-dim broadcasting cases.
                        alignments[i] = 1
                elif prev_is_broadcast != is_broadcast:
                    alignment = alignment_utils.find_max_alignment(
                        shape_utils.get_num_rightmost_static_elements(output_shape, j),
                        dtype,
                    )
                    # Update global_max_alignment when is_broadcast is not the
                    # same as prev_is_broadcast.
                    global_max_alignment = min(global_max_alignment, alignment)
                    if not prev_is_broadcast:
                        # Update alignment for mid-dim broadcasting cases.
                        alignments[i] = alignment
                    break

    # Cap alignments based on global_max_alignment.
    alignments = [
        (
            min(alignment, global_max_alignment)
            if alignment is not None
            else global_max_alignment
        )
        for alignment in alignments
    ]
    return alignments


def _get_input_broadcast_sizes(
    input_accessors, output_accessors, mixed_jagged_dense_indexing, output_volume
) -> List[Optional[List[IntVar]]]:
    input_broadcast_sizes = []
    for input_accessor in input_accessors:
        input_shape = input_accessor.original_shapes

        if mixed_jagged_dense_indexing:
            if _is_jagged_shape(input_shape):
                # broadcast the jagged input shape against the output_shape:
                # in a mixed jagged / dense op the output_shape is the shape
                # of the output jagged Tensor
                output_shape = output_accessors[0].original_shapes
            else:
                # broadcast the dense input shape against the output_volume,
                # as the dense indexing will be done in the output_volume
                output_shape = output_volume
        else:
            # treat all outputs as dense: use output_shape for broadcasting
            output_shape = output_accessors[0].original_shapes

        broadcastable, _ = shape_utils.get_broadcast_max_shape(
            output_shape, input_shape
        )
        if not broadcastable:
            raise RuntimeError(
                "Input shape {} is not compatible with output shape {}!".format(
                    input_shape, output_shape
                )
            )
        extended_input_shape = list(input_shape)
        if input_shape == output_shape:
            input_broadcast_sizes.append(None)
        if input_shape != output_shape:
            extended_input_shape = [IntImm(1)] * len(output_shape)
            extended_input_shape[len(output_shape) - len(input_shape) :] = input_shape
            input_broadcast_sizes.append(extended_input_shape)
    return input_broadcast_sizes


def _get_alignments_and_broadcast_sizes(
    dtype: str,
    input_accessors: List[TensorAccessor],
    output_accessors: List[TensorAccessor],
    mixed_jagged_dense_indexing: bool,
    output_volume: Optional[List[IntVar]],
) -> Tuple[List[int], List[Optional[List[IntVar]]]]:
    """
    Returns Tuple(input_alignments, input_broadcast_sizes)
    """
    # Handle input broadcast.
    output_shape = output_accessors[0].original_shapes

    input_broadcast_sizes = _get_input_broadcast_sizes(
        input_accessors, output_accessors, mixed_jagged_dense_indexing, output_volume
    )

    # In the mixed jagged / dense indexing case, the number of the
    # rightmost non-broadcated static dimensions of the dense inputs
    # to be considered for vectorization can't be larger than the
    # number of the jagged output's inner dimensions (i.e., the
    # dimensions following the JaggedIntVar). Otherwise, there may
    # be an overlap with the jagged dimensions, in which case the
    # vectorization can break.
    max_num_rightmost_dims_considered_for_alignments = (
        len(output_shape) - 1 if mixed_jagged_dense_indexing else len(output_shape)
    )

    # We do not support mixed input / output alignments except for last dim broadcast.
    # The global_max_alignment is the min value of:
    #     1) input shape alignments (with input broadcast in consideration);
    #     2) input tensor accessor alignments (strides, offsets);
    #     3) output shape alignments;
    #     4) output tensor accessor alignments (strides, offsets);

    # Now calculate global_max_alignment based on 2), 3) and 4) first.
    global_max_alignment = min(
        alignment_utils.find_max_alignment(
            shape_utils.get_num_rightmost_static_elements(
                output_shape, max_num_rightmost_dims_considered_for_alignments
            ),
            dtype,
        ),
        tensor_accessor_codegen.find_max_alignment_for_accessors(
            dtype, input_accessors
        ),
        tensor_accessor_codegen.find_max_alignment_for_accessors(
            dtype, output_accessors
        ),
    )

    # Now calculate global_max_alignment based on 1).
    # Also calculate input alignments.
    input_alignments = _get_input_alignments(
        input_accessors,
        input_broadcast_sizes,
        max_num_rightmost_dims_considered_for_alignments,
        output_shape,
        dtype,
        global_max_alignment,
    )
    return input_alignments, input_broadcast_sizes


def get_dynamic_dims(*shapes: List[List[IntVar]]) -> List[IntVar]:
    res = {}
    for shape in shapes:
        for dim in shape:
            if not isinstance(dim, IntImm):
                res[dim._attrs["name"]] = dim
                if isinstance(dim, JaggedIntVar):
                    # the batch_dim and the JaggedDim bounds within the JaggedIntVar
                    # may not be present directly in other input / output shapes,
                    # so we're adding it here separately
                    batch_dim = dim.batch_dim()
                    if not isinstance(batch_dim, IntImm):
                        res[batch_dim._attrs["name"]] = batch_dim
                    for jagged_dim in dim.jagged_dims():
                        min_value = jagged_dim.min_value()
                        if not isinstance(min_value, IntImm):
                            res[min_value._attrs["name"]] = min_value
                        max_value = jagged_dim.max_value()
                        if not isinstance(max_value, IntImm):
                            res[max_value._attrs["name"]] = max_value

    return list(res.values())


def _get_mixed_jagged_dense_config(
    input_accessors: List[TensorAccessor],
    output_accessors: List[TensorAccessor],
) -> Tuple[bool, List[IntVar], bool]:
    """
    Returns Tuple(
        mixed_jagged_dense_indexing,
        output_volume,
        use_jagged_space_indexing,
    )
    """
    # all output shapes are assumed to be the same
    output_shape = output_accessors[0].original_shapes
    input_shapes = [acc.original_shapes for acc in input_accessors]
    jagged_input_shapes = [s for s in input_shapes if _is_jagged_shape(s)]
    dense_input_shapes = [s for s in input_shapes if not _is_jagged_shape(s)]

    if not jagged_input_shapes or not dense_input_shapes:
        # there are either only dense inputs or only jagged inputs:
        # in both cases all inputs will be treated as dense, because
        # the JaggedIntVars and ranks of all the jagged inputs are
        # assumed to be the same
        return False, None, False

    jagged_rank = len(jagged_input_shapes[0])
    max_dense_rank = max(len(s) for s in dense_input_shapes)

    if max_dense_rank <= jagged_rank - 1:
        # the longest dense shape does not overlap with the jagged dims:
        # the jagged inputs can be treated as dense, meaning that the
        # total_length of the jagged inputs (not overlapping with the
        # dense inputs' shapes) will be treated as a single dense dim
        return False, None, False

    jagged_int_var = output_shape[0]
    jagged_max_dense_prefix_shape = jagged_int_var.get_max_dense_shape()
    jagged_suffix_shape = output_shape[1:]
    output_volume = jagged_max_dense_prefix_shape + jagged_suffix_shape

    use_jagged_space_indexing = Target.current()._kwargs.get(
        "use_jagged_space_indexing", False
    )

    # because at least one of the dense inputs overlap with the
    # JaggedIntVar of the jagged inputs, jagged and dense inputs
    # will need different indexing in the generated kernel.
    # output_volume is the smallest rectangular volume fitting
    # all the input (jagged and dense) and outputs (jagged).
    return True, output_volume, use_jagged_space_indexing


def _parse_func_metadata(
    ops: List[Operator],
    inputs: List[Tensor],
    outputs: List[Tensor],
    input_accessors: List[TensorAccessor],
    output_accessors: List[TensorAccessor],
    original_inputs: List[Tensor],
    original_outputs: List[Tensor],
    backend_spec: BackendSpec,
) -> FusedElementwiseMetaData:
    (
        mixed_jagged_dense_indexing,
        output_volume,
        use_jagged_space_indexing,
    ) = _get_mixed_jagged_dense_config(
        input_accessors,
        output_accessors,
    )
    dtype = inputs[0]._attrs["dtype"]
    (input_alignments, input_broadcast_sizes) = _get_alignments_and_broadcast_sizes(
        dtype,
        input_accessors,
        output_accessors,
        mixed_jagged_dense_indexing,
        output_volume,
    )
    max_read_type = backend_spec.get_elementwise_read_backend_type(
        max(input_alignments), dtype
    )
    read_types = [
        backend_spec.get_elementwise_read_backend_type(alignment, dtype)
        for alignment in input_alignments
    ]

    # It's safe to use the maximum alignment for determine op_type, because
    # smaller inputs (i.e. those being broadcasted) will be placed into a
    # larger tmp variable which is valid for selected op_type.
    op_type = backend_spec.get_elementwise_op_backend_type(max(input_alignments), dtype)
    data_type = backend_spec.dtype_to_backend_type(dtype)
    float32_type = backend_spec.dtype_to_backend_type("float32")
    sub_func_metadata, op_type, use_fp32_acc = _get_sub_func_metadata(
        ops,
        data_type,
        op_type,
        backend_spec,
        float32_type,
    )
    dynamic_dims = get_dynamic_dims(*[acc.original_shapes for acc in output_accessors])

    return FusedElementwiseMetaData(
        inputs,
        outputs,
        input_accessors,
        output_accessors,
        original_inputs,
        original_outputs,
        max_read_type,
        read_types,
        op_type,
        data_type,
        input_broadcast_sizes,
        dynamic_dims,
        sub_func_metadata,
        mixed_jagged_dense_indexing,
        output_volume,
        use_jagged_space_indexing,
        use_fp32_acc,
        float32_type,
    )


def gen_int_var_decl_str(
    int_vars: List[IntVar],
) -> str:
    res = []
    for int_var in int_vars:
        if isinstance(int_var, IntImm):
            res.append(str(int_var._attrs["values"][0]))
        elif isinstance(int_var, IntVar):
            res.append(int_var._attrs["name"])
        else:
            raise RuntimeError(
                "A dim must be an IntVar! Current type: {}".format(type(int_var))
            )

    return " , ".join(res) if res else "1"

def gen_int_var_product_str(
    int_vars: List[IntVar],
) -> str:
    res = []
    for int_var in int_vars:
        if isinstance(int_var, IntImm):
            res.append(str(int_var._attrs["values"][0]))
        elif isinstance(int_var, IntVar):
            res.append(int_var._attrs["name"])
        else:
            raise RuntimeError(
                "A dim must be an IntVar! Current type: {}".format(type(int_var))
            )

    return " * ".join(res) if res else "1"


def gen_dynamic_dim_str(
    index_type: str,
    dynamic_dims: List[IntVar],
    has_type: bool,
) -> str:
    type_str = index_type + " " if has_type else ""
    res = ", ".join([type_str + dim._attrs["name"] for dim in dynamic_dims])
    if res:
        res += ", "

    return res


def gen_offsets_str(
    jagged_int_var: JaggedIntVar,
    has_type: bool,
    const_ref: bool,
    name: Optional[str] = None,
) -> str:
    offsets_var_name = jagged_int_var.offsets_var_name()
    offsets_struct_type = jagged_int_var.offsets_struct_type()

    ref_prefix = "const " if const_ref else ""
    ref_suffix = "&" if const_ref else ""
    arg_type = f"{ref_prefix}{offsets_struct_type}{ref_suffix} " if has_type else ""
    arg_name = name if name is not None else offsets_var_name
    offsets = f"{arg_type}{arg_name}, "

    return offsets


def _gen_offsets_str_from_metadata(
    fused_elementwise_metadata: FusedElementwiseMetaData,
    has_type: bool,
    const_ref: bool,
    name: Optional[str] = None,
):
    if fused_elementwise_metadata.mixed_jagged_dense_indexing:
        inputs = fused_elementwise_metadata.inputs
        jagged_input = [t for t in inputs if t.is_jagged()][0]
        jagged_int_var = jagged_input._attrs["shape"][0]

        return gen_offsets_str(
            jagged_int_var=jagged_int_var,
            has_type=has_type,
            const_ref=const_ref,
            name=name,
        )
    else:
        return ""


def _gen_num_elements_calculator(
    fused_elementwise_metadata: FusedElementwiseMetaData,
) -> str:
    if fused_elementwise_metadata.mixed_jagged_dense_indexing:
        if fused_elementwise_metadata.use_jagged_space_indexing:
            # for the jagged space indexing, the num_elements
            # is the number of elements in the output jagged Tensor, hence
            # the usage of the output shape here, not the output volume
            return gen_int_var_product_str(
                fused_elementwise_metadata.output_accessors[0].original_shapes,
            )
        else:
            # for the dense space indexing, the num_elements
            # is the number of elements in the output volume: the smallest
            # rectangular volume that fits the output jagged Tensor, hence
            # the usage of the output volume here, not the output shape
            return gen_int_var_product_str(
                fused_elementwise_metadata.output_volume,
            )
    else:
        # all inputs and outputs are treated as dense:
        # use the output shape for computing num_elements
        return gen_int_var_product_str(
            fused_elementwise_metadata.output_accessors[0].original_shapes,
        )



@registry.reg("rvv.fused_elementwise.gen_function")
def fused_elementwise_gen_function(func_attrs: Dict[str, Any]) -> str:
    """Generates fused_elementwise function definition."""

    custom_libs = Target.current().get_custom_libs(
        os.path.dirname(__file__), "custom_math.h"
    )
    ops = func_attrs["elementwise_ops"]
    inputs = func_attrs["inputs"]
    outputs = func_attrs["outputs"]
    input_accessors = func_attrs["input_accessors"]
    output_accessors = func_attrs["output_accessors"]
    original_inputs = func_attrs["original_inputs"]
    original_outputs = func_attrs["original_outputs"]
    fused_elementwise_metadata = _parse_func_metadata(
        ops,
        inputs,
        outputs,
        input_accessors,
        output_accessors,
        original_inputs,
        original_outputs,
        backend_spec,
    )
    # Dump data types into func_attr for testing purpose.
    func_attrs["max_read_t"] = fused_elementwise_metadata.max_read_t
    # Fused inputs may not be in the same order as the inputs passed to each
    # elementwise op, so we save a tuple. Note that this attribute is different
    # from the read_types field of FusedElementwiseMetaData, where each "read_t"
    # maps to the input at the same index. The "read_types" attribute is only
    # used for testing purpose.
    func_attrs["read_types"] = [
        (inp._attrs["name"], read_t)
        for (inp, read_t) in zip(inputs, fused_elementwise_metadata.read_types)
    ]
    func_attrs["op_t"] = fused_elementwise_metadata.op_t
    func_attrs["data_t"] = fused_elementwise_metadata.data_t

    tensor_accessor_lib = tensor_accessor_codegen.get_libs()
    tensor_accessor_lib_str = "\n\n" + tensor_accessor_lib + "\n\n"

    output_params_decl = ",".join(
        [
            FUNC_DECL_OUTPUT_PARAM_TEMPLATE.render(idx=i)
            for i, _ in enumerate(fused_elementwise_metadata.outputs)
        ]
    )
    input_params_decl = ",".join(
        [
            FUNC_DECL_INPUT_PARAM_TEMPLATE.render(idx=i)
            for i, _ in enumerate(fused_elementwise_metadata.inputs)
        ]
    )
    output_reinterpret = ";\n".join(
        [
            OUTPUT_INIT_TEMPLATE.render(
                read_t=fused_elementwise_metadata.max_read_t, idx=i
            )
            for i, _ in enumerate(fused_elementwise_metadata.outputs)
        ]
    )
    input_reinterpret = ";\n".join(
        [
            INPUT_INIT_TEMPLATE.render(
                read_t=fused_elementwise_metadata.read_types[i], idx=i
            )
            for i, _ in enumerate(fused_elementwise_metadata.inputs)
        ]
    )
    constant = ""
    input_names = []
    for i, _ in enumerate(fused_elementwise_metadata.inputs):
        if i != 1:
            input_names.append(KERNEL_TMP_INPUT_TEMPLATE.render(idx=i))
        else:
            input_names.append(KERNEL_TMP_INPUT_TEMPLATE_FOR_INPUT.render(idx=i))
    output_names = [
        KERNEL_TMP_OUTPUT_TEMPLATE.render(idx=i)
        for i, _ in enumerate(fused_elementwise_metadata.outputs)
    ]
    fused_funcs = gen_function_single_thread(
        fused_elementwise_metadata,
        input_names,
        output_names,
        backend_spec.backend_datatype_convertors,
    )
    function = FUNC_TEMPLATE.render(
        prefix=backend_spec.prefix,
        index_type=backend_spec.index_type,
        head=backend_spec.header_src_template.render(extra_header=""),
        constant=constant,
        custom_libs=custom_libs,
        tensor_accessor_lib=tensor_accessor_lib_str,
        kernel_function="",
        func_name=func_attrs["name"],
        output_params=output_params_decl,
        input_params=input_params_decl,
        dynamic_dims_decl=gen_dynamic_dim_str(
            backend_spec.index_type,
            fused_elementwise_metadata.dynamic_dims,
            has_type=True,
        ),
        dynamic_dims_call=gen_dynamic_dim_str(
            backend_spec.index_type,
            fused_elementwise_metadata.dynamic_dims,
            has_type=False,
        ),
        offsets_decl=_gen_offsets_str_from_metadata(
            fused_elementwise_metadata,
            has_type=True,
            # the offsets are passed
            # by const reference to the function
            const_ref=True,
            name="offsets",
        ),
        offsets_call=_gen_offsets_str_from_metadata(
            fused_elementwise_metadata,
            has_type=False,
            const_ref=False,
            name="offsets",
        ),
        output_reinterpret=output_reinterpret,
        input_reinterpret=input_reinterpret,
        fused_funcs=fused_funcs
    )
    return function


@registry.reg("rvv.fused_elementwise.func_decl")
def fused_elementwise_gen_function_decl(func_attrs):
    """Generates fused_elementwise function declaration."""
    """Generates fused_elementwise function declaration."""

    func_name = func_attrs["name"]
    ops = func_attrs["elementwise_ops"]
    inputs = func_attrs["inputs"]
    outputs = func_attrs["outputs"]
    input_accessors = func_attrs["input_accessors"]
    output_accessors = func_attrs["output_accessors"]
    original_inputs = func_attrs["original_inputs"]
    original_outputs = func_attrs["original_outputs"]
    fused_elementwise_metadata = _parse_func_metadata(
        ops,
        inputs,
        outputs,
        input_accessors,
        output_accessors,
        original_inputs,
        original_outputs,
        backend_spec,
    )
    output_params_decl = ",".join(
        [
            FUNC_DECL_OUTPUT_PARAM_TEMPLATE.render(idx=i)
            for i, _ in enumerate(fused_elementwise_metadata.outputs)
        ]
    )
    input_params_decl = ",".join(
        [
            FUNC_DECL_INPUT_PARAM_TEMPLATE.render(idx=i)
            for i, _ in enumerate(fused_elementwise_metadata.inputs)
        ]
    )

    function_decl = FUNC_DECL_TEMPLATE.render(
        prefix=backend_spec.prefix,
        index_type=backend_spec.index_type,
        func_name=func_name,
        output_params=output_params_decl,
        input_params=input_params_decl,
        dynamic_dims=gen_dynamic_dim_str(
            backend_spec.index_type,
            fused_elementwise_metadata.dynamic_dims,
            has_type=True,
        ),
        offsets=_gen_offsets_str_from_metadata(
            fused_elementwise_metadata,
            has_type=True,
            const_ref=True,
            name="offsets",
        ),
    )
    return function_decl


@registry.reg("rvv.fused_elementwise.func_call")
def fused_elementwise_gen_function_call(func_attrs, indent):
    """Generates fused_elementwise function call."""

    ops = func_attrs["elementwise_ops"]
    inputs = func_attrs["inputs"]
    outputs = func_attrs["outputs"]
    input_accessors = func_attrs["input_accessors"]
    output_accessors = func_attrs["output_accessors"]
    original_inputs = func_attrs["original_inputs"]
    original_outputs = func_attrs["original_outputs"]
    fused_elementwise_metadata = _parse_func_metadata(
        ops,
        inputs,
        outputs,
        input_accessors,
        output_accessors,
        original_inputs,
        original_outputs,
        backend_spec,
    )

    output_params = ",".join([output._attrs["name"] for output in outputs])
    input_params = ",".join([input._attrs["name"] for input in inputs])

    return FUNC_CALL_TEMPLATE.render(
        func_name=func_attrs["name"],
        index_type=backend_spec.index_type,
        calculate_n=_gen_num_elements_calculator(
            fused_elementwise_metadata,
        ),
        int_var_decl_str=gen_int_var_decl_str(
            fused_elementwise_metadata.output_accessors[0].original_shapes,
        ),
        output_params=output_params,
        input_params=input_params,
        dynamic_dims=gen_dynamic_dim_str(
            backend_spec.index_type,
            fused_elementwise_metadata.dynamic_dims,
            has_type=False,
        ),
        offsets=_gen_offsets_str_from_metadata(
            fused_elementwise_metadata,
            has_type=False,
            const_ref=False,
        ),
        indent=indent,
    )
