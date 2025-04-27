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
RVV concatenate function
"""
from aitemplate.backend.target import Target
from aitemplate.backend import registry
from aitemplate.compiler.base import IntImm
from aitemplate.backend.backend_spec import RVVSpec
from aitemplate.backend.common import concatenate_common
from aitemplate.compiler.ops.tensor import concatenate
from aitemplate.utils.shape_utils import is_empty_rank1_tensor
import jinja2
from copy import deepcopy

from typing import List
backend_spec = RVVSpec()
INPUT_SHAPE_DEF_TEMPLATE = jinja2.Template(
    """
{{indent}}{{index_type}} {{input_shape_name}}[] = {
{{indent}}  {{input_dims}}
{{indent}}};
"""
)
DUMMY_KERNEL_TEMPLATE = jinja2.Template(
    """
#include <assert.h>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

void {{func_name}}(
    void *output,
    {{index_type}} *output_shape[],
    const void *inputs[],
    const {{index_type}} *real_input_shapes[],
    const {{index_type}} *all_input_shapes[],
    const bool input_masks[],
    const {{index_type}} concat_dim_sizes[],
    {{index_type}} concat_dim,
    {{index_type}} rank,
    {{index_type}} num_real_inputs,
    {{index_type}} num_all_inputs
    ) {
  // DO NOTHING
}
"""
)

SRC_TEMPLATE = jinja2.Template(
    """
#include <cstdint>
#include <cstring>
#include <functional>
#include <random>
#include <cstddef> // For size_t
#include <cstring> // For memcpy
#include <cstdio>
#include <stdexcept>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>
void {{func_name}}(
    void *output,
    {{index_type}} *output_shape[],
    const void *inputs[],
    const {{index_type}} *real_input_shapes[],
    const {{index_type}} *all_input_shapes[],
    const bool input_masks[],
    const {{index_type}} concat_dim_sizes[],
    {{index_type}} concat_dim,
    {{index_type}} rank,
    {{index_type}} num_real_inputs,
    {{index_type}} num_all_inputs
    ) {

    int64_t out_shape[4];
    for (int i = 0; i < rank; i++) {
        // Each output_shape[i] is a pointer to a single int64_t value.
        out_shape[i] = output_shape[i][0];
    }
    // Example: out_shape = {N, H, W, (channels1 + channels2)}

    // Cast output to {{input_type}} pointer.
    {{input_type}}* out_ptr = ({{input_type}}*) output;

    // current_offset accumulates the channel offset along the concatenation dimension.
    int64_t current_offset = 0;
    int real_index = 0; // index into real_input_shapes array

    // Loop over all inputs.
    for (int i = 0; i < num_all_inputs; i++) {
        // In this implementation, if input_masks[i] is true we copy the input.
        if (!input_masks[i]) {
            continue;
        }
        // Get the i-th input pointer.
        const {{input_type}}* in_ptr = (const {{input_type}}*) inputs[i];

        // Get the shape for this input from real_input_shapes.
        // We assume each input has rank 4.
        int64_t in_shape[4];
        for (int j = 0; j < rank; j++) {
            in_shape[j] = real_input_shapes[real_index][j];
        }
        real_index++;

        // Assume the shape layout is NHWC.
        int64_t N = in_shape[0];
        int64_t H = in_shape[1];
        int64_t W = in_shape[2];
        int64_t C = in_shape[3];

        // For each (n, h, w) location, copy C elements from in_ptr to the appropriate
        // channel position in out_ptr.
        for (int64_t n = 0; n < N; n++) {
            for (int64_t h = 0; h < H; h++) {
                for (int64_t w = 0; w < W; w++) {
                    // Compute base indices for the current (n,h,w) in input and output.
                    // For input: index = ((n * H + h) * W + w) * C.
                    // For output: index = ((n * H + h) * W + w) * (out_channels),
                    // where out_channels = out_shape[3].
                    int64_t in_base = ((n * H + h) * W + w) * C;
                    int64_t out_base = ((n * out_shape[1] + h) * out_shape[2] + w) * out_shape[3];

                    // Copy the C elements from input to output at the appropriate channel offset.
                    std::memcpy(out_ptr + out_base + current_offset,
                                in_ptr + in_base,
                                C * sizeof({{input_type}}));
                }
            }
        }
        // Update the channel offset with the size of the current input's channel dimension.
        current_offset += concat_dim_sizes[i];
    }

return;

  throw std::runtime_error(
      "Unsupported concat kernel specialization!"
  );
}
"""
)


FUNC_DECL_TEMPLATE = jinja2.Template(
    """
void {{func_name}}(
    void * /*output*/,
    {{index_type}} *[] /*output_shape*/,
    const void *[] /*inputs*/,
    const {{index_type}} *[], /* real_input_shapes, representing shapes of those inputs
                                 whose masks are False, i.e. inputs that will be
                                 copied to the output tensor by concat.*/
    const {{index_type}} *[], /* all_input_shapes, including both kinds of inputs,
                                 i.e. not matter input_mask being True or False */
    const bool [] /*input_masks*/,
    const {{index_type}} [] /*concat_dim_sizes*/,
    {{index_type}} /*concat_dim*/,
    {{index_type}} /*rank*/,
    {{index_type}} /*num_real_inputs*/,
    {{index_type}} /*num_all_inputs*/
);
"""
)

FUNC_CALL_TEMPLATE = jinja2.Template(
    """
{{indent}}{

{{indent}}  const void *inputs[] = {
{{indent}}    {{inputs}}
{{indent}}  };

{{real_input_shape_defs}}

{{indent}}  const {{index_type}} *real_input_shapes[] = {
{{indent}}    {{real_input_shapes}}
{{indent}}  };

{{all_input_shape_defs}}

{{indent}}  const {{index_type}} *all_input_shapes[] = {
{{indent}}    {{all_input_shapes}}
{{indent}}  };

{{indent}}  {{index_type}} *{{output}}_shape[] = {
{{indent}}    {{output_dim_refs}}
{{indent}}  };

{{indent}}  {{index_type}} concat_dim_sizes[] = {
{{indent}}    {{concat_dim_sizes}}
{{indent}}  };

{{indent}}  bool input_masks[] = {
{{indent}}    {{input_masks}}
{{indent}}  };

{{indent}}  {{func_name}}(
{{indent}}      {{output_ptr}},
{{indent}}      {{output}}_shape,
{{indent}}      inputs,
{{indent}}      real_input_shapes,
{{indent}}      all_input_shapes,
{{indent}}      input_masks,
{{indent}}      concat_dim_sizes,
{{indent}}      {{concat_dim}}/*concat_dim*/,
{{indent}}      {{rank}}/*rank*/,
{{indent}}      {{num_real_inputs}}/*num_real_inputs*/,
{{indent}}      {{num_all_inputs}}/*num_all_inputs*/
{{indent}}  );
{{indent}}}
"""
)
@registry.reg("rvv.concatenate.func_decl")
def gen_function_decl(func_attrs):
    """Generate function declaration.

    Parameters
    ----------
    func_attrs : Dict[str, Any]
        Stores the operation attributes.
    Returns
    -------
    str
        Rendered function declaration.
    """
    return FUNC_DECL_TEMPLATE.render(
        func_name=func_attrs["name"],
        index_type=backend_spec.index_type
    )


@registry.reg("rvv.concatenate.gen_function")
def gen_function(func_attrs, element_func=None, element_func_def=None):
    """Generates function body.

    Parameters
    ----------
    func_attrs : Dict[str, Any]
        Stores the operation attributes.

    Returns
    -------
    str
        Rendered function body.
    """
    inputs = func_attrs["inputs"]
    original_inputs = func_attrs["original_inputs"]
    concatenate.check_rank(original_inputs, func_attrs["concat_dim"])
    orig_x, _ = concatenate.get_first_non_empty_input_if_any(original_inputs)
    y = func_attrs["outputs"][0]
    input_type = backend_spec.dtype_to_backend_type(orig_x._attrs["dtype"])
    if len(inputs) > 0:
        return SRC_TEMPLATE.render(
            func_name=func_attrs["name"],
            input_type=input_type,
            index_type=backend_spec.index_type,
        )

    return DUMMY_KERNEL_TEMPLATE.render(
        func_name=func_attrs["name"],
        index_type=backend_spec.index_type,
    )


@registry.reg("rvv.concatenate.func_call")
def gen_function_call(func_attrs, indent="  "):
    """Generates function call.

    Parameters
    ----------
    func_attrs : Dict[str, Any]
        Stores the operation attributes.
    indent : str, optional
        Indent for template, by default "  ".

    Returns
    -------
    str
        Rendered function call.
    """
    inputs = func_attrs["inputs"]
    input_accessors = func_attrs["input_accessors"]
    assert len(inputs) == len(input_accessors), (
        "expected inputs and input_accessors to have the same length, but got: "
        f'{len(inputs)}, {len(input_accessors)}, op: {func_attrs["name"]}'
    )
    original_inputs = func_attrs["original_inputs"]
    concatenate.check_rank(original_inputs, func_attrs["concat_dim"])
    orig_x, _ = concatenate.get_first_non_empty_input_if_any(original_inputs)
    y = func_attrs["outputs"][0]
    concat_dim = func_attrs["concat_dim"]

    input_names = [i._attrs["name"] for i in inputs]
    real_input_shape_defs = []
    real_input_shape_names = []
    # It's not uncommon that multiple shape definitions share the same
    # dimension values. In such a case, we could keep a single definition.
    # In some rare cases, this little trick can dramatically reduce the
    # number of lines generated for the relevant concatenate op and thus
    # may improve the compilation time. Currently, we only enable this
    # optimization for cases where we care about compilation time, as
    # for most cases, the unoptimized version can generate more readable
    # code while having little impact to the compilation time.
    seen_input_shape_dims = {}
    input_shape_name_subs = {}
    optimize_for_compilation_time = Target.current()._kwargs.get(
        "optimize_for_compilation_time", False
    )

    def _make_dims_key(dims):
        dim_vals = [
            str(d.value()) if isinstance(d, IntImm) else d._attrs["name"] for d in dims
        ]
        return ",".join(dim_vals)

    non_empty_input, non_empty_idx = concatenate.get_first_non_empty_input_if_any(
        inputs
    )
    non_empty_input_accessor = input_accessors[non_empty_idx]
    for idx, (i, input_accessor) in enumerate(zip(inputs, input_accessors)):
        input_shape_name = f'{i._attrs["name"]}_shape_{idx}'
        orig_input_shape = input_accessor.original_shapes
        if is_empty_rank1_tensor(orig_input_shape):
            orig_dim = orig_input_shape[0]
            orig_input_shape = deepcopy(non_empty_input_accessor.original_shapes)
            orig_input_shape[concat_dim] = orig_dim
        dims = ", ".join([dim._attrs["name"] for dim in orig_input_shape])
        dims_key = _make_dims_key(orig_input_shape)
        seen_shape_name = seen_input_shape_dims.get(dims_key, None)
        if not optimize_for_compilation_time or seen_shape_name is None:
            one_shape_def = INPUT_SHAPE_DEF_TEMPLATE.render(
                indent="      ",
                input_shape_name=input_shape_name,
                input_dims=dims,
                index_type=backend_spec.index_type,
            )
            real_input_shape_defs.append(one_shape_def)
            real_input_shape_names.append(input_shape_name)
            seen_input_shape_dims[dims_key] = input_shape_name
        else:
            real_input_shape_names.append(seen_shape_name)
            input_shape_name_subs[input_shape_name] = seen_shape_name

    y_shape = y._attrs["shape"]
    y_dim_refs = ", ".join(["&" + dim._attrs["name"] for dim in y_shape])

    input_masks = func_attrs["input_masks"]
    input_indices = [idx for idx, m in enumerate(input_masks) if m is True]
    assert len(inputs) == len(input_indices)
    concat_dim_sizes = []
    for idx, mask in enumerate(input_masks):
        if is_empty_rank1_tensor(original_inputs[idx]._attrs["shape"]):
            d = "0"
        elif mask:
            d = "-1"
        else:
            d = str(original_inputs[idx]._attrs["shape"][concat_dim].value())
        concat_dim_sizes.append(d)

    # update dim size for real inputs
    for input_accessor, input_index in zip(input_accessors, input_indices):
        if is_empty_rank1_tensor(input_accessor.original_shapes):
            dim = input_accessor.original_shapes[0]._attrs["name"]
        else:
            dim = input_accessor.original_shapes[concat_dim]._attrs["name"]
        concat_dim_sizes[input_index] = dim

    input_mask_values = ["true" if mask is True else "false" for mask in input_masks]

    # all input shape defs and names, including those that are masked out
    all_input_shape_defs = []
    all_input_shape_names = []
    seen_input_shape_dims = {}
    # first, create shape defs for inputs that have been masked off
    for (
        mask,
        orig_input,
    ) in zip(input_masks, original_inputs):
        if mask is False:
            orig_input_shape_name = f'orig_{orig_input._attrs["name"]}_shape'
            if orig_input_shape_name not in all_input_shape_names:
                dims = ", ".join(
                    [str(dim._attrs["values"][0]) for dim in orig_input._attrs["shape"]]
                )
                dims_key = _make_dims_key(orig_input._attrs["shape"])
                seen_shape_name = seen_input_shape_dims.get(dims_key, None)
                if not optimize_for_compilation_time or seen_shape_name is None:
                    one_shape_def = INPUT_SHAPE_DEF_TEMPLATE.render(
                        indent="      ",
                        input_shape_name=orig_input_shape_name,
                        input_dims=dims,
                        index_type=backend_spec.index_type,
                    )
                    all_input_shape_defs.append(one_shape_def)
                    seen_input_shape_dims[dims_key] = orig_input_shape_name
                    all_input_shape_names.append(orig_input_shape_name)
                else:
                    all_input_shape_names.append(seen_shape_name)
            else:
                all_input_shape_names.append(orig_input_shape_name)
        else:
            all_input_shape_names.append("")
    # update all_input_shapes with real input shapes
    for idx, (input_tensor, input_index) in enumerate(zip(inputs, input_indices)):
        input_shape_name = f'{input_tensor._attrs["name"]}_shape_{idx}'
        sub_name = input_shape_name_subs.get(input_shape_name, None)
        if sub_name is not None:
            input_shape_name = sub_name
        all_input_shape_names[input_index] = input_shape_name

    return FUNC_CALL_TEMPLATE.render(
        indent=indent,
        inputs=",\n      ".join(input_names),
        real_input_shape_defs="".join(real_input_shape_defs),
        real_input_shapes=", ".join(real_input_shape_names),
        all_input_shape_defs="".join(all_input_shape_defs),
        all_input_shapes=", ".join(all_input_shape_names),
        input_masks=", ".join(input_mask_values),
        concat_dim_sizes=", ".join(concat_dim_sizes),
        output_dim_refs=y_dim_refs,
        func_name=func_attrs["name"],
        output=y._attrs["name"],
        output_ptr=y._attrs["name"],
        concat_dim=concat_dim,
        rank=len(orig_x._attrs["shape"]),
        num_real_inputs=len(inputs),
        num_all_inputs=len(original_inputs),
        index_type=backend_spec.index_type,
    )

