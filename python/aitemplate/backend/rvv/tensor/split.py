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
import jinja2
FUNC_CALL_TEMPLATE = jinja2.Template(
    """
{{indent}}{

{{indent}}  void *outputs[] = {
{{indent}}    {{outputs}}
{{indent}}  };

{{output_shape_defs}}

{{indent}}  {{index_type}} **output_shapes[] = {
{{indent}}    {{output_shapes}}
{{indent}}  };

{{indent}}  const {{index_type}} {{input_name}}_shape[] = {
{{indent}}    {{input_dims}}
{{indent}}  };

{{indent}}  {{index_type}} split_sizes[] = {
{{indent}}    {{split_sizes}}
{{indent}}  };

{{indent}}  bool output_masks[] = {
{{indent}}    {{output_masks}}
{{indent}}  };

{{indent}}  {{func_name}}(
{{indent}}      outputs,
{{indent}}      output_shapes,
{{indent}}      output_masks,
{{indent}}      {{input_ptr}},
{{indent}}      {{input_name}}_shape,
{{indent}}      {{real_num_splits}}/*real_num_splits*/,
{{indent}}      {{all_num_splits}}/*all_num_splits*/,
{{indent}}      split_sizes,
{{indent}}      {{split_dim}}/*split_dim*/,
{{indent}}      {{rank}}/*rank*/
{{indent}}  );
{{indent}}}
"""
)

FUNC_DECL_TEMPLATE = jinja2.Template(
    """
void {{func_name}}(
    void *[] /*outputs*/,
    {{index_type}} **[] /*output_shapes*/,
    const bool [] /*output_masks*/,
    const void * /*input*/,
    const {{index_type}} * /*input_shape*/,
    {{index_type}} /*real_num_splits*/,
    {{index_type}} /*all_num_splits*/,
    {{index_type}} [] /*split_sizes*/,
    {{index_type}} /*split_dim*/,
    {{index_type}} /*rank*/
);
"""
)

SRC_TEMPLATE = jinja2.Template(
    """
#include <vector>
#include <assert.h>
#include <iostream>
#include <stdexcept>
#include <string>
static void compute_strides(const int64_t* shape, int rank, int64_t* strides) {
    int64_t acc = 1;
    for (int i = rank - 1; i >= 0; --i) {
        strides[i] = acc;
        acc *= shape[i];
    }
}
void {{func_name}}(
    void* outputs[],
    {{index_type}} **output_shapes[],
    const bool output_masks[],
    const void* input,
    const {{index_type}} *input_shape,
    {{index_type}} real_num_splits,
    {{index_type}} all_num_splits,
    {{index_type}} split_sizes[],
    {{index_type}} split_dim,
    {{index_type}} rank
    ) {

        assert(split_dim >= 0 && split_dim < rank);
    int64_t elem_size = sizeof(uint32_t);

    // Compute input strides (in elements)
    std::vector<int64_t> in_strides(rank);
    compute_strides(input_shape, rank, in_strides.data());
    const uint32_t* input_u32 = reinterpret_cast<const uint32_t*>(input);
    int64_t dimSize = input_shape[split_dim];

    // Check sum of splits
    int64_t sum = 0;
    for (int64_t i = 0; i < all_num_splits; ++i) sum += split_sizes[i];
    if (sum != dimSize) {
        throw std::runtime_error("unmatched split dim size! with sum = " + std::to_string(sum) + " dimSize = " + std::to_string(dimSize));
    }


    // For each split output
    int64_t offset = 0;
    int64_t split_sizes_idx = 0;
    for (int64_t si = 0; si < real_num_splits; ++si) {
        while (!output_masks[split_sizes_idx]) {
            offset += split_sizes[split_sizes_idx];
            split_sizes_idx++;
        }
        // the shape for this output is same as input_shape but
        // shape[split_dim] = split_sizes[si]
        int64_t** out_shape = (output_shapes[si]);
        for (int d = 0; d < rank; ++d) {
            *(out_shape[d]) = input_shape[d];
        }
        *(out_shape[split_dim]) = split_sizes[si];

        void* out_ptr = outputs[si];

        // Decide implementation depending on split_dim
        if (split_dim == 0) {
            // Outer-axis: can simply copy big contiguous blocks
            int64_t blockCount = split_sizes[si];
            int64_t blockStride = in_strides[0]; // elements per first dim
            const uint32_t* in_base = input_u32 + offset * blockStride;
            uint32_t* out_base = reinterpret_cast<uint32_t*>(out_ptr);
            size_t bytes_to_copy = blockCount * blockStride * elem_size;
            std::memcpy(out_base, in_base, bytes_to_copy);
        }
        else if (split_dim == rank - 1) {
            // Innermost axis: contiguous per row, so iterate outer dims and memcpy
            int64_t outer_count = 1;
            for (int d = 0; d < rank - 1; ++d) outer_count *= input_shape[d];
            int64_t chunk_len = split_sizes[si];
            int64_t full_len  = input_shape[split_dim];
            int64_t copy_bytes = chunk_len * elem_size;
            for (int64_t idx = 0; idx < outer_count; ++idx) {
                const uint32_t* in_base  = input_u32 + (idx * full_len + offset);
                uint32_t*       out_base = reinterpret_cast<uint32_t*>(out_ptr) + (idx * chunk_len);
                std::memcpy(out_base, in_base, copy_bytes);
            }
        }
        else {
            // Middle axis: general case - element-by-element copy
            // Compute nested loops over all dims, copying appropriate slice
            std::vector<int64_t> indices(rank, 0);
            // We'll do a simple recursion or nested loops; here is a naive approach:
            auto out_data = reinterpret_cast<uint32_t*>(out_ptr);
            int64_t before = 1;
            for (int d = 0; d < split_dim; ++d) before *= input_shape[d];
            int64_t after = 1;
            for (int d = split_dim + 1; d < rank; ++d) after *= input_shape[d];
            int64_t in_stride = in_strides[split_dim] * elem_size;
            int64_t copy_count = split_sizes[si];
            for (int64_t b0 = 0; b0 < before; ++b0) {
                for (int64_t s0 = 0; s0 < copy_count; ++s0) {
                    for (int64_t a0 = 0; a0 < after; ++a0) {
                        const uint32_t* in_ptr  = input_u32  + (b0 * input_shape[split_dim] * after + (offset + s0) * after + a0);
                        uint32_t*       out_ptr = out_data + (b0 * copy_count   * after + s0 * after + a0);
                        std::memcpy(out_ptr, in_ptr, elem_size);
                    }
                }
            }
        }
        offset += split_sizes[si];
        split_sizes_idx++;
    }
}
"""
)
from aitemplate.backend import registry
from aitemplate.backend.backend_spec import RVVSpec
from aitemplate.backend.common.split_common import OUTPUT_SHAPE_DEF_TEMPLATE
backend_spec = RVVSpec()

@registry.reg("rvv.split.func_decl")
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
        index_type=backend_spec.index_type,
        func_name=func_attrs["name"],
    )


@registry.reg("rvv.split.gen_function")
def gen_function(func_attrs):
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
    return SRC_TEMPLATE.render(
        func_name=func_attrs["name"],
        index_type=backend_spec.index_type,
    )


@registry.reg("rvv.split.func_call")
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
    x = func_attrs["inputs"][0]
    outputs = func_attrs["outputs"]
    split_dim = func_attrs["split_dim"]
    num_splits = len(func_attrs["outputs"])

    output_names = ",\n      ".join([i._attrs["name"] for i in outputs])

    output_shape_defs = []
    output_shape_names = []
    for i in outputs:
        output_shape_name = "{}_shape".format(i._attrs["name"])
        if output_shape_name not in output_shape_names:
            dim_refs = ", ".join(
                ["&" + dim._attrs["name"] for dim in i._attrs["shape"]]
            )
            one_shape_def = OUTPUT_SHAPE_DEF_TEMPLATE.render(
                indent="      ",
                output_shape_name=output_shape_name,
                output_dim_refs=dim_refs,
                index_type=backend_spec.index_type,
            )
            output_shape_defs.append(one_shape_def)
        output_shape_names.append(output_shape_name)

    x_shape = x._attrs["shape"]
    x_dims = ", ".join([dim._attrs["name"] for dim in x_shape])

    split_sizes = ", ".join([str(i) for i in func_attrs["split_sizes"]])

    output_masks_str = ", ".join(
        ["true" if mask is True else "false" for mask in func_attrs["output_masks"]]
    )

    return FUNC_CALL_TEMPLATE.render(
        indent=indent,
        outputs=output_names,
        output_shape_defs="".join(output_shape_defs),
        output_shapes=", ".join(output_shape_names),
        output_masks=output_masks_str,
        input_dims=x_dims,
        func_name=func_attrs["name"],
        input_name=x._attrs["name"],
        input_ptr=x._attrs["name"],
        split_dim=split_dim,
        real_num_splits=len(func_attrs["outputs"]),
        all_num_splits=len(func_attrs["output_masks"]),
        rank=len(x._attrs["shape"]),
        num_splits=num_splits,
        split_sizes=split_sizes,
        index_type=backend_spec.index_type,
    )
