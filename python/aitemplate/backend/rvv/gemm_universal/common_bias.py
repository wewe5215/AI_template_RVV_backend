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
Common codegen functions for gemm with bias.
"""

import jinja2

# pylint: disable=C0301,C0415,R1705

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
#include <sstream>
#include <memory>
#include <ctime>
#include <cstdlib>
#include <stdexcept>
#include <cstring> // For memset

{{extra_code}}


{{instances}}


void {{function_name}} (
    void* a_ptr,
    void* b_ptr,
    void* bias_ptr,
    void* c_ptr,
{% for idx in range(input_ndims) %}
    int64_t* a_dim{{idx}},
{% endfor %}
{% for idx in range(weight_ndims) %}
    int64_t* b_dim{{idx}},
{% endfor %}
{% for idx in range(input_ndims) %}
  {% if idx == input_ndims - 1 %}
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

  if (!bias_ptr) {
    throw std::runtime_error("bias_ptr is null!");
  }

  {{exec_paths}}
  {% for idx in range(input_ndims) %}
      std::cout << "input_ndims{{idx}}: " << *a_dim{{idx}} << std::endl;
  {% endfor %}
  {% for idx in range(weight_ndims) %}
      std::cout << "weight_ndims{{idx}}: " << *b_dim{{idx}} << std::endl;
  {% endfor %}
  {% for idx in range(input_ndims) %}
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


FUNC_DECL_TEMPLATE = jinja2.Template(
    """
void {{func_name}}(
  void*,
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
    int64_t*{% if not loop.last %},{% endif %}
{% endfor %}
);
"""
)
