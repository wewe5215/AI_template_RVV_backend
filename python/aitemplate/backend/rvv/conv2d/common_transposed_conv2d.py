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
common functions for transposed conv2d
"""

import re

from aitemplate.backend.rvv.conv2d import common

def extract_config(
    dtype="float16",
    is_bias=False,
    is_relu=False,
):
    """Populates all available conv2d configs into the op_instance field."""
    import cpu_lib
    if is_bias and is_relu:
        op_kind = cpu_lib.library.Conv2dKind.TransposedConv2dBiasRelu
    elif is_bias:
        op_kind = cpu_lib.library.Conv2dKind.Conv2dBias
    else:
        op_kind = cpu_lib.library.Conv2dKind.TransposedConv2d
    extra_kind = cpu_lib.library.TensorOperation.PassThrough
    # if dtype == "float32": --> TODO: uncomment later
    Layout = cpu_lib.library.LayoutType.NHWC
    return common.extract_config(
        dtype = dtype,
        op_kind = op_kind,
        extra_kind = extra_kind,
        Layout = Layout)
