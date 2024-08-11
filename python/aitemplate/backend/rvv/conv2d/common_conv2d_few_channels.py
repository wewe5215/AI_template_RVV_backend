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
common functions for conv2d op with few channels(< 8)
"""

from aitemplate.backend.rvv.conv2d import common
from aitemplate.utils import alignment


def extract_config(func_attrs, dtype="float16", unary_op_name="Identity"):
    import cpu_lib
    if unary_op_name == "ReLu":
        op_kind = cpu_lib.library.Conv2dKind.Conv2dBiasRelu
    elif unary_op_name == "Identity":
        op_kind = cpu_lib.library.Conv2dKind.Conv2dBias
    extra_kind = cpu_lib.library.TensorOperation.PassThrough
    # if dtype == "float32": --> TODO: uncomment later
    conv2d_specialization = cpu_lib.conv2d_operation.Conv2DSpecialization.ConvNhwcF32
    return common.extract_config(
        dtype = dtype,
        op_kind = op_kind,
        extra_kind = extra_kind,
        conv2d_specialization = conv2d_specialization)
