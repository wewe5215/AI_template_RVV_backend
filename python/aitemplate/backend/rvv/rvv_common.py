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
RVV common functions for codegen.
"""
from typing import Dict

DTYPE_TO_RVVTYPE: Dict[str, str] = {
    "float16": "float16_t",
    "float32": "float32_t",
    "float": "float32_t",
    "int64": "int64_t",
    "int32": "int32_t",
    "int16": "int16_t",
    "int8": "int8_t",
    "uint64": "uint64_t",
    "uint32": "uint32_t",
    "uint16": "uint16_t",
    "uint8": "uint8_t",
    "bool": "bool",
}



def dtype_to_rvv_type(dtype: str):
    """Returns the corresponding rvv type."""
    rvv_type = DTYPE_TO_RVVTYPE.get(dtype)

    if rvv_type is None:
        raise NotImplementedError("RVV - Unsupported dtype: {}".format(dtype))
    return rvv_type

