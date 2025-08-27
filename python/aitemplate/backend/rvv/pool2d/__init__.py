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
RVV pool2d init
"""
from aitemplate.backend.rvv.pool2d import avg_pool2d, max_pool2d, \
    avg_pool2d_transpose, avg_pool2d_cnhw, avg_pool2d_cnhw_transpose, \
    max_pool2d_transpose

__all__ = [
    "avg_pool2d",
    "max_pool2d",
    "avg_pool2d_transpose",
    "avg_pool2d_cnhw",
    "avg_pool2d_cnhw_transpose",
    "max_pool2d_transpose",
]
