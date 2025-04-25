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
# flake8: noqa
"""
RVV backend codegen functions.
"""
from aitemplate.backend.rvv import (
    rvv_common,
    lib_template,
    target_def,
    utils,
)
# from aitemplate.backend.rvv.common import *
from aitemplate.backend.rvv.conv2d import *
from aitemplate.backend.rvv.conv2d_cnhw import *
from aitemplate.backend.rvv.conv2d_cnhw_pruning import *
# from aitemplate.backend.rvv.conv3d import *
from aitemplate.backend.rvv.elementwise import *
# from aitemplate.backend.rvv.embedding import *
# from aitemplate.backend.rvv.gemm_special import *
from aitemplate.backend.rvv.gemm_universal import *
# from aitemplate.backend.rvv.gemm_epilogue_vistor import *
# from aitemplate.backend.rvv.jagged import *
# from aitemplate.backend.rvv.layernorm_sigmoid_mul import *
from aitemplate.backend.rvv.padding import *
from aitemplate.backend.rvv.pool2d import *
# from aitemplate.backend.rvv.reduce import *
# from aitemplate.backend.rvv.softmax import *
from aitemplate.backend.rvv.tensor import *
# from aitemplate.backend.rvv.upsample import *
from aitemplate.backend.rvv.view_ops import *
# from aitemplate.backend.rvv.vision_ops import *
# from aitemplate.backend.rvv.attention import *
# from aitemplate.backend.rvv.groupnorm import *
# from aitemplate.backend.rvv.b2b_bmm import *
