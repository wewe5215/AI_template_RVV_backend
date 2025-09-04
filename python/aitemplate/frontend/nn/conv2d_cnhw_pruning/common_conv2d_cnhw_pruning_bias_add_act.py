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
common module for conv2d bias act residual add
"""
from aitemplate.compiler import ops
from aitemplate.frontend.nn.module import Module
from aitemplate.frontend.nn.parameter import Parameter

# pylint: disable=C0103


class Conv2dCNHWPruningBiasAddAct(Module):
    def __init__(
        self,
        op_name,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding=0,
        dilation=1,
        groups=1,
        dtype="float32",
        pruning_ratio=0.5,
    ):
        super().__init__()
        self.weight = Parameter(
            shape=[out_channels, kernel_size, kernel_size, int((in_channels // groups) * (1 - pruning_ratio))],
            dtype=dtype,
        )
        self.bias = Parameter(shape=[out_channels], dtype=dtype)
        self.weight_indice = Parameter( # out_channels / 8 stands for each tile is with 8 rows this may change after profiling on tile size
            shape=[out_channels, (kernel_size * kernel_size * int((in_channels // groups) * (1 - pruning_ratio)))],
            dtype="uint16_t"
        )
        op_func = getattr(ops, op_name)
        self.op = op_func(stride=stride, pad=padding, dilate=dilation, group=groups, pruning_ratio=pruning_ratio)

    def forward(self, *args):
        assert len(args) == 2
        x = args[0]
        r = args[1]
        return self.op(x, self.weight.tensor(), self.bias.tensor(), self.weight_indice.tensor(), r)
