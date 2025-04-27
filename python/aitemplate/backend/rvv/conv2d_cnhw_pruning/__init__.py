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
rvv conv2d module init
"""
from aitemplate.backend.rvv.conv2d_cnhw_pruning import (
    conv2d_cnhw_pruning,
    conv2d_cnhw_pruning_bias,
    conv2d_cnhw_pruning_bias_add,
    conv2d_cnhw_pruning_bias_add_hardswish,
    conv2d_cnhw_pruning_bias_add_relu,
    # conv2d_cnhw_pruning_bias_few_channels,
    conv2d_cnhw_pruning_bias_hardswish,
    # conv2d_cnhw_pruning_bias_hardswish_few_channels,
    conv2d_cnhw_pruning_bias_relu,
    conv2d_cnhw_pruning_bias_relu6,
    # conv2d_cnhw_pruning_bias_relu_few_channels,
    conv2d_cnhw_pruning_bias_sigmoid,
    # depthwise convolution with pruning is not supported
    # conv2d_cnhw_pruning_depthwise,
    # conv2d_cnhw_pruning_depthwise_bias,
    # conv2d_cnhw_pruning_depthwise_bias_relu,
    # conv2d_cnhw_pruning_depthwise_bias_relu6,
    # transposed_conv2d_cnhw_pruning,
    # transposed_conv2d_cnhw_pruning_bias,
)
