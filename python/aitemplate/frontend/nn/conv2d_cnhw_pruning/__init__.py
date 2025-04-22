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
modules for conv2d
"""
from aitemplate.frontend.nn.conv2d_cnhw_pruning.conv2d_cnhw_pruning import Conv2dCNHWPruning
from aitemplate.frontend.nn.conv2d_cnhw_pruning.conv2d_cnhw_pruning_bias import Conv2dCNHWPruningBias
from aitemplate.frontend.nn.conv2d_cnhw_pruning.conv2d_cnhw_pruning_bias_add_hardswish import (
    Conv2dCNHWPruningBiasAddHardswish,
)
from aitemplate.frontend.nn.conv2d_cnhw_pruning.conv2d_cnhw_pruning_bias_add_relu import Conv2dCNHWPruningBiasAddRelu
from aitemplate.frontend.nn.conv2d_cnhw_pruning.conv2d_cnhw_pruning_bias_add import Conv2dCNHWPruningBiasAdd
# from aitemplate.frontend.nn.conv2d_cnhw_pruning.conv2d_cnhw_pruning_bias_few_channels import Conv2dCNHWPruningBiasFewChannels
from aitemplate.frontend.nn.conv2d_cnhw_pruning.conv2d_cnhw_pruning_bias_hardswish import Conv2dCNHWPruningBiasHardswish
# from aitemplate.frontend.nn.conv2d_cnhw_pruning.conv2d_cnhw_pruning_bias_hardswish_few_channels import (
#     Conv2dCNHWPruningBiasHardswishFewChannels,
# )
from aitemplate.frontend.nn.conv2d_cnhw_pruning.conv2d_cnhw_pruning_bias_relu import Conv2dCNHWPruningBiasRelu
from aitemplate.frontend.nn.conv2d_cnhw_pruning.conv2d_cnhw_pruning_bias_relu6 import Conv2dCNHWPruningBiasRelu6
# from aitemplate.frontend.nn.conv2d_cnhw_pruning.conv2d_cnhw_pruning_bias_relu_few_channels import (
#     Conv2dCNHWPruningBiasReluFewChannels,
# )
from aitemplate.frontend.nn.conv2d_cnhw_pruning.conv2d_cnhw_pruning_bias_sigmoid import Conv2dCNHWPruningBiasSigmoid
from aitemplate.frontend.nn.conv2d_cnhw_pruning.conv2d_cnhw_pruning_depthwise import Conv2dCNHWPruningDepthwise
from aitemplate.frontend.nn.conv2d_cnhw_pruning.conv2d_cnhw_pruning_depthwise_bias import Conv2dCNHWPruningDepthwiseBias
from aitemplate.frontend.nn.conv2d_cnhw_pruning.conv2d_cnhw_pruning_depthwise_bias_relu import Conv2dCNHWPruningDepthwiseBiasRelu
from aitemplate.frontend.nn.conv2d_cnhw_pruning.conv2d_cnhw_pruning_depthwise_bias_relu6 import Conv2dCNHWPruningDepthwiseBiasRelu6
# from aitemplate.frontend.nn.conv2d_cnhw_pruning.transposed_conv2d_cnhw_pruning_bias import ConvTranspose2dBias
# from aitemplate.frontend.nn.conv2d_cnhw_pruning.transposed_conv2d_cnhw_pruning_bias_relu import (
#     ConvTranspose2dBiasRelu,
# )
