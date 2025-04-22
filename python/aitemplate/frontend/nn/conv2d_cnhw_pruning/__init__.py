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
from aitemplate.frontend.nn.conv2d_cnhw.conv2d_cnhw import Conv2dCNHW
from aitemplate.frontend.nn.conv2d_cnhw.conv2d_cnhw_bias import Conv2dCNHWBias
from aitemplate.frontend.nn.conv2d_cnhw.conv2d_cnhw_bias_add_hardswish import (
    Conv2dCNHWBiasAddHardswish,
)
from aitemplate.frontend.nn.conv2d_cnhw.conv2d_cnhw_bias_add_relu import Conv2dCNHWBiasAddRelu
from aitemplate.frontend.nn.conv2d_cnhw.conv2d_cnhw_bias_add import Conv2dCNHWBiasAdd
# from aitemplate.frontend.nn.conv2d_cnhw.conv2d_cnhw_bias_few_channels import Conv2dCNHWBiasFewChannels
from aitemplate.frontend.nn.conv2d_cnhw.conv2d_cnhw_bias_hardswish import Conv2dCNHWBiasHardswish
# from aitemplate.frontend.nn.conv2d_cnhw.conv2d_cnhw_bias_hardswish_few_channels import (
#     Conv2dCNHWBiasHardswishFewChannels,
# )
from aitemplate.frontend.nn.conv2d_cnhw.conv2d_cnhw_bias_relu import Conv2dCNHWBiasRelu
from aitemplate.frontend.nn.conv2d_cnhw.conv2d_cnhw_bias_relu6 import Conv2dCNHWBiasRelu6
# from aitemplate.frontend.nn.conv2d_cnhw.conv2d_cnhw_bias_relu_few_channels import (
#     Conv2dCNHWBiasReluFewChannels,
# )
from aitemplate.frontend.nn.conv2d_cnhw.conv2d_cnhw_bias_sigmoid import Conv2dCNHWBiasSigmoid
from aitemplate.frontend.nn.conv2d_cnhw.conv2d_cnhw_depthwise import Conv2dCNHWDepthwise
from aitemplate.frontend.nn.conv2d_cnhw.conv2d_cnhw_depthwise_bias import Conv2dCNHWDepthwiseBias
from aitemplate.frontend.nn.conv2d_cnhw.conv2d_cnhw_depthwise_bias_relu import Conv2dCNHWDepthwiseBiasRelu
from aitemplate.frontend.nn.conv2d_cnhw.conv2d_cnhw_depthwise_bias_relu6 import Conv2dCNHWDepthwiseBiasRelu6
# from aitemplate.frontend.nn.conv2d_cnhw.transposed_conv2d_cnhw_bias import ConvTranspose2dBias
# from aitemplate.frontend.nn.conv2d_cnhw.transposed_conv2d_cnhw_bias_relu import (
#     ConvTranspose2dBiasRelu,
# )
