import enum
from enum import auto
from typing import List
from aitemplate.utils.cpu_lib import library
class Conv2DSpecialization(enum.Enum):
    # ConvNchwF32 = auto()
    # ConvNchwF16 = auto()
    ConvNhwcF32 = auto()
    ConvNhwcF16 = auto()
    ConvNhwcQd8F32Qc8w = auto()
    ConvNhwcQd8F16Qc8w = auto()
    ConvNhwcQc8 = auto()
    ConvNhwcQs8 = auto()
    ConvNhwcQu8 = auto()
    ConvCNHWF32 = auto()
    ConvCNHWF16 = auto()
NHWC_KINDS = {
    Conv2DSpecialization.ConvNhwcF32,
    Conv2DSpecialization.ConvNhwcF16,
    Conv2DSpecialization.ConvNhwcQd8F32Qc8w,
    Conv2DSpecialization.ConvNhwcQd8F16Qc8w,
    Conv2DSpecialization.ConvNhwcQc8,
    Conv2DSpecialization.ConvNhwcQs8,
    Conv2DSpecialization.ConvNhwcQu8,
}

CNHW_KINDS = {
    Conv2DSpecialization.ConvCNHWF32,
    Conv2DSpecialization.ConvCNHWF16,
}
BIAS_KINDS = {
    library.Conv2dKind.Conv2dBias,
    library.Conv2dKind.Conv2dBiasRelu,
    library.Conv2dKind.Conv2dBiasRelu6,
    library.Conv2dKind.Conv2dBiasAdd,
    library.Conv2dKind.Conv2dBiasReluAdd,
    library.Conv2dKind.Conv2dBiasRelu6Add,
    library.Conv2dKind.Conv2dBiasSigmoid,
    library.Conv2dKind.Conv2dBiasAddRelu,
    library.Conv2dKind.Conv2dBiasAddRelu6,
    library.Conv2dKind.Conv2dDepthwiseBias,
    library.Conv2dKind.Conv2dDepthwiseBiasAdd,
    library.Conv2dKind.Conv2dDepthwiseBiasRelu,
    library.Conv2dKind.Conv2dDepthwiseBiasRelu6,
    library.Conv2dKind.Conv2dDepthwiseBiasAddRelu,
    library.Conv2dKind.Conv2dDepthwiseBiasAddRelu6,
    library.Conv2dKind.Conv2dBiasTranspose,
    library.Conv2dKind.Conv2dBiasReluTranspose,
    library.Conv2dKind.Conv2dBiasRelu6Transpose,
    library.Conv2dKind.Conv2dDepthwiseBiasTranspose,
    library.Conv2dKind.Conv2dDepthwiseBiasReluTranspose,
    library.Conv2dKind.Conv2dDepthwiseBiasRelu6Transpose,
}

TRANSPOSE_AFTER_CONV_KINDS = {
    library.Conv2dKind.Conv2dBiasTranspose,
    library.Conv2dKind.Conv2dBiasReluTranspose,
    library.Conv2dKind.Conv2dBiasRelu6Transpose,
    library.Conv2dKind.Conv2dDepthwiseBiasTranspose,
    library.Conv2dKind.Conv2dDepthwiseBiasReluTranspose,
    library.Conv2dKind.Conv2dDepthwiseBiasRelu6Transpose,
}

DEPTHWISE_KINDS = {
    library.Conv2dKind.Conv2dDepthwise,
    library.Conv2dKind.Conv2dDepthwiseBias,
    library.Conv2dKind.Conv2dDepthwiseBiasAdd,
    library.Conv2dKind.Conv2dDepthwiseBiasRelu,
    library.Conv2dKind.Conv2dDepthwiseBiasRelu6,
    library.Conv2dKind.Conv2dDepthwiseBiasAddRelu,
    library.Conv2dKind.Conv2dDepthwiseBiasAddRelu6,
    library.Conv2dKind.Conv2dDepthwiseBiasTranspose,
    library.Conv2dKind.Conv2dDepthwiseBiasReluTranspose,
    library.Conv2dKind.Conv2dDepthwiseBiasRelu6Transpose,
}

RELU_KINDS = {
    library.Conv2dKind.Conv2dBiasRelu,
    library.Conv2dKind.Conv2dBiasAddRelu,
    library.Conv2dKind.Conv2dBiasReluAdd,
    library.Conv2dKind.Conv2dDepthwiseBiasRelu,
    library.Conv2dKind.Conv2dDepthwiseBiasAddRelu,
    library.Conv2dKind.Conv2dBiasReluTranspose,
    library.Conv2dKind.Conv2dDepthwiseBiasReluTranspose,
}

RELU6_KINDS = {
    library.Conv2dKind.Conv2dBiasRelu6,
    library.Conv2dKind.Conv2dBiasAddRelu6,
    library.Conv2dKind.Conv2dBiasRelu6Add,
    library.Conv2dKind.Conv2dDepthwiseBiasRelu6,
    library.Conv2dKind.Conv2dDepthwiseBiasAddRelu6,
    library.Conv2dKind.Conv2dBiasRelu6Transpose,
    library.Conv2dKind.Conv2dDepthwiseBiasRelu6Transpose,
}

BINARY_OP_KIND = {
    library.TensorOperation.Add,
    library.TensorOperation.Mul,
    library.TensorOperation.Div,
    library.TensorOperation.Sub
}

BINARY_FLAG_OP_KIND = {
    library.TensorOperation.Max,
    library.TensorOperation.Min,
    library.TensorOperation.Sqrtdiff
}
Conv2DSpecializationTag = {
    # the code format for ConvNchw is different from the one of ConvNhwc
    # TODO : handle it independently
    # Conv2DSpecialization.ConvNchwF32: "convolution2d_nchw_f32",
    # Conv2DSpecialization.ConvNchwF16: "convolution2d_nchw_f16",
    Conv2DSpecialization.ConvNhwcF32: "convolution2d_nhwc_f32",
    Conv2DSpecialization.ConvNhwcF16: "convolution2d_nhwc_f16",
    Conv2DSpecialization.ConvNhwcQd8F32Qc8w: "convolution2d_nhwc_qd8_f32_qc8w",
    Conv2DSpecialization.ConvNhwcQd8F16Qc8w: "convolution2d_nhwc_qd8_f16_qc8w",
    Conv2DSpecialization.ConvNhwcQc8: "convolution2d_nhwc_qc8",
    Conv2DSpecialization.ConvNhwcQs8: "convolution2d_nhwc_qs8",
    Conv2DSpecialization.ConvNhwcQu8: "convolution2d_nhwc_qu8",
    Conv2DSpecialization.ConvCNHWF32: "input_T_convolution2d_nhwc_f32",
}