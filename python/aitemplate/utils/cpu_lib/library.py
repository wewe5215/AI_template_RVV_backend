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
import enum
from dataclasses import dataclass
from enum import auto


class DataType(enum.Enum):
    b1 = auto()
    u4 = auto()
    u8 = auto()
    u16 = auto()
    u32 = auto()
    u64 = auto()
    s4 = auto()
    s8 = auto()
    s16 = auto()
    s32 = auto()
    s64 = auto()
    f16 = auto()
    bf16 = auto()
    f32 = auto()
    f64 = auto()
    qs8 = auto()
    qu8 = auto()
    invalid = auto()


#
DataTypeNames = {
    DataType.b1: "b1",
    DataType.u4: "u4",
    DataType.u8: "u8",
    DataType.u16: "u16",
    DataType.u32: "u32",
    DataType.u64: "u64",
    DataType.s4: "s4",
    DataType.s8: "s8",
    DataType.s16: "s16",
    DataType.s32: "s32",
    DataType.s64: "s64",
    DataType.f16: "f16",
    DataType.bf16: "bf16",
    DataType.f32: "f32",
    DataType.f64: "f64",
    DataType.qs8: "qs8",
    DataType.qu8: "qu8",
}

DataTypeTag = {
    DataType.u8: "uint8_t",
    DataType.u16: "uint16_t",
    DataType.u32: "uint32_t",
    DataType.u64: "uint64_t",
    DataType.s8: "int8_t",
    DataType.s16: "int16_t",
    DataType.s32: "int32_t",
    DataType.s64: "int64_t",
    DataType.f16: "__fp16",
    DataType.bf16: "__bf16",
    DataType.f32: "float",
    DataType.f64: "double",
    DataType.qs8: "int8_t",
    DataType.qu8: "uint8_t",
}

DataTypeSize = {
    DataType.b1: 1,
    DataType.u4: 4,
    DataType.u8: 8,
    DataType.u16: 16,
    DataType.u32: 32,
    DataType.u64: 64,
    DataType.s4: 4,
    DataType.s8: 8,
    DataType.s16: 16,
    DataType.s32: 32,
    DataType.s64: 64,
    DataType.f16: 16,
    DataType.bf16: 16,
    DataType.f32: 32,
    DataType.f64: 64,
    DataType.qs8: 8,
    DataType.qu8: 8,

}


class LayoutType(enum.Enum):
    ColumnMajor = auto()
    RowMajor = auto()
    NWC = auto()
    KXC = auto()
    NWK = auto()
    NCW = auto()
    KCX = auto()
    NKW = auto()
    NHWC = auto()
    KYXC = auto()
    NHWK = auto()
    NCHW = auto()
    CNHW = auto()
    KCYX = auto()
    NKWH = auto()
    NDHWC = auto()
    KZYXC = auto()
    NDHWK = auto()
    NCDHW = auto()
    KCZYX = auto()
    NKDHW = auto()
    G_NHW_C = auto()
    G_K_YX_C = auto()
    G_NHW_K = auto()
    NHWGC = auto()
    KYXGC = auto()
    NHWGK = auto()
    GNHWC = auto()
    GKYXC = auto()
    GNHWK = auto()
    GNWC = auto()
    GKXC = auto()
    GNWK = auto()


LayoutTag = {
    LayoutType.NHWC: "nhwc",
    LayoutType.NCHW: "nchw",
    LayoutType.ColumnMajor: "column",
    LayoutType.RowMajor: "row",
    LayoutType.CNHW: "cnhw"
}


#
class OperationKind(enum.Enum):
    Gemm = auto()
    Conv1d = auto()
    Conv2d = auto()
    Conv3d = auto()
    Softmax = auto()
    LayerNorm = auto()
    GroupNorm = auto()


OperationKindNames = {
    OperationKind.Gemm: "gemm",
    OperationKind.Conv1d: "conv1d",
    OperationKind.Conv2d: "conv2d",
    OperationKind.Conv3d: "conv3d",
    OperationKind.Softmax: "softmax",
    OperationKind.LayerNorm: "layernorm",
    OperationKind.GroupNorm: "groupnorm",
}


class Conv2dKind(enum.Enum):
    Conv2d = auto()
    Conv2dBias = auto()
    Conv2dBiasAdd = auto()
    Conv2dBiasRelu = auto()
    Conv2dBiasRelu6 = auto()
    Conv2dBiasAddRelu = auto()
    Conv2dBiasAddRelu6 = auto()
    Conv2dBiasReluAdd = auto()
    Conv2dBiasRelu6Add = auto()
    Conv2dBiasSigmoid = auto()
    GroupConv2dBiasRelu = auto()
    GroupConv2dBiasRelu6 = auto()
    TransposedConv2d = auto()
    TransposedConv2dBiasRelu = auto()
    Conv2dDepthwise = auto()
    Conv2dDepthwiseBias = auto()
    Conv2dDepthwiseBiasAdd = auto()
    Conv2dDepthwiseBiasRelu = auto()
    Conv2dDepthwiseBiasRelu6 = auto()
    Conv2dDepthwiseBiasAddRelu = auto()
    Conv2dDepthwiseBiasAddRelu6 = auto()


Conv2dKindNames = {
    Conv2dKind.Conv2d: "conv2d",
    Conv2dKind.Conv2dBias: "conv2d_bias",
    Conv2dKind.Conv2dBiasAdd: "conv2d_bias_add",
    Conv2dKind.Conv2dBiasRelu: "conv2d_bias_relu",
    Conv2dKind.Conv2dBiasRelu6: "conv2d_bias_relu6",
    Conv2dKind.Conv2dBiasAddRelu: "conv2d_bias_add_relu",
    Conv2dKind.Conv2dBiasAddRelu6: "conv2d_bias_add_relu6",
    Conv2dKind.Conv2dBiasReluAdd: "conv2d_bias_relu_add",
    Conv2dKind.Conv2dBiasRelu6Add: "conv2d_bias_relu6_add",
    Conv2dKind.Conv2dBiasSigmoid: "conv2d_bias_sigmoid",
    Conv2dKind.GroupConv2dBiasRelu: "group_conv2d_bias_relu",
    Conv2dKind.GroupConv2dBiasRelu6: "group_conv2d_bias_relu6",
    Conv2dKind.TransposedConv2d: "transposed_conv2d",
    Conv2dKind.TransposedConv2dBiasRelu: "transposed_conv2d_bias_relu",
    Conv2dKind.Conv2dDepthwise: "conv2d_depthwise",
    Conv2dKind.Conv2dDepthwiseBias: "conv2d_depthwise_bias",
    Conv2dKind.Conv2dDepthwiseBiasAdd: "conv2d_depthwise_bias_add",
    Conv2dKind.Conv2dDepthwiseBiasRelu: "conv2d_depthwise_bias_relu",
    Conv2dKind.Conv2dDepthwiseBiasRelu6: "conv2d_depthwise_bias_relu6",
    Conv2dKind.Conv2dDepthwiseBiasAddRelu: "conv2d_depthwise_bias_add_relu",
    Conv2dKind.Conv2dDepthwiseBiasAddRelu6: "conv2d_depthwise_bias_add_relu6",
}


class GemmKind(enum.Enum):
    Gemm = auto()
    GemmBias = auto()
    DynamicGemm = auto()
    BatchGemm = auto()


GemmKindNames = {
    GemmKind.Gemm: "gemm",
    GemmKind.GemmBias: "gemm_bias",
    GemmKind.BatchGemm: "batch_gemm",
    GemmKind.DynamicGemm: "dynamic_gemm",
}


class TensorOperation(enum.Enum):
    PassThrough = auto()
    Add = auto()
    Sub = auto()
    Copysign = auto()
    Div = auto()
    Mul = auto()
    Max = auto()
    Min = auto()
    Sqrtdiff = auto()
    UnaryAbs = auto()
    UnaryBankersRounding = auto()
    UnaryCeiling = auto()
    UnaryClamp = auto()
    UnaryCvt = auto()
    UnaryCopy = auto()
    UnaryElu = auto()
    UnaryFloor = auto()
    UnaryGelu = auto()
    UnaryHardwish = auto()
    UnaryLeakyRelu = auto()
    UnaryLog = auto()
    UnaryNeg = auto()
    UnarySigmoid = auto()
    UnarySquare = auto()
    UnarySqrt = auto()
    UnaryRsqrt = auto()
    UnaryTanh = auto()
    UnaryTrunc = auto()

#
TensorOperationTag = {
    TensorOperation.PassThrough: "PassThrough",
    TensorOperation.Add: "add_nd",
    TensorOperation.Sub: "subtract_nd",
    TensorOperation.Copysign: "copysign_nd",
    TensorOperation.Div: "divide_nd",
    TensorOperation.Mul: "multiply_nd",
    TensorOperation.Max: "maximum_nd",
    TensorOperation.Min: "minimum_nd",
    TensorOperation.Sqrtdiff: "squared_difference_nd",
    TensorOperation.UnaryAbs: "abs_nc",
    TensorOperation.UnaryBankersRounding: "bankers_rounding_nc",
    TensorOperation.UnaryCeiling: "ceiling_nc",
    TensorOperation.UnaryCvt: "convert_nc",
    TensorOperation.UnaryCopy: "copy_nc",
    TensorOperation.UnaryElu: "elu_nc",
    TensorOperation.UnaryFloor: "floor_nc",
    TensorOperation.UnaryGelu: "gelu_nc",
    TensorOperation.UnaryHardwish: "hardswish_nc",
    TensorOperation.UnaryLeakyRelu: "leaky_relu_nc",
    TensorOperation.UnaryLog: "log_nc",
    TensorOperation.UnaryNeg: "negate_nc",
    TensorOperation.UnarySigmoid: "sigmoid_nc",
    TensorOperation.UnarySquare: "square_nc",
    TensorOperation.UnarySqrt: "square_root_nc",
    TensorOperation.UnaryRsqrt: "reciprocal_square_root_nc",
    TensorOperation.UnaryTanh: "tanh_nc",
    TensorOperation.UnaryTrunc: "truncation_nc",
}


@dataclass
class TensorDesc:
    element: DataType
    layout: LayoutType
