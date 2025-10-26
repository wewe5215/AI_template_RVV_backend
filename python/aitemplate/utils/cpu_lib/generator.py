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

import copy

from aitemplate.utils.cpu_lib import (
    conv2d_operation as conv,
    gemm_operation as gemm,
    conv2d_pruning_operation as conv_prune,
    # groupnorm_operation as groupnorm,
    # layernorm_operation as layernorm,
    library,
    # softmax_operation as softmax,
)


###########################################################################################################
# Convolution for 2D Fwd operations (Layout : NHWC)
def CreateConv2dFwdOperator(manifest, operation_kind, out_element_op, out_data_op=""):
    in_element_op = library.TensorOperation.PassThrough
    conv2d_specialization = [
        conv.Conv2DSpecialization.ConvNhwcF32,
        conv.Conv2DSpecialization.ConvNhwcF16,
        conv.Conv2DSpecialization.ConvCNHWF32,
    ]

    operations = []
    for conv2d_spec in conv2d_specialization:
        if conv2d_spec == conv.Conv2DSpecialization.ConvNhwcF32:
            data_type = library.DataType.f32
            layout_type = library.LayoutType.NHWC
        elif conv2d_spec == conv.Conv2DSpecialization.ConvNhwcF16:
            data_type = library.DataType.f16
            layout_type = library.LayoutType.NHWC
        elif conv2d_spec == conv.Conv2DSpecialization.ConvCNHWF32:
            data_type = library.DataType.f32
            layout_type = library.LayoutType.CNHW

        a_element_desc = library.TensorDesc(
            data_type, layout_type
        )
        b_element_desc = library.TensorDesc(
            data_type, layout_type
        )
        c_element_desc = library.TensorDesc(
            data_type, layout_type
        )
        new_operation = conv.Conv2DOperation(
            operation_kind=operation_kind,
            extra_kind=out_element_op,
            A=a_element_desc,
            B=b_element_desc,
            C=c_element_desc,
            a_elem_op=in_element_op,
            b_elem_op=in_element_op,
            epilogue_functor=out_element_op,
            conv2d_specialization=conv2d_spec,
        )
        manifest.append(new_operation)
        operations.append(new_operation)

    return operations
def CreateConv2dPruningFwdOperator(manifest, operation_kind, out_element_op):
    in_element_op = library.TensorOperation.PassThrough
    conv2d_spec = conv_prune.Conv2DSpecialization.ConvCNHWF32

    operations = []
    data_type = library.DataType.f32
    layout_type = library.LayoutType.CNHW

    a_element_desc = library.TensorDesc(
        data_type, layout_type
    )
    b_element_desc = library.TensorDesc(
        data_type, layout_type
    )
    c_element_desc = library.TensorDesc(
        data_type, layout_type
    )
    LMUL_Setting = [1, 2, 4, 8]
    for LMUL in LMUL_Setting:
        for tile_size in range(3, 13):
            new_operation = conv_prune.Conv2D_Pruning_Operation(
                operation_kind=operation_kind,
                extra_kind=out_element_op,
                A=a_element_desc,
                B=b_element_desc,
                C=c_element_desc,
                a_elem_op=in_element_op,
                b_elem_op=in_element_op,
                epilogue_functor=out_element_op,
                conv2d_specialization=conv2d_spec,
                LMUL=LMUL,
                tile_size=tile_size
            )
            manifest.append(new_operation)
            operations.append(new_operation)

    return operations
# Gemm Fwd operations (Layout : RCR)
def CreateGemmFwdOperator(manifest, operation_kind, out_element_op, out_data_op=""):
    in_element_op = library.TensorOperation.PassThrough
    gemm_specialization = [
        gemm.GemmSpecialization.GemmRCR_f16,
        gemm.GemmSpecialization.GemmRCR_f32,
    ]

    operations = []
    for gemm_spec in gemm_specialization:
        if gemm_spec == gemm.GemmSpecialization.GemmRCR_f32:
            a_element_desc = library.TensorDesc(
                library.DataType.f32, library.LayoutType.RowMajor
            )
            b_element_desc = library.TensorDesc(
                library.DataType.f32, library.LayoutType.ColumnMajor
            )
            c_element_desc = library.TensorDesc(
                library.DataType.f32, library.LayoutType.RowMajor
            )
            new_operation = gemm.GemmOperation(
                operation_kind=operation_kind,
                extra_kind=out_element_op,
                A=a_element_desc,
                B=b_element_desc,
                C=c_element_desc,
                a_elem_op=in_element_op,
                b_elem_op=in_element_op,
                epilogue_functor=out_element_op,
                gemm_specialization=gemm_spec,
            )
        else:
            a_element_desc = library.TensorDesc(
                library.DataType.f16, library.LayoutType.RowMajor
            )
            b_element_desc = library.TensorDesc(
                library.DataType.f16, library.LayoutType.ColumnMajor
            )
            c_element_desc = library.TensorDesc(
                library.DataType.f16, library.LayoutType.RowMajor
            )
            new_operation = gemm.GemmOperation(
                operation_kind=operation_kind,
                extra_kind=out_element_op,
                A=a_element_desc,
                B=b_element_desc,
                C=c_element_desc,
                a_elem_op=in_element_op,
                b_elem_op=in_element_op,
                epilogue_functor=out_element_op,
                gemm_specialization=gemm_spec,
            )
        manifest.append(new_operation)
        operations.append(new_operation)

    return operations
def GenerateTensorOp(manifest):
    # Conv2d
    CreateConv2dFwdOperator(
        manifest,
        library.Conv2dKind.Conv2d,
        library.TensorOperation.PassThrough,
    )
    # Conv2dBias
    CreateConv2dFwdOperator(
        manifest,
        library.Conv2dKind.Conv2dBias,
        library.TensorOperation.PassThrough,
    )
    # Conv2dBiasRelu
    CreateConv2dFwdOperator(
        manifest,
        library.Conv2dKind.Conv2dBiasRelu,
        library.TensorOperation.PassThrough,
    )
    # Conv2dBiasRelu6
    CreateConv2dFwdOperator(
        manifest,
        library.Conv2dKind.Conv2dBiasRelu6,
        library.TensorOperation.PassThrough,
    )
    # Conv2dBiasAdd
    CreateConv2dFwdOperator(
        manifest,
        library.Conv2dKind.Conv2dBiasAdd,
        library.TensorOperation.Add,
    )
    # Conv2dBiasReluAdd
    CreateConv2dFwdOperator(
        manifest,
        library.Conv2dKind.Conv2dBiasReluAdd,
        library.TensorOperation.Add,
    )
    # Conv2dBiasRelu6Add
    CreateConv2dFwdOperator(
        manifest,
        library.Conv2dKind.Conv2dBiasRelu6Add,
        library.TensorOperation.Add,
    )
    # Conv2dBiasAddRelu
    CreateConv2dFwdOperator(
        manifest,
        library.Conv2dKind.Conv2dBiasAddRelu,
        library.TensorOperation.Add,
    )
    # Conv2dBiasAddRelu6
    CreateConv2dFwdOperator(
        manifest,
        library.Conv2dKind.Conv2dBiasAddRelu6,
        library.TensorOperation.Add,
    )
    # Conv2dDepthwise
    CreateConv2dFwdOperator(
        manifest,
        library.Conv2dKind.Conv2dDepthwise,
        library.TensorOperation.PassThrough,
    )
    # Conv2dDepthwiseBias
    CreateConv2dFwdOperator(
        manifest,
        library.Conv2dKind.Conv2dDepthwiseBias,
        library.TensorOperation.PassThrough,
    )
    # Conv2dDepthwiseBiasAdd
    CreateConv2dFwdOperator(
        manifest,
        library.Conv2dKind.Conv2dDepthwiseBiasAdd,
        library.TensorOperation.Add,
    )
    # Conv2dDepthwiseBiasRelu
    CreateConv2dFwdOperator(
        manifest,
        library.Conv2dKind.Conv2dDepthwiseBiasRelu,
        library.TensorOperation.PassThrough,
    )
    # Conv2dDepthwiseBiasRelu6
    CreateConv2dFwdOperator(
        manifest,
        library.Conv2dKind.Conv2dDepthwiseBiasRelu6,
        library.TensorOperation.PassThrough,
    )
    # Conv2dDepthwiseBiasAddRelu
    CreateConv2dFwdOperator(
        manifest,
        library.Conv2dKind.Conv2dDepthwiseBiasAddRelu,
        library.TensorOperation.Add,
    )
    # Conv2dDepthwiseBiasAddRelu6
    CreateConv2dFwdOperator(
        manifest,
        library.Conv2dKind.Conv2dDepthwiseBiasAddRelu6,
        library.TensorOperation.Add,
    )

    # Conv2dBiasTranspose
    CreateConv2dFwdOperator(
        manifest,
        library.Conv2dKind.Conv2dBiasTranspose,
        library.TensorOperation.Transpose,
    )

    # Conv2dBiasReluTranspose
    CreateConv2dFwdOperator(
        manifest,
        library.Conv2dKind.Conv2dBiasReluTranspose,
        library.TensorOperation.Transpose,
    )

    # Conv2dBiasRelu6Transpose
    CreateConv2dFwdOperator(
        manifest,
        library.Conv2dKind.Conv2dBiasRelu6Transpose,
        library.TensorOperation.Transpose,
    )

    # Conv2dDepthwiseBiasTranspose
    CreateConv2dFwdOperator(
        manifest,
        library.Conv2dKind.Conv2dDepthwiseBiasTranspose,
        library.TensorOperation.Transpose,
    )

    # Conv2dDepthwiseBiasReluTranspose
    CreateConv2dFwdOperator(
        manifest,
        library.Conv2dKind.Conv2dDepthwiseBiasReluTranspose,
        library.TensorOperation.Transpose,
    )

    # Conv2dDepthwiseBiasRelu6Transpose
    CreateConv2dFwdOperator(
        manifest,
        library.Conv2dKind.Conv2dDepthwiseBiasRelu6Transpose,
        library.TensorOperation.Transpose,
    )

    CreateConv2dPruningFwdOperator(
        manifest,
        library.Conv2dPruningKind.Conv2dPruning,
        library.TensorOperation.PassThrough,
    )

    CreateConv2dPruningFwdOperator(
        manifest,
        library.Conv2dPruningKind.Conv2dPruningBias,
        library.TensorOperation.PassThrough,
    )

    CreateConv2dPruningFwdOperator(
        manifest,
        library.Conv2dPruningKind.Conv2dPruningBiasAdd,
        library.TensorOperation.Add,
    )

    CreateConv2dPruningFwdOperator(
        manifest,
        library.Conv2dPruningKind.Conv2dPruningBiasRelu,
        library.TensorOperation.PassThrough,
    )

    CreateConv2dPruningFwdOperator(
        manifest,
        library.Conv2dPruningKind.Conv2dPruningBiasRelu6,
        library.TensorOperation.PassThrough,
    )

    CreateConv2dPruningFwdOperator(
        manifest,
        library.Conv2dPruningKind.Conv2dPruningBiasAddRelu,
        library.TensorOperation.Add,
    )

    CreateConv2dPruningFwdOperator(
        manifest,
        library.Conv2dPruningKind.Conv2dPruningBiasAddRelu6,
        library.TensorOperation.Add,
    )

    CreateConv2dPruningFwdOperator(
        manifest,
        library.Conv2dPruningKind.Conv2dPruningBiasReluAdd,
        library.TensorOperation.Add,
    )

    CreateConv2dPruningFwdOperator(
        manifest,
        library.Conv2dPruningKind.Conv2dPruningBiasRelu6Add,
        library.TensorOperation.Add,
    )
    # # Conv2dBiasSigmoid
    # CreateConv2dFwdOperator(
    #     manifest,
    #     library.Conv2dKind.GroupConv2dBiasRelu,
    #     library.TensorOperation.AddSigmoid,
    # )
    # # TransposedConv2d
    # CreateConv2dBwdOperator(
    #     manifest,
    #     library.Conv2dKind.TransposedConv2d,
    #     library.TensorOperation.PassThrough,
    #     library.MemoryDataOperation.MemorySet,
    # )
    # # TransposedConv2dBiasRelu
    # CreateConv2dBwdBiasOperator(
    #     manifest,
    #     library.Conv2dKind.TransposedConv2dBiasRelu,
    #     library.TensorOperation.AddRelu,
    #     library.MemoryDataOperation.MemorySet,
    # )
    # Gemm_RCR_Bias
    CreateGemmFwdOperator(
         manifest,
         library.GemmKind.GemmBias,
         library.TensorOperation.PassThrough,
    )
    CreateGemmFwdOperator(
         manifest,
         library.GemmKind.GemmBiasAdd,
         library.TensorOperation.Add,
    )


def GenerateRV64GCV_ZVFH(manifest, rvv_version):
    GenerateTensorOp(manifest)

