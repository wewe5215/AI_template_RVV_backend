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
    # gemm_operation as gemm,
    # groupnorm_operation as groupnorm,
    # layernorm_operation as layernorm,
    library,
    # softmax_operation as softmax,
)


###########################################################################################################
# Convolution for 2D Fwd operations
def CreateConv2dFwdOperator(manifest, operation_kind, out_element_op, out_data_op=""):
    a_element_desc = library.TensorDesc(
        library.DataType.f16, library.LayoutType.NHWC
    )
    b_element_desc = library.TensorDesc(
        library.DataType.f16, library.LayoutType.NHWC
    )
    c_element_desc = library.TensorDesc(
        library.DataType.f16, library.LayoutType.NHWC
    )

    in_element_op = library.TensorOperation.PassThrough
    conv2d_specialization = [
        conv.Conv2DSpecialization.ConvNhwcF32,
        conv.Conv2DSpecialization.ConvNhwcF16,
    ]

    operations = []
    for conv2d_spec in conv2d_specialization:
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


# # Convolution for 2D Bwd operations
# def CreateConv2dBwdOperator(manifest, operation_kind, out_element_op, out_data_op=""):
#     a_element_desc = library.TensorDesc(library.DataType.f16, library.LayoutType.GNWC)
#     b_element_desc = library.TensorDesc(library.DataType.f16, library.LayoutType.GKXC)
#     c_element_desc = library.TensorDesc(library.DataType.f16, library.LayoutType.GNWK)

#     in_element_op = library.TensorOperation.PassThrough

#     conv2d_specialization = [
#         conv.Conv2DSpecialization.ConvBwdDataDefault,
#         conv.Conv2DSpecialization.ConvBwd1x1S1P0,
#     ]
#     gemm_spec = conv.Conv2DSpecialization.GemmDefault

#     operations = []
#     for conv2d_spec in conv2d_specialization:
#             new_operation = conv.Conv2DOperation(
#                 operation_kind=operation_kind,
#                 extra_kind=out_element_op,
#                 A=a_element_desc,
#                 B=b_element_desc,
#                 C=c_element_desc,
#                 a_elem_op=in_element_op,
#                 b_elem_op=in_element_op,
#                 epilogue_functor=out_element_op,
#                 c_data_op=out_data_op,
#                 conv2d_specialization=conv2d_spec,
#                 gemm_specialization=gemm_spec,
#             )
#             manifest.append(new_operation)
#             operations.append(new_operation)
#     return operations


# # Convolution for 2D Bwd + Bias operations
# def CreateConv2dBwdBiasOperator(
#     manifest, operation_kind, out_element_op, out_data_op=""
# ):
#     a_element_desc = library.TensorDesc(library.DataType.f16, library.LayoutType.GNHWK)
#     b_element_desc = library.TensorDesc(library.DataType.f16, library.LayoutType.GKYXC)
#     c_element_desc = library.TensorDesc(library.DataType.f16, library.LayoutType.GNHWC)

#     in_element_op = library.TensorOperation.PassThrough

#     conv2d_specialization = [
#         conv.Conv2DSpecialization.ConvBwdDataDefault,
#         conv.Conv2DSpecialization.ConvBwd1x1S1P0,
#     ]
#     gemm_spec = conv.Conv2DSpecialization.GemmDefault

#     operations = []
#     for conv2d_spec in conv2d_specialization:
#             new_operation = conv.Conv2DOperation(
#                 operation_kind=operation_kind,
#                 extra_kind=out_element_op,
#                 A=a_element_desc,
#                 B=b_element_desc,
#                 C=c_element_desc,
#                 a_elem_op=in_element_op,
#                 b_elem_op=in_element_op,
#                 epilogue_functor=out_element_op,
#                 c_data_op=out_data_op,
#                 conv2d_specialization=conv2d_spec,
#                 gemm_specialization=gemm_spec
#             )
#             manifest.append(new_operation)
#             operations.append(new_operation)
#     return operations



def GenerateTensorOp(manifest):
    # Conv2d
    CreateConv2dFwdOperator(
        manifest,
        library.Conv2dKind.GroupConv2dBiasRelu,
        library.TensorOperation.PassThrough,
    )
    # Conv2dBias
    # CreateConv2dFwdOperator(
    #     manifest,
    #     library.Conv2dKind.GroupConv2dBiasRelu,
    #     library.TensorOperation.Add,
    # )
    # # Conv2dBiasRelu
    # CreateConv2dFwdOperator(
    #     manifest,
    #     library.Conv2dKind.GroupConv2dBiasRelu,
    #     library.TensorOperation.AddRelu,
    # )
    # # Conv2dBiasAdd
    # CreateConv2dFwdOperator(
    #     manifest,
    #     library.Conv2dKind.GroupConv2dBiasRelu,
    #     library.TensorOperation.AddAdd,
    # )
    # # Conv2dBiasReluAdd
    # CreateConv2dFwdOperator(
    #     manifest,
    #     library.Conv2dKind.GroupConv2dBiasRelu,
    #     library.TensorOperation.AddReluAdd,
    # )
    # # Conv2dBiasAddRelu
    # CreateConv2dFwdOperator(
    #     manifest,
    #     library.Conv2dKind.GroupConv2dBiasRelu,
    #     library.TensorOperation.AddAddRelu,
    # )
    # # Conv2dBiasSigmoid
    # CreateConv2dFwdOperator(
    #     manifest,
    #     library.Conv2dKind.GroupConv2dBiasRelu,
    #     library.TensorOperation.AddSigmoid,
    #     library.MemoryDataOperation.MemorySet,
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


def GenerateRV64GCV_ZVFH(manifest, rvv_version):
    GenerateTensorOp(manifest)

