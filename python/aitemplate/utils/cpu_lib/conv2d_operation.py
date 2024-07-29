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
from copy import deepcopy
from dataclasses import dataclass
from enum import auto
from typing import List

import jinja2

# import library

from aitemplate.utils.cpu_lib import library


class Conv2DSpecialization(enum.Enum):
    ConvFwdDefault = auto()
    ConvFwd1x1P0 = auto()
    ConvFwd1x1S1P0 = auto()
    ConvFwdOddC = auto()
    GemmDefault = auto()
    MNKPadding = auto()
    ConvBwdDataDefault = auto()
    ConvBwd1x1S1P0 = auto()


Conv2DSpecializationTag = {
    Conv2DSpecialization.ConvFwdDefault: "ck::tensor_operation::device::ConvolutionForwardSpecialization::Default",
    Conv2DSpecialization.ConvFwd1x1P0: "ck::tensor_operation::device::ConvolutionForwardSpecialization::Filter1x1Pad0",
    Conv2DSpecialization.ConvFwd1x1S1P0: "ck::tensor_operation::device::ConvolutionForwardSpecialization::Filter1x1Stride1Pad0",
    Conv2DSpecialization.ConvFwdOddC: "ck::tensor_operation::device::ConvolutionForwardSpecialization::OddC",
    Conv2DSpecialization.GemmDefault: "ck::tensor_operation::device::GemmSpecialization::Default",
    Conv2DSpecialization.MNKPadding: "ck::tensor_operation::device::GemmSpecialization::MNKPadding",
    Conv2DSpecialization.ConvBwdDataDefault: "ck::tensor_operation::device::ConvolutionBackwardDataSpecialization::Default",
    Conv2DSpecialization.ConvBwd1x1S1P0: "ck::tensor_operation::device::ConvolutionBackwardDataSpecialization::Filter1x1Stride1Pad0",
}

@dataclass
class Conv2DOperation:
    operation_kind: library.Conv2dKind
    extra_kind: library.TensorOperation
    A: library.TensorDesc
    B: library.TensorDesc
    C: library.TensorDesc
    a_elem_op: library.TensorOperation
    b_elem_op: library.TensorOperation
    epilogue_functor: library.TensorOperation
    c_data_op: library.MemoryDataOperation
    conv2d_specialization: Conv2DSpecialization
    gemm_specialization: Conv2DSpecialization

    def __str__(self) -> str:
        io_name = "{conv2d_kind}_{conv2d_specialization}_{gemm_specialization}_{a_dtype}{b_dtype}{c_dtype}_{a_layout}_{b_layout}_{c_layout}".format(
            conv2d_kind=library.Conv2dKindNames[self.operation_kind],
            conv2d_specialization=self.conv2d_specialization.value,
            gemm_specialization=self.gemm_specialization.value,
            a_dtype=library.ShortDataTypeNames[self.A.element],
            b_dtype=library.ShortDataTypeNames[self.B.element],
            c_dtype=library.ShortDataTypeNames[self.C.element],
            a_layout=library.ShortLayoutTypeNames[self.A.layout],
            b_layout=library.ShortLayoutTypeNames[self.B.layout],
            c_layout=library.ShortLayoutTypeNames[self.C.layout],
        )
        tile_name = str(self.tile_desc)
        return "{io_name}_{tile_name}_{epilogue_functor}".format(
            io_name=io_name,
            tile_name=tile_name,
            epilogue_functor=library.ShortTensorOperationNames[self.epilogue_functor],
        )

    def accumulator_type(self):
        return library.DataType.f32

    def emit(self) -> str:
        template = jinja2.Template(
            """"""
        )
        return template.render(
            name=self.__str__(),
            InLayout=library.LayoutTag[self.A.layout],
            WeiLayout=library.LayoutTag[self.B.layout],
            OutLayout=library.LayoutTag[self.C.layout],
            ADType=library.DataTypeTag[self.A.element],
            BDType=library.DataTypeTag[self.B.element],
            CDType=library.DataTypeTag[self.C.element],
            AccDType=library.DataTypeTag[library.DataType.f32],
            CShuffleDType=library.DataTypeTag[self.C.element],
            A_elem_op=library.TensorOperationTag[self.a_elem_op],
            B_elem_op=library.TensorOperationTag[self.b_elem_op],
            epilogue_functor=library.TensorOperationTag[self.epilogue_functor],
            C_data_op=library.MemoryDataOperationTag.get(self.c_data_op, -1),
            Conv2DSpecialization=Conv2DSpecializationTag[self.conv2d_specialization],
            GemmSpecialization=Conv2DSpecializationTag[self.gemm_specialization],
            func=library.ShortTensorOperationNames[self.epilogue_functor],
        )


if __name__ == "__main__":
    A = library.TensorDesc(library.DataType.f16, library.LayoutType.RowMajor)
    B = library.TensorDesc(library.DataType.f16, library.LayoutType.ColumnMajor)
    C = library.TensorDesc(library.DataType.f16, library.LayoutType.RowMajor)
    Conv2DOp = Conv2DOperation(
        operation_kind=library.Conv2dKind.Conv2d,
        extra_kind=library.TensorOperation.PassThrough,
        A=A,
        B=B,
        C=C,
        a_elem_op=library.TensorOperation.PassThrough,
        b_elem_op=library.TensorOperation.PassThrough,
        epilogue_functor=library.TensorOperation.PassThrough,
        c_data_op="",
        conv2d_specialization=Conv2DSpecialization.ConvFwdDefault,
        gemm_specialization=Conv2DSpecialization.GemmDefault,
    )
    print(str(Conv2DOp))
    print(Conv2DOp.emit())
