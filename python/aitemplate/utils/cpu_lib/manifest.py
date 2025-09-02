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

#
# \file generator.py
#
# \brief Generates the cpu Library's instances
#

import os.path
import re

from aitemplate.utils.cpu_lib.library import OperationKind, OperationKindNames


class Manifest:
    def __init__(self, args=None):
        self.operations = {}
        self.args = args
        self.operation_count = 0
        self.operations_by_name = {}

        self.kernel_filter = ""
        self.kernel_filter_list = []
        self.kernel_names = []
        self.operations_enabled = []
        self.selected_kernels = []
        self.ignore_kernels = []
        self.curr_build_dir = "."
        # self.filter_by_cc = True

        if self.args:
            self.kernel_filter = self.args.kernels
            self.curr_build_dir = args.curr_build_dir

            if args.operations == "all":
                self.operations_enabled = []
            else:
                operations_list = [
                    OperationKind.Gemm,
                    OperationKind.Conv2d,
                    OperationKind.Softmax,
                ]
                self.operations_enabled = [
                    x
                    for x in operations_list
                    if OperationKindNames[x] in args.operations.split(",")
                ]

            if args.kernels == "all":
                self.kernel_names = []
            else:
                self.kernel_names = [x for x in args.kernels.split(",") if x != ""]

            self.ignore_kernels = [x for x in args.ignore_kernels.split(",") if x != ""]
            self.kernel_filter_list = []

    def _filter_string_matches(self, filter_string, haystack):
        """Returns true if all substrings appear in the haystack in order"""
        substrings = filter_string.split("*")
        for sub in substrings:
            idx = haystack.find(sub)
            if idx < 0:
                return False
            haystack = haystack[idx + len(sub) :]
        return True

    def filter(self, operation):
        """Filtering operations based on various criteria"""
        enabled = True

        if (
            len(self.operations_enabled)
            and operation.operation_kind not in self.operations_enabled
        ):
            return False
        # eliminate duplicates
        if str(operation) in self.operations_by_name.keys():
            return False
        # Filter based on list of valid substrings
        if len(self.kernel_names):
            name = str(operation)
            enabled = False

            # compare against the include list
            for name_substr in self.kernel_names:
                if self._filter_string_matches(name_substr, name):
                    enabled = True
                    break

            # compare against the exclude list
            for name_substr in self.ignore_kernels:
                if self._filter_string_matches(name_substr, name):
                    enabled = False
                    break

        # todo: filter based on compute data type
        return enabled

    def append(self, operation):
        """
        Inserts the operation.
        operation_kind -> configuration_name -> []
        """
        if self.filter(operation):
            self.selected_kernels.append(str(operation))

            self.operations_by_name[str(operation)] = operation

            # add the configuration
            configuration_name = str(operation)

            if operation.operation_kind not in self.operations.keys():
                self.operations[operation.operation_kind] = {}
            if (
                operation.extra_kind
                not in self.operations[operation.operation_kind].keys()
            ):
                self.operations[operation.operation_kind][operation.extra_kind] = {}
            
            if (
                operation.A.layout
                not in self.operations[operation.operation_kind][
                    operation.extra_kind
                ].keys()
            ):
                self.operations[operation.operation_kind][operation.extra_kind][operation.A.layout] = {}
            
            if (
                configuration_name
                not in self.operations[operation.operation_kind][
                    operation.extra_kind
                ].keys()
            ):
                self.operations[operation.operation_kind][operation.extra_kind][operation.A.layout][configuration_name] = []


            self.operations[operation.operation_kind][operation.extra_kind][operation.A.layout][configuration_name].append(operation)
            self.operation_count += 1

if __name__ == "__main__":
    from aitemplate.backend.rvv.utils import Args
    from aitemplate.utils.cpu_lib.generator import GenerateRV64GCV_ZVFH
    from aitemplate.utils.cpu_lib.library import Conv2dPruningKind, TensorOperation, LayoutType
    arch="rv64gcv1_zfh_zvfh"
    args = Args(arch)
    manifest = Manifest(args)
    GenerateRV64GCV_ZVFH(manifest, args.rvv_version)
    operations = manifest.operations[Conv2dPruningKind.Conv2dPruningBias][TensorOperation.PassThrough][LayoutType.CNHW]
    for instance_idx, (op_name, op_list) in enumerate(operations.items()):
        for op in op_list:                          # op is now a single object
            LMUL = op.LMUL
            tile_size=op.tile_size
            print(f"{op} with LMUL = {LMUL}, tile_size = {tile_size}")