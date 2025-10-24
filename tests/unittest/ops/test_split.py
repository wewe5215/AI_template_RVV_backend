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
import logging
import unittest
import os
import torch
import importlib
dt = importlib.import_module("aitemplate.testing.detect_target")
dt.IS_CPU_BACKEND = True
dt = importlib.import_module("aitemplate.compiler.compiler")
dt.IS_REMOTE_COMPILE = False
from aitemplate.compiler import compile_model, ops, Model
from aitemplate.compiler.base import IntVarTensor
from aitemplate.frontend import IntImm, IntVar, Tensor
from aitemplate.testing import detect_target
from aitemplate.testing.test_utils import (
    get_random_torch_tensor,
    get_torch_empty_tensor,
)
from aitemplate.utils import shape_utils


class SplitTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(SplitTestCase, self).__init__(*args, **kwargs)
        self.test_count = 0

    def _run_split_remote(
        self,
        *,
        input_shape,
        split_size_or_sections,
        output_masks=None,
        dim=None,
        input_type="float32",
        rebuild=True,
    ):
        logging.info(
            f"Test input shape {input_shape}, "
            f"split_size_or_sections={split_size_or_sections}, dim={dim}, output_masks={output_masks}"
        )

        split_op = ops.split()
        # generate torch reference result
        X_pt = get_random_torch_tensor(input_shape, input_type)
        if isinstance(split_size_or_sections, (list, tuple)):
            split_size_or_sections_pt = [
                shape_utils.convert_IntVar_to_int(d) for d in split_size_or_sections
            ]
        else:
            split_size_or_sections_pt = shape_utils.convert_IntVar_to_int(
                split_size_or_sections
            )
        Ys_pt = (
            torch.split(X_pt, split_size_or_sections_pt)
            if dim is None
            else torch.split(X_pt, split_size_or_sections_pt, dim)
        )
        if output_masks is not None:
            Ys_pt = [y_pt for idx, y_pt in enumerate(Ys_pt) if output_masks[idx]]
        target = detect_target(xnnpack_path="/Users/wewe5215/Desktop/XNNPACK")
        X = Tensor(shape=input_shape, dtype=input_type, name="input_0", is_input=True)
        Ys = (
            split_op(
                X,
                split_size_or_sections,
            )
            if dim is None
            else split_op(
                X,
                split_size_or_sections,
                dim,
            )
        )
        if output_masks is not None:
            split_op.remove_output_at(
                [idx for idx, mask in enumerate(output_masks) if not mask]
            )
            Ys = split_op._attrs["outputs"]
        torch.testing.assert_close(len(Ys_pt), len(Ys), atol=1e-6, rtol=1e-6)

        y_shapes = []
        for idx, Y in enumerate(Ys):
            Y._attrs["name"] = f"output_{idx}"
            Y._attrs["is_output"] = True
            y_shape = [d._attrs["values"][0] for d in Y._attrs["shape"]]
            logging.info(f"AITemplate output_{idx} shape: {y_shape}")
            y_shapes.append(y_shape)

        dll_name = f"test_{self.test_count}.so"
        if rebuild:
            module = compile_model(Ys, target, "./tmp", "split", dll_name=dll_name, remote_compile=dt.IS_REMOTE_COMPILE)
        else:
            module = Model(os.path.join("./tmp", "split", dll_name))

        outputs = {
            f"output_{idx}": get_torch_empty_tensor(y_shape, input_type)
            for idx, y_shape in enumerate(y_shapes)
        }
        module.run_with_tensors([X_pt], outputs)

        for idx, y_pt in enumerate(Ys_pt):
            y = outputs[f"output_{idx}"]
            self.assertTrue(torch.equal(y_pt, y))
            print(f"pass output {idx}!")
        self.test_count += 1

    def _run_split(
        self,
        *,
        input_shape,
        split_size_or_sections,
        output_masks=None,
        dim=None,
        input_type="float16",
    ):
        logging.info(
            f"Test input shape {input_shape}, "
            f"split_size_or_sections={split_size_or_sections}, dim={dim}"
        )

        split_op = ops.split()
        # generate torch reference result
        X_pt = get_random_torch_tensor(input_shape, input_type)
        if isinstance(split_size_or_sections, (list, tuple)):
            split_size_or_sections_pt = [
                shape_utils.convert_IntVar_to_int(d) for d in split_size_or_sections
            ]
        else:
            split_size_or_sections_pt = shape_utils.convert_IntVar_to_int(
                split_size_or_sections
            )
        Ys_pt = (
            torch.split(X_pt, split_size_or_sections_pt)
            if dim is None
            else torch.split(X_pt, split_size_or_sections_pt, dim)
        )
        if output_masks is not None:
            Ys_pt = [y_pt for idx, y_pt in enumerate(Ys_pt) if output_masks[idx]]
        target = detect_target()
        X = Tensor(shape=input_shape, dtype=input_type, name="input_0", is_input=True)
        Ys = (
            split_op(
                X,
                split_size_or_sections,
            )
            if dim is None
            else split_op(
                X,
                split_size_or_sections,
                dim,
            )
        )
        if output_masks is not None:
            split_op.remove_output_at(
                [idx for idx, mask in enumerate(output_masks) if not mask]
            )
            Ys = split_op._attrs["outputs"]
        self.assertEqual(len(Ys_pt), len(Ys))

        y_shapes = []
        for idx, Y in enumerate(Ys):
            Y._attrs["name"] = f"output_{idx}"
            Y._attrs["is_output"] = True
            y_shape = [d._attrs["values"][0] for d in Y._attrs["shape"]]
            logging.info(f"AITemplate output_{idx} shape: {y_shape}")
            y_shapes.append(y_shape)

        dll_name = f"test_{self.test_count}.so"
        module = compile_model(Ys, target, "./tmp", "split", dll_name=dll_name)

        outputs = {
            f"output_{idx}": get_torch_empty_tensor(y_shape, input_type)
            for idx, y_shape in enumerate(y_shapes)
        }
        module.run_with_tensors([X_pt], outputs)

        for idx, y_pt in enumerate(Ys_pt):
            y = outputs[f"output_{idx}"]
            self.assertTrue(torch.equal(y_pt, y))
        self.test_count += 1

    def _run_batch_split(
        self,
        *,
        batch_sizes,
        input_shape,
        split_size_or_sections,
        dim=None,
        input_type="float32",
    ):
        logging.info(
            f"Batch test: batch_sizes {batch_sizes}, input shape {input_shape}, "
            f"split_size_or_sections={split_size_or_sections}, dim={dim}"
        )

        split_op = ops.split()

        target = detect_target()
        X = Tensor(
            shape=[IntVar(values=batch_sizes, name="input_batch_0"), *input_shape],
            dtype=input_type,
            name="input_0",
            is_input=True,
        )
        Ys = (
            split_op(X, split_size_or_sections)
            if dim is None
            else split_op(X, split_size_or_sections, dim)
        )

        for idx, Y in enumerate(Ys):
            Y._attrs["name"] = f"output_{idx}"
            Y._attrs["is_output"] = True

        dll_name = f"test_{self.test_count}.so"
        module = compile_model(Ys, target, "./tmp", "split", dll_name=dll_name)

        for batch in batch_sizes:
            logging.info(f"checking batch: {batch}")

            # generate torch reference result
            X_pt = get_random_torch_tensor([batch, *input_shape], input_type)
            Ys_pt = (
                torch.split(X_pt, split_size_or_sections)
                if dim is None
                else torch.split(X_pt, split_size_or_sections, dim)
            )

            self.assertEqual(len(Ys_pt), len(Ys))

            y_shapes = [Y_pt.size() for Y_pt in Ys_pt]
            outputs = {
                f"output_{idx}": get_torch_empty_tensor(y_shape, input_type)
                for idx, y_shape in enumerate(y_shapes)
            }
            module.run_with_tensors(
                [X_pt],
                outputs,
            )

            for idx, y_pt in enumerate(Ys_pt):
                y = outputs[f"output_{idx}"]
                self.assertTrue(torch.equal(y_pt, y))
            self.test_count += 1

    def _run_batch_split_remote(
        self,
        *,
        batch_sizes,
        input_shape,
        split_size_or_sections,
        dim=None,
        input_type="float32",
    ):
        logging.info(
            f"Batch test: batch_sizes {batch_sizes}, input shape {input_shape}, "
            f"split_size_or_sections={split_size_or_sections}, dim={dim}"
        )

        split_op = ops.split()

        target = detect_target(xnnpack_path="/Users/wewe5215/Desktop/XNNPACK")
        X = Tensor(
            shape=[IntVar(values=batch_sizes, name="input_batch_0"), *input_shape],
            dtype=input_type,
            name="input_0",
            is_input=True,
        )
        Ys = (
            split_op(X, split_size_or_sections)
            if dim is None
            else split_op(X, split_size_or_sections, dim)
        )

        for idx, Y in enumerate(Ys):
            Y._attrs["name"] = f"output_{idx}"
            Y._attrs["is_output"] = True

        dll_name = f"test_{self.test_count}.so"
        module = compile_model(Ys, target, "./tmp", "split", dll_name=dll_name)

        for batch in batch_sizes:
            logging.info(f"checking batch: {batch}")

            # generate torch reference result
            X_pt = get_random_torch_tensor([batch, *input_shape], input_type)
            Ys_pt = (
                torch.split(X_pt, split_size_or_sections)
                if dim is None
                else torch.split(X_pt, split_size_or_sections, dim)
            )

            self.assertEqual(len(Ys_pt), len(Ys))

            y_shapes = [Y_pt.size() for Y_pt in Ys_pt]
            outputs = {
                f"output_{idx}": get_torch_empty_tensor(y_shape, input_type)
                for idx, y_shape in enumerate(y_shapes)
            }
            module.run_with_tensors(
                [X_pt],
                outputs,
            )

            for idx, y_pt in enumerate(Ys_pt):
                y = outputs[f"output_{idx}"]
                self.assertTrue(torch.equal(y_pt, y))
            self.test_count += 1
    @unittest.skipIf(detect_target().name() == "rvv", "Not supported by RVV.")
    def test_split(self):
        self._run_split(input_shape=[1], split_size_or_sections=1, dim=0)
        self._run_split(input_shape=[2, 1], split_size_or_sections=1, dim=0)
        self._run_split(input_shape=[2, 3], split_size_or_sections=2, dim=1)
        self._run_split(input_shape=[2, 3, 4], split_size_or_sections=10, dim=1)
        self._run_split(input_shape=[2, 3, 4], split_size_or_sections=4, dim=2)
        self._run_split(input_shape=[8, 6, 4], split_size_or_sections=2, dim=0)
        self._run_split(input_shape=[8, 6, 4], split_size_or_sections=3, dim=0)
        self._run_split(input_shape=[4097, 128, 64], split_size_or_sections=1024, dim=0)
        self._run_split(input_shape=[4097, 128, 64], split_size_or_sections=32, dim=1)

        self._run_split(input_shape=[1], split_size_or_sections=[1], dim=0)
        self._run_split(input_shape=[8, 6, 4], split_size_or_sections=[2, 3, 3], dim=0)
        self._run_split(input_shape=[8, 6, 4], split_size_or_sections=(5, 1), dim=1)
        self._run_split(input_shape=[8, 6, 4], split_size_or_sections=(2, 2), dim=2)

        self._run_split(
            input_shape=[2, 3, 4],
            split_size_or_sections=IntVarTensor(IntImm(10)),
            dim=1,
        )
        self._run_split(
            input_shape=[8, 6, 4],
            split_size_or_sections=[2, IntVarTensor(IntImm(3)), 3],
            dim=0,
        )

        # some special cases
        self._run_split(input_shape=[2, 0, 4], split_size_or_sections=4, dim=-2)
        self._run_split(input_shape=[2, 0, 4], split_size_or_sections=0, dim=-2)
        self._run_split(input_shape=[2, 0, 4], split_size_or_sections=2, dim=-1)
        self._run_split(input_shape=[2, 0, 7], split_size_or_sections=[2, 3, 2], dim=-1)
        self._run_split(input_shape=[32, 8], split_size_or_sections=[8, 0, 0], dim=-1)

    @unittest.skipIf(detect_target().name() == "rocm", "Not supported by rocm.")
    @unittest.skipIf(detect_target().name() == "cuda", "Not supported by cuda.")
    def test_split_rvv(self):
        self._run_split_remote(input_shape=[1], split_size_or_sections=1, dim=0)
        self._run_split_remote(input_shape=[2, 1], split_size_or_sections=1, dim=0)
        self._run_split_remote(input_shape=[2, 3], split_size_or_sections=2, dim=1)
        self._run_split_remote(input_shape=[2, 3, 4], split_size_or_sections=10, dim=1)
        self._run_split_remote(input_shape=[2, 3, 4], split_size_or_sections=4, dim=2)
        self._run_split_remote(input_shape=[8, 6, 4], split_size_or_sections=2, dim=0)
        self._run_split_remote(input_shape=[8, 6, 4], split_size_or_sections=3, dim=0)
        self._run_split_remote(input_shape=[4097, 128, 64], split_size_or_sections=1024, dim=0)
        self._run_split_remote(input_shape=[4097, 128, 64], split_size_or_sections=32, dim=1)

        self._run_split_remote(input_shape=[1], split_size_or_sections=[1], dim=0)
        self._run_split_remote(input_shape=[8, 6, 4], split_size_or_sections=[2, 3, 3], dim=0)
        self._run_split_remote(input_shape=[8, 6, 4], split_size_or_sections=(5, 1), dim=1)
        self._run_split_remote(input_shape=[8, 6, 4], split_size_or_sections=(2, 2), dim=2)

        self._run_split_remote(
            input_shape=[2, 3, 4],
            split_size_or_sections=IntVarTensor(IntImm(10)),
            dim=1,
        )
        self._run_split_remote(
            input_shape=[8, 6, 4],
            split_size_or_sections=[2, IntVarTensor(IntImm(3)), 3],
            dim=0,
        )

        # some special cases
        self._run_split_remote(input_shape=[2, 0, 4], split_size_or_sections=4, dim=-2)
        self._run_split_remote(input_shape=[2, 0, 4], split_size_or_sections=0, dim=-2)
        self._run_split_remote(input_shape=[2, 0, 4], split_size_or_sections=2, dim=-1)
        self._run_split_remote(input_shape=[2, 0, 7], split_size_or_sections=[2, 3, 2], dim=-1)
        self._run_split_remote(input_shape=[32, 8], split_size_or_sections=[8, 0, 0], dim=-1)

    @unittest.skipIf(detect_target().name() == "rvv", "Not supported by RVV.")
    def test_split_with_mask(self):
        self._run_split(
            input_shape=[8, 6, 4],
            split_size_or_sections=[2, 3, 3],
            output_masks=[True, False, True],
            dim=0,
        )
        self._run_split(
            input_shape=[8, 6, 4],
            split_size_or_sections=(5, 1),
            output_masks=[True, False],
            dim=1,
        )
        self._run_split(
            input_shape=[8, 6, 4],
            split_size_or_sections=(2, 2),
            output_masks=[False, True],
            dim=2,
        )

    @unittest.skipIf(detect_target().name() == "rocm", "Not supported by rocm.")
    @unittest.skipIf(detect_target().name() == "cuda", "Not supported by cuda.")
    def test_split_with_mask_rvv(self):
        self._run_split_remote(
            input_shape=[8, 6, 4],
            split_size_or_sections=[2, 3, 3],
            output_masks=[True, False, True],
            dim=0,
        )
        self._run_split_remote(
            input_shape=[8, 6, 4],
            split_size_or_sections=(5, 1),
            output_masks=[True, False],
            dim=1,
        )
        self._run_split_remote(
            input_shape=[8, 6, 4],
            split_size_or_sections=(2, 2),
            output_masks=[False, True],
            dim=2,
        )
    # TODO: batch_split for rvv (currently only support split and split with mask)
    @unittest.skipIf(detect_target().name() == "rvv", "Not supported by RVV.")
    def test_batch_split(self):
        if detect_target().name() == "rvv":
            func = self._run_batch_split_remote
            input_type = "float32"
        else:
            func = self._run_batch_split
            input_type = "float16"
        func(
            batch_sizes=[1, 1], input_shape=[2, 1], split_size_or_sections=1, dim=1, input_type=input_type
        )
        func(
            batch_sizes=[3, 4], input_shape=[2, 3, 4], split_size_or_sections=2, dim=2, input_type=input_type
        )
        func(
            batch_sizes=[3, 4], input_shape=[2, 3, 4], split_size_or_sections=2, dim=3, input_type=input_type
        )
        func(
            batch_sizes=[11, 5, 9],
            input_shape=[2, 9, 4],
            split_size_or_sections=[2, 4, 3],
            dim=2, input_type=input_type
        )

        func(
            batch_sizes=[11, 5, 9],
            input_shape=[4, 0, 4],
            split_size_or_sections=2,
            dim=1, input_type=input_type
        )
        func(
            batch_sizes=[11, 5, 9],
            input_shape=[4, 0, 5],
            split_size_or_sections=[1, 2, 2],
            dim=3, input_type=input_type
        )

    @unittest.skipIf(detect_target().name() == "rvv", "Not supported by RVV.")
    @unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
    def test_split_float(self):
        self._run_split(
            input_shape=[8, 6, 4],
            split_size_or_sections=(2, 4),
            dim=1,
            input_type="float",
        )
        self._run_batch_split(
            batch_sizes=[11, 5, 9],
            input_shape=[2, 9, 4],
            split_size_or_sections=[2, 4, 3],
            dim=2,
        )

    @unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
    @unittest.skipIf(detect_target().name() == "rvv", "Not supported by RVV.")
    def test_split_bfloat16(self):
        self._run_split(
            input_shape=[2, 3], split_size_or_sections=2, dim=1, input_type="bfloat16"
        )
        self._run_batch_split(
            batch_sizes=[3, 4],
            input_shape=[2, 3, 4],
            split_size_or_sections=2,
            dim=2,
            input_type="bfloat16",
        )
if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
