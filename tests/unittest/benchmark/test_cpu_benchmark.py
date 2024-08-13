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
import itertools
import json
import logging
import unittest
import uuid

import torch

from aitemplate.compiler import ops
from aitemplate.compiler.base import Tensor
from aitemplate.compiler.compiler import compile_model

from aitemplate.testing import detect_target
from aitemplate.testing.benchmark_ait import make_input_output_pools, run_benchmark
from aitemplate.testing.benchmark_pt import benchmark_torch_function
from aitemplate.utils import shape_utils

NK_SHAPES = ((8314, 3072), (6912, 8314))
INPUT_POOL_SIZE = 20
BATCH_SIZES = (
    1,
)


_LOGGER = logging.getLogger(__name__)


class GemmRCRModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        return torch.nn.functional.linear(a, b)


class GemmRCRFunction:
    def __init__(self, inputs_pool):
        self._it_pool = 0
        self._as = [t["a"] for t in inputs_pool]
        self._bs = [t["b"] for t in inputs_pool]
        self._inputs_pool_size = len(inputs_pool)
        self._module = GemmRCRModule()

    def next_input(self):
        self._it_pool += 1
        self._it_pool %= self._inputs_pool_size
        return self._as[self._it_pool], self._bs[self._it_pool]

    def __call__(self):
        return self._module(*self.next_input())


def build_ait_module_gemm_rcr(*, ms, n, k, split_k, test_name):
    target = detect_target(use_fp16_acc=True)
    input_params = {
        "dtype": "float32",
        "is_input": True,
    }
    a = Tensor(shape=[shape_utils.gen_int_var_min_max(ms), k], name="a", **input_params)
    b = Tensor(shape=[n, k], name="b", **input_params)
    bias = Tensor(shape=[n], name="bias", **input_params)
    OP = ops.gemm_rcr_bias()
    OP._attrs["split_k_hints"] = (split_k,)
    output = OP(a, b, bias)
    output._attrs["name"] = "output"
    output._attrs["is_output"] = True
    return compile_model(output, target, "./tmp", test_name=test_name)


def eval_pt_gemm_rcr(*, m, n, k):
    input_params = {
        "dtype": torch.float32,
        "device": "cpu",
    }
    a = torch.rand(m, k, **input_params)
    b = torch.rand(n, k, **input_params)
    bias = torch.rand(n, **input_params)
    output = torch.nn.functional.linear(a, b, bias).to(torch.float32)
    return {"a": a, "b": b, "bias": bias, "output": output}


class TestGemmRCRBenchmark(unittest.TestCase):
    def test_benchmark(self):
        split_ks = sorted(set(range(1, 6)).union([2**i for i in range(5)]))
        for split_k, (n, k) in itertools.product(split_ks, NK_SHAPES):
            NUM_ITERS = 10
            NUM_WARMUP_ITERS = 1
            ait_module = build_ait_module_gemm_rcr(
                ms=BATCH_SIZES,
                n=n,
                k=k,
                split_k=split_k,
                test_name=f"gemm_rcr_{split_k=}_{uuid.uuid4().hex}",
            )
            for m in BATCH_SIZES:
                mnk = {"m": m, "n": n, "k": k}
                _LOGGER.warning(f"mnk={mnk}, split_k={split_k}")
                inputs_pool, outputs_pool = make_input_output_pools(
                    pool_size=INPUT_POOL_SIZE,
                    eval_pt_func=lambda: eval_pt_gemm_rcr(**mnk),
                    input_filter_func=lambda name, _: not name.startswith("output"),
                    output_filter_func=lambda name, _: name.startswith("output"),
                )
                gemm_rcr_function = GemmRCRFunction(inputs_pool)

                pt_outputs = eval_pt_gemm_rcr(**mnk)
                ait_outputs = {"output": torch.empty_like(pt_outputs["output"])}
                _LOGGER.info("ait_module.run_with_tensors")
                ait_module.run_with_tensors(
                    {k: v for k, v in pt_outputs.items() if k != "output"},
                    ait_outputs,
                )
                _LOGGER.info("ait_module.run_with_tensors done")
                torch.testing.assert_close(
                    ait_outputs["output"], pt_outputs["output"], rtol=1, atol=1
                )
                mean_runtime_ait = run_benchmark(
                    ait_module=ait_module,
                    inputs_pool=inputs_pool,
                    outputs_pool=outputs_pool,
                    num_iters=NUM_ITERS,
                    num_warmup_iters=NUM_WARMUP_ITERS,
                )

                mean_runtime_pt = benchmark_torch_function(
                    iters=NUM_ITERS, function=gemm_rcr_function
                )

                benchmark_results = {
                    "function": "gemm_rcr_bias",
                    "mean_runtime_ait_ms": round(mean_runtime_ait, 5),
                    "mean_runtime_pt_ms": round(mean_runtime_pt, 5),
                    **mnk,
                }
                _LOGGER.warning(
                    f"Benchmark results {json.dumps(benchmark_results, separators=(',', ':'))}",
                )


if __name__ == "__main__":
    unittest.main()
