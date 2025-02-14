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
"""benchmark for resnet50"""

import os

import click
import numpy as np
import torch
from aitemplate.compiler import compile_model, Model

from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target
from modeling.resnet import build_resnet_backbone
from weight_utils import export_to_torch_tensor
import subprocess

def mark_output(y):
    """Different to PyTorch, we need to explicit mark output tensor for optimization,

    Parameters
    ----------
    y : List[Tensor]
        List of output tensors
    """
    if type(y) is not tuple:
        y = (y,)
    for i in range(len(y)):
        y[i]._attrs["is_output"] = True
        y[i]._attrs["name"] = "output_%d" % (i)
        y_shape = [d._attrs["values"][0] for d in y[i]._attrs["shape"]]
        print("output_{} shape: {}".format(i, y_shape))


def compile_module(model_name, batch_size, **kwargs):

    if model_name != "resnet50":
        raise NotImplementedError

    model_name = f"{model_name}_{batch_size}"
    target = detect_target(**kwargs)
    # Create input tensor, need to specify the shape, dtype and is_input flag
    x = Tensor(
        shape=[batch_size, 224, 224, 3], dtype="float32", name="input0", is_input=True
    )
    model = build_resnet_backbone(50, activation="ReLU")
    # Mark all parameters with name same to PyTorch name convention
    model.name_parameter_tensor()
    # Forward the input tensor to the model, get output tensor
    y = model(x)
    # Mark output tensor
    mark_output(y)
    # Compile the model
    module = compile_model(y, target, "./tmp", model_name)
    return module

def transfer_files(files, target_user: str, target_ip: str, target_dir: str):
    for f in files:
        print(f"[Host] Transferring '{f}' to {target_user}@{target_ip}:{target_dir} ...")
        subprocess.run(["scp", f, f"{target_user}@{target_ip}:{target_dir}"], check=True)
    print("[Host] All files transferred successfully.")

def benchmark(model_name, batch_size, mod=None, graph_mode=True):
    # Load params
    weights_file = f"weights_file_{batch_size}.npz"
    io_file = f"io_tensors_{batch_size}.npz"
    ait_model = export_to_torch_tensor()
    np_weights = {}
    for k, v in ait_model.items():
        np_weights[k] = v.detach().cpu().numpy().astype(np.float32)
    print(ait_model.items())
    np.savez_compressed(weights_file, **np_weights)


    # prepare input/output tensor
    x_input = torch.randn([batch_size, 224, 224, 3])
    x_input = x_input.contiguous()
    y_output = torch.zeros([batch_size, 1, 1, 1000])
    y_output = y_output.contiguous()
    x_input_np = x_input.cpu().detach().numpy().astype(np.float32)
    y_output_np = y_output.cpu().detach().numpy().astype(np.float32)
    
    # Suppose io_data is a dictionary like: {"x_input": x_input_np, "y_output": y_output_np}
    io_data = {"x_input": x_input_np, "y_output": y_output_np}

    # Save to a compressed NPZ file
    np.savez_compressed(io_file, **io_data)
    

    
    
    print(f"[Host] Saved weights to: {weights_file}")
    print(f"Input/output tensors have been saved to {io_file}")

    files = [weights_file, io_file]
    target_user = "riscv"                # Your RISC-V board username
    target_ip   = "192.168.96.48"              # Your RISC-V board IP address
    target_dir  = f"/home/{target_user}/Desktop/AITemplate_Benchmark_on_XNNPACK" # Target directory to store files
    transfer_files(files, target_user, target_ip, target_dir)
    # # warm up
    # t, _, __ = mod.benchmark_with_tensors(
    #     [x_input],
    #     [y_output],
    #     count=100,
    #     repeat=4,
    #     graph_mode=graph_mode,
    # )
    # # benchmark
    # t, _, __ = mod.benchmark_with_tensors(
    #     [x_input],
    #     [y_output],
    #     count=100,
    #     repeat=4,
    #     graph_mode=graph_mode,
    # )
    # print(f"batch_size: {batch_size}, latency: {t}")
    # dev_flag = os.environ.get("HIP_VISIBLE_DEVICES", "-1")
    # dev_flag = dev_flag.replace(",", "_")
    # with open(f"resnet50_ait_benchmark_dev_{dev_flag}.txt", "a") as f:
    #     f.write(f"batch_size: {batch_size}, latency: {t}\n")


@click.command()
@click.option(
    "--use-fp16-acc",
    type=bool,
    default=False,
    help="Whether to use FP16 for accumulation (similar to TensorRT)",
)
@click.option("--use-graph", type=bool, default=True, help="Whether to use CUDA graph")
@click.option("--batch-size", type=int, default=0, help="Batch size")
def main(use_fp16_acc=False, use_graph=True, batch_size=0):
    use_graph = False
    if batch_size < 1:
        for bs in (1, 2, 4, 8, 16, 32, 64, 128, 256):
            # compile_module("resnet50", bs, use_fp16_acc=use_fp16_acc)
            benchmark("resnet50", bs, graph_mode=use_graph)
    else:
        # compile_module("resnet50", batch_size, use_fp16_acc=use_fp16_acc)
        benchmark("resnet50", batch_size, graph_mode=use_graph)


if __name__ == "__main__":
    main()
