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
import time
import click
import numpy as np
import torch
from aitemplate.compiler import compile_model, Model

from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target
from modeling.resnet import build_resnet_backbone
from weight_utils import timm_export, export_to_torch_tensor
import subprocess
from aitemplate.utils.remote_send_receive_files import (
    transfer_folder, 
    check_remote_file_exists, 
    retrieve_confirmation_file, 
    poll_for_confirmation,
    TARGET_USER,
    TARGET_IP,
    remote_run_program_send_back_result
)
target_dir  = f"/home/{TARGET_USER}/Desktop/AITemplate_Benchmark_on_XNNPACK" # Target directory to store files

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


def compile_module(model_name, batch_size, depth):
    target = detect_target()
    # Create input tensor, need to specify the shape, dtype and is_input flag
    x = Tensor(
        shape=[batch_size, 224, 224, 3], dtype="float32", name="input0", is_input=True
    )
    model = build_resnet_backbone(depth, activation="ReLU")
    # Mark all parameters with name same to PyTorch name convention
    model.name_parameter_tensor()
    # Forward the input tensor to the model, get output tensor
    y = model(x)
    # Mark output tensor
    mark_output(y)
    # Compile the model
    module = compile_model(y, target, "./tmp", model_name, remote_compile = True)
    return module


def benchmark(model_name, batch_size, mod=None, graph_mode=True, depth=50):
    metadata_folder = f"metadata_{model_name}_{batch_size}"
    os.makedirs(metadata_folder, exist_ok=True)
    weights_file = f"{metadata_folder}/weights_file_{batch_size}.npz"
    io_file = f"{metadata_folder}/io_tensors_{batch_size}.npz"
    timm_exporter = timm_export(f"resnet{depth}", pretrained=True)
    ait_params = timm_exporter.export_model(half=False)
    np_weights = {}
    for k, v in ait_params.items():
        np_weights[k] = v.detach().cpu().numpy().astype(np.float32)
    np.savez_compressed(weights_file, **np_weights)

    # prepare input/output tensor
    torch_dtype = torch.float32
    x_ait = torch.rand([batch_size, 224, 224, 3], dtype=torch_dtype, device="cpu")
    x_ait -= torch.tensor([0.485, 0.456, 0.406])
    x_ait /= torch.tensor([0.229, 0.224, 0.225])
    y_ait = torch.zeros([batch_size, 1, 1, 1000], dtype=torch_dtype, device="cpu")
    x_input = torch.randn([batch_size, 224, 224, 3])
    x_ait = x_ait.contiguous()
    y_ait = torch.zeros([batch_size, 1, 1, 1000])
    y_ait = y_ait.contiguous()
    x_input_np = x_ait.cpu().detach().numpy().astype(np.float32)
    y_output_np = y_ait.cpu().detach().numpy().astype(np.float32)
    io_data = {"x_input": x_input_np, "y_output": y_output_np}
    np.savez_compressed(io_file, **io_data)
    transfer_folder(metadata_folder, TARGET_USER, TARGET_IP, target_dir)
    remote_run_program_send_back_result(target_dir, "static/run_benchmark_on_riscv.py", model_name, batch_size, is_benchmark=True)

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
    depth = 50
    model_name = f"resnet{depth}"
    if batch_size < 1:
        for bs in (1, 2, 4):
            compile_module(model_name, batch_size, depth)
            benchmark(model_name, bs, graph_mode=use_graph, depth=depth)
    else:
        compile_module(model_name, batch_size, depth)
        benchmark(model_name, batch_size, graph_mode=use_graph, depth=depth)
if __name__ == "__main__":
    main()
