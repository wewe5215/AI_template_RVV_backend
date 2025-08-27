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
import unittest
import os
import torch
import numpy as np
import subprocess
from aitemplate.compiler import compile_model
from aitemplate.compiler.base import Tensor
from aitemplate.testing import detect_target
from modeling.resnet import build_resnet_backbone
from weight_utils import timm_export, export_to_torch_tensor
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

class ResNetVerification(unittest.TestCase):
    def test_resnet(self):
        batch_size = 1
        depth = 50
        model_name = f"cnhw_resnet{depth}"
        compile_module(model_name, batch_size, depth)
        torch_dtype = torch.float32
        timm_exporter = timm_export(f"resnet{depth}", pretrained=True)
        ait_params = timm_exporter.export_model(half=False)
        np_weights = {}
        for k, v in ait_params.items():
            np_weights[k] = v.detach().cpu().numpy().astype(np.float32)
        metadata_folder = f"metadata_{model_name}_{batch_size}"
        os.makedirs(metadata_folder, exist_ok=True)
        weights_file = f"{metadata_folder}/weights_file_{batch_size}.npz"
        io_file = f"{metadata_folder}/io_tensors_{batch_size}.npz"

        np.savez_compressed(weights_file, **np_weights)
        # ait model expects NHWC format
        x_ait = torch.rand([batch_size, 224, 224, 3], dtype=torch_dtype, device="cpu")
        # center the input wrt the training data for numerical stability
        x_ait -= torch.tensor([0.485, 0.456, 0.406])
        x_ait /= torch.tensor([0.229, 0.224, 0.225])
        y_ait = torch.zeros([batch_size, 1, 1, 1000], dtype=torch_dtype, device="cpu")
        x_input_np = x_ait.cpu().detach().numpy().astype(np.float32)
        y_output_np = y_ait.cpu().detach().numpy().astype(np.float32)
        # Suppose io_data is a dictionary like: {"x_input": x_input_np, "y_output": y_output_np}
        io_data = {"x_input": x_input_np, "y_output": y_output_np}
        np.savez_compressed(io_file, **io_data)
        transfer_folder(metadata_folder, TARGET_USER, TARGET_IP, target_dir)
        remote_run_program_send_back_result(target_dir, "static/test_correctness_on_riscv.py", model_name, batch_size)

        pt_model = timm_exporter.pt_model.to(dtype=torch_dtype, device="cpu")
        pt_model.eval()
        # torch model expects NCHW format
        x_pt = torch.transpose(x_ait, 1, 3).contiguous()
        with torch.no_grad():
            y_pt = pt_model(x_pt)

        output_file = f"output_file_{model_name}_{batch_size}.npz"
        output_np = np.load(output_file, allow_pickle=True)
        y_ait = torch.from_numpy(output_np["y_output"].reshape([batch_size, 1000]))
        torch.testing.assert_close(
            y_pt, y_ait.reshape([batch_size, 1000]), rtol=1e-1, atol=1e-1
        )


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
