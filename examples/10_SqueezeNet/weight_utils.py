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
"""
Script for converting a SqueezeNet model from timm to AITemplate format.
Tested on SqueezeNet v1.0 / v1.1.
"""

import pickle
import click
import numpy as np
import timm
import torch
from aitemplate.testing import detect_target
import torchvision.models as models

class timm_export:
    def __init__(self, model_name, pretrained=True):
        self.model_name = model_name
        if model_name not in ["squeezenet1_0", "squeezenet1_1"]:
            raise NotImplementedError(f"Only squeezenet1_0 and squeezenet1_1 are supported, got {model_name}")
        print(timm.list_models())
        with torch.no_grad():
            # Create the timm SqueezeNet model.
            self.pt_model = models.squeezenet1_0(pretrained=True)
        self.pt_state = self.pt_model.state_dict()

    def export_model(self, half=False):
        fused_model = {}
        for param_name in self.pt_state.keys():
            self.transform_params(param_name, fused_model)
        # Replace dots with underscores in keys for AITemplate.
        ait_model = {k.replace(".", "_"): weight for k, weight in fused_model.items()}
        if detect_target().name() == "rvv":
            float_params = {}
            for k, v in ait_model.items():
                float_params[k] = v.detach().cpu()
            return float_params
        if half:
            half_params = {}
            for k, v in ait_model.items():
                half_params[k] = v.detach().cuda().half().contiguous()
            return half_params
        return ait_model

    def transform_params(self, param_name, fused_model):
        weight = self.pt_state[param_name]
        # For convolution weights (4D tensors), permute from NCHW to NHWC.
        if weight.ndim == 4:
            # Permute: (out_channels, in_channels, kernel_h, kernel_w) -> (out_channels, kernel_h, kernel_w, in_channels)
            transformed_weight = weight.permute(0, 2, 3, 1).contiguous()
            fused_model[param_name] = transformed_weight
            # If a corresponding bias exists, copy it.
            bias_key = param_name.replace("weight", "bias")
            if bias_key in self.pt_state:
                fused_model[bias_key] = self.pt_state[bias_key]
        else:
            # For other parameters (e.g. linear weights/biases), copy directly.
            fused_model[param_name] = weight


def export_to_torch_tensor(model_name="squeezenet1_0"):
    if model_name not in ["squeezenet1_0", "squeezenet1_1"]:
        raise NotImplementedError
    timm2ait = timm_export(model_name)
    ait_model = timm2ait.export_model(half=False)
    return ait_model


@click.command()
@click.option("--param-path", type=str, default="squeezenet1_0.pkl")
def export_to_numpy(param_path):
    ait_model = export_to_torch_tensor(model_name="squeezenet1_0")
    np_weights = {}
    for k, v in ait_model.items():
        np_weights[k] = v.detach().cpu().numpy().astype(np.float32)
    with open(param_path, "wb") as f:
        pickle.dump(np_weights, f)


if __name__ == "__main__":
    export_to_numpy()
