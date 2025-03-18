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
import re
import numpy as np
import timm
import torch
from aitemplate.testing import detect_target
import torchvision.models as models
CONV_WEIGHT_PATTERN = re.compile(r"features\.\d+\.(squeeze|expand1x1|expand3x3)\.weight")
name_mapping = {
    "features.0.weight": "conv1.weight",
    "features.0.bias": "conv1.bias",
    "features.3.squeeze.weight": "fire2.squeeze.weight",
    "features.3.squeeze.bias": "fire2.squeeze.bias",
    "features.3.expand1x1.weight": "fire2.expand1x1.weight",
    "features.3.expand1x1.bias": "fire2.expand1x1.bias",
    "features.3.expand3x3.weight": "fire2.expand3x3.weight",
    "features.3.expand3x3.bias": "fire2.expand3x3.bias",
    "features.4.squeeze.weight": "fire3.squeeze.weight",
    "features.4.squeeze.bias": "fire3.squeeze.bias",
    "features.4.expand1x1.weight": "fire3.expand1x1.weight",
    "features.4.expand1x1.bias": "fire3.expand1x1.bias",
    "features.4.expand3x3.weight": "fire3.expand3x3.weight",
    "features.4.expand3x3.bias": "fire3.expand3x3.bias",
    "features.5.squeeze.weight": "fire4.squeeze.weight",
    "features.5.squeeze.bias": "fire4.squeeze.bias",
    "features.5.expand1x1.weight": "fire4.expand1x1.weight",
    "features.5.expand1x1.bias": "fire4.expand1x1.bias",
    "features.5.expand3x3.weight": "fire4.expand3x3.weight",
    "features.5.expand3x3.bias": "fire4.expand3x3.bias",
    "features.7.squeeze.weight": "fire5.squeeze.weight",
    "features.7.squeeze.bias": "fire5.squeeze.bias",
    "features.7.expand1x1.weight": "fire5.expand1x1.weight",
    "features.7.expand1x1.bias": "fire5.expand1x1.bias",
    "features.7.expand3x3.weight": "fire5.expand3x3.weight",
    "features.7.expand3x3.bias": "fire5.expand3x3.bias",
    "features.8.squeeze.weight": "fire6.squeeze.weight",
    "features.8.squeeze.bias": "fire6.squeeze.bias",
    "features.8.expand1x1.weight": "fire6.expand1x1.weight",
    "features.8.expand1x1.bias": "fire6.expand1x1.bias",
    "features.8.expand3x3.weight": "fire6.expand3x3.weight",
    "features.8.expand3x3.bias": "fire6.expand3x3.bias",
    "features.9.squeeze.weight": "fire7.squeeze.weight",
    "features.9.squeeze.bias": "fire7.squeeze.bias",
    "features.9.expand1x1.weight": "fire7.expand1x1.weight",
    "features.9.expand1x1.bias": "fire7.expand1x1.bias",
    "features.9.expand3x3.weight": "fire7.expand3x3.weight",
    "features.9.expand3x3.bias": "fire7.expand3x3.bias",
    "features.10.squeeze.weight": "fire8.squeeze.weight",
    "features.10.squeeze.bias": "fire8.squeeze.bias",
    "features.10.expand1x1.weight": "fire8.expand1x1.weight",
    "features.10.expand1x1.bias": "fire8.expand1x1.bias",
    "features.10.expand3x3.weight": "fire8.expand3x3.weight",
    "features.10.expand3x3.bias": "fire8.expand3x3.bias",
    "features.12.squeeze.weight": "fire9.squeeze.weight",
    "features.12.squeeze.bias": "fire9.squeeze.bias",
    "features.12.expand1x1.weight": "fire9.expand1x1.weight",
    "features.12.expand1x1.bias": "fire9.expand1x1.bias",
    "features.12.expand3x3.weight": "fire9.expand3x3.weight",
    "features.12.expand3x3.bias": "fire9.expand3x3.bias",
    "classifier.1.weight": "conv10.weight",
    "classifier.1.bias": "conv10.bias",
}
class timm_export:
    def __init__(self, model_name, pretrained=True):
        self.model_name = model_name
        if model_name not in ["squeezenet1_0", "squeezenet1_1"]:
            raise NotImplementedError(f"Only squeezenet1_0 and squeezenet1_1 are supported, got {model_name}")
        # print(timm.list_models())
        with torch.no_grad():
            # Create the timm SqueezeNet model.
            self.pt_model = models.squeezenet1_0(pretrained=True)
        self.pt_state = self.pt_model.state_dict()

    def export_model(self, half=False):
        fused_model = {}
        for param_name in self.pt_state.keys():
            print(f'pt_state param_name: {param_name}')
            self.transform_params(param_name, fused_model)
        # Replace dots with underscores in keys for AITemplate.
        ait_model = {k.replace(".", "_"): weight.float() for k, weight in fused_model.items()}
        if detect_target().name() == "rvv":
            self.export_conv0(ait_model, fused_model)
        if half:
            half_params = {}
            for k, v in ait_model.items():
                half_params[k] = v.detach().contiguous()
            return half_params
        return ait_model

    def export_conv0(self, ait_model, fuse_model, half=False):
        pt_name = "conv1.weight"
        x = (fuse_model[pt_name])  # Ensure this is (96, 3, 7, 7)
        
        conv_w = torch.zeros((96, 7, 7, 4), device=x.device)  # Ensure it's on the same device
        conv_w[:, :, :, :3] = x  # Copy weights

        if half:
            conv_w = conv_w.half()  # Convert to half precision if needed

        # Ensure correct storage in ait_model
        ait_model[pt_name.replace(".", "_")] = conv_w

        # print(f"conv1_weight: device={conv_w.device}, contiguous={conv_w.is_contiguous()}, dtype={conv_w.dtype}")

    def transform_params(self, param_name, fused_model):
        if param_name == "features.0.weight":
            fused_model["conv1.weight"] = (self.pt_state["features.0.weight"]).permute(0, 2, 3, 1).contiguous()
            fused_model["conv1.bias"] = (self.pt_state["features.0.bias"]).contiguous()
        elif param_name == "classifier.1.weight":
            fused_model["conv10.weight"] = (self.pt_state["classifier.1.weight"]).permute(0, 2, 3, 1).contiguous()
            fused_model["conv10.bias"] = (self.pt_state["classifier.1.bias"]).contiguous()
        elif CONV_WEIGHT_PATTERN.search(param_name) is not None:
            # Permute: (out_channels, in_channels, kernel_h, kernel_w) -> (out_channels, kernel_h, kernel_w, in_channels)
            transformed_weight = (self.pt_state[param_name]).permute(0, 2, 3, 1).contiguous()
            fused_model[name_mapping[param_name]] = transformed_weight
            # If a corresponding bias exists, copy it.
            bias_key = param_name.replace("weight", "bias")
            if bias_key in self.pt_state:
                fused_model[name_mapping[bias_key]] = (self.pt_state[bias_key]).contiguous()
        else:
            pass


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
