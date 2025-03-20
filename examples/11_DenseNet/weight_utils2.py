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
Script for converting a DenseNet-121 model from timm to AITemplate.
"""

import pickle
import re
import click
import numpy as np
import timm
import torch
from aitemplate.testing import detect_target
pattern = r"^features\.denseblock(\d+)\.denselayer(\d+)\.conv(\d+)\.weight$"
CONV_WEIGHT_PATTERN = re.compile(pattern)

class timm_export:
    def __init__(self, model_name, pretrained=True):
        self.model_name = model_name
        if model_name != "densenet121":
            raise NotImplementedError("Only DenseNet-121 is supported in this version.")
        with torch.no_grad():
            self.pt_model = timm.create_model(model_name, pretrained=pretrained, num_classes=1000)
        self.pt_state = self.pt_model.state_dict()

    def export_model(self, half=False):
        fused_model = {}
        # Process every parameter in the state dict.
        for param_name in self.pt_state.keys():
            print(f'param_name:{param_name}')
            self.transform_params(param_name, fused_model)
        # Replace "." with "_" in keys to match AITemplate naming.
        ait_model = {k.replace(".", "_"): weight for k, weight in fused_model.items()}
        # Handle target-specific conversions.
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

    def fuse_conv_bn_weights(
        self, conv_w, conv_b, bn_rm, bn_rv, bn_eps, bn_w, bn_b, transpose=False
    ):
        conv_w = torch.tensor(conv_w)
        bn_rm = torch.tensor(bn_rm)
        bn_rv = torch.tensor(bn_rv)
        conv_b = torch.tensor(conv_b) if conv_b is not None else torch.zeros_like(bn_rm)
        bn_w = torch.tensor(bn_w) if bn_w is not None else torch.ones_like(bn_rm)
        bn_b = torch.tensor(bn_b) if bn_b is not None else torch.zeros_like(bn_rm)
        bn_eps = torch.tensor(bn_eps)

        bn_var_rsqrt = torch.rsqrt(bn_rv + bn_eps)

        if transpose:
            shape = [1, -1] + [1] * (len(conv_w.shape) - 2)
        else:
            shape = [-1, 1] + [1] * (len(conv_w.shape) - 2)

        conv_w = conv_w * (bn_w * bn_var_rsqrt).reshape(shape)
        conv_b = (conv_b - bn_rm) * bn_var_rsqrt * bn_w + bn_b

        # Convert from NCHW to NHWC.
        conv_w = conv_w.permute(0, 2, 3, 1).contiguous()
        for arr in [conv_w.numpy(), conv_b.numpy()]:
            if np.isnan(arr).any():
                print("fuse bn error")
        return conv_w, conv_b

    def transform_conv0(self):
        # For DenseNet-121, the first conv is at features.conv0,
        # and its following BN is at features.norm0.
        conv_w = self.pt_state["features.conv0.weight"]
        bn_w = self.pt_state["features.norm0.weight"]
        bn_b = self.pt_state["features.norm0.bias"]
        bn_rm = self.pt_state["features.norm0.running_mean"]
        bn_rv = self.pt_state["features.norm0.running_var"]
        fused_w, fused_b = self.fuse_conv_bn_weights(
            conv_w, None, bn_rm, bn_rv, 1e-5, bn_w, bn_b
        )
        return fused_w, fused_b
    

    def fuse_conv_double_bn_weights(
            self, conv_w, conv_b,
            bn1_rm, bn1_rv, bn1_eps, bn1_w, bn1_b,
            bn2_rm, bn2_rv, bn2_eps, bn2_w, bn2_b,
            transpose=False
        ):
        # Convert inputs to tensors.
        conv_w = torch.tensor(conv_w)
        conv_b = torch.tensor(conv_b) if conv_b is not None else torch.zeros_like(torch.tensor(bn1_rm))
        
        bn1_rm = torch.tensor(bn1_rm)
        bn1_rv = torch.tensor(bn1_rv)
        bn1_eps = torch.tensor(bn1_eps)
        bn1_w = torch.tensor(bn1_w) if bn1_w is not None else torch.ones_like(bn1_rm)
        bn1_b = torch.tensor(bn1_b) if bn1_b is not None else torch.zeros_like(bn1_rm)
        
        bn2_rm = torch.tensor(bn2_rm)
        bn2_rv = torch.tensor(bn2_rv)
        bn2_eps = torch.tensor(bn2_eps)
        bn2_w = torch.tensor(bn2_w) if bn2_w is not None else torch.ones_like(bn2_rm)
        bn2_b = torch.tensor(bn2_b) if bn2_b is not None else torch.zeros_like(bn2_rm)
        
        # Compute effective parameters for BN1.
        bn1_scale = bn1_w / torch.sqrt(bn1_rv + bn1_eps)
        bn1_bias  = -bn1_rm * bn1_scale + bn1_b
        # Compute effective parameters for BN2.
        bn2_scale = bn2_w / torch.sqrt(bn2_rv + bn2_eps)
        bn2_bias  = -bn2_rm * bn2_scale + bn2_b
        
        # Compose the two BN layers: the overall effective transformation is
        # x' = (x*bn1_scale + bn1_bias)*bn2_scale + bn2_bias.
        effective_scale = bn1_scale * bn2_scale
        effective_bias  = bn2_scale * bn1_bias + bn2_bias

        # Fuse with convolution parameters.
        if transpose:
            shape = [1, -1] + [1] * (len(conv_w.shape) - 2)
        else:
            shape = [-1, 1] + [1] * (len(conv_w.shape) - 2)

        conv_w_fused = conv_w * effective_scale.reshape(shape)
        conv_b_fused = conv_b * effective_scale + effective_bias

        # Optionally convert conv_w_fused from NCHW to NHWC.
        conv_w_fused = conv_w_fused.permute(0, 2, 3, 1).contiguous()

        for arr in [conv_w_fused.numpy(), conv_b_fused.numpy()]:
            if np.isnan(arr).any():
                print("fuse double BN error")
        return conv_w_fused, conv_b_fused

    def transform_conv_with_double_bn(self, pt_state):
        # Assume:
        #   - The convolution is at key "features.denseblock4.denselayer16.conv2.weight"
        #   - BN1 (denselayer16.norm2) is given by keys:
        #         "features.denseblock4.denselayer16.norm2.weight", "bias", "running_mean", "running_var", eps assumed to be 1e-5.
        #   - BN2 (features.norm5) is given by keys:
        #         "features.norm5.weight", "features.norm5.bias", "features.norm5.running_mean", "features.norm5.running_var", eps assumed to be 1e-5.
        
        conv_w = pt_state["features.denseblock4.denselayer16.conv2.weight"]
        conv_b = None  # Or if present, the convolution bias.
        
        bn1_w = pt_state["features.denseblock4.denselayer16.norm2.weight"]
        bn1_b = pt_state["features.denseblock4.denselayer16.norm2.bias"]
        bn1_rm = pt_state["features.denseblock4.denselayer16.norm2.running_mean"]
        bn1_rv = pt_state["features.denseblock4.denselayer16.norm2.running_var"]
        
        bn2_w = pt_state["features.norm5.weight"]
        bn2_b = pt_state["features.norm5.bias"]
        bn2_rm = pt_state["features.norm5.running_mean"]
        bn2_rv = pt_state["features.norm5.running_var"]

        fused_w, fused_b = self.fuse_conv_double_bn_weights(
            conv_w, conv_b,
            bn1_rm, bn1_rv, 1e-5, bn1_w, bn1_b,
            bn2_rm, bn2_rv, 1e-5, bn2_w, bn2_b,
            transpose=False
        )
        return fused_w, fused_b


    def transform_params(self, param_name, fused_model):
        if param_name == "features.conv0.weight":
            fused_w, fused_b = self.transform_conv0()
            # Store the fused parameters under the "stem" key.
            fused_model["stem.conv0.weight"] = fused_w
            fused_model["stem.conv0.bias"] = fused_b
        elif param_name == "classifier.weight":
            # Map DenseNet's classifier layer to "fc" in AITemplate.
            fused_model["fc.weight"] = self.pt_state["classifier.weight"]
            fused_model["fc.bias"] = self.pt_state["classifier.bias"]
        elif param_name == "features.denseblock4.denselayer16.conv2.weight":
            fused_w, fused_b = self.transform_conv_with_double_bn(self.pt_state)
            fused_model["denseblock4.0.block.15.conv2.weight"] = fused_w
            fused_model["denseblock4.0.block.15.conv2.bias"] = fused_b
        elif CONV_WEIGHT_PATTERN.search(param_name) is not None:
            # For other parameters (dense block layers, transitions, etc.),
            # simply copy them over.
            bn_w_name = param_name.replace("conv", "norm")
            conv_w = self.pt_state[param_name]
            print(f'bn_w_name = {bn_w_name}, param_name = {param_name}')
            bn_w = self.pt_state[bn_w_name]
            bn_b = self.pt_state[bn_w_name.replace("weight", "bias")]
            bn_rm = self.pt_state[bn_w_name.replace("weight", "running_mean")]
            bn_rv = self.pt_state[bn_w_name.replace("weight", "running_var")]
            fused_w, fused_b = self.fuse_conv_bn_weights(
                conv_w, None, bn_rm, bn_rv, 1e-5, bn_w, bn_b
            )
            match = re.match(pattern, param_name)
            block_num = match.group(1)
            layer_num = int(match.group(2))
            conv_num = match.group(3)
            # Compute the new layer index as j-1.
            new_layer_index = layer_num - 1
            new_key_weight = f"denseblock{block_num}.0.block.{new_layer_index}.conv{conv_num}.weight"
            new_key_bias = new_key_weight.replace("weight", "bias")
            fused_model[new_key_weight] = fused_w
            fused_model[new_key_bias] = fused_b
    def transform_downsample(self, param_name):
        assert "downsample" in param_name
        tags = param_name.split(".")
        block_tag = ".".join(tags[:2])
        conv_w = self.pt_state[f"{block_tag}.downsample.0.weight"]
        bn_w = self.pt_state[f"{block_tag}.downsample.1.weight"]
        bn_b = self.pt_state[f"{block_tag}.downsample.1.bias"]
        bn_rm = self.pt_state[f"{block_tag}.downsample.1.running_mean"]
        bn_rv = self.pt_state[f"{block_tag}.downsample.1.running_var"]
        fused_w, fused_b = self.fuse_conv_bn_weights(
            conv_w, None, bn_rm, bn_rv, 1e-5, bn_w, bn_b
        )
        return fused_w, fused_b

    def export_conv0(self, ait_model, fuse_model):
        pt_name = "stem.conv1.weight"
        x = fuse_model[pt_name]
        conv_w = torch.zeros((64, 7, 7, 4))
        conv_w[:, :, :, :3] = x
        ait_model[pt_name.replace(".", "_")] = conv_w


def export_to_torch_tensor(model_name="densenet121"):
    if model_name != "densenet121":
        raise NotImplementedError("Only DenseNet-121 is supported in this version.")
    timm2ait = timm_export(model_name)
    ait_model = timm2ait.export_model(half=False)
    return ait_model

# @click.command()
# @click.option("--param-path", type=str, default="densenet121.pkl")
def export_to_numpy():
    ait_model = export_to_torch_tensor()
    np_weights = {}
    for k, v in ait_model.items():
        np_weights[k] = v.detach().cpu().numpy().astype(np.float32)

    # with open(param_path, "wb") as f:
    #     pickle.dump(np_weights, f)

if __name__ == "__main__":
    export_to_numpy()