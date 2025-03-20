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
pattern_transition = r"^features\.transition(\d+)\.conv\.weight$"
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
                print(f'f = {k}')
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

    

    def fuse_bn_post(self, conv_w, conv_b, bn_w, bn_b, bn_mean, bn_var, eps=1e-5):
        # conv_w: [out, in, kH, kW], conv_b: [out] or None
        # if conv_b is None:
        #     conv_b = torch.zeros(conv_w.shape[0])  # create zero bias if none
        # Compute scaling factor and fused bias
        conv_w = torch.tensor(conv_w, dtype=torch.float32)
        bn_mean = torch.tensor(bn_mean, dtype=torch.float32)
        bn_var = torch.tensor(bn_var, dtype=torch.float32)
        conv_b = torch.tensor(conv_b, dtype=torch.float32) if conv_b is not None else torch.zeros(conv_w.shape[0])
        bn_w = torch.tensor(bn_w, dtype=torch.float32) if bn_w is not None else torch.ones_like(bn_mean)
        bn_b = torch.tensor(bn_b, dtype=torch.float32) if bn_b is not None else torch.zeros_like(bn_mean)
        eps = torch.tensor(eps)

        bn_var_rsqrt = torch.rsqrt(bn_var + eps)
        shape = [-1, 1] + [1] * (len(conv_w.shape) - 2)
        # Scale weight (out channels)
        w_factor = bn_w * bn_var_rsqrt  # shape [out]
        fused_w = conv_w * w_factor.reshape(shape)
        fused_b = (conv_b - bn_mean) * bn_var_rsqrt * bn_w + bn_b
        for arr in [fused_w.numpy(), fused_b.numpy()]:
            if np.isnan(arr).any():
                print("fuse bn error")
        return fused_w, fused_b

    # Helper: fuse BN into conv (pre-conv scenario)
    def fuse_bn_pre(self, conv_w, conv_b, bn_w, bn_b, bn_mean, bn_var, eps=1e-5):
        # conv_w: [out, in, kH, kW], conv_b: [out] or None
        conv_w = torch.tensor(conv_w, dtype=torch.float32)
        bn_mean = torch.tensor(bn_mean, dtype=torch.float32)
        bn_var = torch.tensor(bn_var)
        conv_b = torch.tensor(conv_b, dtype=torch.float32) if conv_b is not None else torch.zeros(conv_w.shape[0])
        bn_w = torch.tensor(bn_w, dtype=torch.float32) if bn_w is not None else torch.ones_like(bn_mean)
        bn_b = torch.tensor(bn_b, dtype=torch.float32) if bn_b is not None else torch.zeros_like(bn_mean)
        eps = torch.tensor(eps)
        bn_var_rsqrt = torch.rsqrt(bn_var + eps)
        # Scale factors for each input channel
        A = bn_w * bn_var_rsqrt              # shape [in]
        B = bn_b - bn_mean * A               # shape [in]
        # 1) Scale conv weights on the input channel dimension
        fused_w = conv_w * A.reshape(1, -1, 1, 1)
        # 2) Compute new bias for each output channel
        # Sum conv_w over kernel spatial dims for each out,in pair
        weight_sum = fused_w.sum(dim=(2, 3))  # shape [out, in]
        fused_b = weight_sum.matmul(B) + conv_b  # shape [out], add conv_b if it existed
        return fused_w, fused_b

    def transform_params(self, name, fused_model):
        if name.endswith(".weight"):
            if "conv0.weight" in name:  
                # Initial conv (post-conv BN norm0)
                bn_prefix = name.replace("conv0", "norm0")
                conv_w = self.pt_state["features.conv0.weight"]
                bn_w = self.pt_state["features.norm0.weight"]
                bn_b = self.pt_state["features.norm0.bias"]
                bn_mean = self.pt_state["features.norm0.running_mean"]
                bn_var = self.pt_state["features.norm0.running_var"]
                fused_w, fused_b = self.fuse_bn_post(conv_w, None, bn_w, bn_b, bn_mean, bn_var)
                fused_w = fused_w.permute(0, 2, 3, 1).contiguous()
                fused_model["stem.conv0.weight"] = fused_w
                fused_model["stem.conv0.bias"] = fused_b
            elif "denselayer" in name and name.endswith("conv1.weight"):
                # DenseLayer bottleneck conv1 (pre-conv BN is norm1, post-conv BN is norm2)
                # Fuse pre-conv BN (norm1) into conv1
                bn_w_name = name.replace("conv", "norm")
                bn1_w = self.pt_state[bn_w_name]
                bn1_b = self.pt_state[bn_w_name.replace("weight", "bias")]
                bn1_mean = self.pt_state[bn_w_name.replace("weight", "running_mean")]
                bn1_var = self.pt_state[bn_w_name.replace("weight", "running_var")]
                conv1_w = self.pt_state.get(name)  # original conv1 weight
                fused_w, fused_b = self.fuse_bn_pre(
                    conv1_w, None, bn1_w, bn1_b, bn1_mean, bn1_var
                )
                # Permute to NHWC and store
                fused_w = fused_w.permute(0, 2, 3, 1).contiguous()

                match = re.match(pattern, name)
                block_num = match.group(1)
                layer_num = int(match.group(2))
                conv_num = match.group(3)
                # Compute the new layer index as j-1.
                new_layer_index = layer_num - 1
                new_key_weight = f"denseblock{block_num}.0.block.{new_layer_index}.conv{conv_num}.weight"
                new_key_bias = new_key_weight.replace("weight", "bias")
                fused_model[new_key_weight] = fused_w
                fused_model[new_key_bias]  = fused_b
            elif "denselayer" in name and name.endswith("conv2.weight"):
                bn_w_name = name.replace("conv", "norm")
                bn1_w = self.pt_state[bn_w_name]
                bn1_b = self.pt_state[bn_w_name.replace("weight", "bias")]
                bn1_mean = self.pt_state[bn_w_name.replace("weight", "running_mean")]
                bn1_var = self.pt_state[bn_w_name.replace("weight", "running_var")]
                conv1_w = self.pt_state.get(name)  # original conv1 weight
                fused_w, fused_b = self.fuse_bn_pre(
                    conv1_w, None, bn1_w, bn1_b, bn1_mean, bn1_var
                )
                # Permute to NHWC and store
                fused_w = fused_w.permute(0, 2, 3, 1).contiguous()

                match = re.match(pattern, name)
                block_num = match.group(1)
                layer_num = int(match.group(2))
                conv_num = match.group(3)
                # Compute the new layer index as j-1.
                new_layer_index = layer_num - 1
                new_key_weight = f"denseblock{block_num}.0.block.{new_layer_index}.conv{conv_num}.weight"
                new_key_bias = new_key_weight.replace("weight", "bias")
                fused_model[new_key_weight] = fused_w
                fused_model[new_key_bias]  = fused_b
            elif "transition" in name and name.endswith("conv.weight"):
                # Transition layer conv (pre-conv BN named "norm")
                prefix = name.rsplit(".conv.weight", 1)[0]  # e.g., "features.transition1"
                conv_w = self.pt_state[name]
                bn_w = self.pt_state[prefix + ".norm.weight"]
                bn_b = self.pt_state[prefix + ".norm.bias"]
                bn_mean = self.pt_state[prefix + ".norm.running_mean"]
                bn_var = self.pt_state[prefix + ".norm.running_var"]
                fused_w, fused_b = self.fuse_bn_pre(
                    conv_w, None, bn_w, bn_b, bn_mean, bn_var
                )
                fused_w = fused_w.permute(0, 2, 3, 1).contiguous()
                match = re.match(pattern_transition, name)
                block_num = match.group(1)
                new_key_weight = f"transition{block_num}.0.conv.weight"
                new_key_bias = new_key_weight.replace("weight", "bias")
                fused_model[new_key_weight] = fused_w
                fused_model[new_key_bias]  = fused_b
            elif name == "classifier.weight":
                # Fully connected classifier weight – fuse final norm5 BN into it
                # DenseNet norm5 is features.norm5
                bn_w =  torch.tensor(self.pt_state["features.norm5.weight"])
                bn_b =  torch.tensor(self.pt_state["features.norm5.bias"])
                bn_mean =  torch.tensor(self.pt_state["features.norm5.running_mean"])
                bn_var =  torch.tensor(self.pt_state["features.norm5.running_var"])
                fc_w =  torch.tensor(self.pt_state["classifier.weight"])
                fc_b =  torch.tensor(self.pt_state["classifier.bias"])
                # Fuse BN post-convolution (here convolution is just the identity for features into FC)
                # We treat each input feature like an "output channel" of a conv for formula:
                epsilon = 1e-5
                # Compute the BN scale factor: bn_w / sqrt(bn_var + epsilon)
                s = bn_w * torch.rsqrt(bn_var + epsilon)  # shape: [1024]

                # Fuse the FC weights by scaling each column with s.
                fused_fc_w = fc_w * s.unsqueeze(0)  # shape: [1000, 1024]

                # The batch norm transformation is:
                #   out = s * x + (bn_b - s * bn_mean)
                # So the fused FC layer becomes:
                #   y = fc_w * out + fc_b
                #     = (fc_w * s) * x + [fc_w @ (bn_b - s * bn_mean) + fc_b]
                # Thus, the new bias is:
                fused_fc_b = fc_b + fc_w.matmul(bn_b - s * bn_mean)  # shape: [1000]

                fused_model["fc.weight"] = fused_fc_w
                fused_model["fc.bias"]   = fused_fc_b
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
