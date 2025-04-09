"""
Script for converting a MobileNetV2 model from torchvision to AITemplate.
This script loads the pretrained MobileNetV2 model from torchvision,
fuses convolution + batch normalization layers, and exports the weights
in a format (NHWC) suitable for AITemplate.
"""

import pickle
import re
import numpy as np
import torch
import torchvision.models as models
from aitemplate.testing import detect_target
from aitemplate.compiler import ops
import timm
from torchvision.models import MobileNet_V2_Weights
# -----------------------------------------------------------------------------
# Utility: Fuse convolution and batch norm (post-conv scenario)
# -----------------------------------------------------------------------------
def fuse_conv_bn(conv_w, bn_w, bn_b, bn_mean, bn_var, bn_eps=1e-5):
    """
    Fuses convolution weights with batch normalization parameters.
    The fused weights and bias are computed as:
    
       bn_var_rsqrt = 1 / sqrt(bn_var + bn_eps)
       fused_w = conv_w * (bn_w * bn_var_rsqrt).reshape(-1, 1, 1, 1)
       fused_b = bn_b - bn_mean * bn_w * bn_var_rsqrt
       
    Args:
        conv_w: Convolution weights (tensor).
        bn_w: BatchNorm weight (gamma).
        bn_b: BatchNorm bias (beta).
        bn_mean: BatchNorm running mean.
        bn_var: BatchNorm running variance.
        bn_eps: BatchNorm epsilon.
        
    Returns:
        fused_w, fused_b
    """
    conv_w = torch.tensor(conv_w)
    bn_w = torch.tensor(bn_w)
    bn_b = torch.tensor(bn_b)
    bn_mean = torch.tensor(bn_mean)
    bn_var = torch.tensor(bn_var)
    bn_eps = torch.tensor(bn_eps)

    bn_var_rsqrt = torch.rsqrt(bn_var + bn_eps)
    # Reshape scaling factor to broadcast over conv kernel dimensions.
    scale = (bn_w * bn_var_rsqrt).reshape(-1, 1, 1, 1)
    fused_w = conv_w * scale
    fused_b = bn_b - bn_mean * bn_w * bn_var_rsqrt
    return fused_w, fused_b

# -----------------------------------------------------------------------------
# MobileNetV2 Export Class (using torchvision)
# -----------------------------------------------------------------------------
class export_mobilenet:
    def __init__(self, model_name, pretrained=True):
        self.model_name = model_name
        if model_name != "mobilenetv2":
            raise NotImplementedError("Only MobileNetV2 is supported in this version.")
        with torch.no_grad():
            # Load MobileNetV2 from torchvision
            weights = MobileNet_V2_Weights.IMAGENET1K_V1
            self.pt_model = models.mobilenet_v2(weights=weights)
        self.pt_state = self.pt_model.state_dict()

    def export_model(self, half=False):
        fused_model = {}
        # Process every parameter in the state dict.
        for param_name in self.pt_state.keys():
            print(f'param_name = {param_name}')
            self.transform_params(param_name, fused_model)
        # Replace periods with underscores in keys to match AITemplate naming conventions.
        ait_model = {k.replace(".", "_"): weight for k, weight in fused_model.items()}
        # Target-specific conversion if needed.
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

    def transform_params(self, name, fused_model):
        # ----- Stem block fusion -----
        # In torchvision MobileNetV2, the stem block is at features[0]:
        # Convolution weights: "features.0.0.weight"
        # BatchNorm parameters: "features.0.1.weight", "features.0.1.bias",
        #                       "features.0.1.running_mean", and "features.0.1.running_var"
        parts = name.split(".")
        block_index = int(parts[1])
        if name.startswith("features.0.0.weight"):
            conv_w = self.pt_state[name]
            bn_w = self.pt_state["features.0.1.weight"]
            bn_b = self.pt_state["features.0.1.bias"]
            bn_mean = self.pt_state["features.0.1.running_mean"]
            bn_var = self.pt_state["features.0.1.running_var"]
            fused_w, fused_b = fuse_conv_bn(conv_w, bn_w, bn_b, bn_mean, bn_var)
            # Permute from NCHW to NHWC (if required by AITemplate)
            fused_w = fused_w.permute(0, 2, 3, 1).contiguous()
            fused_model["stem.conv.weight"] = fused_w
            fused_model["stem.conv.bias"] = fused_b
        elif name.startswith("features.18.0.weight"):
            conv_w = self.pt_state[name]
            bn_w = self.pt_state["features.18.1.weight"]
            bn_b = self.pt_state["features.18.1.bias"]
            bn_mean = self.pt_state["features.18.1.running_mean"]
            bn_var = self.pt_state["features.18.1.running_var"]
            fused_w, fused_b = fuse_conv_bn(conv_w, bn_w, bn_b, bn_mean, bn_var)
            # Permute from NCHW to NHWC (if required by AITemplate)
            fused_w = fused_w.permute(0, 2, 3, 1).contiguous()
            fused_model["final.conv.weight"] = fused_w
            fused_model["final.conv.bias"] = fused_b
        elif name.startswith("features.1.conv.0.0.weight") or name.startswith("features.1.conv.1.weight"):
            conv_index = int(parts[3])
            conv_w = self.pt_state[name]
            print(f'block_index = {block_index}, conv_index = {conv_index}')
            if conv_index == 0: # features.1.conv.0.0.weight
                conv_sub_index = int(parts[4])
                bn_index = str(int(conv_sub_index) + 1)
                bn_key_prefix = f"features.{block_index}.conv.{conv_index}.{bn_index}"
                print(f'bn_key_prefix = {bn_key_prefix}')
                bn_w = self.pt_state[bn_key_prefix + ".weight"]
                bn_b = self.pt_state[bn_key_prefix + ".bias"]
                bn_mean = self.pt_state[bn_key_prefix + ".running_mean"]
                bn_var = self.pt_state[bn_key_prefix + ".running_var"]
                new_key = f'features.{block_index-1}.depthwise.conv.weight'
            elif conv_index == 1:
                bn_index = str(int(conv_index) + 1)
                bn_key_prefix = f"features.{block_index}.conv.{bn_index}"
                bn_w = self.pt_state[bn_key_prefix + ".weight"]
                bn_b = self.pt_state[bn_key_prefix + ".bias"]
                bn_mean = self.pt_state[bn_key_prefix + ".running_mean"]
                bn_var = self.pt_state[bn_key_prefix + ".running_var"]
                new_key = f'features.{block_index-1}.projection.conv.weight'
            else:
                pass
            fused_w, fused_b = fuse_conv_bn(conv_w, bn_w, bn_b, bn_mean, bn_var)
            fused_w = fused_w.permute(0, 2, 3, 1).contiguous()
            fused_model[new_key] = fused_w
            fused_model[new_key.replace("weight", "bias")] = fused_b
        # ----- Inverted Residual Blocks -----
        # For the inverted residual blocks, assume keys are of the form:
        #   "features.<i>.conv.<j>.weight"  (with i >= 1)
        # where we assume the BN for that convolution is stored in the subsequent index.
        elif name.startswith("features.") and block_index > 1 and (("conv.0.0" in name) or ("conv.1.0") in name or ("conv.2") in name) and name.endswith("weight"):
            conv_index = int(parts[3])
            if conv_index == 0 or conv_index == 1: # features.2.conv.1.0.weight
                conv_sub_index = int(parts[4])
                conv_w = self.pt_state[name]
                bn_index = str(int(conv_sub_index) + 1)
                bn_key_prefix = f"features.{block_index}.conv.{conv_index}.{bn_index}"
                bn_w = self.pt_state[bn_key_prefix + ".weight"]
                bn_b = self.pt_state[bn_key_prefix + ".bias"]
                bn_mean = self.pt_state[bn_key_prefix + ".running_mean"]
                bn_var = self.pt_state[bn_key_prefix + ".running_var"]
                if conv_index == 0:
                    new_key = f'features.{block_index-1}.expansion.conv.weight'
                else:
                    new_key = f'features.{block_index-1}.depthwise.conv.weight'
            elif conv_index == 2:
                conv_w = self.pt_state[name]
                bn_index = str(int(conv_index) + 1)
                bn_key_prefix = f"features.{block_index}.conv.{bn_index}"
                bn_w = self.pt_state[bn_key_prefix + ".weight"]
                bn_b = self.pt_state[bn_key_prefix + ".bias"]
                bn_mean = self.pt_state[bn_key_prefix + ".running_mean"]
                bn_var = self.pt_state[bn_key_prefix + ".running_var"]
                new_key = f'features.{block_index-1}.projection.conv.weight'
            fused_w, fused_b = fuse_conv_bn(conv_w, bn_w, bn_b, bn_mean, bn_var)
            fused_w = fused_w.permute(0, 2, 3, 1).contiguous()
            fused_model[new_key] = fused_w
            fused_model[new_key.replace("weight", "bias")] = fused_b

        # ----- Classifier -----
        elif name.startswith("classifier"):
            fused_model["fc.weight"] = self.pt_state["classifier.1.weight"]
            fused_model["fc.bias"] = self.pt_state["classifier.1.bias"]

    def export_to_torch_tensor(self, half=False):
        return self.export_model(half=half)

def export_to_torch_tensor(model_name="mobilenetv2"):
    if model_name != "mobilenetv2":
        raise NotImplementedError("Only MobileNetV2 is supported in this version.")
    exporter = export_mobilenet(model_name)
    ait_model = exporter.export_to_torch_tensor(half=False)
    return ait_model

def export_to_numpy():
    ait_model = export_to_torch_tensor()
    np_weights = {}
    for k, v in ait_model.items():
        np_weights[k] = v.detach().cpu().numpy().astype(np.float32)
    # Optionally, save to file:
    # with open("mobilenetv2.pkl", "wb") as f:
    #     pickle.dump(np_weights, f)
    return np_weights

if __name__ == "__main__":
    weights = export_to_numpy()
    # For inspection: print keys and shapes.
    for k, v in weights.items():
        print(k, v.shape)
