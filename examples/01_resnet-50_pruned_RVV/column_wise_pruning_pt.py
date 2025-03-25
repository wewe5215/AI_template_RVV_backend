import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from huggingface_hub import Repository, create_repo
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import torchvision
import evaluate
import torchvision.transforms as transforms
import timm
import re
import argparse
import json
import logging
import math
import os
import random
from pathlib import Path
import numpy as np

vlen = 256
from group_op_and_lmul import fetch_lmul_for_op

CONV_WEIGHT_PATTERN = re.compile(r"conv\d+\.weight")
logger = get_logger(__name__)
layers_to_prune = [
    'layer1.0.conv1.weight', 'layer1.0.conv2.weight', 'layer1.0.conv3.weight',
    'layer1.1.conv1.weight', 'layer1.1.conv2.weight', 'layer1.1.conv3.weight',
    'layer1.2.conv1.weight', 'layer1.2.conv2.weight', 'layer1.2.conv3.weight',
    'layer2.0.conv1.weight', 'layer2.0.conv2.weight', 'layer2.0.conv3.weight',
    'layer2.1.conv1.weight', 'layer2.1.conv2.weight', 'layer2.1.conv3.weight',
    'layer2.2.conv1.weight', 'layer2.2.conv2.weight', 'layer2.2.conv3.weight',
    'layer2.3.conv1.weight', 'layer2.3.conv2.weight', 'layer2.3.conv3.weight',
    'layer3.0.conv1.weight', 'layer3.0.conv2.weight', 'layer3.0.conv3.weight',
    'layer3.1.conv1.weight', 'layer3.1.conv2.weight', 'layer3.1.conv3.weight',
    'layer3.2.conv1.weight', 'layer3.2.conv2.weight', 'layer3.2.conv3.weight',
    'layer3.3.conv1.weight', 'layer3.3.conv2.weight', 'layer3.3.conv3.weight',
    'layer3.4.conv1.weight', 'layer3.4.conv2.weight', 'layer3.4.conv3.weight',
    'layer3.5.conv1.weight', 'layer3.5.conv2.weight', 'layer3.5.conv3.weight',
    'layer4.0.conv1.weight', 'layer4.0.conv2.weight', 'layer4.0.conv3.weight',
    'layer4.1.conv1.weight', 'layer4.1.conv2.weight', 'layer4.1.conv3.weight',
    'layer4.2.conv1.weight', 'layer4.2.conv2.weight', 'layer4.2.conv3.weight'
]

def f32_data_pruning_column_wise_with_ratio(weight, nr, mr, pruning_ratio):
    """
    Performs column-wise pruning on a 2D weight array using a given pruning ratio.
    
    Parameters:
      weight: a 2D numpy array of shape (output_channel, input_channel)
      nr: an integer multiplier for the recorded indices
      mr: block size (number of rows per block)
      pruning_ratio: fraction of columns to prune (e.g., 0.5 means prune bottom 50% columns)
                     
    Returns:
      mask: a 1D numpy array (flattened binary mask)
      indices: a 1D numpy array (dtype uint16) with the recorded column indices
    """
    output_channel, input_channel = weight.shape
    indices = []  # For storing selected column indices (from the first row in each block).
    mask = []     # Flattened binary mask.
    for i in range(0, output_channel, mr):
        end_offset = min(mr, output_channel - i)
        block = weight[i:i+end_offset, :]  # (end_offset, input_channel)
        accumulator = np.sum(np.abs(block), axis=0)
        keep_count = int(np.ceil((1 - pruning_ratio) * input_channel))
        if np.all(accumulator == accumulator[0]):
            for j in range(end_offset):
                for k in range(input_channel):
                    mask.append(1 if k < keep_count else 0)
                    if j == 0:
                        indices.append(k)
        else:
            threshold = np.percentile(accumulator, pruning_ratio * 100)
            for j in range(end_offset):
                for k in range(input_channel):
                    select = (accumulator[k] >= threshold) if (input_channel % 2 == 0) else (accumulator[k] > threshold)
                    if select:
                        mask.append(1)
                        if j == 0:
                            indices.append(k)
                    else:
                        mask.append(0)
    indices = np.array(indices, dtype=np.uint16)
    mask = np.array(mask, dtype=np.uint8)
    return mask, indices

def perform_pruning(model, pruning_ratio=0.5):
    """
    Applies custom column-wise pruning to the specified layers in the model.
    """
    batch_size = 1
    weight_to_lmul = fetch_lmul_for_op(batch_size)
    for layer_name in layers_to_prune:
        # Split into module name and parameter name.
        module_name, weight_name = layer_name.rsplit('.', 1)
        module = dict(model.named_modules())[module_name]
        key = layer_name.replace('.', '_')  # e.g., "layer1_0_conv1_weight"
        lmul = int(weight_to_lmul[key])
        if lmul in [1, 4]:
            mr = 7
        elif lmul == 2:
            mr = 8
        else:
            mr = 3
        nr = int(lmul * (vlen / 32))
        
        # Convert weight tensor to NumPy from CPU.
        weight_tensor = getattr(module, weight_name).detach().cpu().numpy().astype(np.float32)
        output_channel, input_channel, kernel_height, kernel_width = weight_tensor.shape
        weight_2d = weight_tensor.reshape(output_channel, kernel_height * kernel_width * input_channel)
        mask, indice = f32_data_pruning_column_wise_with_ratio(weight_2d, nr, mr, pruning_ratio)
        
        # Convert mask back to PyTorch tensor and reshape to original weight shape,
        # then move it to the device of the module's weight.
        custom_mask = torch.from_numpy(mask).reshape(weight_tensor.shape).to(getattr(module, weight_name).device)
        
        # Apply the custom mask using PyTorch's pruning API.
        prune.CustomFromMask.apply(module, weight_name, custom_mask)
        print(f"Applied custom mask on {layer_name} with shape {weight_tensor.shape}")
    
    # Optionally, print the mask shapes.
    for name, module in model.named_modules():
        if hasattr(module, 'weight_mask'):
            print(f"Layer: {name}, Mask shape: {module.weight_mask.shape}")

def evaluate_model(model, dataloader, device):
    """
    Evaluates the model on the provided dataloader and returns accuracy.
    """
    model.eval()
    correct = 0
    total = 0
    batch_idx = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            batch_idx += 1
            if batch_idx % 10 == 0:
                print(f"Processed {batch_idx} batches, current accuracy: {100 * correct / total:.2f}%")
    final_accuracy = 100 * correct / total
    print(f"Final evaluation accuracy: {final_accuracy:.2f}%")
    return final_accuracy

if __name__ == "__main__":
    # Set device to GPU if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load a pretrained ResNet-50 model from timm.
    model_name = "resnet50"
    model = timm.create_model(model_name, pretrained=True, num_classes=1000)
    model.to(device)
    
    # Perform custom column-wise pruning on the model.
    perform_pruning(model, pruning_ratio=0.5)
    
    # Define test transformations.
    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    # Load your local dataset (ImageFolder structure expected).
    data_dir = "/data/imagenet/train"  # Update this path to your dataset.
    test_dataset = torchvision.datasets.ImageFolder(data_dir, transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
    
    # Run inference and check accuracy after pruning.
    accuracy = evaluate_model(model, test_loader, device)
    print(f"Test Accuracy after pruning: {accuracy:.2f}%")
