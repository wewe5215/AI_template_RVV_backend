import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from tqdm.auto import tqdm
import torchvision
import torchvision.transforms as transforms
import re
import argparse
import json
import logging
import math
import os
logger = get_logger(__name__)
import random
from pathlib import Path
import numpy as np
from group_op_and_lmul import fetch_lmul_for_op
FINAL_SPARSITY = 0.75                # P  (keep 25 %)
MASK_UPDATE_F  = 1                   # every iteration
WARMUP_EPOCHS  = 10
TOTAL_EPOCHS   = 100
MOMENTUM       = 0.875
WEIGHT_DECAY   = 2e-5                # paper: L1, here as L2; small diff in practice
INIT_LR        = 0.256
VLEN = 256
weight_to_lmul = fetch_lmul_for_op(1, 'chosen_lmul_bs')
layers_to_prune = [
    # Uncomment or adjust the layers you want to prune:
    # 'layer1.0.conv1.weight', 'layer1.0.conv2.weight', 'layer1.0.conv3.weight',
    # 'layer1.1.conv1.weight', 'layer1.1.conv2.weight', 'layer1.1.conv3.weight',
    # 'layer1.2.conv1.weight', 'layer1.2.conv2.weight', 'layer1.2.conv3.weight',
    # 'layer2.0.conv1.weight', 
    'layer2.0.conv2.weight', 'layer2.0.conv3.weight',
    'layer2.1.conv1.weight', 'layer2.1.conv2.weight', 'layer2.1.conv3.weight',
    'layer2.2.conv1.weight', 'layer2.2.conv2.weight', 'layer2.2.conv3.weight',
    'layer2.3.conv1.weight', 'layer2.3.conv2.weight', 'layer2.3.conv3.weight',
    'layer3.0.downsample.0.weight',
    'layer3.0.conv1.weight', 'layer3.0.conv2.weight', 'layer3.0.conv3.weight',
    'layer3.1.conv1.weight', 'layer3.1.conv2.weight', 'layer3.1.conv3.weight',
    'layer3.2.conv1.weight', 'layer3.2.conv2.weight', 'layer3.2.conv3.weight',
    'layer3.3.conv1.weight', 'layer3.3.conv2.weight', 'layer3.3.conv3.weight',
    'layer3.4.conv1.weight', 'layer3.4.conv2.weight', 'layer3.4.conv3.weight',
    'layer3.5.conv1.weight', 'layer3.5.conv2.weight', 'layer3.5.conv3.weight',
    'layer4.0.downsample.0.weight',
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
    indices = []  # Store selected column indices (from the first row in each block)
    mask = []     # Flattened binary mask
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

def evaluate(model: nn.Module, dataloader: DataLoader, device):
    model.eval(); correct = total = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds  = logits.argmax(1)
            total += y.size(0)
            correct += (preds == y).sum().item()
    return 100.0 * correct / total

def magnitude_attention_mask(t: torch.Tensor, pruning_ratio: float) -> torch.Tensor:
    """Return a 0/1 mask keeping top‑(1−p) of |t|."""
    flat = t.abs().flatten()
    k = int((1.0 - pruning_ratio) * flat.numel())
    if k == 0:
        thr = float('inf')
    else:
        thr = flat.topk(k, sorted=True).values[-1]
    binary_mask = (t.abs() > thr).float()

    # update attention
    z = 1
    abs_weight = t.abs()
    max_val = abs_weight.max()
    min_val = abs_weight.min()
    denom = max_val - min_val + 1e-8  # avoid zero division
    att_base = (1 - pruning_ratio) ** z  # lower bound
    normed = (abs_weight - min_val) / denom
    att_important = normed * (1 - att_base) + att_base

    attention = torch.where(binary_mask.bool(), att_important, att_base)
    return binary_mask, attention


def apply_map_masks(model: nn.Module, layers_to_prune, p_ratio: float):
    """Re‑compute & apply masks on designated layers."""

    for layer_name, module in model.named_modules():
        if not isinstance(module, torch.nn.Conv2d):
            continue
        if 'layer' not in layer_name:
            continue
        if 'layer1' in layer_name or \
        (
            'layer2.0' in layer_name and \
            ('conv1' in layer_name or 'downsample' in layer_name)
        ):
            print(f"skip {layer_name}")
            continue
        key = f"{layer_name}.weight".replace('.', '_')
        lmul = int(weight_to_lmul[key])
        if lmul in [1, 4]:
            mr = 7
        elif lmul == 2:
            mr = 8
        else:
            mr = 3
        nr = int(lmul * (VLEN / 32))
        mask_in_device, attention = magnitude_attention_mask(module.weight, p_ratio).cpu().numpy()
        output_channel, input_channel, kernel_height, kernel_width = module.weight.shape
        mask_in_device_2d = mask_in_device.reshape(output_channel, kernel_height * kernel_width * input_channel)
        mask, indice = f32_data_pruning_column_wise_with_ratio(mask_in_device_2d, nr, mr, p_ratio)
        custom_mask = torch.from_numpy(mask.astype(np.float32)) \
                           .view_as(module.weight).to(module.weight.device)
        prune.CustomFromMask.apply(module, 'weight', custom_mask)
        module._map_attention = attention

def train_one_epoch(epoch, model, dataloader, optimizer, scheduler, accelerator,
                    criterion, start_prune_epoch, end_prune_epoch):
    device = accelerator.device
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(dataloader, disable=not accelerator.is_local_main_process)

    for step, (images, labels) in enumerate(progress_bar):
        images, labels = images.to(device), labels.to(device)

        # --- MAP mask update every F=1 iters after warm‑up ---
        if start_prune_epoch <= epoch <= end_prune_epoch:
            progress = (epoch - start_prune_epoch) / max(1, (end_prune_epoch - start_prune_epoch))
            current_p = FINAL_SPARSITY * (progress ** 3)  # cubic schedule
            apply_map_masks(model, layers_to_prune, current_p)

        with accelerator.accumulate(model):
            outputs = model(images)
            loss = criterion(outputs, labels)
            accelerator.backward(loss)
            optimizer.step(); optimizer.zero_grad()
            if isinstance(scheduler, CosineAnnealingLR):
                scheduler.step()
        running_loss += loss.item()
        progress_bar.set_description(f"E{epoch+1} L{running_loss/(step+1):.4f}")

def warmup_cosine_scheduler(optimizer, warmup_steps, total_steps, min_lr=0.0):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, warmup_steps))
        # Cosine decay
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr, cosine_decay)
    return LambdaLR(optimizer, lr_lambda)

def train(args, accelerator, train_data, eval_data, model, is_regression=False):
    train_dataset, train_dataloader = train_data
    eval_dataset, eval_dataloader = eval_data
    if torch.cuda.is_available():
        device = torch.device("cuda")
    # Use all model parameters (or split by weight decay if desired).
    optimizer = torch.optim.SGD(model.parameters(), lr=INIT_LR, momentum=MOMENTUM,
                                weight_decay=WEIGHT_DECAY, nesterov=True)

    # Calculate the number of update steps per epoch.
    total_steps = TOTAL_EPOCHS * math.ceil(len(train_dataloader))
    warmup_steps = WARMUP_EPOCHS * math.ceil(len(train_dataloader))
    lr_scheduler = warmup_cosine_scheduler(optimizer, warmup_steps, total_steps)

    # Prepare the model, optimizer, dataloader, and scheduler with accelerator.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )
    criterion = nn.CrossEntropyLoss()
    best_acc = 0.0
    for epoch in range(TOTAL_EPOCHS):
        train_one_epoch(epoch, model, train_dataloader, optimizer, lr_scheduler, accelerator,
                        criterion, WARMUP_EPOCHS, TOTAL_EPOCHS)

        if accelerator.is_local_main_process:
            acc = evaluate(model, eval_dataloader, accelerator.device)
            logger.info(f"Epoch {epoch+1}: Val Acc {acc:.2f}%")
            best_acc = max(best_acc, acc)

    if accelerator.is_local_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        torch.save(accelerator.unwrap_model(model).state_dict(),
                   os.path.join(args.output_dir, 'map_resnet50_imagenet.pth'))
        logger.info(f"Best validation accuracy: {best_acc:.2f}%")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a pruned ResNet-50 model for image classification")
    parser.add_argument('--train_dir', type=str, default='/data/imagenet/train')
    parser.add_argument('--val_dir',   type=str, default='/data/imagenet/val')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--per_device_batch', type=int, default=256)  # 4×256 = 1024 global
    args = parser.parse_args()

    # Set device.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Create accelerator.
    accelerator = Accelerator(mixed_precision="no")  # Set fp16=True if using mixed precision.
    # Set seed for reproducibility.
    set_seed(42)
    # Define data transforms for training and evaluation.
    train_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    # Load your training and evaluation datasets (update paths accordingly).
    train_dataset = torchvision.datasets.ImageFolder(args.train_dir, transform=train_transforms)
    eval_dataset = torchvision.datasets.ImageFolder(args.val_dir, transform=test_transforms)
    train_dataloader = DataLoader(train_dataset, batch_size=args.per_device_batch, shuffle=True, num_workers=4)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.per_device_batch, shuffle=False, num_workers=4)
    
    train_data = (train_dataset, train_dataloader)
    eval_data = (eval_dataset, eval_dataloader)
    
    # Load pretrained ResNet-50.
    import torchvision.models as models
    from torchvision.models import resnet50, ResNet50_Weights
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model.to(device)
    # Train the model.
    train(args, accelerator, train_data, eval_data, model, is_regression=False)
    # Evaluate the pruned model.
    final_accuracy = evaluate(model, eval_dataloader, device)
    print(f"Test Accuracy after pruning and retraining: {final_accuracy:.2f}%")
