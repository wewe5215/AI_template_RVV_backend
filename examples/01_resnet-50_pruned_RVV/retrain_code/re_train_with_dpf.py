import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR, LambdaLR
from tqdm.auto import tqdm
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
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
from utils import MyPruningMethod
#from group_op_and_lmul import fetch_lmul_for_op
FINAL_SPARSITY = 0.75                # P  (keep 25 %)
WARMUP_EPOCHS  = 0
TOTAL_EPOCHS   = 90
MOMENTUM       = 0.9
WEIGHT_DECAY   = 1e-4
VLEN = 256
milestones = [WARMUP_EPOCHS, TOTAL_EPOCHS]
#weight_to_lmul = fetch_lmul_for_op(1, 'chosen_lmul_bs')

cached_pruning_method = {}

weight_to_lmul = {
    "stem_conv1_weight": 2,
    "layer1_0_conv1_weight": 2,
    "layer1_0_conv2_weight": 2,
    "layer1_0_downsample_0_weight": 8,
    "layer1_0_conv3_weight": 8,
    "layer1_1_conv1_weight": 4,
    "layer1_1_conv2_weight": 2,
    "layer1_1_conv3_weight": 8,
    "layer1_2_conv1_weight": 4,
    "layer1_2_conv2_weight": 2,
    "layer1_2_conv3_weight": 8,
    "layer2_0_conv1_weight": 8,
    "layer2_0_conv2_weight": 4,
    "layer2_0_downsample_0_weight": 4,
    "layer2_0_conv3_weight": 4,
    "layer2_1_conv1_weight": 2,
    "layer2_1_conv2_weight": 2,
    "layer2_1_conv3_weight": 4,
    "layer2_2_conv1_weight": 2,
    "layer2_2_conv2_weight": 2,
    "layer2_2_conv3_weight": 4,
    "layer2_3_conv1_weight": 2,
    "layer2_3_conv2_weight": 2,
    "layer2_3_conv3_weight": 4,
    "layer3_0_conv1_weight": 2,
    "layer3_0_conv2_weight": 4,
    "layer3_0_downsample_0_weight": 2,
    "layer3_0_conv3_weight": 2,
    "layer3_1_conv1_weight": 2,
    "layer3_1_conv2_weight": 2,
    "layer3_1_conv3_weight": 2,
    "layer3_2_conv1_weight": 2,
    "layer3_2_conv2_weight": 2,
    "layer3_2_conv3_weight": 2,
    "layer3_3_conv1_weight": 2,
    "layer3_3_conv2_weight": 2,
    "layer3_3_conv3_weight": 2,
    "layer3_4_conv1_weight": 2,
    "layer3_4_conv2_weight": 2,
    "layer3_4_conv3_weight": 2,
    "layer3_5_conv1_weight": 2,
    "layer3_5_conv2_weight": 2,
    "layer3_5_conv3_weight": 2,
    "layer4_0_conv1_weight": 2,
    "layer4_0_conv2_weight": 4,
    "layer4_0_downsample_0_weight": 4,
    "layer4_0_conv3_weight": 1,
    "layer4_1_conv1_weight": 1,
    "layer4_1_conv2_weight": 2,
    "layer4_1_conv3_weight": 1,
    "layer4_2_conv1_weight": 1,
    "layer4_2_conv2_weight": 2,
    "layer4_2_conv3_weight": 1,
}

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
    # comment out indices if needed
    # indices = []  # Store selected column indices (from the first row in each block)
    mask = []     # Flattened binary mask
    for i in range(0, output_channel, mr):
        end_offset = min(mr, output_channel - i)
        block = weight[i:i+end_offset, :]  # (end_offset, input_channel)
        accumulator = block.abs().sum(dim=0)
        keep_count = int(torch.ceil((1 - pruning_ratio) * torch.tensor(input_channel, dtype=torch.float32)).item())
        if torch.all(accumulator == accumulator[0]):
            # for j in range(end_offset):
            #     for k in range(input_channel):
                    block_mask = torch.zeros((end_offset, input_channel), dtype=torch.uint8, device=weight.device)
                    block_mask[:, :keep_count] = 1
                    # if j == 0:
                    #     indices.append(k)
        else:
            # threshold = np.percentile(accumulator, pruning_ratio * 100)
            threshold = torch.kthvalue(accumulator, input_channel - keep_count + 1).values.item()
            block_mask = (accumulator >= threshold).to(torch.uint8).repeat(end_offset, 1)
            # for j in range(end_offset):
            #     for k in range(input_channel):
                    # select = (accumulator[k] >= threshold) if (input_channel % 2 == 0) else (accumulator[k] > threshold)
                    # if select:
                    #     mask.append(1)
                    #     # if j == 0:
                    #     #     indices.append(k)
                    # else:
                    #     mask.append(0)
    # indices = np.array(indices, dtype=np.uint16)
        mask.append(block_mask)
    mask = torch.cat(mask, dim=0).flatten()
    # mask = np.array(mask, dtype=np.uint8)
    return mask#, indices

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

def get_weight_threshold(model, rate):
    importance_all = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            # ---- 1. get the *leaf* grad tensor if you used torch.nn.utils.prune ----
            weight_leaf = getattr(module, 'weight_orig', module.weight)
            grad_leaf   = weight_leaf.grad          # <-- will be leaf grad

            # ---- 2. choose importance ------------------------------------------------
            if args.prune_imp == 'grad' and grad_leaf is not None:
                importance = grad_leaf.abs()
            elif args.prune_imp == 'syn' and grad_leaf is not None:
                importance = (weight_leaf * grad_leaf).abs()
            elif args.prune_imp == 'L2':
                importance = weight_leaf.pow(2)
            else:                       # default L1
                importance = weight_leaf.abs()

            importance_all.append(importance.flatten().to('cuda'))

    if not importance_all:
        raise RuntimeError("No valid importance scores collected for pruning.")
    importance_all = torch.cat(importance_all)
    k = int(len(importance_all) * rate)
    if k == 0:
        threshold = float('inf')
    else:
        threshold = importance_all.kthvalue(k).values.item()
    return threshold

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
            # print(f"skip {layer_name}")
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
        weight_leaf = getattr(module, 'weight_orig', module.weight)
        grad_leaf   = weight_leaf.grad

        pruned_count = int(p_ratio * weight_leaf.shape[1] * weight_leaf.shape[2] * weight_leaf.shape[3])
        if pruned_count == 0:
            continue

        if args.prune_imp == 'L2':
            mat = weight_leaf.pow(2)
        elif args.prune_imp == 'grad' and grad_leaf is not None:
            mat = grad_leaf.abs()
        elif args.prune_imp == 'syn' and grad_leaf is not None:
            mat = (weight_leaf * grad_leaf).abs()
        else:
            mat = weight_leaf.abs()

        output_channel, input_channel, kernel_height, kernel_width = module.weight.shape
        mat = mat.reshape(output_channel, kernel_height * kernel_width * input_channel)

        # pad the output_channel dimension to be multiple of mr
        padding = output_channel % mr
        if padding > 0:
            padding = mr - padding # pad bottom
            mat = F.pad(mat, (0,0,0,padding), mode='constant', value=0)

        height, width = mat.shape
        mat = mat.reshape(-1, mr, width) # height//mr, mr, width
        mat = torch.sum(mat, 1)          # height//mr, width

        threshold = torch.kthvalue(mat, pruned_count).values # height//mr
        threshold = torch.unsqueeze(threshold, 1) # height//mr, 1
        mask = mat > threshold  # height//mr, width
        mask = torch.unsqueeze(mask, 1)  # height//mr, 1, width
        mask = mask.repeat(1, mr, 1)     # height//mr, mr, width
        mask = mask.reshape(-1, width)   # height, width
        if padding:
            mask = mask[:-padding, :]

        mask = mask.reshape(output_channel, input_channel, kernel_height, kernel_width)

        if layer_name not in cached_pruning_method.keys():
            method = MyPruningMethod.apply(module, name='weight', mask=mask)
            cached_pruning_method[layer_name] = method
            print(f"cache pruning method: {layer_name}")
        else:
            method = cached_pruning_method[layer_name]
            method.update_mask(module, mask)


def get_cubic_pruning_ratio(global_step, dataloader):
    return FINAL_SPARSITY - FINAL_SPARSITY * (1 - (global_step) / (TOTAL_EPOCHS * len(dataloader)))**3

def train_one_epoch(epoch, model, dataloader, optimizer, scheduler, warmup_scheduler, accelerator,
                    criterion, start_prune_epoch, end_prune_epoch, warmup_steps_total):
    device = accelerator.device
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(dataloader, disable=not accelerator.is_local_main_process)
    global_step = epoch * len(dataloader)
    for step, (images, labels) in enumerate(progress_bar):
        images, labels = images.to(device), labels.to(device)
        if epoch >= WARMUP_EPOCHS and (global_step + 1) % args.frequency == 0 and (epoch + 1) <= TOTAL_EPOCHS:
            target_sparsity = FINAL_SPARSITY
            if target_sparsity > 0:
                apply_map_masks(model, layers_to_prune, target_sparsity)

        with accelerator.accumulate(model):
            outputs = model(images)
            loss = criterion(outputs, labels)
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
        
        if WARMUP_EPOCHS > 0 and global_step < warmup_steps_total:
            warmup_scheduler.step()
        running_loss += loss.item()
        progress_bar.set_description(f"E{epoch+1} L{running_loss/(step+1):.4f}")

    if epoch >= WARMUP_EPOCHS:
            scheduler.step()

    if accelerator.is_local_main_process:
        if start_prune_epoch <= epoch <= end_prune_epoch:
            print(f"Epoch {epoch+1}: exploration, target_sparsity={target_sparsity:.4f}")
        else:
            print(f"Epoch {epoch+1}: exploitation (mask fixed)")

def train(args, accelerator, train_data, eval_data, model, is_regression=False):
    train_dataset, train_dataloader = train_data
    eval_dataset, eval_dataloader = eval_data
    if torch.cuda.is_available():
        device = torch.device("cuda")
    base_batch_size = 256
    actual_batch_size = args.per_device_batch * accelerator.num_processes
    INIT_LR = 0.1 * (actual_batch_size / base_batch_size)
    # Use all model parameters (or split by weight decay if desired).
    optimizer = torch.optim.SGD(model.parameters(), lr=INIT_LR, momentum=MOMENTUM,
                                weight_decay=WEIGHT_DECAY, nesterov=True)

    # Calculate the number of update steps per epoch.
    total_steps = TOTAL_EPOCHS * math.ceil(len(train_dataloader) / args.gradient_accumulation_steps) # divide with gradient
    warmup_steps = WARMUP_EPOCHS * math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    def linear_warmup(step):
        return min(1.0, step / warmup_steps)
    if WARMUP_EPOCHS > 0:
        warmup_scheduler = LambdaLR(optimizer, lr_lambda=linear_warmup)
    else:
        warmup_scheduler = None

    lr_scheduler = MultiStepLR(optimizer, milestones=[30, 60, 80], gamma=0.1)

    # Prepare the model, optimizer, dataloader, and scheduler with accelerator.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )
    criterion = nn.CrossEntropyLoss()
    best_acc = 0.0
    for epoch in range(TOTAL_EPOCHS):
        train_one_epoch(epoch, model, train_dataloader, optimizer, lr_scheduler, warmup_scheduler, accelerator,
                        criterion, WARMUP_EPOCHS, TOTAL_EPOCHS, warmup_steps)
        if accelerator.is_local_main_process:
            acc = evaluate(model, eval_dataloader, accelerator.device)
            print(f"Epoch {epoch+1}: Val Acc {acc:.2f}%")
            best_acc = max(best_acc, acc)
        torch.cuda.empty_cache()


    if accelerator.is_local_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        torch.save(accelerator.unwrap_model(model).state_dict(),
                   os.path.join(args.output_dir, 'map_resnet50_imagenet.pth'))
        print(f"Best validation accuracy: {best_acc:.2f}%")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a pruned ResNet-50 model for image classification")
    parser.add_argument('--train_dir', type=str, default='/data/imagenet/train')
    parser.add_argument('--val_dir',   type=str, default='/data/imagenet/val')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--per_device_batch', type=int, default=256)  # 4×256 = 1024 global
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of gradient accumulation steps")
    parser.add_argument("--frequency", type=int, default=1, help="frequency to update mask and attention")
    parser.add_argument("--prune_imp", type=str, default='syn')
    args = parser.parse_args()

    # Set device.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Create accelerator.
    accelerator = Accelerator(mixed_precision="no", gradient_accumulation_steps=args.gradient_accumulation_steps)  # Set fp16=True if using mixed precision.
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
    train_dataloader = DataLoader(train_dataset, batch_size=args.per_device_batch, shuffle=True, num_workers=8, pin_memory=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.per_device_batch, shuffle=False, num_workers=8, pin_memory=True)
    
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
