import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
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
N = 3
M = 4
result_file = f"results_prune_{N}_{M}_pruning.txt"
vlen = 256
from group_op_and_lmul import fetch_lmul_for_op

CONV_WEIGHT_PATTERN = re.compile(r"conv\d+\.weight")
logger = get_logger(__name__)

# List of ResNet-50 convolution weight parameters to prune.
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

def f32_data_pruning_N_M_with_ratio(weight, mr, N, M):
    """
    Applies an N:M pruning pattern to a 2D weight array, where in each row, for every group of M columns,
    only N columns are retained (i.e., set to 1 in the mask) and the rest are pruned (set to 0).
    
    Parameters:
      weight: A 2D numpy array of shape (output_channel, input_channel) representing the weight matrix.
      mr: Block size (number of rows per block). This parameter is preserved for compatibility,
          but the pruning is applied row-wise.
      N: Number of columns to retain (non-zero) in each group.
      M: Group size, i.e., the total number of columns in each group.
                     
    Returns:
      mask: A 2D numpy array of shape (output_channel, input_channel) containing binary values 
            (1 for retained weights, 0 for pruned weights).
      indices: A list of lists, where each inner list contains the column indices (in the original row)
               that are retained for the corresponding row.
    """
    output_channel, input_channel = weight.shape
    mask = np.zeros_like(weight, dtype=np.uint8)
    indices = []
    
    # Process each row independently.
    for i in range(output_channel):
        row_mask = np.zeros(input_channel, dtype=np.uint8)
        row_indices = []
        # Process the row in groups of M columns.
        for start in range(0, input_channel, M):
            end = min(start + M, input_channel)
            group = weight[i, start:end]
            # In case the group is smaller than M, adjust N accordingly.
            current_N = min(N, end - start)
            # Identify the indices (within the group) of the top current_N elements by absolute value.
            top_indices = np.argsort(-np.abs(group))[:current_N]
            # Set the corresponding positions in the mask to 1.
            row_mask[start:end][top_indices] = 1
            # Record the actual column indices in the original row.
            row_indices.extend(list(np.arange(start, end)[top_indices]))
        mask[i, :] = row_mask
        indices.append(row_indices)
        
    return mask, indices
def perform_pruning(model, N, M):
    """
    Applies custom column-wise pruning to ResNet-50 convolution layers.
    """
    batch_size = 1
    for layer_name in layers_to_prune:
        module_name, weight_name = layer_name.rsplit('.', 1)
        module = dict(model.named_modules())[module_name]
        key = layer_name.replace('.', '_')  # e.g., "layer1_0_conv1_weight"
        # Convert weight tensor to NumPy from CPU.
        weight_tensor = getattr(module, weight_name).detach().cpu().numpy().astype(np.float32)
        output_channel, input_channel, kernel_height, kernel_width = weight_tensor.shape
        weight_2d = weight_tensor.reshape(output_channel, kernel_height * kernel_width * input_channel)
        mask, indice = f32_data_pruning_N_M_with_ratio(weight_2d, 1, N, M)
        # Convert mask back to PyTorch tensor and reshape to original weight shape, 
        # then move it to the device of the module's weight.
        custom_mask = torch.from_numpy(mask).reshape(weight_tensor.shape).to(getattr(module, weight_name).device)
        prune.CustomFromMask.apply(module, weight_name, custom_mask)
        print(f"Applied custom mask on {layer_name} with shape {weight_tensor.shape}")
    
    # Optionally, print the mask shapes.
    for name, module in model.named_modules():
        if hasattr(module, 'weight_mask'):
            print(f"Layer: {name}, Mask shape: {module.weight_mask.shape}")

def evaluate_model(model, dataloader, device):
    """
    Evaluates the model on the provided dataloader and returns accuracy.
    Prints progress every 10 batches.
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
    result_line = f"Final evaluation accuracy: {final_accuracy:.2f}%"
    print(result_line)
    if accelerator.is_local_main_process:
        abs_path = os.path.abspath(result_file)
        print(f"Results will be written to: {abs_path}")
        with open(result_file, "a") as f:
            f.write(result_line + "\n")
    return final_accuracy

def validate(args, accelerator, eval_data, model, is_regression):
    """
    Simple validation function that evaluates the model.
    """
    eval_dataset, eval_dataloader = eval_data
    return evaluate_model(model, eval_dataloader, accelerator.device)

def train(args, accelerator, train_data, eval_data, model, is_regression=False):
    """Train the ResNet-50 model after applying pruning."""
    train_dataset, train_dataloader = train_data

    # Use all model parameters (or split by weight decay if desired).
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)

    # Scheduler: use your preferred scheduler; here we assume get_scheduler is defined.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    # For example, you can use a linear scheduler (or replace with torch.optim.lr_scheduler)
    from transformers import get_scheduler
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )
    # Prepare with accelerator.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    # Apply custom pruning to the model.
    result_line = f"apply {N} : {M} pruning "
    print(result_line)
    if accelerator.is_local_main_process:
        abs_path = os.path.abspath(result_file)
        print(f"Results will be written to: {abs_path}")
        with open(result_file, "a") as f:
            f.write(result_line + "\n")
    perform_pruning(model, N, M)

    # Recalculate total steps in case the dataloader changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Optionally initialize trackers.
    if args.with_tracking:
        experiment_config = vars(args)
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("resnet50_pruning", experiment_config)

    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    # Optionally resume from a checkpoint.
    if args.resume_from_checkpoint:
        checkpoint_path = args.resume_from_checkpoint
        accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
        accelerator.load_state(checkpoint_path)
        # Extract epoch or step information from checkpoint if needed.
        # (This part may require additional customization.)
    
    progress_bar.update(completed_steps)

    # Define a standard classification loss.
    criterion = nn.CrossEntropyLoss()

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        total_loss = 0
        for step, (images, labels) in enumerate(train_dataloader):
            images = images.to(accelerator.device)
            labels = labels.to(accelerator.device)
            
            with accelerator.accumulate(model):
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.detach().float()
                accelerator.backward(loss)
                if args.max_grad_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1
            
            if completed_steps >= args.max_train_steps:
                break

        result_line = f"Epoch {epoch+1} completed. Average loss: {total_loss / (step+1):.4f}"
        print(result_line)
        if accelerator.is_local_main_process:
            abs_path = os.path.abspath(result_file)
            print(f"Results will be written to: {abs_path}")
            with open(result_file, "a") as f:
                f.write(result_line + "\n")
        # Evaluate after each epoch.
        validate(args, accelerator, eval_data, model, is_regression)
    for layer_name in layers_to_prune:
        module_name, weight_name = layer_name.rsplit('.', 1)
        module = dict(model.named_modules())[module_name]
        # Only remove if the module was pruned
        if hasattr(module, f"{weight_name}_orig"):
            prune.remove(module, weight_name)
        else:
            print(f"Layer {layer_name} was not pruned, skipping removal.")
    accelerator.save_state(args.output_dir)
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a pruned ResNet-50 model for image classification")
    parser.add_argument("--model_name_or_path", type=str, default="resnet50", help="Pretrained model name or path")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the final model")
    parser.add_argument("--adam_epsilon", type=float, default=1e-8, help="Epsilon for Adam optimizer")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument("--num_train_epochs", type=int, default=35, help="Number of training epochs")
    parser.add_argument("--max_train_steps", type=int, default=None, help="Maximum training steps")
    parser.add_argument("--per_device_train_batch_size", type=int, default=32, help="Train batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of gradient accumulation steps")
    parser.add_argument("--num_warmup_steps", type=int, default=0, help="Warmup steps for lr scheduler")
    parser.add_argument("--lr_scheduler_type", type=str, default="linear", help="Type of lr scheduler")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm")
    parser.add_argument("--with_tracking", action="store_true", help="Enable experiment tracking")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--pruning", action="store_true", help="Whether to apply pruning")
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
    train_data_dir = "/data/imagenet/train"
    eval_data_dir = "/data/imagenet/val"
    train_dataset = torchvision.datasets.ImageFolder(train_data_dir, transform=train_transforms)
    eval_dataset = torchvision.datasets.ImageFolder(eval_data_dir, transform=test_transforms)
    train_dataloader = DataLoader(train_dataset, batch_size=args.per_device_train_batch_size, shuffle=True, num_workers=4)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.per_device_train_batch_size, shuffle=False, num_workers=4)
    
    train_data = (train_dataset, train_dataloader)
    eval_data = (eval_dataset, eval_dataloader)
    
    # Load pretrained ResNet-50.
    model = models.__dict__[args.model_name_or_path](pretrained=True)
    model.to(device)
    
    # Train the model.
    train(args, accelerator, train_data, eval_data, model, is_regression=False)
    
    # Evaluate the pruned model.
    final_accuracy = evaluate_model(model, eval_dataloader, device)
    print(f"Test Accuracy after pruning and retraining: {final_accuracy:.2f}%")
