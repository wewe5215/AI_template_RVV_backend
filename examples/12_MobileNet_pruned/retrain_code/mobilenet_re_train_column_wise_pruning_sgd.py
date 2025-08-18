# Copyright (c) 2025
# ------------------------------------------------------------
# MobileNet V2 column‑wise pruning + retraining (SGD ImageNet recipe)
#  • Helpers (pruning, evaluation) unchanged
#  • train() uses SGD + StepLR (classic ImageNet schedule), torchvision v1 pretrained weight
# ------------------------------------------------------------

import argparse
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torchvision
import torchvision.transforms as transforms
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

LOGGER = get_logger(__name__)
pruning_ratio = 0.75
result_file = f"results_prune_{int(pruning_ratio*100)}_mobilenetv2_sgd.txt"

# -----------------------------------------------------------------------------
# Column‑wise pruning helper (unchanged)
# -----------------------------------------------------------------------------
def f32_data_pruning_column_wise_with_ratio(weight, nr, mr, pruning_ratio):
    out_c, in_c = weight.shape
    mask = []
    indices = []
    for i in range(0, out_c, mr):
        end = min(mr, out_c - i)
        block = weight[i:i+end, :]
        score = np.sum(np.abs(block), axis=0)
        keep = int(np.ceil((1 - pruning_ratio) * in_c))
        if np.all(score == score[0]):
            topk = np.arange(keep)
        else:
            topk = score.argsort()[::-1][:keep]
        block_mask = np.zeros(in_c, dtype=np.uint8)
        block_mask[topk] = 1
        for _ in range(end):
            mask.append(block_mask)
        indices.extend(topk)
    return np.vstack(mask).flatten(), np.array(indices, dtype=np.uint16)

# -----------------------------------------------------------------------------
# Pruning rule for MobileNetV2 (unchanged)
# -----------------------------------------------------------------------------
_EXCLUDE_NAME_SUBSTR = [
    "features.0.",   # stem
    "features.1.",   # block1
    "features.2.",   # block2
    "features.3.",
    "features.4.conv.0.0",  # stage‑2 expand
    "classifier"            # final FC
]

def _should_skip(name: str, module: nn.Module) -> bool:
    if any(sub in name for sub in _EXCLUDE_NAME_SUBSTR):
        return True
    if isinstance(module, nn.Conv2d) and module.groups == module.in_channels:
        return True
    return False


def perform_pruning(model: nn.Module, pruning_ratio: float = 0.5):
    skipped = pruned = 0
    for name, module in model.named_modules():
        if not isinstance(module, nn.Conv2d):
            continue
        if _should_skip(name, module):
            skipped += 1
            continue
        mr = 8
        nr = int(2 * (256 / 32))
        w = module.weight.detach().cpu().numpy().astype(np.float32)
        oc, ic, kh, kw = w.shape
        mask_flat, _ = f32_data_pruning_column_wise_with_ratio(
            w.reshape(oc, ic * kh * kw), nr, mr, pruning_ratio)
        mask = torch.from_numpy(mask_flat.reshape(w.shape)).to(module.weight.device)
        prune.CustomFromMask.apply(module, name="weight", mask=mask)
        pruned += 1
    LOGGER.info(f"Pruned {pruned} conv layers, skipped {skipped}, ratio={pruning_ratio}")

# -----------------------------------------------------------------------------
# Evaluation (unchanged)
# -----------------------------------------------------------------------------
def evaluate_model(model, loader, device):
    model.eval()
    total = correct = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(1)
            total += y.size(0)
            correct += (preds == y).sum().item()
    return 100 * correct / total

# -----------------------------------------------------------------------------
# train(): SGD + StepLR (classic ImageNet schedule)
# -----------------------------------------------------------------------------
def train(args, accelerator, train_data, eval_data, model):
    train_ds, train_loader = train_data
    val_ds, val_loader     = eval_data

    # SGD optimizer and StepLR
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.learning_rate,       # default=0.1
        momentum=0.9,
        weight_decay=args.weight_decay  # default=1e-4
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=30,
        gamma=0.1
    )

    # Prepare for distributed/mixed-precision
    model, optimizer, train_loader = accelerator.prepare(
        model, optimizer, train_loader
    )
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for x, y in tqdm(train_loader,
                         disable=not accelerator.is_local_main_process,
                         desc=f"Epoch {epoch+1}/{args.epochs}"):
            x, y = x.to(accelerator.device), y.to(accelerator.device)
            with accelerator.accumulate(model):
                out = model(x)
                loss = criterion(out, y)
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
            running_loss += loss.item()

        # update LR
        scheduler.step()

        train_loss = running_loss / len(train_loader)
        val_acc = evaluate_model(model, val_loader, accelerator.device)
        accelerator.print(
            f"Epoch {epoch+1}/{args.epochs} — "
            f"train loss: {train_loss:.4f} — "
            f"val acc: {val_acc:.2f}% — "
            f"lr: {scheduler.get_last_lr()[0]:.1e}"
        )

        # save best model
        if accelerator.is_local_main_process and val_acc > best_acc:
            best_acc = val_acc
            args.output_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                accelerator.unwrap_model(model).state_dict(),
                args.output_dir / "mobilenetv2_pruned_best.pt"
            )
    accelerator.print(f"Best validation accuracy: {best_acc:.2f}%")

# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Prune+retrain MobileNetV2 with SGD on ImageNet")
    p.add_argument("--output_dir",    required=True, type=Path)
    p.add_argument("--batch_size",    default=128,    type=int)
    p.add_argument("--epochs",        default=90,     type=int)
    p.add_argument("--learning_rate", default=0.1,    type=float,
                   help="initial LR; decayed by 10× at epochs 30,60")
    p.add_argument("--weight_decay",  default=1e-4,   type=float)
    p.add_argument("--grad_accum",    default=1,      type=int)
    p.add_argument("--max_grad_norm", default=1.0,    type=float)
    p.add_argument("--pruning_ratio", default=0.75,    type=float)
    args = p.parse_args()

    set_seed(42)

    # Transforms
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])

    train_ds = torchvision.datasets.ImageFolder(
        "/data/imagenet/train", transform=train_tf
    )
    val_ds   = torchvision.datasets.ImageFolder(
        "/data/imagenet/val",   transform=val_tf
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=8, pin_memory=True
    )
    val_loader   = DataLoader(
        val_ds,   batch_size=args.batch_size, shuffle=False,
        num_workers=8, pin_memory=True
    )

    accelerator = Accelerator()
    device = accelerator.device

    # Use torchvision v1 pretrained weights
    model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    model.to(device)
    print(f"Pruning with ratio = {args.pruning_ratio}")
    perform_pruning(model, pruning_ratio=args.pruning_ratio)

    train(args, accelerator, (train_ds, train_loader), (val_ds, val_loader), model)
