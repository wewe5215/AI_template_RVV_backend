# Copyright (c) 2025
# ------------------------------------------------------------
# MobileNet V2 column‑wise pruning + retraining
#  • Helpers (pruning, evaluation) unchanged
#  • train() updated: SGD optimizer + StepLR (ImageNet recipe)
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
from torchvision.models import mobilenet_v2

LOGGER = get_logger(__name__)
pruning_ratio = 0.5
result_file = f"results_prune_{int(pruning_ratio*100)}_mobilenetv2.txt"
# -----------------------------------------------------------------------------
#  Original helper: column‑wise mask generator (unchanged)
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
    "features.1.",   # first inverted residual block
    "features.2.",   # second inverted residual block
    "features.3.",
    "features.4.conv.0.0",  # first expansion conv of stage‑2
    "classifier",           # generic fully‑connected layer
]

def _should_skip(name: str, module: nn.Module) -> bool:
    # skip by explicit substrings
    print(f'name = {name}, weight shape = {module.weight.shape}')
    if any(sub in name for sub in _EXCLUDE_NAME_SUBSTR):
        print(f'skip {name} because of _EXCLUDE_NAME_SUBSTR')
        return True
    # skip depth‑wise conv  (groups == in_channels)
    if isinstance(module, nn.Conv2d) and module.groups == module.in_channels:
        print(f'skip {name} because of depthwise conv')
        return True
    return False

def perform_pruning(model: nn.Module, pruning_ratio: float = 0.5):
    skipped, pruned = 0, 0
    for name, module in model.named_modules():
        if not isinstance(module, nn.Conv2d):
            continue
        if _should_skip(name=name, module=module):
            skipped += 1
            continue
        print(f"perform pruning on {name}")
        # simple static params for demo
        mr = 8
        nr = int(2 * (256 / 32)) # could be tuned per layer
        w = module.weight.detach().cpu().numpy().astype(np.float32)
        oc, ic, kh, kw = w.shape
        mask_flat, _ = f32_data_pruning_column_wise_with_ratio(
            w.reshape(oc, ic * kh * kw), nr, mr, pruning_ratio)
        mask = torch.from_numpy(mask_flat.reshape(w.shape)).to(module.weight.device)
        prune.CustomFromMask.apply(module, name="weight", mask=mask)
        pruned += 1
    LOGGER.info(f"Pruned {pruned} conv layers, skipped {skipped}, ratio={pruning_ratio}")

# -----------------------------------------------------------------------------
#  Original evaluation (unchanged)
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
#  NEW train() – replaces StepLR with warm‑up + cosine, keeps original signature
# -----------------------------------------------------------------------------

def train(args, accelerator, train_data, eval_data, model):
    train_ds, train_loader = train_data
    val_ds,   val_loader   = eval_data

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  betas=(0.9, 0.999), eps=args.adam_epsilon)
    total_steps = args.epochs * math.ceil(len(train_loader) / args.grad_accum)
    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    def lr_lambda(step):
        # which epoch are we in?
        epoch = step // steps_per_epoch
        if epoch < 30:
            return 1.0
        elif epoch < 65:
            return 0.1
        elif epoch < 85:
            return 0.01
        elif epoch < 105:
            return 0.001
        elif epoch < 110:
            return 0.0001
        else:
            return 0.00001

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    model, optimizer, train_loader, scheduler = accelerator.prepare(model, optimizer, train_loader, scheduler)
    criterion = nn.CrossEntropyLoss()

    best = 0.0
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        num_batches  = 0

        prog = tqdm(train_loader, disable=not accelerator.is_local_main_process)
        for step, (x, y) in enumerate(prog):
            x, y = x.to(accelerator.device), y.to(accelerator.device)
            with accelerator.accumulate(model):
                out  = model(x)
                loss = criterion(out, y)

                # accumulate for reporting
                running_loss += loss.item()
                num_batches  += 1

                accelerator.backward(loss)
                if args.max_grad_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            if step % 50 == 0 and accelerator.is_local_main_process:
                prog.set_description(f"E{epoch+1} L{loss.item():.4f}")

        # compute average train loss
        avg_train_loss = running_loss / num_batches

        # evaluate validation accuracy
        acc = evaluate_model(model, val_loader, accelerator.device)

        accelerator.print(
            f"Epoch {epoch+1}/{args.epochs} — "
            f"train loss {avg_train_loss:.4f} — "
            f"val acc {acc:.2f}% — "
            f"lr {scheduler.get_last_lr()[0]:.2e}"
        )
        result_line = f"Epoch {epoch+1}/{args.epochs} — train loss {avg_train_loss:.4f} — val acc {acc:.2f}% — lr {scheduler.get_last_lr()[0]:.2e}"
        if accelerator.is_local_main_process:
            abs_path = os.path.abspath(result_file)
            print(f"Results will be written to: {abs_path}")
            with open(result_file, "a") as f:
                f.write(result_line + "\n")

        # save best
        if acc > best and accelerator.is_local_main_process:
            best = acc
            abs_path = os.path.abspath(result_file)
            accelerator.print(f"Results will be written to: {abs_path}")
            with open(result_file, "a") as f:
                f.write(f'current best = {best:.2f}%\n')
            args.output_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                accelerator.unwrap_model(model).state_dict(),
                args.output_dir / "mobilenetv2_pruned_best.pt"
            )

    accelerator.print(f"Best accuracy: {best:.2f}%")

# -----------------------------------------------------------------------------
#  Entrypoint (minimal args — other originals retained if desired)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    #p.add_argument("--data_dir", required=True, type=Path)
    p.add_argument("--output_dir", required=True, type=Path)
    p.add_argument("--batch_size", default=16, type=int)
    p.add_argument("--epochs", default=200, type=int)
    p.add_argument("--warmup_epochs", default=10, type=int)
    p.add_argument("--learning_rate", default=1e-5, type=float)
    p.add_argument("--pruning_ratio", default=0.5, type=float)
    p.add_argument("--adam_epsilon", default=1e-8, type=float)
    p.add_argument("--grad_accum", default=1, type=int)
    p.add_argument("--max_grad_norm", default=1.0, type=float)
    args = p.parse_args()

    set_seed(42)

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    train_ds = torchvision.datasets.ImageFolder("/home/wewe5215/imagenet/train", transform=train_tf)
    val_ds   = torchvision.datasets.ImageFolder("/home/wewe5215/imagenet/val",   transform=val_tf)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=8, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    accelerator = Accelerator()
    device = accelerator.device

    model = mobilenet_v2(pretrained=True)
    model.to(device)
    print(f'perform pruning with pruning_ratio = {args.pruning_ratio}')
    perform_pruning(model, pruning_ratio=args.pruning_ratio)

    train(args, accelerator, (train_ds, train_loader), (val_ds, val_loader), model)
