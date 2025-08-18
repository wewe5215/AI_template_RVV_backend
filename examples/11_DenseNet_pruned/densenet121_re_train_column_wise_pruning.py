# Copyright (c) 2025
# ------------------------------------------------------------
# MobileNet V2 column‑wise pruning + retraining (keep original helpers, revise
#   *training* only)
# ------------------------------------------------------------
#  • f32_data_pruning_column_wise_with_ratio, perform_pruning, evaluate_model
#    are left untouched (copied from the user’s baseline).
#  • train() now uses AdamW + linear warm‑up → cosine decay, and supports
#    gradient accumulation + mixed precision via 🤗 Accelerate.
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
from torchvision.models import densenet121, DenseNet121_Weights

LOGGER = get_logger(__name__)
pruning_ratio = 0.75
result_file = f"results_prune_{int(pruning_ratio*100)}_densenet121.txt"
# -----------------------------------------------------------------------------
#  Original helper: column‑wise mask generator (unchanged)
# -----------------------------------------------------------------------------
vlen = 256
mapping_denseblock3_data = {
    (128, 1, 1, 256): 2,
    (128, 1, 1, 288): 2,
    (128, 1, 1, 320): 2,
    (128, 1, 1, 352): 2,
    (128, 1, 1, 384): 2,
    (128, 1, 1, 416): 2,
    (128, 1, 1, 448): 2,
    (128, 1, 1, 480): 2,
    (128, 1, 1, 512): 2,
    (128, 1, 1, 544): 2,
    (128, 1, 1, 576): 2,
    (128, 1, 1, 608): 2,
    (128, 1, 1, 640): 2,
    (128, 1, 1, 672): 2,
    (128, 1, 1, 704): 2,
    (128, 1, 1, 736): 2,
    (128, 1, 1, 768): 2,
    (128, 1, 1, 800): 2,
    (128, 1, 1, 832): 2,
    (128, 1, 1, 864): 2,
    (128, 1, 1, 896): 2,
    (128, 1, 1, 928): 1,
    (128, 1, 1, 960): 2,
    (128, 1, 1, 992): 2,
}
mapping_denseblock4_data = {
    (128, 1, 1, 512): 4,
    (128, 1, 1, 544): 4,
    (128, 1, 1, 576): 4,
    (128, 1, 1, 608): 4,
    (128, 1, 1, 640): 4,
    (128, 1, 1, 672): 4,
    (128, 1, 1, 704): 4,
    (128, 1, 1, 736): 4,
    (128, 1, 1, 768): 4,
    (128, 1, 1, 800): 4,
    (128, 1, 1, 832): 4,
    (128, 1, 1, 864): 4,
    (128, 1, 1, 896): 4,
    (128, 1, 1, 928): 4,
    (128, 1, 1, 960): 4,
    (128, 1, 1, 992): 4,
}
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
#  Original perform_pruning (adapted only for MobileNetV2 layer names ― keep logic)
# -----------------------------------------------------------------------------
_EXCLUDE_NAME_SUBSTR = [
    "features.conv0.",   # stem
    "features.denseblock1.",   # first inverted residual block
    "features.transition1.",   # second inverted residual block
    "features.denseblock2.",
    "features.transition2.",  # first expansion conv of stage‑2
    "classifier",           # generic fully‑connected layer
]

def _should_skip(name: str, module: nn.Module) -> bool:
    # skip by explicit substrings
    print(f'name = {name}, weight shape = {module.weight.shape}')
    if any(sub in name for sub in _EXCLUDE_NAME_SUBSTR):
        print(f'skip {name} because of _EXCLUDE_NAME_SUBSTR')
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
        w = module.weight.detach().cpu().numpy().astype(np.float32)
        oc, ic, kh, kw = w.shape
        # simple static params for demo
        lmul = 2
        if "denseblock3" in name:
            if "conv2" in name:
                lmul = 2
            else:
                dic_key = (oc, kh, kw, ic)
                lmul = mapping_denseblock3_data[dic_key]
        elif "denseblock4" in name:
            if "conv2" in name:
                lmul = 1
            else:
                dic_key = (oc, kh, kw, ic)
                lmul = mapping_denseblock4_data[dic_key]
        elif "transition3" in name:
            lmul = 2
        # Calculate nr based on your mapping (using weight_to_lmul) and vlen.
        nr = lmul * (vlen / 32)  # 32 for float32
        # print(f'key = {key} being pruned with lmul = {weight_to_lmul[key]}, value.ndim = {value.ndim}')
        if lmul == 1 or lmul == 4:
            mr = 7
        elif lmul == 2:
            mr = 8
        else:
            mr = 3
        print(f'using lmul = {lmul}')
        
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
    final_accuracy = 100 * correct / total
    return final_accuracy

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

    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

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
    p.add_argument("--batch_size", default=32, type=int)
    p.add_argument("--epochs", default=200, type=int)
    p.add_argument("--warmup_epochs", default=10, type=int)
    p.add_argument("--learning_rate", default=1e-4, type=float)
    p.add_argument("--pruning_ratio", default=0.75, type=float)
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

    train_ds = torchvision.datasets.ImageFolder("/data/imagenet/train", transform=train_tf)
    val_ds   = torchvision.datasets.ImageFolder("/data/imagenet/val",   transform=val_tf)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=8, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    accelerator = Accelerator()
    device = accelerator.device

    model = densenet121(weights=DenseNet121_Weights.DEFAULT)
    model.to(device)
    print(f'perform pruning with pruning_ratio = {args.pruning_ratio}')
    perform_pruning(model, pruning_ratio=args.pruning_ratio)

    train(args, accelerator, (train_ds, train_loader), (val_ds, val_loader), model)
