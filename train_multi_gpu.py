"""
train.py
Multi-GPU friendly version for Kaggle / local CUDA.

Chạy ví dụ:
    python train.py
    python train.py --model roberta-base --epochs 3 --lr 2e-5 --batch_size 16
    python train.py --use_amp --multi_gpu
"""

import os
import sys
import argparse
import time
import random
import numpy as np

import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

import wandb
from tqdm import tqdm

# ── Local imports ──────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from dataset.data_loader import get_dataloaders
from models.fake_news_model import get_model
from utils.metrics import compute_metrics


# Argument parser

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Fake News Classifier (Multi-GPU ready)")

    p.add_argument("--model", type=str, default="distilbert-base-uncased",
                   help="HuggingFace model name")
    p.add_argument("--max_len", type=int, default=512, help="Max token length")
    p.add_argument("--batch_size", type=int, default=16,
                   help="Per-step batch size. With multi-GPU, global batch ~= batch_size x n_gpu")
    p.add_argument("--epochs", type=int, default=3, help="Số epoch")
    p.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    p.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay")
    p.add_argument("--dropout", type=float, default=0.3, help="Dropout probability")
    p.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup steps ratio")
    p.add_argument("--num_workers", type=int, default=2, help="DataLoader workers")
    p.add_argument("--freeze_base", action="store_true", help="Đóng băng transformer backbone")
    p.add_argument("--wandb_project", type=str, default="fake-news-detection")
    p.add_argument("--wandb_run", type=str, default=None, help="Tên WandB run (tuỳ chọn)")
    p.add_argument("--ckpt_dir", type=str, default="checkpoints")
    p.add_argument("--seed", type=int, default=42)

    # Multi-GPU / speed options
    p.add_argument("--multi_gpu", action="store_true",
                   help="Dùng tất cả GPU khả dụng bằng torch.nn.DataParallel")
    p.add_argument("--use_amp", action="store_true",
                   help="Bật mixed precision (AMP) để giảm VRAM và tăng tốc")
    p.add_argument("--grad_accum_steps", type=int, default=1,
                   help="Gradient accumulation steps")
    p.add_argument("--wandb_entity", type=str, default=None,
                   help="WandB entity (username hoặc team name)")
    return p.parse_args()


# Reproducibility

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Helpers

def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)) else model


def build_model(args: argparse.Namespace, device: torch.device) -> tuple[nn.Module, int]:
    base_model = get_model(
        model_name=args.model,
        dropout_p=args.dropout,
        freeze_base=args.freeze_base,
    )

    n_gpu = torch.cuda.device_count() if device.type == "cuda" else 0

    if device.type == "cuda":
        base_model = base_model.to(device)

    if args.multi_gpu and n_gpu > 1:
        print(f"[Info] Multi-GPU enabled with DataParallel on {n_gpu} GPUs")
        model = nn.DataParallel(base_model)
    else:
        if args.multi_gpu and n_gpu <= 1:
            print("[Warn] --multi_gpu được bật nhưng chỉ có 1 GPU khả dụng. Sẽ chạy single GPU.")
        model = base_model

    return model, n_gpu


# Training / Validation epoch

def run_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: AdamW | None,
    scheduler,
    device: torch.device,
    is_train: bool,
    scaler: torch.cuda.amp.GradScaler | None,
    grad_accum_steps: int = 1,
) -> dict:
    """Một vòng epoch. Trả về dict metrics."""

    model.train(is_train)
    total_loss, all_preds, all_labels = 0.0, [], []

    if is_train and optimizer is not None:
        optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(tqdm(loader, desc="Train" if is_train else "Val  ", leave=False), start=1):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attn_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        with torch.set_grad_enabled(is_train):
            use_amp_now = bool(is_train and scaler is not None and device.type == "cuda")
            with torch.cuda.amp.autocast(enabled=use_amp_now):
                logits = model(input_ids, attn_mask)
                loss = criterion(logits, labels)
                loss_for_backward = loss / grad_accum_steps if is_train else loss

            if is_train and optimizer is not None:
                if scaler is not None and device.type == "cuda":
                    scaler.scale(loss_for_backward).backward()
                else:
                    loss_for_backward.backward()

                if step % grad_accum_steps == 0 or step == len(loader):
                    if scaler is not None and device.type == "cuda":
                        scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                    if scaler is not None and device.type == "cuda":
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()

                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=-1).detach().cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.detach().cpu().tolist())

    n = max(len(all_labels), 1)
    metrics = compute_metrics(all_labels, all_preds)
    metrics["loss"] = total_loss / n
    return metrics


# Main

def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(args.seed)

    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)] if gpu_count > 0 else []

    print(f"\n{'=' * 60}")
    print("  Fake News Detection — Training")
    print(f"  Model       : {args.model}")
    print(f"  Device      : {device}")
    print(f"  GPU count   : {gpu_count}")
    if gpu_names:
        print(f"  GPU names   : {gpu_names}")
    print(f"  Multi-GPU   : {args.multi_gpu}")
    print(f"  AMP         : {args.use_amp}")
    print(f"  Batch size  : {args.batch_size}")
    print(f"  Grad accum  : {args.grad_accum_steps}")
    print(f"{'=' * 60}\n")

    # ── WandB ──────────────────────────────────
    model_slug = args.model.split("/")[-1]
    run_name = args.wandb_run or f"{model_slug}_ep{args.epochs}_lr{args.lr}"
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        config={**vars(args), "detected_gpu_count": gpu_count, "gpu_names": gpu_names},
    )

    # ── DataLoaders ────────────────────────────
    loaders = get_dataloaders(
        model_name=args.model,
        max_len=args.max_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # ── Model ──────────────────────────────────
    model, n_gpu = build_model(args, device)
    wandb.watch(unwrap_model(model), log="gradients", log_freq=100)

    # ── Loss & Optimizer ───────────────────────
    criterion = nn.CrossEntropyLoss()

    trainable_named_params = [(n, p) for n, p in unwrap_model(model).named_parameters() if p.requires_grad]
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    params = [
        {
            "params": [p for n, p in trainable_named_params if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in trainable_named_params if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(params, lr=args.lr)

    # ── Scheduler ──────────────────────────────
    steps_per_epoch = len(loaders["train"])
    optimizer_steps_per_epoch = (steps_per_epoch + args.grad_accum_steps - 1) // args.grad_accum_steps
    total_steps = optimizer_steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # ── AMP scaler ─────────────────────────────
    scaler = torch.cuda.amp.GradScaler(enabled=(args.use_amp and device.type == "cuda"))

    # ── Checkpoint dir ─────────────────────────
    os.makedirs(args.ckpt_dir, exist_ok=True)
    best_val_loss = float("inf")
    best_val_f1 = float("-inf")
    best_val_acc = float("-inf")
    ckpt_name = f"best_{model_slug}_ep{args.epochs}_lr{args.lr}.pt"
    best_ckpt = os.path.join(args.ckpt_dir, ckpt_name)

    # ── Training loop ──────────────────────────
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_m = run_epoch(
            model=model,
            loader=loaders["train"],
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            is_train=True,
            scaler=scaler,
            grad_accum_steps=args.grad_accum_steps,
        )
        val_m = run_epoch(
            model=model,
            loader=loaders["val"],
            criterion=criterion,
            optimizer=None,
            scheduler=None,
            device=device,
            is_train=False,
            scaler=None,
            grad_accum_steps=1,
        )

        elapsed = time.time() - t0
        current_lr = scheduler.get_last_lr()[0] if scheduler is not None else args.lr

        print(
            f"  Train → loss: {train_m['loss']:.4f}  acc: {train_m['accuracy']:.4f}  f1: {train_m['f1']:.4f}\n"
            f"  Val   → loss: {val_m['loss']:.4f}  acc: {val_m['accuracy']:.4f}  f1: {val_m['f1']:.4f}  ({elapsed:.1f}s)"
        )

        wandb.log({
            "epoch": epoch,
            "train/loss": train_m["loss"],
            "train/accuracy": train_m["accuracy"],
            "train/f1": train_m["f1"],
            "val/loss": val_m["loss"],
            "val/accuracy": val_m["accuracy"],
            "val/f1": val_m["f1"],
            "lr": current_lr,
            "system/gpu_count": n_gpu,
        })

        # Save best checkpoint without DataParallel "module." prefix
        if val_m["loss"] < best_val_loss:
            best_val_loss = val_m["loss"]
            best_val_f1 = val_m["f1"]
            best_val_acc = val_m["accuracy"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_name": args.model,
                    "state_dict": unwrap_model(model).state_dict(),
                    "val_loss": best_val_loss,
                    "val_f1": best_val_f1,
                    "val_accuracy": best_val_acc,
                    "args": vars(args),
                    "gpu_count": n_gpu,
                },
                best_ckpt,
            )
            print(f"  ✅ Best checkpoint saved → {best_ckpt}  (val_loss={best_val_loss:.4f})")

    wandb.summary["best_val_loss"] = best_val_loss
    wandb.summary["best_val_f1"] = best_val_f1
    wandb.summary["best_val_accuracy"] = best_val_acc
    wandb.summary["checkpoint"] = best_ckpt
    wandb.summary["gpu_count"] = n_gpu

    print(f"\n🏁 Training hoàn tất. Best val_loss = {best_val_loss:.4f}")
    wandb.finish()


if __name__ == "__main__":
    main()
