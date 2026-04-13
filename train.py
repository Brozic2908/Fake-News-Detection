"""
train.py
Thành viên 2 (kiến trúc + training loop) & Thành viên 3 (WandB, checkpoint)

Chạy:
    python train.py
    python train.py --model roberta-base --epochs 5 --lr 2e-5
"""

import os
import sys
import argparse
import time

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
from utils.metrics import compute_metrics   # accuracy + f1 nhanh trong train


# Argument parser
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Fake News Classifier")

    p.add_argument("--model",       type=str,   default="distilbert-base-uncased",
                   help="HuggingFace model name (distilbert / bert-base-uncased / roberta-base)")
    p.add_argument("--max_len",     type=int,   default=512,  help="Max token length")
    p.add_argument("--batch_size",  type=int,   default=32,   help="Batch size")
    p.add_argument("--epochs",      type=int,   default=1,    help="Số epoch")
    p.add_argument("--lr",          type=float, default=2e-5, help="Learning rate")
    p.add_argument("--weight_decay",type=float, default=1e-2, help="Weight decay")
    p.add_argument("--dropout",     type=float, default=0.3,  help="Dropout probability")
    p.add_argument("--warmup_ratio",type=float, default=0.1,  help="Warmup steps ratio")
    p.add_argument("--num_workers", type=int,   default=0,    help="DataLoader workers")
    p.add_argument("--freeze_base", action="store_true",      help="Đóng băng transformer backbone")
    p.add_argument("--wandb_project", type=str, default="fake-news-detection")
    p.add_argument("--wandb_run",   type=str,   default=None, help="Tên WandB run (tuỳ chọn)")
    p.add_argument("--ckpt_dir",    type=str,   default="checkpoints")
    p.add_argument("--seed",        type=int,   default=42)

    return p.parse_args()


# Reproducibility
def set_seed(seed: int) -> None:
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Training / Validation epoch
def run_epoch(
    model      : nn.Module,
    loader,
    criterion  : nn.Module,
    optimizer  : AdamW | None,
    scheduler,
    device     : torch.device,
    is_train   : bool,
) -> dict:
    """Một vòng epoch. Trả về dict metrics."""

    model.train(is_train)
    total_loss, all_preds, all_labels = 0.0, [], []

    with torch.set_grad_enabled(is_train):
        for batch in tqdm(loader, desc="Train" if is_train else "Val ", leave=False):
            input_ids  = batch["input_ids"].to(device)
            attn_mask  = batch["attention_mask"].to(device)
            labels     = batch["label"].to(device)

            logits = model(input_ids, attn_mask)       # (B, 2)
            loss   = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=-1).cpu().tolist()
            all_preds  .extend(preds)
            all_labels .extend(labels.cpu().tolist())

    n       = len(all_labels)
    metrics = compute_metrics(all_labels, all_preds)
    metrics["loss"] = total_loss / n
    return metrics


# Main
def main() -> None:
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(args.seed)
    print(f"\n{'='*55}")
    print(f"  Fake News Detection — Training")
    print(f"  Model  : {args.model}")
    print(f"  Device : {device}")
    print(f"{'='*55}\n")

    # ── WandB ──────────────────────────────────
    model_slug = args.model.split("/")[-1]   # vd: "roberta-base"
    run_name   = args.wandb_run or f"{model_slug}_ep{args.epochs}_lr{args.lr}"
    wandb.init(
        project = args.wandb_project,
        name    = run_name,
        config  = vars(args),
    )

    # ── DataLoaders ────────────────────────────
    loaders = get_dataloaders(
        model_name  = args.model,
        max_len     = args.max_len,
        batch_size  = args.batch_size,
        num_workers = args.num_workers,
    )

    # ── Model ──────────────────────────────────
    model = get_model(
        model_name  = args.model,
        dropout_p   = args.dropout,
        freeze_base = args.freeze_base,
    ).to(device)
    wandb.watch(model, log="gradients", log_freq=100)

    # ── Loss & Optimizer ───────────────────────
    criterion = nn.CrossEntropyLoss()

    # Tách weight decay: không apply cho bias / LayerNorm
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    params = [
        {"params": [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters()
                    if     any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = AdamW(params, lr=args.lr)

    # ── Scheduler (linear warmup + linear decay) ──
    total_steps  = len(loaders["train"]) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler    = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # ── Checkpoint dir ─────────────────────────
    os.makedirs(args.ckpt_dir, exist_ok=True)
    best_val_loss = float("inf")
    ckpt_name  = f"best_{model_slug}_ep{args.epochs}_lr{args.lr}.pt"
    best_ckpt  = os.path.join(args.ckpt_dir, ckpt_name)

    # ── Training loop ──────────────────────────
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_m = run_epoch(model, loaders["train"], criterion, optimizer, scheduler, device, is_train=True)
        val_m   = run_epoch(model, loaders["val"],   criterion, None,      None,      device, is_train=False)

        elapsed = time.time() - t0
        print(
            f"  Train → loss: {train_m['loss']:.4f}  acc: {train_m['accuracy']:.4f}  f1: {train_m['f1']:.4f}\n"
            f"  Val   → loss: {val_m['loss']:.4f}  acc: {val_m['accuracy']:.4f}  f1: {val_m['f1']:.4f}  "
            f"({elapsed:.1f}s)"
        )

        # WandB log
        wandb.log({
            "epoch"          : epoch,
            "train/loss"     : train_m["loss"],
            "train/accuracy" : train_m["accuracy"],
            "train/f1"       : train_m["f1"],
            "val/loss"       : val_m["loss"],
            "val/accuracy"   : val_m["accuracy"],
            "val/f1"         : val_m["f1"],
            "lr"             : scheduler.get_last_lr()[0],
        })

        # Lưu best checkpoint theo val_loss
        if val_m["loss"] < best_val_loss:
            best_val_loss  = val_m["loss"]
            best_val_f1    = val_m["f1"]
            best_val_acc   = val_m["accuracy"]
            torch.save(
                {
                    "epoch"      : epoch,
                    "model_name" : args.model,
                    "state_dict" : model.state_dict(),
                    "val_loss"   : best_val_loss,
                    "val_f1"     : val_m["f1"],
                    "val_accuracy": val_m["accuracy"],
                    "args"       : vars(args),
                },
                best_ckpt,
            )
            print(f"  ✅ Best checkpoint saved → {best_ckpt}  (val_loss={best_val_loss:.4f})")

    # ── WandB summary — hiện ở cột ngoài cùng bảng so sánh các run ──
    wandb.summary["best_val_loss"]     = best_val_loss
    wandb.summary["best_val_f1"]       = best_val_f1
    wandb.summary["best_val_accuracy"] = best_val_acc
    wandb.summary["checkpoint"]        = best_ckpt
    
    print(f"\n🏁 Training hoàn tất. Best val_loss = {best_val_loss:.4f}")
    wandb.finish()


if __name__ == "__main__":
    main()