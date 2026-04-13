import os
import sys
import json
import argparse

import torch
import torch.nn as nn
import wandb
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

# Local imports
sys.path.insert(0, os.path.dirname(__file__))
from dataset.data_loader import get_dataloaders
from models.fake_news_model import get_model


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate fine-tuned Fake News model")

    p.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="Path tới checkpoint .pt",
    )
    p.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["val", "test"],
        help="Đánh giá trên tập val hoặc test",
    )
    p.add_argument("--batch_size", type=int, default=32, help="Batch size khi evaluate")
    p.add_argument("--num_workers", type=int, default=2, help="DataLoader workers")
    p.add_argument(
        "--save_json",
        type=str,
        default=None,
        help="Đường dẫn lưu metrics ra json",
    )

    # WandB
    p.add_argument("--use_wandb", action="store_true", help="Bật log evaluate lên WandB")
    p.add_argument("--wandb_project", type=str, default="fake-news-detection", help="Tên project WandB")
    p.add_argument("--wandb_run", type=str, default=None, help="Tên run WandB cho evaluate")

    return p.parse_args()


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)

        preds = torch.argmax(logits, dim=-1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / max(len(all_labels), 1)
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary", zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, digits=4, zero_division=0)

    metrics = {
        "loss": avg_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "labels": all_labels,
        "preds": all_preds,
    }
    return metrics


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[Eval] Loading checkpoint: {args.ckpt_path}")
    ckpt = torch.load(args.ckpt_path, map_location=device)

    saved_args = ckpt.get("args", {})
    model_name = ckpt.get("model_name", saved_args.get("model", "distilbert-base-uncased"))
    dropout = saved_args.get("dropout", 0.3)
    freeze_base = False

    print(f"[Eval] Device      : {device}")
    print(f"[Eval] Model       : {model_name}")
    print(f"[Eval] Split       : {args.split}")
    print(f"[Eval] Batch size  : {args.batch_size}")

    loaders = get_dataloaders(
        model_name=model_name,
        max_len=saved_args.get("max_len", 512),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = get_model(
        model_name=model_name,
        num_classes=2,
        dropout_p=dropout,
        freeze_base=freeze_base,
    ).to(device)

    model.load_state_dict(ckpt["state_dict"], strict=True)
    print("[Eval] Checkpoint loaded successfully.")

    criterion = nn.CrossEntropyLoss()
    loader = loaders["val"] if args.split == "val" else loaders["test"]

    run = None
    if args.use_wandb:
        ckpt_name = os.path.basename(args.ckpt_path).replace(".pt", "")
        run_name = args.wandb_run or f"eval_{ckpt_name}_{args.split}"
        run = wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                "ckpt_path": args.ckpt_path,
                "split": args.split,
                "batch_size": args.batch_size,
                "model_name": model_name,
            },
        )

    metrics = evaluate(model, loader, criterion, device)

    print("\n" + "=" * 60)
    print(f"Results on {args.split.upper()} set")
    print("=" * 60)
    print(f"Loss      : {metrics['loss']:.4f}")
    print(f"Accuracy  : {metrics['accuracy']:.4f}")
    print(f"Precision : {metrics['precision']:.4f}")
    print(f"Recall    : {metrics['recall']:.4f}")
    print(f"F1-score  : {metrics['f1']:.4f}")
    print("\nConfusion Matrix:")
    print(metrics["confusion_matrix"])
    print("\nClassification Report:")
    print(metrics["classification_report"])

    if args.use_wandb and run is not None:
        wandb.log({
            f"{args.split}/loss": metrics["loss"],
            f"{args.split}/accuracy": metrics["accuracy"],
            f"{args.split}/precision": metrics["precision"],
            f"{args.split}/recall": metrics["recall"],
            f"{args.split}/f1": metrics["f1"],
        })

        wandb.log({
            f"{args.split}/confusion_matrix": wandb.plot.confusion_matrix(
                probs=None,
                y_true=metrics["labels"],
                preds=metrics["preds"],
                class_names=["real", "fake"],
            )
        })

        wandb.summary[f"{args.split}_loss"] = metrics["loss"]
        wandb.summary[f"{args.split}_accuracy"] = metrics["accuracy"]
        wandb.summary[f"{args.split}_precision"] = metrics["precision"]
        wandb.summary[f"{args.split}_recall"] = metrics["recall"]
        wandb.summary[f"{args.split}_f1"] = metrics["f1"]

    if args.save_json:
        os.makedirs(os.path.dirname(args.save_json), exist_ok=True)
        save_obj = {
            "loss": metrics["loss"],
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "confusion_matrix": metrics["confusion_matrix"],
            "classification_report": metrics["classification_report"],
        }
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(save_obj, f, ensure_ascii=False, indent=2)
        print(f"[Eval] Metrics saved to: {args.save_json}")

    if args.use_wandb and run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()