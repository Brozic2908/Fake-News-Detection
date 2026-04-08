import os
import re
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


class FakeNewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts.tolist()
        self.labels = labels.tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }


def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def standardize_columns(df):
    rename_map = {}

    for col in df.columns:
        col_clean = col.lower().strip()

        if col_clean in ["title"]:
            rename_map[col] = "title"
        elif col_clean in ["text", "content", "article"]:
            rename_map[col] = "text"
        elif col_clean in ["label", "labels"]:
            rename_map[col] = "label"

    df = df.rename(columns=rename_map)

    required_cols = ["title", "text", "label"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Thiếu cột bắt buộc: {missing_cols}. Các cột hiện có: {df.columns.tolist()}"
        )

    return df


def preprocess_dataframe(df, split_name="dataset"):
    print(f"\n===== PREPROCESS {split_name.upper()} =====")
    print("Raw shape:", df.shape)
    print("Raw columns:", df.columns.tolist())

    df = standardize_columns(df)
    df = df[["title", "text", "label"]].copy()

    df["title"] = df["title"].apply(clean_text)
    df["text"] = df["text"].apply(clean_text)

    # Gộp title + text thành content
    df["content"] = (df["title"] + " " + df["text"]).str.strip()
    df["content"] = df["content"].apply(lambda x: re.sub(r"\s+", " ", x))

    # Làm sạch label
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    df = df[df["label"].isin([0, 1])]

    # Bỏ content rỗng
    df = df[df["content"] != ""]

    # Bỏ trùng theo content
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["content"]).reset_index(drop=True)
    removed_dup = before_dedup - len(df)

    print("Cleaned shape:", df.shape)
    print("Removed duplicates:", removed_dup)
    print("Label distribution:")
    print(df["label"].value_counts())

    # Thống kê độ dài
    df["word_count"] = df["content"].apply(lambda x: len(x.split()))
    print("Word count stats:")
    print(df["word_count"].describe())

    return df


def load_parquet_splits(data_dir):
    train_path = os.path.join(data_dir, "train-00000-of-00001.parquet")
    val_path = os.path.join(data_dir, "validation-00000-of-00001.parquet")
    test_path = os.path.join(data_dir, "test-00000-of-00001.parquet")

    for path in [train_path, val_path, test_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Không tìm thấy file: {path}")

    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    test_df = pd.read_parquet(test_path)

    return train_df, val_df, test_df


def save_processed_data(train_df, val_df, test_df, output_dir="data/processed"):
    os.makedirs(output_dir, exist_ok=True)

    train_save = train_df[["title", "text", "content", "label"]].copy()
    val_save = val_df[["title", "text", "content", "label"]].copy()
    test_save = test_df[["title", "text", "content", "label"]].copy()

    train_path = os.path.join(output_dir, "train_clean.csv")
    val_path = os.path.join(output_dir, "validation_clean.csv")
    test_path = os.path.join(output_dir, "test_clean.csv")

    train_save.to_csv(train_path, index=False, encoding="utf-8-sig")
    val_save.to_csv(val_path, index=False, encoding="utf-8-sig")
    test_save.to_csv(test_path, index=False, encoding="utf-8-sig")

    print("\n===== SAVED PROCESSED CSV FILES =====")
    print("Train:", train_path)
    print("Validation:", val_path)
    print("Test:", test_path)


def get_dataloaders(
    data_dir="data/raw",
    model_name="distilbert-base-uncased",
    max_length=128,
    batch_size=32,
    num_workers=0,
    save_csv=True,
    output_dir="data/processed"
):
    raw_train_df, raw_val_df, raw_test_df = load_parquet_splits(data_dir)

    train_df = preprocess_dataframe(raw_train_df, split_name="train")
    val_df = preprocess_dataframe(raw_val_df, split_name="validation")
    test_df = preprocess_dataframe(raw_test_df, split_name="test")

    # Lưu dữ liệu đã tiền xử lý ra file CSV
    if save_csv:
        save_processed_data(train_df, val_df, test_df, output_dir=output_dir)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_dataset = FakeNewsDataset(
        texts=train_df["content"],
        labels=train_df["label"],
        tokenizer=tokenizer,
        max_length=max_length
    )

    val_dataset = FakeNewsDataset(
        texts=val_df["content"],
        labels=val_df["label"],
        tokenizer=tokenizer,
        max_length=max_length
    )

    test_dataset = FakeNewsDataset(
        texts=test_df["content"],
        labels=test_df["label"],
        tokenizer=tokenizer,
        max_length=max_length
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader, test_loader, tokenizer


if __name__ == "__main__":
    train_loader, val_loader, test_loader, tokenizer = get_dataloaders(
        data_dir="data/raw",
        model_name="distilbert-base-uncased",
        max_length=128,
        batch_size=32,
        num_workers=0,
        save_csv=True,
        output_dir="data/processed"
    )

    print("\n===== CHECK ONE BATCH =====")
    batch = next(iter(train_loader))

    print("input_ids shape:", batch["input_ids"].shape)
    print("attention_mask shape:", batch["attention_mask"].shape)
    print("labels shape:", batch["labels"].shape)

    print("\nSample labels:", batch["labels"][:10])
    print("First sample input_ids[:20]:", batch["input_ids"][0][:20])
    print("First sample attention_mask[:20]:", batch["attention_mask"][0][:20])