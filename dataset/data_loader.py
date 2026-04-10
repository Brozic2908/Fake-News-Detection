import os
import re
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

DEFAULT_MODEL_NAME = "distilbert-base-uncased"
DEFAULT_MAX_LEN    = 512 # Vì count mean = 412
DEFAULT_BATCH_SIZE = 32

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

# Dataset
class FakeNewsDataset(Dataset):
    """
    Đọc file CSV đã xử lý, tokenize văn bản và trả về tensor.
 
    Mỗi sample gồm:
        input_ids      : (max_len,)  LongTensor
        attention_mask : (max_len,)  LongTensor
        label          : ()          LongTensor  (0 = Fake, 1 = Real)
    """
    
    def __init__(
        self,
        dataframe: pd.DataFrame,
        tokenizer: AutoTokenizer,
        max_len: int = DEFAULT_MAX_LEN,
    ) -> None:
        self.df        = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len   = max_len
 
        # Ghép title + text để tận dụng cả hai trường
        self.texts = (
            self.df["title"].fillna("") + " [SEP] " + self.df["text"].fillna("")
        ).tolist()
        self.labels = self.df["label"].tolist()
    
    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        encoding = self.tokenizer(
            self.texts[idx],
            max_length      = self.max_len,
            padding         = "max_length",   # pad về max_len
            truncation      = True,
            return_tensors  = "pt",
        )
 
        return {
            "input_ids"      : encoding["input_ids"].squeeze(0),       # (max_len,)
            "attention_mask" : encoding["attention_mask"].squeeze(0),  # (max_len,)
            "label"          : torch.tensor(self.labels[idx], dtype=torch.long),
        }
        
def get_dataloaders(
    model_name : str = DEFAULT_MODEL_NAME,
    max_len    : int = DEFAULT_MAX_LEN,
    batch_size : int = DEFAULT_BATCH_SIZE,
    num_workers: int = 0,
    processed_dir: str = PROCESSED_DIR,
) -> dict[str, DataLoader]:
    """
    Đọc train_clean.csv / validation_clean.csv / test_clean.csv,
    tạo tokenizer và trả về dict DataLoader.
 
    Returns
    -------
    {
        "train" : DataLoader,
        "val"   : DataLoader,
        "test"  : DataLoader,
    }
    """
    # ── Tokenizer ──────────────────────────────
    print(f"[DataLoader] Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
 
    # ── Đọc CSV ────────────────────────────────
    splits = {
        "train" : os.path.join(processed_dir, "train_clean.csv"),
        "val"   : os.path.join(processed_dir, "validation_clean.csv"),
        "test"  : os.path.join(processed_dir, "test_clean.csv"),
    }
 
    dataloaders: dict[str, DataLoader] = {}
 
    for split_name, csv_path in splits.items():
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"Không tìm thấy file: {csv_path}\n"
                f"Hãy chạy script tiền xử lý trước, hoặc kiểm tra đường dẫn."
            )
 
        df = pd.read_csv(csv_path)
        print(f"[DataLoader] {split_name:5s} → {len(df):,} samples")
 
        dataset = FakeNewsDataset(df, tokenizer, max_len)
 
        shuffle = (split_name == "train")
        dataloaders[split_name] = DataLoader(
            dataset,
            batch_size  = batch_size,
            shuffle     = shuffle,
            num_workers = num_workers,
            pin_memory  = torch.cuda.is_available(),
        )
 
    return dataloaders
 
if __name__ == "__main__":
    loaders = get_dataloaders(batch_size=32)
 
    for split, loader in loaders.items():
        batch = next(iter(loader))
        print(
            f"\n[{split}] "
            f"input_ids: {batch['input_ids'].shape}  "
            f"attention_mask: {batch['attention_mask'].shape}  "
            f"label: {batch['label'].shape}"
        )
        # Kỳ vọng: [32, 256]  [32, 256]  [32]
 
    print("\n✅ DataLoader hoạt động bình thường!")