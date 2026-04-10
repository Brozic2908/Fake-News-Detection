"""
preprocess.py  (chạy một lần trước khi train)

Đọc file .parquet trong data/raw/ → làm sạch → lưu CSV vào data/processed/

Chạy:
    python preprocess.py
"""

import os
import re
import pandas as pd

RAW_DIR       = os.path.join("data", "raw")
PROCESSED_DIR = os.path.join("data", "processed")

SPLIT_FILES = {
    "train"      : "train-00000-of-00001.parquet",
    "validation" : "validation-00000-of-00001.parquet",
    "test"       : "test-00000-of-00001.parquet",
}

OUTPUT_FILES = {
    "train"      : "train_clean.csv",
    "validation" : "validation_clean.csv",
    "test"       : "test_clean.csv",
}


# ─────────────────────────────────────────────
def clean_text(text: str) -> str:
    """Làm sạch và chuẩn hoá khoảng trắng."""
    if pd.isna(text):
        return ""
    text = str(text).replace("\n", " ").replace("\r", " ").replace("\t", " ")
    text = re.sub(r"https?://\S+|www\.\S+", " ", text) # Thêm lọc URL từ File 1
    text = re.sub(r"<[^>]+>", " ", text)               # Thêm lọc HTML từ File 1
    text = re.sub(r"\s+", " ", text).strip()
    return text

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Đồng bộ tên cột để tránh lỗi định dạng."""
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
    return df

def process_split(split_name: str) -> pd.DataFrame:
    parquet_path = os.path.join(RAW_DIR, SPLIT_FILES[split_name])
    print(f"\n[Preprocess] Đang xử lý: {split_name.upper()}")
    
    df = pd.read_parquet(parquet_path)
    df = standardize_columns(df)
    
    # Chỉ giữ lại các cột cần thiết
    if not all(col in df.columns for col in ["title", "text", "label"]):
        raise ValueError(f"Thiếu cột bắt buộc trong {split_name}")
    df = df[["title", "text", "label"]].copy()

    # Làm sạch text
    df["title"] = df["title"].apply(clean_text)
    df["text"]  = df["text"].apply(clean_text)
    
    # Ép kiểu nhãn (label) và loại bỏ nhãn lỗi
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    df = df[df["label"].isin([0, 1])]

    # Lọc bỏ các dòng mà title và text đều rỗng
    df = df[(df["title"] != "") | (df["text"] != "")].copy()

    # Tạo cột content tạm thời để check duplicate (giống File 2)
    df["content_temp"] = (df["title"] + " " + df["text"]).str.strip()
    
    # Lọc trùng (Deduplication)
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["content_temp"]).reset_index(drop=True)
    after_dedup = len(df)
    
    # Xoá cột tạm và trả về dataframe sạch
    df = df.drop(columns=["content_temp"])
    
    print(f"  -> Original: {before_dedup:,} | Cleaned: {after_dedup:,} (Đã loại {before_dedup - after_dedup:,} dòng lỗi/trùng)")
    return df


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    for split_name, out_file in OUTPUT_FILES.items():
        df = process_split(split_name)
        out_path = os.path.join(PROCESSED_DIR, out_file)
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"  -> Đã lưu: {out_path}\n")
        
    print("✅ TIỀN XỬ LÝ HOÀN TẤT!")

if __name__ == "__main__":
    main()