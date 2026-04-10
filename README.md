# 📰 Dự án Phát hiện Tin giả (Fake News Detection)

Dự án này ứng dụng các mô hình ngôn ngữ lớn (Transformers như BERT/RoBERTa/DistilBERT) và PyTorch để phân loại tin tức là Thật (Real) hay Giả (Fake).

## 📂 Cấu trúc Thư mục

```text
fake-news-detection/
│
├── data/                   # Chứa dữ liệu (Thành viên 1 quản lý)
│   ├── raw/                # Dữ liệu gốc tải về từ Hugging Face
│   └── processed/          # Dữ liệu đã qua tiền xử lý (nếu có lưu lại)
│
├── models/                 # Chứa code định nghĩa mạng Nơ-ron (Thành viên 2)
│   ├── __init__.py
│   └── fake_news_model.py  # Định nghĩa class FakeNewsClassifier (DistilBERT + Head)
│
├── dataset/                # Chứa code xử lý dữ liệu (Thành viên 1)
│   ├── __init__.py
│   └── data_loader.py      # Định nghĩa class FakeNewsDataset và hàm get_dataloaders
│
├── utils/                  # Các hàm tiện ích dùng chung
│   ├── __init__.py
│   └── metrics.py          # Hàm tính F1-score, vẽ Confusion Matrix (Thành viên 3)
│
├── checkpoints/            # Nơi lưu trữ trọng số mô hình tốt nhất (.pt / .pth)
│
├── train.py                # Script chính để chạy huấn luyện (Thành viên 2 & 3)
├── evaluate.py             # Script đánh giá mô hình trên tập test (Thành viên 3)
├── inference.py            # Script chạy thử với một đoạn tin tức bất kỳ (Thành viên 3)
│
├── requirements.txt        # Danh sách các thư viện cần cài đặt
└── README.md               # Hướng dẫn chạy code và phân công công việc
```

---

## 🚀 Hướng dẫn Cài đặt & Chạy Code

**1. Cài đặt môi trường**
```bash
# Clone repository
git clone https://github.com/Brozic2908/Fake-News-Detection
cd fake-news-detection

# Khởi tạo môi trường ảo (Khuyến nghị)
python -m venv venv
source venv/bin/activate  # Trên Windows: venv\Scripts\activate

# Cài đặt thư viện
pip install -r requirements.txt

# Bước 1: Tiền xử lý (một lần duy nhất)
python preprocess.py

# Bước 2: Test DataLoader
python dataset/data_loader.py

# Bước 3: Test Model forward pass
python models/fake_news_model.py

# Bước 4: Train (WandB sẽ hỏi login lần đầu)
python train.py --model distilbert-base-uncased --epochs 4
# hoặc dùng RoBERTa
python train.py --model roberta-base --epochs 5 --lr 2e-5
```

**2. Quy trình chạy script**
- **Kiểm tra dữ liệu:** `python dataset/data_loader.py` (Đảm bảo in ra tensor shape chuẩn).
- **Huấn luyện mô hình:** `python train.py` (Quá trình này sẽ yêu cầu đăng nhập Wandb).
- **Đánh giá mô hình:** `python evaluate.py` (Sử dụng checkpoint tốt nhất trong thư mục `checkpoints/`).
- **Thử nghiệm dự đoán:** `python inference.py --text "Nhập câu tin tức vào đây"`

---

## 📋 Phân công Nhiệm vụ & Theo dõi Tiến độ

### 👤 Thành viên 1: Data Engineer (Xử lý Dữ liệu & Dataloader)
**Mục tiêu:** Chuẩn bị đầu vào chuẩn xác, output ra batch tensor.
-  Tải dataset (vd: từ Hugging Face).
-  Xử lý giá trị rỗng (NaN) và gom nhóm nhãn (vd: từ 6 nhãn xuống 2 nhãn True/False).
-  Khởi tạo Hugging Face Tokenizer (chuyển đổi văn bản thành `input_ids` và `attention_mask`).
-  Code class kế thừa `torch.utils.data.Dataset` (padding/truncation về độ dài cố định).
-  Viết hàm trả ra `DataLoader` cho Train/Val/Test với `batch_size` phù hợp.
-  **Test:** Chạy độc lập `data_loader.py` in ra test thử một batch tensor có shape chuẩn (vd: `[32, 128]`).
- Tìm hiểu các thuật toán đang có và lưu file docs cho thành viên 2

### 👤 Thành viên 2: Model Engineer (Thiết kế Kiến trúc & Training Loop)
**Mục tiêu:** Định nghĩa kiến trúc Transformer và đảm bảo pipeline huấn luyện chạy mượt mà.
-  Viết class kiến trúc mô hình kế thừa `nn.Module` trong `models/fake_news_model.py` (Pre-trained Transformer + Linear Classifier Head).
-  Khởi tạo Loss Function (vd: `CrossEntropyLoss`) và Optimizer (`AdamW` + Scheduler).
-  Viết vòng lặp huấn luyện (Training Loop) chuẩn của PyTorch trong `train.py`.
-  **Test:** Chạy thử vài epoch đầu tiên không bị lỗi tràn RAM (OOM) hay lệch shape tensor.

### 👤 Thành viên 3: MLOps & Evaluation (Wandb, Đánh giá & Suy luận)
**Mục tiêu:** Track log huấn luyện, tính toán metrics và xây dựng script dự đoán.
-  Tích hợp Weights & Biases (`wandb.init()`) vào `train.py` để log `train_loss`, `val_loss`, `accuracy`.
-  Viết hàm `evaluate()` trong `evaluate.py`: tính F1-score, Precision, Recall và vẽ Confusion Matrix (dùng `scikit-learn`).
-  Code chức năng lưu `state_dict` cho Best Checkpoint (val_loss thấp nhất) vào thư mục `checkpoints/`.
-  Viết script `inference.py` để nhập tin tức bất kỳ và in ra kết quả Fake/Real.
-  Đảm bảo có link project Wandb ở chế độ Public.

---

## 📝 Phân chia Báo cáo (Report & Slides)
*(Thực hiện sau khi ráp xong code - Thời gian dự kiến: 3 ngày cuối)*

-  **Thành viên 1:** Viết phần Tổng quan tài liệu (Literature Review) và Mô tả bộ dữ liệu, mô tả các thuật toán đang có.
-  **Thành viên 2:** Vẽ sơ đồ kiến trúc mô hình, giải thích thuật toán, loss function và các siêu tham số (Hyperparameters).
-  **Thành viên 3:** Chụp ảnh log từ Wandb, vẽ biểu đồ, viết phân tích đánh giá hiệu năng và so sánh kết quả. 
-  **Cả nhóm:** Tổng hợp source code, link Wandb, xuất Report/Slide ra PDF và nén thành `Group_XX.zip` để nộp.

🔗 **Link Wandb Public:** `[Chèn link Wandb của nhóm vào đây]`