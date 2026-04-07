# Fake-News
Dự án này được sử dụng để phân biệt tin tức giả hay tin tức thật

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
├── requirements.txt        # Danh sách các thư viện cần cài đặt (pip install -r ...)
└── README.md               # Hướng dẫn cách chạy code (Rất quan trọng khi nộp bài)