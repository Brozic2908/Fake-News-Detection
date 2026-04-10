import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


# Model
class FakeNewsClassifier(nn.Module):
    """
    Kiến trúc:
        [Input tokens]
              ↓
        Pre-trained Transformer  (DistilBERT / BERT / RoBERTa)
              ↓  CLS token embedding  (hidden_size,)
        LayerNorm  →  Dropout
              ↓
        Linear (hidden_size → hidden_size // 2)
        GELU
        Dropout
              ↓
        Linear (hidden_size // 2 → num_classes)
              ↓
        Logits  (num_classes,)

    Parameters
    ----------
    model_name  : tên / đường dẫn HuggingFace checkpoint
    num_classes : số nhãn (mặc định 2: Fake / Real)
    dropout_p   : xác suất dropout
    freeze_base : True → đóng băng toàn bộ transformer (chỉ train head)
    """

    def __init__(
        self,
        model_name  : str   = "distilbert-base-uncased",
        num_classes : int   = 2,
        dropout_p   : float = 0.3,
        freeze_base : bool  = False,
    ) -> None:
        super().__init__()

        self.model_name = model_name

        # ── Backbone ───────────────────────────
        config           = AutoConfig.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name, config=config)
        hidden_size      = config.hidden_size   # 768 cho BERT/RoBERTa, 768 cho DistilBERT

        # Đóng băng backbone nếu cần (tốt cho máy yếu / giai đoạn thử nghiệm)
        if freeze_base:
            for param in self.transformer.parameters():
                param.requires_grad = False

        # ── Classification head ────────────────
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_size // 2, num_classes),
        )

        # Khởi tạo trọng số Linear theo phân phối chuẩn nhỏ
        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self) -> None:
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids      : torch.Tensor,
        attention_mask : torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        input_ids      : (batch, seq_len)
        attention_mask : (batch, seq_len)

        Returns
        -------
        logits : (batch, num_classes)
        """
        outputs = self.transformer(
            input_ids      = input_ids,
            attention_mask = attention_mask,
        )

        # Lấy CLS token (vị trí 0) từ last_hidden_state
        # last_hidden_state : (batch, seq_len, hidden_size)
        cls_output = outputs.last_hidden_state[:, 0, :]   # (batch, hidden_size)

        logits = self.classifier(cls_output)              # (batch, num_classes)
        return logits

    # ------------------------------------------------------------------
    def count_parameters(self) -> dict:
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}


# Factory function
def get_model(
    model_name  : str   = "distilbert-base-uncased",
    num_classes : int   = 2,
    dropout_p   : float = 0.3,
    freeze_base : bool  = False,
) -> FakeNewsClassifier:
    """
    Tạo và trả về FakeNewsClassifier.

    Ví dụ:
        model = get_model("roberta-base")
        model = get_model("bert-base-uncased", dropout_p=0.2)
    """
    model = FakeNewsClassifier(
        model_name  = model_name,
        num_classes = num_classes,
        dropout_p   = dropout_p,
        freeze_base = freeze_base,
    )
    params = model.count_parameters()
    print(
        f"[Model] {model_name} | "
        f"Total params: {params['total']:,} | "
        f"Trainable: {params['trainable']:,}"
    )
    return model


# Quick test  (python models/fake_news_model.py)
if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Test] Device: {DEVICE}")

    model = get_model("distilbert-base-uncased").to(DEVICE)

    # Tạo batch giả: batch_size=4, seq_len=128
    batch_size, seq_len = 4, 128
    dummy_ids  = torch.randint(0, 30522, (batch_size, seq_len)).to(DEVICE)
    dummy_mask = torch.ones(batch_size, seq_len, dtype=torch.long).to(DEVICE)

    with torch.no_grad():
        logits = model(dummy_ids, dummy_mask)

    print(f"[Test] Output logits shape: {logits.shape}")   # kỳ vọng: [4, 2]
    assert logits.shape == (batch_size, 2), "Shape không đúng!"
    print("✅ Model forward pass thành công!")