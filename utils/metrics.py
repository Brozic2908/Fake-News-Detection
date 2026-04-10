"""
utils/metrics.py
Thành viên 3 - MLOps & Evaluation

Hàm compute_metrics() dùng trong train.py và evaluate.py.
Hàm plot_confusion_matrix() dùng riêng trong evaluate.py.
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")           # non-interactive backend (server / headless)
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)


LABEL_NAMES = ["Fake (0)", "Real (1)"]


def compute_metrics(
    y_true: list[int],
    y_pred: list[int],
) -> dict[str, float]:
    """
    Tính accuracy, F1 (macro), precision, recall.
    Trả về dict — tiện log lên WandB.
    """
    return {
        "accuracy"  : accuracy_score(y_true, y_pred),
        "f1"        : f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision" : precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall"    : recall_score(y_true, y_pred, average="macro", zero_division=0),
    }


def plot_confusion_matrix(
    y_true     : list[int],
    y_pred     : list[int],
    save_path  : str  = "confusion_matrix.png",
    normalize  : bool = True,
) -> str:
    """
    Vẽ và lưu confusion matrix.

    Parameters
    ----------
    normalize : True → hiển thị tỷ lệ phần trăm, False → số tuyệt đối

    Returns
    -------
    Đường dẫn file ảnh đã lưu.
    """
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fmt = ".2%"
    else:
        fmt = "d"

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.get_cmap("Blues"))
    plt.colorbar(im, ax=ax)

    ax.set(
        xticks     = np.arange(len(LABEL_NAMES)),
        yticks     = np.arange(len(LABEL_NAMES)),
        xticklabels= LABEL_NAMES,
        yticklabels= LABEL_NAMES,
        xlabel     = "Predicted label",
        ylabel     = "True label",
        title      = "Confusion Matrix" + (" (normalized)" if normalize else ""),
    )
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = f"{cm[i, j]:{fmt}}"
            ax.text(j, i, val, ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[Metrics] Confusion matrix saved → {save_path}")
    return save_path


def print_classification_report(
    y_true: list[int],
    y_pred: list[int],
) -> None:
    print("\n" + classification_report(y_true, y_pred, target_names=LABEL_NAMES))