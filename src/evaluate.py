"""Thước đo và hình vẽ đánh giá dùng chung cho mọi mô hình."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)

from .config import LABELS


def metrics(y_true, y_pred) -> dict:
    return {
        "macro_f1": f1_score(y_true, y_pred, average="macro", labels=LABELS, zero_division=0),
        "macro_precision": precision_score(y_true, y_pred, average="macro", labels=LABELS, zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", labels=LABELS, zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", labels=LABELS, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
    }


def per_class(y_true, y_pred) -> pd.DataFrame:
    rep = classification_report(y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0)
    return pd.DataFrame({l: rep[l] for l in LABELS}).T[["precision", "recall", "f1-score", "support"]]


def plot_confusion(y_true, y_pred, title, path, normalize=True):
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    show = cm / cm.sum(axis=1, keepdims=True) if normalize else cm
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(show, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    ax.set_xticks(range(len(LABELS)), LABELS, rotation=45, ha="right")
    ax.set_yticks(range(len(LABELS)), LABELS)
    ax.set_xlabel("Nhãn dự đoán")
    ax.set_ylabel("Nhãn thật")
    ax.set_title(title)
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            v = show[i, j]
            ax.text(j, i, f"{v:.2f}\n({cm[i, j]})" if normalize else str(cm[i, j]),
                    ha="center", va="center", fontsize=7,
                    color="white" if v > (0.5 if normalize else show.max() / 2) else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return cm
