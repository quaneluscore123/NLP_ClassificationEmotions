"""Khảo sát dữ liệu (EDA): thống kê, độ dài, phân bố nhãn, emoji, từ đặc trưng."""
from __future__ import annotations

import collections
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import FIGURES, LABEL_COL, LABELS, TABLES, TEXT_COL
from .preprocess import EMOJI_RE

EMOTICON_RE = re.compile(r"[:;=][\)\(dDpPoO]|\^\^|-_-|<3")
BROKEN_DIACRITIC_RE = re.compile(r"\s[̀-ͯ]")


def split_stats(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for k, df in splits.items():
        toks = df[TEXT_COL].str.split()
        n_tok = toks.str.len()
        vocab = {w for t in toks for w in t}
        rows.append({
            "split": k, "n_sentences": len(df), "n_tokens(syllable)": int(n_tok.sum()),
            "vocab_size(syllable)": len(vocab), "mean_len": round(n_tok.mean(), 2),
            "median_len": float(n_tok.median()), "p95_len": float(n_tok.quantile(.95)),
            "max_len": int(n_tok.max()),
            "dup_text": int(df[TEXT_COL].duplicated().sum()),
            "has_emoji": int(df[TEXT_COL].map(lambda x: bool(EMOJI_RE.search(x))).sum()),
            "has_emoticon": int(df[TEXT_COL].map(lambda x: bool(EMOTICON_RE.search(x))).sum()),
            "broken_diacritic": int(df[TEXT_COL].map(lambda x: bool(BROKEN_DIACRITIC_RE.search(x))).sum()),
            "has_per_token": int(df[TEXT_COL].map(lambda x: "per" in x.split()).sum()),
        })
    return pd.DataFrame(rows)


def label_distribution(splits) -> pd.DataFrame:
    tab = pd.DataFrame({k: df[LABEL_COL].value_counts() for k, df in splits.items()}).reindex(LABELS)
    for k in splits:
        tab[f"{k}_%"] = (tab[k] / tab[k].sum() * 100).round(2)
    return tab


def run_eda(splits: dict[str, pd.DataFrame], log: dict | None = None) -> dict:
    out = {}
    st = split_stats(splits)
    st.to_csv(TABLES / "eda_split_stats.csv", index=False)
    out["split_stats"] = st
    ld = label_distribution(splits)
    ld.to_csv(TABLES / "eda_label_distribution.csv")
    out["label_distribution"] = ld

    # Hình 1: phân bố nhãn
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(LABELS))
    w = 0.27
    for i, k in enumerate(splits):
        ax.bar(x + (i - 1) * w, ld[f"{k}_%"], w, label=k)
    ax.set_xticks(x, LABELS)
    ax.set_ylabel("% số câu")
    ax.set_title("Phân bố nhãn cảm xúc theo tập (UIT-VSMEC)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "eda_label_distribution.png", dpi=150)
    plt.close(fig)

    # Hình 2: độ dài câu theo nhãn (train)
    tr = splits["train"].copy()
    tr["len"] = tr[TEXT_COL].str.split().str.len()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(tr["len"].clip(upper=80), bins=40, color="#4C72B0")
    axes[0].set_xlabel("Số âm tiết/câu (cắt ở 80)")
    axes[0].set_ylabel("Số câu")
    axes[0].set_title("Phân bố độ dài câu (train)")
    axes[1].boxplot([tr.loc[tr[LABEL_COL] == l, "len"] for l in LABELS], showfliers=False)
    axes[1].set_xticks(range(1, len(LABELS) + 1), LABELS, rotation=30)
    axes[1].set_title("Độ dài câu theo nhãn (train, ẩn outlier)")
    fig.tight_layout()
    fig.savefig(FIGURES / "eda_length.png", dpi=150)
    plt.close(fig)
    len_by_label = tr.groupby(LABEL_COL)["len"].describe().reindex(LABELS).round(2)
    len_by_label.to_csv(TABLES / "eda_length_by_label.csv")
    out["length_by_label"] = len_by_label

    # Emoji phổ biến theo nhãn (train)
    rows = []
    for l in LABELS:
        c = collections.Counter(e for s in tr.loc[tr[LABEL_COL] == l, TEXT_COL] for e in EMOJI_RE.findall(s))
        rows.append({"label": l, "n_sent_with_emoji": int(tr.loc[tr[LABEL_COL] == l, TEXT_COL]
                                                         .map(lambda s: bool(EMOJI_RE.search(s))).sum()),
                     "top_emoji": " ".join(f"{e}({n})" for e, n in c.most_common(6))})
    emo = pd.DataFrame(rows)
    emo.to_csv(TABLES / "eda_emoji_by_label.csv", index=False)
    out["emoji_by_label"] = emo

    # Âm tiết phổ biến nhất (toàn train) và theo nhãn (đã loại dấu câu)
    allc = collections.Counter(w for s in tr[TEXT_COL] for w in s.split() if re.search(r"\w", w))
    top = pd.DataFrame(allc.most_common(30), columns=["token", "count"])
    top.to_csv(TABLES / "eda_top_tokens.csv", index=False)
    out["top_tokens"] = top
    if log:
        pd.Series(log, name="count").to_csv(TABLES / "eda_cleaning_log.csv")
        out["cleaning_log"] = log
    return out
