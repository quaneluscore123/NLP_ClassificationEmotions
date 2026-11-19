"""Tải và đọc dữ liệu UIT-VSMEC (giữ nguyên split train/valid/test gốc của tác giả)."""
from __future__ import annotations

import urllib.request

import pandas as pd

from .config import DATA_RAW, DATA_PROCESSED, HF_BASE, LABEL_COL, LABELS, SPLIT_FILES, TEXT_COL


def download(force: bool = False) -> None:
    for split, fname in SPLIT_FILES.items():
        dst = DATA_RAW / fname
        if dst.exists() and not force:
            continue
        print(f"Tải {fname} ...")
        urllib.request.urlretrieve(f"{HF_BASE}/{fname}", dst)


def load_raw() -> dict[str, pd.DataFrame]:
    download()
    out = {}
    for split, fname in SPLIT_FILES.items():
        df = pd.read_json(DATA_RAW / fname)
        df = df[[TEXT_COL, LABEL_COL]].copy()
        df[TEXT_COL] = df[TEXT_COL].astype(str)
        assert set(df[LABEL_COL]) <= set(LABELS), set(df[LABEL_COL]) - set(LABELS)
        out[split] = df.reset_index(drop=True)
    return out


def clean_splits(raw: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], dict]:
    """Làm sạch tối thiểu, có ghi log:
    - bỏ câu rỗng;
    - bỏ bản ghi trùng hoàn toàn (câu + nhãn) TRONG tập train;
    - với câu trùng trong train nhưng khác nhãn: giữ nhãn xuất hiện đầu tiên;
    - KHÔNG sửa valid/test (để so sánh được với kết quả công bố), chỉ báo cáo chồng lấn."""
    log = {}
    d = {k: v.copy() for k, v in raw.items()}
    for k in d:
        n0 = len(d[k])
        d[k] = d[k][d[k][TEXT_COL].str.strip() != ""]
        log[f"{k}_empty_removed"] = n0 - len(d[k])
    tr = d["train"]
    n0 = len(tr)
    tr = tr.drop_duplicates([TEXT_COL, LABEL_COL])
    log["train_exact_dup_removed"] = n0 - len(tr)
    n1 = len(tr)
    tr = tr.drop_duplicates([TEXT_COL], keep="first")
    log["train_conflicting_label_removed"] = n1 - len(tr)
    d["train"] = tr.reset_index(drop=True)
    train_set = set(d["train"][TEXT_COL])
    for k in ("valid", "test"):
        log[f"{k}_overlap_with_train"] = int(d[k][TEXT_COL].isin(train_set).sum())
        d[k] = d[k].reset_index(drop=True)
    return d, log


def save_processed(name: str, splits: dict[str, pd.DataFrame], col: str = "text") -> None:
    safe = name.replace("/", "_").replace("+", "plus").replace("-", "minus")
    for k, df in splits.items():
        df.to_csv(DATA_PROCESSED / f"{safe}_{k}.csv", index=False, encoding="utf-8")
