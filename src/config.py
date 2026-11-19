"""Cấu hình chung: đường dẫn, seed, nhãn."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"
PREDICTIONS = RESULTS / "predictions"
MODELS = ROOT / "models"

for _p in (DATA_RAW, DATA_PROCESSED, TABLES, FIGURES, PREDICTIONS, MODELS):
    _p.mkdir(parents=True, exist_ok=True)

SEED = 42

# Nguồn dữ liệu (bản phân phối công khai của UIT-VSMEC trên Hugging Face, giữ nguyên split gốc)
HF_BASE = "https://huggingface.co/datasets/tridm/UIT-VSMEC/resolve/main"
SPLIT_FILES = {"train": "train.json", "valid": "valid.json", "test": "test.json"}

TEXT_COL = "Sentence"
LABEL_COL = "Emotion"
LABELS = ["Enjoyment", "Sadness", "Anger", "Fear", "Disgust", "Surprise", "Other"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
