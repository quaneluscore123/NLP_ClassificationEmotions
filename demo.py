"""Demo dự đoán cảm xúc cho câu tiếng Việt bằng 2 mô hình đã huấn luyện (chạy run_all.py trước).

    python demo.py                                   # chế độ tương tác
    python demo.py "sản phẩm dùng chán quá, phí tiền"  # dự đoán 1 câu
"""
import json
import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import joblib
import numpy as np

from src.config import MODELS
from src.preprocess import CONFIGS, preprocess

VI = {"Enjoyment": "Vui thích", "Sadness": "Buồn", "Anger": "Giận dữ", "Fear": "Sợ hãi",
      "Disgust": "Ghê tởm/khinh", "Surprise": "Ngạc nhiên", "Other": "Khác"}


class Predictor:
    def __init__(self):
        b = joblib.load(MODELS / "baseline.joblib")
        self.base, self.base_cfg, self.base_name = b["pipe"], b["preprocess"], b["name"]
        import tensorflow as tf
        from src import bilstm  # noqa: F401  (đăng ký lớp MaskedMaxMeanPool trước khi load)
        self.tf = tf
        meta = json.loads((MODELS / "rnn_meta.json").read_text(encoding="utf-8"))
        self.rnn = tf.keras.models.load_model(MODELS / "rnn.keras")
        self.rnn_name, self.labels = meta["variant"], meta["labels"]
        self.rnn_cfg = CONFIGS[meta["preprocess"]]
        self.w = meta.get("ensemble_w_logreg")
        self.vec = tf.keras.layers.TextVectorization(standardize=None, split="whitespace",
                                                     output_sequence_length=meta["max_len"],
                                                     vocabulary=meta["vocab"][2:])  # bỏ "" và [UNK]

    def predict(self, text):
        out = {}
        tb = preprocess(text, self.base_cfg)
        pb = self.base.predict_proba([tb])[0] if hasattr(self.base, "predict_proba") else None
        out[self.base_name] = (self.base.predict([tb])[0], dict(zip(self.base.classes_, pb)) if pb is not None else None, tb)
        tr = preprocess(text, self.rnn_cfg)
        pr = self.rnn.predict(self.vec(self.tf.constant([tr])).numpy(), verbose=0)[0]
        out[self.rnn_name] = (self.labels[int(np.argmax(pr))], dict(zip(self.labels, pr)), tr)
        if self.w is not None and self.w < 1.0 and pb is not None:  # w = 1 -> trùng LogReg
            pb_ordered = np.array([dict(zip(self.base.classes_, pb))[l] for l in self.labels])
            pe = self.w * pb_ordered + (1 - self.w) * pr
            out[f"Ensemble (w={self.w})"] = (self.labels[int(np.argmax(pe))], dict(zip(self.labels, pe)), "")
        return out


def show(p, text):
    print(f"\nCâu: {text}")
    for name, (lab, proba, t) in p.predict(text).items():
        print(f"  [{name}] -> {lab} ({VI[lab]})" + (f"   | đã tiền xử lý: {t}" if t else ""))
        if proba:
            top = sorted(proba.items(), key=lambda kv: -kv[1])[:3]
            print("     top-3: " + ", ".join(f"{k} {v:.2f}" for k, v in top))


if __name__ == "__main__":
    p = Predictor()
    if len(sys.argv) > 1:
        show(p, " ".join(sys.argv[1:]))
    else:
        print("Nhập câu (Enter trống để thoát):")
        while True:
            s = input("> ").strip()
            if not s:
                break
            show(p, s)
