"""Mô hình cải tiến: (Bi)LSTM với TensorFlow/Keras.

- TextVectorization chỉ adapt trên tập train (vocab không lấy từ valid/test).
- Early stopping theo macro-F1 trên tập valid; khôi phục trọng số tốt nhất.
- Cố định seed (Python/NumPy/TF) + bật op determinism để chạy lại cho cùng kết quả trên CPU."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, asdict

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight

from .config import LABELS, LABEL2ID


@dataclass
class RNNConfig:
    name: str = "BiLSTM"
    bidirectional: bool = True
    emb_dim: int = 128
    units: int = 64
    max_len: int = 60
    max_tokens: int = 20000
    dropout: float = 0.5
    spatial_dropout: float = 0.2
    class_weight: bool = True
    lr: float = 1e-3
    batch_size: int = 32
    epochs: int = 25
    patience: int = 4
    pooling: str = "last"          # last (trạng thái cuối) | maxmean (max+mean pooling có mask)
    l2: float = 0.0

    def as_dict(self):
        return asdict(self)


def set_seed(seed: int):
    tf.keras.utils.set_random_seed(seed)  # python, numpy, tf
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


class MacroF1EarlyStopping(tf.keras.callbacks.Callback):
    def __init__(self, X_val, y_val, patience=4):
        super().__init__()
        self.X_val, self.y_val, self.patience = X_val, y_val, patience
        self.best, self.best_weights, self.wait, self.best_epoch = -1.0, None, 0, 0
        self.history = []

    def on_epoch_end(self, epoch, logs=None):
        pred = self.model.predict(self.X_val, verbose=0, batch_size=256).argmax(1)
        f1 = f1_score(self.y_val, pred, average="macro")
        logs = logs or {}
        logs["val_macro_f1"] = f1
        self.history.append({"epoch": epoch + 1, "loss": logs.get("loss"),
                             "val_loss": logs.get("val_loss"), "val_macro_f1": f1})
        if f1 > self.best:
            self.best, self.best_epoch, self.wait = f1, epoch + 1, 0
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)


def build_vectorizer(train_texts, cfg: RNNConfig):
    vec = tf.keras.layers.TextVectorization(max_tokens=cfg.max_tokens, standardize=None,
                                            split="whitespace", output_mode="int",
                                            output_sequence_length=cfg.max_len)
    vec.adapt(tf.constant(list(train_texts)))
    return vec


@tf.keras.utils.register_keras_serializable(package="vsmec")
class MaskedMaxMeanPool(tf.keras.layers.Layer):
    """Ghép max-pooling và mean-pooling theo thời gian, bỏ qua vị trí padding (id = 0)."""

    def call(self, seq, ids):
        m = tf.cast(tf.not_equal(ids, 0), seq.dtype)[..., None]
        mx = tf.reduce_max(seq + (1.0 - m) * -1e9, axis=1)
        mean = tf.reduce_sum(seq * m, axis=1) / tf.maximum(tf.reduce_sum(m, axis=1), 1.0)
        return tf.concat([mx, mean], axis=-1)

    def compute_mask(self, inputs, mask=None):
        return None


def build_model(vocab_size: int, cfg: RNNConfig) -> tf.keras.Model:
    inp = tf.keras.Input(shape=(cfg.max_len,), dtype="int64")
    reg = tf.keras.regularizers.l2(cfg.l2) if cfg.l2 else None
    x = tf.keras.layers.Embedding(vocab_size, cfg.emb_dim, mask_zero=True,
                                  embeddings_regularizer=reg)(inp)
    x = tf.keras.layers.SpatialDropout1D(cfg.spatial_dropout)(x)
    seq = cfg.pooling != "last"
    lstm = tf.keras.layers.LSTM(cfg.units, return_sequences=seq)
    x = tf.keras.layers.Bidirectional(lstm)(x) if cfg.bidirectional else lstm(x)
    if seq:
        x = MaskedMaxMeanPool()(x, inp)
    x = tf.keras.layers.Dropout(cfg.dropout)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(cfg.dropout)(x)
    out = tf.keras.layers.Dense(len(LABELS), activation="softmax")(x)
    model = tf.keras.Model(inp, out, name=cfg.name)
    model.compile(optimizer=tf.keras.optimizers.Adam(cfg.lr),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def encode_labels(y):
    return np.array([LABEL2ID[l] for l in y], dtype="int64")


def train_rnn(X_tr, y_tr, X_va, y_va, cfg: RNNConfig, seed: int):
    """Huấn luyện 1 lần với 1 seed. Trả về dict gồm model, vectorizer, lịch sử, thời gian."""
    set_seed(seed)
    vec = build_vectorizer(X_tr, cfg)
    A_tr = vec(tf.constant(list(X_tr))).numpy()
    A_va = vec(tf.constant(list(X_va))).numpy()
    t_tr, t_va = encode_labels(y_tr), encode_labels(y_va)
    model = build_model(vec.vocabulary_size(), cfg)
    cw = None
    if cfg.class_weight:
        w = compute_class_weight("balanced", classes=np.arange(len(LABELS)), y=t_tr)
        cw = {i: float(v) for i, v in enumerate(w)}
    es = MacroF1EarlyStopping(A_va, t_va, cfg.patience)
    t0 = time.time()
    model.fit(A_tr, t_tr, validation_data=(A_va, t_va), epochs=cfg.epochs,
              batch_size=cfg.batch_size, class_weight=cw, callbacks=[es], verbose=0, shuffle=True)
    return {"model": model, "vectorizer": vec, "history": es.history, "best_epoch": es.best_epoch,
            "val_macro_f1": es.best, "train_sec": round(time.time() - t0, 1),
            "n_params": model.count_params(), "vocab_size": vec.vocabulary_size()}


def predict_proba(run, texts):
    A = run["vectorizer"](tf.constant(list(texts))).numpy()
    return run["model"].predict(A, verbose=0, batch_size=256)


def predict_labels(run, texts):
    return [LABELS[i] for i in predict_proba(run, texts).argmax(1)]
