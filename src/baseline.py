"""Baseline: TF-IDF + mô hình tuyến tính (Naive Bayes / Logistic Regression / Linear SVM).

Vectorizer nằm TRONG sklearn Pipeline nên chỉ được fit trên tập train -> không rò rỉ
thông tin IDF/vocab từ valid/test. Siêu tham số được chọn theo macro-F1 trên tập valid."""
from __future__ import annotations

import itertools
import time

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from .config import SEED
from .evaluate import metrics

WS_TOKEN = r"(?u)\S+"  # văn bản đã được tiền xử lý & tách token bằng khoảng trắng (giữ emoji, emo_*)

FEATURES = {
    "word_1-1": dict(kind="word", ngram=(1, 1)),
    "word_1-2": dict(kind="word", ngram=(1, 2)),
    "word_1-3": dict(kind="word", ngram=(1, 3)),
    "char_2-5": dict(kind="char", ngram=(2, 5)),
    "word_1-2+char_2-5": dict(kind="union"),
}


def make_vectorizer(kind="word", ngram=(1, 2), min_df=1, sublinear_tf=True):
    if kind == "word":
        return TfidfVectorizer(token_pattern=WS_TOKEN, lowercase=False, ngram_range=ngram,
                               min_df=min_df, sublinear_tf=sublinear_tf)
    if kind == "char":
        return TfidfVectorizer(analyzer="char_wb", lowercase=False, ngram_range=ngram,
                               min_df=2, sublinear_tf=sublinear_tf, max_features=200_000)
    if kind == "union":
        return FeatureUnion([("w", make_vectorizer("word", (1, 2), min_df, sublinear_tf)),
                             ("c", make_vectorizer("char", (2, 5), min_df, sublinear_tf))])
    raise ValueError(kind)


def make_model(name: str, **hp):
    if name == "MultinomialNB":
        return MultinomialNB(alpha=hp.get("alpha", 1.0))
    if name == "ComplementNB":
        return ComplementNB(alpha=hp.get("alpha", 1.0))
    if name == "LogReg":
        return LogisticRegression(C=hp.get("C", 1.0), class_weight=hp.get("class_weight"),
                                  max_iter=3000, random_state=SEED)
    if name == "LinearSVC":
        return LinearSVC(C=hp.get("C", 1.0), class_weight=hp.get("class_weight"), random_state=SEED)
    raise ValueError(name)


GRIDS = {
    "MultinomialNB": {"alpha": [0.01, 0.05, 0.1, 0.3, 0.5, 1.0]},
    "ComplementNB": {"alpha": [0.05, 0.1, 0.3, 0.5, 1.0]},
    "LogReg": {"C": [0.5, 1, 2, 5, 10, 20], "class_weight": [None, "balanced"]},
    "LinearSVC": {"C": [0.05, 0.1, 0.3, 0.5, 1.0], "class_weight": [None, "balanced"]},
}


def build_pipeline(model_name, feature="word_1-2", **hp):
    f = FEATURES[feature]
    vec = make_vectorizer(f["kind"], f.get("ngram", (1, 2)))
    return Pipeline([("tfidf", vec), ("clf", make_model(model_name, **hp))])


def grid_search_val(model_name, X_tr, y_tr, X_va, y_va, feature="word_1-2", grid=None):
    """Thử toàn bộ lưới siêu tham số, fit trên train, chấm điểm trên valid.
    Trả về (bảng kết quả, siêu tham số tốt nhất theo macro-F1 valid)."""
    grid = grid or GRIDS[model_name]
    keys = list(grid)
    rows = []
    for vals in itertools.product(*grid.values()):
        hp = dict(zip(keys, vals))
        t0 = time.time()
        pipe = build_pipeline(model_name, feature, **hp).fit(X_tr, y_tr)
        m = metrics(y_va, pipe.predict(X_va))
        rows.append({"model": model_name, "feature": feature, **{k: str(v) for k, v in hp.items()},
                     **{f"val_{k}": v for k, v in m.items()}, "fit_sec": round(time.time() - t0, 2),
                     "_hp": hp})
    df = pd.DataFrame(rows).sort_values("val_macro_f1", ascending=False).reset_index(drop=True)
    best_hp = df.loc[0, "_hp"]
    return df.drop(columns="_hp"), best_hp


def top_features(pipe: Pipeline, k=12) -> pd.DataFrame:
    """Đặc trưng có trọng số lớn nhất mỗi lớp (chỉ cho mô hình tuyến tính trên word TF-IDF)."""
    vec, clf = pipe.named_steps["tfidf"], pipe.named_steps["clf"]
    names = vec.get_feature_names_out()
    coef = clf.coef_ if hasattr(clf, "coef_") else clf.feature_log_prob_
    rows = []
    for ci, label in enumerate(clf.classes_):
        idx = coef[ci].argsort()[::-1][:k]
        rows.append({"label": label, "top_features": ", ".join(names[i] for i in idx)})
    return pd.DataFrame(rows)
