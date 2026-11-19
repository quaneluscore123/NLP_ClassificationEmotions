"""Chạy TOÀN BỘ pipeline từ dữ liệu thô -> kết quả cuối cùng.

    python run_all.py            # đầy đủ (3 seed cho mô hình RNN, ~20-25 phút trên CPU)
    python run_all.py --fast     # nhanh (1 seed, ít biến thể RNN) để kiểm tra

Mọi bảng/hình được ghi vào results/, mô hình vào models/, dữ liệu trung gian vào data/processed/.
Quy tắc: mọi lựa chọn (tiền xử lý, đặc trưng, siêu tham số, cấu hình RNN) đều dựa trên
macro-F1 của tập VALID. Tập TEST chỉ dùng ở bước 6 để báo cáo."""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import (FIGURES, LABEL_COL, LABELS, MODELS, PREDICTIONS, RESULTS, SEED, TABLES,
                        TEXT_COL)
from src.data import clean_splits, load_raw, save_processed
from src.eda import run_eda
from src.evaluate import metrics, per_class, plot_confusion
from src.preprocess import CONFIGS, preprocess_series
from src import baseline as B


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def env_info():
    import sklearn, tensorflow as tf, pyvi
    import importlib.metadata as md
    info = {"python": sys.version.split()[0], "platform": platform.platform(),
            "numpy": np.__version__, "pandas": pd.__version__, "scikit-learn": sklearn.__version__,
            "tensorflow": tf.__version__, "pyvi": md.version("pyvi"),
            "matplotlib": matplotlib.__version__, "seed": SEED}
    (RESULTS / "environment.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


def paired_bootstrap(y, p_a, p_b, n=1000, seed=SEED):
    """Bootstrap có ghép cặp trên tập test: phân phối của macro-F1(B) - macro-F1(A)."""
    from sklearn.metrics import f1_score
    rng = np.random.default_rng(seed)
    y, p_a, p_b = map(np.asarray, (y, p_a, p_b))
    diffs = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        diffs.append(f1_score(y[idx], p_b[idx], average="macro", labels=LABELS, zero_division=0)
                     - f1_score(y[idx], p_a[idx], average="macro", labels=LABELS, zero_division=0))
    diffs = np.array(diffs)
    return {"mean_diff": float(diffs.mean()), "ci95_low": float(np.percentile(diffs, 2.5)),
            "ci95_high": float(np.percentile(diffs, 97.5)), "p_B_not_better": float((diffs <= 0).mean())}


def bar(df, x, y, title, path, err=None, xlabel="", ylim=None):
    fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(df)), 4))
    ax.bar(df[x], df[y], yerr=df[err] if err else None, capsize=4, color="#4C72B0")
    for i, v in enumerate(df[y]):
        ax.text(i, v + 0.004, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_ylabel(y)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    if ylim:
        ax.set_ylim(*ylim)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(fast=False):
    t_start = time.time()
    info = env_info()
    log(f"Môi trường: {info}")

    # ---------------- 1. Dữ liệu + EDA ----------------
    raw = load_raw()
    splits, clean_log = clean_splits(raw)
    log(f"Làm sạch: {clean_log}")
    run_eda(splits, clean_log)
    y = {k: splits[k][LABEL_COL].tolist() for k in splits}

    # ---------------- 2. Tiền xử lý (mọi cấu hình) ----------------
    X = {}
    examples = []
    sample_idx = [3, 7, 15, 42, 100, 2021]
    for name, cfg in CONFIGS.items():
        X[name] = {k: preprocess_series(splits[k][TEXT_COL], cfg) for k in splits}
        save_processed(name, {k: pd.DataFrame({"text": X[name][k], "label": y[k]}) for k in splits})
        for i in sample_idx:
            examples.append({"config": name, "idx": i, "text": X[name]["train"][i]})
    ex = pd.DataFrame(examples).pivot(index="idx", columns="config", values="text")
    ex.insert(0, "original", [splits["train"][TEXT_COL][i] for i in ex.index])
    ex.to_csv(TABLES / "preprocess_examples.csv", encoding="utf-8")
    pd.DataFrame([c.as_dict() for c in CONFIGS.values()]).to_csv(TABLES / "preprocess_configs.csv", index=False)
    log("Đã tiền xử lý 6 cấu hình.")

    # ---------------- 3. Khảo sát tiền xử lý (baseline, word 1-2 gram) ----------------
    rows, all_grid = [], []
    for name in CONFIGS:
        for model in ("MultinomialNB", "LogReg"):
            g, hp = B.grid_search_val(model, X[name]["train"], y["train"], X[name]["valid"], y["valid"])
            g.insert(0, "preprocess", name)
            all_grid.append(g)
            best = g.iloc[0]
            rows.append({"preprocess": name, "model": model, "best_hp": json.dumps(hp),
                         "val_macro_f1": best.val_macro_f1, "val_accuracy": best.val_accuracy})
    prep_abl = pd.DataFrame(rows)
    prep_abl.to_csv(TABLES / "ablation_preprocessing.csv", index=False)
    pd.concat(all_grid).to_csv(TABLES / "grid_preprocessing_all.csv", index=False)
    piv = prep_abl.pivot(index="preprocess", columns="model", values="val_macro_f1").reindex(list(CONFIGS))
    fig, ax = plt.subplots(figsize=(9, 4))
    piv.plot.bar(ax=ax, rot=20)
    ax.set_ylabel("Macro-F1 (valid)")
    ax.set_title("Khảo sát tiền xử lý (TF-IDF word 1-2gram)")
    ax.set_ylim(piv.min().min() - 0.05, piv.max().max() + 0.03)
    fig.tight_layout()
    fig.savefig(FIGURES / "ablation_preprocessing.png", dpi=150)
    plt.close(fig)
    best_prep = piv.mean(axis=1).idxmax()  # tốt nhất trung bình trên 2 mô hình
    log(f"Khảo sát tiền xử lý xong. Tốt nhất (valid, TB 2 mô hình): {best_prep}\n{piv}")

    # ---------------- 4. Khảo sát đặc trưng + so sánh mô hình baseline ----------------
    rows, all_grid = [], []
    Xp = X[best_prep]
    for feat in B.FEATURES:
        for model in ("MultinomialNB", "LogReg", "LinearSVC"):
            g, hp = B.grid_search_val(model, Xp["train"], y["train"], Xp["valid"], y["valid"], feature=feat)
            all_grid.append(g)
            rows.append({"feature": feat, "model": model, "best_hp": json.dumps(hp),
                         "val_macro_f1": g.iloc[0].val_macro_f1, "val_accuracy": g.iloc[0].val_accuracy})
    feat_abl = pd.DataFrame(rows)
    feat_abl.to_csv(TABLES / "ablation_features.csv", index=False)
    pd.concat(all_grid).to_csv(TABLES / "grid_features_all.csv", index=False)
    piv = feat_abl.pivot(index="feature", columns="model", values="val_macro_f1").reindex(list(B.FEATURES))
    fig, ax = plt.subplots(figsize=(9, 4))
    piv.plot.bar(ax=ax, rot=15)
    ax.set_ylabel("Macro-F1 (valid)")
    ax.set_title(f"Khảo sát đặc trưng n-gram ({best_prep})")
    ax.set_ylim(piv.min().min() - 0.05, piv.max().max() + 0.03)
    fig.tight_layout()
    fig.savefig(FIGURES / "ablation_features.png", dpi=150)
    plt.close(fig)
    log(f"Khảo sát đặc trưng xong:\n{piv}")

    # chọn cấu hình tốt nhất (theo valid) cho MỖI loại mô hình baseline
    baseline_sel = {}
    for model in ("MultinomialNB", "LogReg", "LinearSVC"):
        r = feat_abl[feat_abl.model == model].sort_values("val_macro_f1", ascending=False).iloc[0]
        baseline_sel[model] = {"feature": r.feature, "hp": json.loads(r.best_hp), "val_macro_f1": r.val_macro_f1}

    # ---------------- 5. Mô hình RNN (TensorFlow) ----------------
    from src.bilstm import RNNConfig, train_rnn, predict_proba
    import tensorflow as tf
    seeds = [SEED] if fast else [SEED, SEED + 1, SEED + 2]
    # Mỗi biến thể chỉ thay ĐÚNG 1 yếu tố so với R3 (cấu hình chính) -> khảo sát có kiểm soát
    rnn_variants = [
        ("R1_LSTM_last", "P2_+teencode", RNNConfig(name="LSTM", bidirectional=False)),
        ("R2_BiLSTM_last", "P2_+teencode", RNNConfig(name="BiLSTM")),
        ("R3_BiLSTM_maxmean", "P2_+teencode", RNNConfig(name="BiLSTM", pooling="maxmean")),
        ("R4_BiLSTM_maxmean_segment", "P3_+segment", RNNConfig(name="BiLSTM", pooling="maxmean")),
        ("R5_BiLSTM_maxmean_noCW", "P2_+teencode", RNNConfig(name="BiLSTM", pooling="maxmean", class_weight=False)),
        ("R6_BiLSTM_maxmean_emb256", "P2_+teencode", RNNConfig(name="BiLSTM", pooling="maxmean", emb_dim=256)),
    ]
    if fast:
        rnn_variants = [v for v in rnn_variants if v[0] in ("R1_LSTM_last", "R3_BiLSTM_maxmean")]
    rnn_cfgs = {v: c for v, _, c in rnn_variants}
    rnn_rows, rnn_runs, histories = [], {}, []
    for vname, prep, cfg in rnn_variants:
        for s in seeds:
            run = train_rnn(X[prep]["train"], y["train"], X[prep]["valid"], y["valid"], cfg, s)
            # Dự đoán test được LƯU LẠI nhưng KHÔNG dùng để chọn cấu hình.
            run["test_proba"] = predict_proba(run, X[prep]["test"])
            run["val_proba"] = predict_proba(run, X[prep]["valid"])
            run["prep"] = prep
            rnn_runs[(vname, s)] = run
            for h in run["history"]:
                histories.append({"variant": vname, "seed": s, **h})
            rnn_rows.append({"variant": vname, "preprocess": prep, "seed": s, **cfg.as_dict(),
                             "val_macro_f1": run["val_macro_f1"], "best_epoch": run["best_epoch"],
                             "train_sec": run["train_sec"], "n_params": run["n_params"],
                             "vocab_size": run["vocab_size"]})
            log(f"{vname} seed={s}: val macro-F1={run['val_macro_f1']:.4f} "
                f"(epoch {run['best_epoch']}, {run['train_sec']}s)")
    rnn_df = pd.DataFrame(rnn_rows)
    rnn_df.to_csv(TABLES / "rnn_runs.csv", index=False)
    pd.DataFrame(histories).to_csv(TABLES / "rnn_histories.csv", index=False)
    rnn_sum = (rnn_df.groupby("variant", sort=False)
               .agg(preprocess=("preprocess", "first"), bidirectional=("bidirectional", "first"),
                    emb_dim=("emb_dim", "first"), class_weight=("class_weight", "first"),
                    val_macro_f1_mean=("val_macro_f1", "mean"), val_macro_f1_std=("val_macro_f1", "std"),
                    best_epoch_mean=("best_epoch", "mean"), n_params=("n_params", "first"),
                    train_sec_mean=("train_sec", "mean")).reset_index())
    rnn_sum["val_macro_f1_std"] = rnn_sum["val_macro_f1_std"].fillna(0)
    rnn_sum.to_csv(TABLES / "ablation_rnn.csv", index=False)
    bar(rnn_sum, "variant", "val_macro_f1_mean", "Khảo sát RNN (macro-F1 valid, TB ± std theo seed)",
        FIGURES / "ablation_rnn.png", err="val_macro_f1_std",
        ylim=(rnn_sum.val_macro_f1_mean.min() - 0.08, rnn_sum.val_macro_f1_mean.max() + 0.05))
    best_rnn = rnn_sum.sort_values("val_macro_f1_mean", ascending=False).iloc[0].variant
    log(f"RNN tốt nhất theo valid: {best_rnn}\n{rnn_sum}")

    # đường học của biến thể tốt nhất
    hd = pd.DataFrame(histories)
    hd = hd[hd.variant == best_rnn]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for s, g in hd.groupby("seed"):
        axes[0].plot(g.epoch, g.loss, "-", label=f"train loss s{s}")
        axes[0].plot(g.epoch, g.val_loss, "--", label=f"val loss s{s}")
        axes[1].plot(g.epoch, g.val_macro_f1, "-o", ms=3, label=f"seed {s}")
    axes[0].set_xlabel("epoch"); axes[0].set_title(f"Loss — {best_rnn}"); axes[0].legend(fontsize=7)
    axes[1].set_xlabel("epoch"); axes[1].set_title("Macro-F1 trên valid"); axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / "rnn_learning_curve.png", dpi=150)
    plt.close(fig)

    # ---------------- 6. Đánh giá cuối trên TEST (chỉ các cấu hình đã chọn) ----------------
    final_rows, preds = [], {}
    maj = pd.Series(y["train"]).mode()[0]
    preds["Majority"] = [maj] * len(y["test"])
    final_rows.append({"model": "Majority class (tham chiếu)", "preprocess": "-", "feature": "-",
                       "hp": "-", "val_macro_f1": metrics(y["valid"], [maj] * len(y["valid"]))["macro_f1"],
                       **{f"test_{k}": v for k, v in metrics(y["test"], preds["Majority"]).items()}})
    fitted = {}
    for model, sel in baseline_sel.items():
        pipe = B.build_pipeline(model, sel["feature"], **sel["hp"]).fit(Xp["train"], y["train"])
        fitted[model] = pipe
        key = f"TF-IDF+{model}"
        preds[key] = pipe.predict(Xp["test"]).tolist()
        final_rows.append({"model": key, "preprocess": best_prep, "feature": sel["feature"],
                           "hp": json.dumps(sel["hp"]), "val_macro_f1": sel["val_macro_f1"],
                           **{f"test_{k}": v for k, v in metrics(y["test"], preds[key]).items()}})
    # RNN: báo cáo TB ± std trên các seed cho biến thể LSTM (R1) và biến thể tốt nhất
    rnn_report = ["R1_LSTM_last", best_rnn] if best_rnn != "R1_LSTM_last" else ["R1_LSTM_last"]
    for vname in rnn_report:
        ms = []
        for s in seeds:
            run = rnn_runs[(vname, s)]
            p = [LABELS[i] for i in run["test_proba"].argmax(1)]
            ms.append(metrics(y["test"], p))
            if s == SEED:
                preds[vname] = p
        mdf = pd.DataFrame(ms)
        row = {"model": vname,
               "preprocess": rnn_runs[(vname, SEED)]["prep"], "feature": "embedding học từ đầu",
               "hp": json.dumps({k: v for k, v in rnn_cfgs[vname].as_dict().items()
                                 if k in ("bidirectional", "emb_dim", "units", "class_weight")}),
               "val_macro_f1": rnn_sum.set_index("variant").loc[vname, "val_macro_f1_mean"]}
        for k in mdf.columns:
            row[f"test_{k}"] = mdf[k].mean()
            row[f"test_{k}_std"] = mdf[k].std() if len(seeds) > 1 else 0.0
        final_rows.append(row)
    # ---- Ensemble mềm: P = w * P_LogReg + (1 - w) * P_RNN ; w chọn trên VALID ----
    lr_pipe = fitted["LogReg"]
    order = [list(lr_pipe.classes_).index(l) for l in LABELS]   # sắp cột xác suất theo LABELS
    P_lr_va = lr_pipe.predict_proba(Xp["valid"])[:, order]
    P_lr_te = lr_pipe.predict_proba(Xp["test"])[:, order]
    from sklearn.metrics import f1_score
    ws = np.round(np.arange(0, 1.01, 0.1), 2)
    w_rows = []
    for w in ws:
        f1s = [f1_score(y["valid"], [LABELS[i] for i in (w * P_lr_va + (1 - w) * rnn_runs[(best_rnn, s)]["val_proba"]).argmax(1)],
                        average="macro") for s in seeds]
        w_rows.append({"w_logreg": w, "val_macro_f1_mean": np.mean(f1s), "val_macro_f1_std": np.std(f1s, ddof=1) if len(f1s) > 1 else 0})
    wdf = pd.DataFrame(w_rows)
    wdf.to_csv(TABLES / "ensemble_weight_valid.csv", index=False)
    best_w = float(wdf.sort_values("val_macro_f1_mean", ascending=False).iloc[0].w_logreg)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.errorbar(wdf.w_logreg, wdf.val_macro_f1_mean, yerr=wdf.val_macro_f1_std, marker="o", capsize=3)
    ax.axvline(best_w, ls="--", color="gray")
    ax.set_xlabel("w (trọng số TF-IDF+LogReg); 0 = chỉ BiLSTM, 1 = chỉ LogReg")
    ax.set_ylabel("Macro-F1 (valid)"); ax.set_title("Chọn trọng số ensemble trên valid")
    fig.tight_layout(); fig.savefig(FIGURES / "ensemble_weight.png", dpi=150); plt.close(fig)
    ens_name = "Ensemble LogReg+" + best_rnn.split("_", 1)[1]
    ens_proba42 = None
    ms = []
    for s in (seeds if best_w < 1.0 else []):  # w=1 => ensemble trùng LogReg, không báo cáo riêng
        P = best_w * P_lr_te + (1 - best_w) * rnn_runs[(best_rnn, s)]["test_proba"]
        p = [LABELS[i] for i in P.argmax(1)]
        ms.append(metrics(y["test"], p))
        if s == SEED:
            preds[ens_name] = p
            ens_proba42 = P
    mdf = pd.DataFrame(ms)
    row = None if best_w >= 1.0 else {"model": ens_name, "preprocess": f"{best_prep} / {rnn_runs[(best_rnn, SEED)]['prep']}",
           "feature": "TF-IDF + embedding", "hp": json.dumps({"w_logreg": best_w}),
           "val_macro_f1": float(wdf.val_macro_f1_mean.max())}
    if row is not None:
        for k in mdf.columns:
            row[f"test_{k}"] = mdf[k].mean()
            row[f"test_{k}_std"] = mdf[k].std() if len(seeds) > 1 else 0.0
        final_rows.append(row)
    log(f"Ensemble: w_logreg={best_w}")

    final = pd.DataFrame(final_rows)
    final.to_csv(TABLES / "final_comparison_test.csv", index=False)
    log(f"Bảng so sánh trên TEST:\n{final[['model', 'val_macro_f1', 'test_macro_f1', 'test_accuracy', 'test_weighted_f1']]}")

    fig, ax = plt.subplots(figsize=(9, 4))
    fr = final.iloc[1:]
    ax.bar(fr.model, fr.test_macro_f1, yerr=fr.get("test_macro_f1_std"), capsize=4, color="#55A868", label="Test")
    ax.scatter(fr.model, fr.val_macro_f1, color="black", zorder=3, label="Valid")
    for i, v in enumerate(fr.test_macro_f1):
        ax.text(i, v + 0.012, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_ylabel("Macro-F1"); ax.set_title("So sánh mô hình (cấu hình chọn theo valid)")
    ax.set_ylim(0.3, max(fr.test_macro_f1.max(), fr.val_macro_f1.max()) + 0.06); ax.legend()
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    fig.tight_layout(); fig.savefig(FIGURES / "final_comparison.png", dpi=150); plt.close(fig)

    # per-class + confusion matrix cho baseline tốt nhất và RNN tốt nhất
    best_base = max(baseline_sel, key=lambda m: baseline_sel[m]["val_macro_f1"])
    best_base_key = f"TF-IDF+{best_base}"
    pcs = []
    for key in ["TF-IDF+MultinomialNB", best_base_key, best_rnn] + ([ens_name] if ens_name in preds else []):
        pc = per_class(y["test"], preds[key]); pc.insert(0, "model", key); pcs.append(pc)
        plot_confusion(y["test"], preds[key], f"Confusion matrix (test) — {key}",
                       FIGURES / f"confusion_{key.replace('+', '_')}.png")
    pcs = pd.concat(pcs).reset_index(names="label").drop_duplicates(["model", "label"])
    pcs.to_csv(TABLES / "per_class_test.csv", index=False)
    pv = pcs.pivot(index="label", columns="model", values="f1-score").reindex(LABELS)
    fig, ax = plt.subplots(figsize=(9, 4))
    pv.plot.bar(ax=ax, rot=0); ax.set_ylabel("F1 (test)"); ax.set_title("F1 theo từng lớp cảm xúc")
    fig.tight_layout(); fig.savefig(FIGURES / "per_class_f1.png", dpi=150); plt.close(fig)

    # kiểm định bootstrap: RNN tốt nhất (seed 42) vs baseline tốt nhất
    boot = paired_bootstrap(y["test"], preds[best_base_key], preds[best_rnn])
    boot.update({"A": best_base_key, "B": f"{best_rnn} (seed {SEED})"})
    boot_nb = paired_bootstrap(y["test"], preds["TF-IDF+MultinomialNB"], preds[best_base_key])
    boot_nb.update({"A": "TF-IDF+MultinomialNB", "B": best_base_key})
    boots = [boot_nb, boot]
    if ens_name in preds:
        boot_ens = paired_bootstrap(y["test"], preds[best_base_key], preds[ens_name])
        boot_ens.update({"A": best_base_key, "B": f"{ens_name} (seed {SEED})"})
        boots.append(boot_ens)
    pd.DataFrame(boots).to_csv(TABLES / "bootstrap_test.csv", index=False)
    log(f"Bootstrap: {boots}")

    # ---------------- 7. Phân tích lỗi ----------------
    pred_df = pd.DataFrame({"text": splits["test"][TEXT_COL], "text_processed": Xp["test"], "true": y["test"]})
    for k, v in preds.items():
        if k != "Majority":
            pred_df[f"pred_{k}"] = v
    run42 = rnn_runs[(best_rnn, SEED)]
    pred_df[f"conf_{best_rnn}"] = run42["test_proba"].max(1).round(3)
    if ens_proba42 is not None:
        pred_df[f"conf_{ens_name}"] = ens_proba42.max(1).round(3)
    pred_df["len"] = pred_df.text.str.split().str.len()
    pred_df.to_csv(PREDICTIONS / "test_predictions.csv", index=False, encoding="utf-8")

    b, r = f"pred_{best_base_key}", f"pred_{best_rnn}"
    pred_df["base_ok"] = pred_df[b] == pred_df.true
    pred_df["rnn_ok"] = pred_df[r] == pred_df.true
    agree = pd.crosstab(pred_df.base_ok, pred_df.rnn_ok, rownames=[f"{best_base_key} đúng"],
                        colnames=[f"{best_rnn} đúng"])
    agree.to_csv(TABLES / "error_agreement.csv")
    # cặp nhầm lẫn phổ biến nhất
    for key, col in ((best_base_key, b), (best_rnn, r)):
        wrong = pred_df[pred_df[col] != pred_df.true]
        pairs = (wrong.groupby(["true", col]).size().sort_values(ascending=False).head(10)
                 .reset_index(name="count").rename(columns={col: "pred"}))
        pairs.insert(0, "model", key)
        pairs.to_csv(TABLES / f"error_top_confusions_{key.replace('+', '_')}.csv", index=False)
    # lỗi theo độ dài câu
    pred_df["len_bin"] = pd.cut(pred_df.len, [0, 5, 10, 20, 40, 1000], labels=["1-5", "6-10", "11-20", "21-40", ">40"])
    from sklearn.metrics import f1_score
    lrows = []
    for bn, g in pred_df.groupby("len_bin", observed=True):
        lrows.append({"len_bin": bn, "n": len(g),
                      f"acc_{best_base_key}": g.base_ok.mean(), f"acc_{best_rnn}": g.rnn_ok.mean()})
    pd.DataFrame(lrows).to_csv(TABLES / "error_by_length.csv", index=False)
    # mẫu lỗi: cả 2 sai (ưu tiên RNN tự tin cao) + chỉ 1 mô hình sai
    both = pred_df[~pred_df.base_ok & ~pred_df.rnn_ok].sort_values(f"conf_{best_rnn}", ascending=False)
    only_rnn = pred_df[pred_df.base_ok & ~pred_df.rnn_ok]
    only_base = pred_df[~pred_df.base_ok & pred_df.rnn_ok]
    cols = ["text", "true", b, r, f"conf_{best_rnn}"]
    err = pd.concat([both.head(8)[cols].assign(case="cả hai sai"),
                     only_base.head(4)[cols].assign(case=f"chỉ {best_base_key} sai"),
                     only_rnn.head(4)[cols].assign(case=f"chỉ {best_rnn} sai")])
    err.to_csv(TABLES / "error_examples.csv", encoding="utf-8")

    # đặc trưng nổi bật của LogReg (diễn giải)
    if "LogReg" in fitted and baseline_sel["LogReg"]["feature"].startswith("word"):
        B.top_features(fitted["LogReg"]).to_csv(TABLES / "logreg_top_features.csv", index=False)
    else:
        lr_word = B.build_pipeline("LogReg", "word_1-2", **baseline_sel["LogReg"]["hp"]).fit(Xp["train"], y["train"])
        B.top_features(lr_word).to_csv(TABLES / "logreg_top_features.csv", index=False)

    # ---------------- 8. Lưu mô hình cho demo ----------------
    joblib.dump({"pipe": fitted["LogReg"], "preprocess": CONFIGS[best_prep], "name": "TF-IDF+LogReg"},
                MODELS / "baseline.joblib")
    run42["model"].save(MODELS / "rnn.keras")
    (MODELS / "rnn_meta.json").write_text(json.dumps({
        "variant": best_rnn, "preprocess": run42["prep"],
        "max_len": int(rnn_cfgs[best_rnn].max_len), "ensemble_w_logreg": best_w,
        "vocab": run42["vectorizer"].get_vocabulary(), "labels": LABELS}, ensure_ascii=False), encoding="utf-8")

    summary = {"cleaning": clean_log, "best_preprocess": best_prep, "baseline_selection": baseline_sel,
               "best_rnn": best_rnn, "ensemble": {"name": ens_name, "w_logreg": best_w},
               "seeds": seeds, "bootstrap": boots,
               "runtime_min": round((time.time() - t_start) / 60, 1), "env": info}
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str),
                                          encoding="utf-8")
    log(f"HOÀN TẤT trong {summary['runtime_min']} phút. Xem results/.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    main(**vars(ap.parse_args()))
