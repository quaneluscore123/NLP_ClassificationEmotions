"""Sinh notebook NLP_BTL_UIT_VSMEC.ipynb (chạy: python notebooks/build_notebook.py)."""
from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []


def md(s):
    C.append(nbf.v4.new_markdown_cell(s.strip()))


def code(s):
    C.append(nbf.v4.new_code_cell(s.strip()))


md("""
# Phân loại cảm xúc văn bản tiếng Việt — UIT-VSMEC
**Bài tập lớn Xử lý ngôn ngữ tự nhiên**

Pipeline: *dữ liệu thô → khảo sát/tiền xử lý → biểu diễn (TF-IDF / embedding) → mô hình (baseline tuyến tính vs. BiLSTM) → đánh giá (macro-F1) → phân tích lỗi.*

- **Đầu vào:** 1 câu bình luận tiếng Việt (mạng xã hội).
- **Đầu ra:** 1 trong 7 nhãn: Enjoyment, Sadness, Anger, Fear, Disgust, Surprise, Other.
- **Thước đo chính:** macro-F1 (dữ liệu mất cân bằng), kèm accuracy, weighted-F1, precision/recall từng lớp.
- **Quy tắc:** mọi lựa chọn dựa trên tập **valid**; tập **test** chỉ dùng để báo cáo cuối.

Mã nguồn chi tiết nằm trong thư mục `src/`; notebook này gọi lại các hàm đó và hiển thị kết quả.
""")

md("## 0. Cài đặt môi trường\nTrên Google Colab: bỏ comment 3 dòng đầu của cell dưới để tải mã nguồn và cài thư viện ([mở notebook bằng Colab](https://colab.research.google.com/github/quaneluscore123/NLP_ClassificationEmotions/blob/quan/notebooks/NLP_BTL_UIT_VSMEC.ipynb)).")
code("""
# !git clone https://github.com/quaneluscore123/NLP_ClassificationEmotions.git
# %cd NLP_ClassificationEmotions
# !pip install -q -r requirements.txt
import os, sys, json
from pathlib import Path
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
ROOT = Path.cwd() if (Path.cwd() / "src").exists() else Path.cwd().parent
os.chdir(ROOT); sys.path.insert(0, str(ROOT))
import pandas as pd, numpy as np
from IPython.display import Image, display
pd.set_option("display.max_colwidth", 120); pd.set_option("display.width", 200)
print("Thư mục dự án:", ROOT)
""")

md("""
## 1. Dữ liệu
**Nguồn:** UIT-VSMEC (Ho et al., 2019) — 6.927 câu bình luận Facebook tiếng Việt, gán nhãn thủ công 7 cảm xúc.
Dùng bản phân phối công khai trên Hugging Face (`tridm/UIT-VSMEC`), giữ nguyên split train/valid/test của tác giả.
""")
code("""
from src.config import LABELS, TEXT_COL, LABEL_COL
from src.data import load_raw, clean_splits
raw = load_raw()
splits, clean_log = clean_splits(raw)
print(clean_log)
splits["train"].sample(8, random_state=1)
""")

md("### 1.1 Thống kê (EDA)")
code("""
from src.eda import run_eda
eda = run_eda(splits, clean_log)
display(eda["split_stats"])
display(eda["label_distribution"])
""")
code("""
display(Image("results/figures/eda_label_distribution.png"))
display(Image("results/figures/eda_length.png"))
display(eda["length_by_label"]); display(eda["emoji_by_label"]); display(eda["top_tokens"].head(20).T)
""")
md("""
**Nhận xét:** dữ liệu mất cân bằng (Enjoyment ~28% so với Surprise ~4%) → dùng **macro-F1** làm thước đo chính và thử `class_weight`.
Câu ngắn (trung vị 12 âm tiết), nhiều emoji/emoticon, teencode, từ tục; token `per` là tên người đã được ẩn danh; một số câu lỗi dấu tổ hợp bị tách rời.
""")

md("""
## 2. Tiền xử lý
Mỗi bước là một cờ để làm khảo sát có kiểm soát (ablation). Các bước đều là **quy tắc cố định** (không học tham số) nên không gây rò rỉ dữ liệu.

| Cấu hình | Mô tả |
|---|---|
| P0_raw | chỉ chữ thường |
| P1_normalize | + chuẩn Unicode NFC, sửa dấu tách rời, số→`num`, tách dấu câu, rút gọn ký tự lặp, emoticon→token (`:))`→`emo_cuoi`), emoji thành token riêng |
| P2_+teencode | + chuẩn hóa teencode (`ko`→`không`, `dc`→`được`...) |
| P3_+segment | + tách từ tiếng Việt (pyvi: `nước mắt`→`nước_mắt`) |
| P4_+stopwords | + loại hư từ (giữ phủ định / từ mức độ) |
| P5_-emoji/emoticon | như P3 nhưng xóa emoji & emoticon (kiểm tra vai trò của chúng) |
""")
code("""
from src.preprocess import CONFIGS, preprocess
s = "t cu ̃ người cu ̀ người nghê ̀. ko biết đc gì :)))) 😡😡 xem tới đây ứa nước mắt"
for name, cfg in CONFIGS.items():
    print(f"{name:20s} -> {preprocess(s, cfg)}")
pd.read_csv("results/tables/preprocess_examples.csv", index_col=0)
""")

md("""
## 3. Chạy toàn bộ thực nghiệm
`run_all.py` thực hiện: khảo sát tiền xử lý → khảo sát đặc trưng n-gram → grid-search siêu tham số baseline (trên valid) →
huấn luyện 6 biến thể RNN × 3 seed → đánh giá test → bootstrap → phân tích lỗi → lưu mô hình.

Mất khoảng 25–30 phút trên CPU. Đặt `RUN_FULL = False` để chỉ đọc lại kết quả đã lưu trong `results/`.
""")
code("""
RUN_FULL = False
if RUN_FULL or not Path("results/summary.json").exists():
    import run_all
    run_all.main(fast=False)
summary = json.loads(Path("results/summary.json").read_text(encoding="utf-8"))
print(json.dumps({k: summary[k] for k in ["best_preprocess", "baseline_selection", "best_rnn", "seeds", "runtime_min"]},
                 indent=2, ensure_ascii=False))
""")

md("## 4. Phương pháp 1 — Baseline TF-IDF + mô hình tuyến tính\n### 4.1 Khảo sát tiền xử lý (TF-IDF word 1–2gram, macro-F1 trên valid)")
code("""
t = pd.read_csv("results/tables/ablation_preprocessing.csv")
display(t.pivot(index="preprocess", columns="model", values="val_macro_f1").reindex(list(CONFIGS)).round(4))
display(Image("results/figures/ablation_preprocessing.png"))
""")
md("### 4.2 Khảo sát đặc trưng n-gram và mô hình")
code("""
t = pd.read_csv("results/tables/ablation_features.csv")
display(t.pivot(index="feature", columns="model", values="val_macro_f1").round(4))
display(Image("results/figures/ablation_features.png"))
display(pd.read_csv("results/tables/logreg_top_features.csv"))
""")
md("Ví dụ huấn luyện nhanh baseline ngay trong notebook (cùng hàm với `run_all.py`):")
code("""
from src import baseline as B
from src.evaluate import metrics
from src.preprocess import preprocess_series
cfg = CONFIGS[summary["best_preprocess"]]
X = {k: preprocess_series(v[TEXT_COL], cfg) for k, v in splits.items()}
y = {k: v[LABEL_COL].tolist() for k, v in splits.items()}
sel = summary["baseline_selection"]["LogReg"]
pipe = B.build_pipeline("LogReg", sel["feature"], **sel["hp"]).fit(X["train"], y["train"])  # vectorizer fit trên train
print("Cấu hình:", sel)
print("valid:", {k: round(v, 4) for k, v in metrics(y["valid"], pipe.predict(X["valid"])).items()})
""")

md("""
## 5. Phương pháp 2 — (Bi)LSTM với TensorFlow/Keras
Kiến trúc: `TextVectorization (adapt trên train)` → `Embedding(mask_zero)` → `SpatialDropout` → `BiLSTM` → (pooling) → `Dense(64)` → `Softmax(7)`.
Early stopping theo **macro-F1 valid**, khôi phục trọng số tốt nhất; `class_weight="balanced"`; mỗi cấu hình chạy 3 seed.
""")
code("""
t = pd.read_csv("results/tables/ablation_rnn.csv")
display(t.round(4))
display(Image("results/figures/ablation_rnn.png"))
display(Image("results/figures/rnn_learning_curve.png"))
""")

md("## 6. Kết quả trên tập test (bảng so sánh chính)")
code("""
f = pd.read_csv("results/tables/final_comparison_test.csv")
cols = [c for c in ["model", "preprocess", "feature", "val_macro_f1", "test_macro_f1", "test_macro_f1_std",
                    "test_macro_precision", "test_macro_recall", "test_weighted_f1", "test_accuracy"] if c in f]
display(f[cols].round(4))
display(Image("results/figures/final_comparison.png"))
display(pd.read_csv("results/tables/bootstrap_test.csv").round(4))
""")
code("""
pc = pd.read_csv("results/tables/per_class_test.csv")
display(pc.pivot(index="label", columns="model", values="f1-score").reindex(LABELS).round(3))
display(Image("results/figures/per_class_f1.png"))
import glob
for p in sorted(glob.glob("results/figures/confusion_*.png")):
    display(Image(p))
""")

md("## 7. Phân tích lỗi")
code("""
for p in sorted(glob.glob("results/tables/error_top_confusions_*.csv")):
    display(pd.read_csv(p))
display(pd.read_csv("results/tables/error_agreement.csv"))
display(pd.read_csv("results/tables/error_by_length.csv").round(3))
pd.read_csv("results/tables/error_examples.csv", index_col=0)
""")
md("""
**Diễn giải chính** (chi tiết ở Mục 5.7 của báo cáo `report/NLP_BTL_MaNhom.pdf`):
- *Anger ↔ Disgust*: câu chửi bới có ranh giới nhãn mờ; từ tục xuất hiện ở cả hai lớp.
- *Câu hỏi tu từ → Surprise*: "sao", ":))", "nhỉ" là từ khóa của Surprise nhưng câu thực chất là phản bác (Other).
- *Từ tiêu cực trong câu tích cực / thành ngữ*: "khắt khe... bỏ thói hư tật xấu" (biết ơn) bị đoán Disgust.
- *Other* không có từ khóa riêng nên bị kéo sang lớp có cảm xúc.
- *BiLSTM quá tự tin* (xác suất ≈ 1 khi sai), dấu hiệu quá khớp, nên ensemble không có lợi.
""")

md("## 8. Demo")
code("""
from demo import Predictor, show
p = Predictor()
for s in ["sản phẩm dùng được 2 hôm đã hỏng, phí tiền vãi",
          "hàng về đẹp quá, shop đóng gói cẩn thận 😍",
          "trời ơi giao hàng nhanh không tưởng luôn á",
          "nhìn con gián trong hộp mà sợ muốn chết"]:
    show(p, s)
""")

nb["cells"] = C
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = Path(__file__).parent / "NLP_BTL_UIT_VSMEC.ipynb"
nbf.write(nb, out)
print("Đã ghi", out)
