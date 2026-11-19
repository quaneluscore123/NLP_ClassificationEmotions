# Phân loại cảm xúc đánh giá sản phẩm tiếng Việt

Bài tập lớn môn **Xử lý ngôn ngữ tự nhiên**: nhận diện cảm xúc của câu bình luận/đánh giá tiếng Việt thành
**7 lớp**: Enjoyment, Sadness, Anger, Fear, Disgust, Surprise, Other.
Thực nghiệm trên bộ dữ liệu chuẩn **UIT-VSMEC**, gồm baseline **TF-IDF + Naive Bayes / Logistic Regression / Linear SVM**
và phương án đối chứng **(Bi)LSTM bằng TensorFlow/Keras**.

## 1. Mục tiêu

- Phát biểu bài toán: đầu vào là một câu tiếng Việt, đầu ra là một nhãn cảm xúc (phân loại đa lớp, đơn nhãn).
- Xây dựng pipeline chạy lại được: dữ liệu thô → EDA/tiền xử lý → biểu diễn → mô hình → đánh giá → phân tích lỗi.
- So sánh công bằng ít nhất 2 phương án trên cùng tập test bằng **macro-F1** (dữ liệu mất cân bằng), có khảo sát có
  kiểm soát (tiền xử lý, n-gram, kiến trúc RNN) và phân tích lỗi.

## 2. Kết quả chính (tập test, 693 câu)

| Mô hình (cấu hình chọn theo valid) | Macro-F1 valid | **Macro-F1 test** | Accuracy test |
|---|---|---|---|
| Đoán lớp đa số (tham chiếu) | 0.068 | 0.062 | 0.278 |
| TF-IDF (word 1–2 + char 2–5) + Multinomial NB | 0.540 | 0.529 | 0.563 |
| **TF-IDF (word 1–2) + Logistic Regression** | **0.569** | **0.590** | **0.612** |
| TF-IDF (word 1–2 + char 2–5) + Linear SVM | 0.562 | 0.583 | 0.603 |
| LSTM 1 chiều (TB ± std, 3 seed) | 0.486 | 0.517 ± 0.006 | 0.540 |
| BiLSTM max+mean pooling, emb 256 (3 seed) | 0.508 | 0.512 ± 0.024 | 0.529 |

- TF-IDF + LogReg tốt nhất. Mô hình này vượt Naive Bayes có ý nghĩa thống kê (paired bootstrap, CI 95% của ΔF1 là [0.021; 0.102]).
- (Bi)LSTM học embedding từ đầu bị quá khớp trên ~5,5 nghìn câu nên thấp hơn baseline. Ensemble LogReg + BiLSTM chọn
  trọng số trên valid ra w = 1.0, tức là không có lợi.
- Khảo sát tiền xử lý cho thấy:
  - giữ emoji/emoticon và chuẩn hóa teencode có ích;
  - tách từ (pyvi) và loại stopwords làm giảm kết quả trên văn bản mạng xã hội.

Số liệu đầy đủ: [results/tables/final_comparison_test.csv](results/tables/final_comparison_test.csv); phân tích chi tiết trong báo cáo.

## 3. Dữ liệu

- **UIT-VSMEC** (Ho et al., 2019): 6.927 câu bình luận Facebook tiếng Việt, gán nhãn thủ công 7 cảm xúc.
  Split gốc gồm train 5.548 / valid 686 / test 693 (sau khi bỏ 4 câu trùng, train còn 5.544).
- Nguồn tải: Hugging Face [`tridm/UIT-VSMEC`](https://huggingface.co/datasets/tridm/UIT-VSMEC).
  `src/data.py` tự tải về `data/raw/` khi chạy lần đầu.
- Giấy phép: dữ liệu công bố cho mục đích nghiên cứu. Repository chỉ chứa mã tải, không phân phối lại dữ liệu.
  Tên người trong dữ liệu đã được tác giả ẩn danh thành token `per`.
- Lưu ý phạm vi: UIT-VSMEC là bình luận mạng xã hội đa chủ đề, không riêng đánh giá sản phẩm (xem mục *Hạn chế* trong báo cáo).

## 4. Cấu trúc thư mục

```
NLP_ClassificationEmotions/
├── README.md
├── requirements.txt
├── run_all.py                 # chạy TOÀN BỘ pipeline → results/, models/
├── demo.py                    # demo dự đoán (CLI / tương tác)
├── src/
│   ├── config.py              # đường dẫn, seed, nhãn
│   ├── data.py                # tải + làm sạch (giữ split gốc)
│   ├── eda.py                 # thống kê, hình EDA
│   ├── resources.py           # từ điển teencode, emoticon, stopwords
│   ├── preprocess.py          # tiền xử lý tiếng Việt, 6 cấu hình ablation (P0–P5)
│   ├── baseline.py            # TF-IDF + NB / LogReg / LinearSVC, grid-search theo valid
│   ├── bilstm.py              # (Bi)LSTM TensorFlow, early stopping theo macro-F1 valid
│   └── evaluate.py            # macro-F1, báo cáo từng lớp, confusion matrix
├── notebooks/
│   ├── NLP_BTL_UIT_VSMEC.ipynb  # notebook trình bày từng bước (đã chạy, có output)
│   └── build_notebook.py        # sinh lại notebook
├── data/
│   ├── raw/                   # train/valid/test.json (tự tải)
│   └── processed/             # văn bản sau tiền xử lý cho từng cấu hình (kết quả trung gian)
├── results/
│   ├── tables/                # mọi bảng số liệu (.csv)
│   ├── figures/               # mọi hình (.png)
│   ├── predictions/           # dự đoán từng câu test của các mô hình
│   ├── summary.json           # cấu hình được chọn, bootstrap, thời gian chạy
│   └── environment.json       # phiên bản thư viện
├── models/                    # mô hình đã huấn luyện cho demo (sinh bởi run_all.py, không commit)
└── report/
    ├── build_report.py        # sinh báo cáo từ results/
    ├── NLP_BTL_MaNhom.docx
    └── NLP_BTL_MaNhom.pdf
```

## 5. Môi trường

- Python 3.10+ (đã kiểm thử với Python 3.12.2 trên Windows 11, chỉ dùng CPU).
- Thư viện: xem [requirements.txt](requirements.txt). Các phiên bản chính: numpy 2.2.6, pandas 3.0.3,
  scikit-learn 1.9.1, tensorflow 2.21.0, pyvi 0.1.1, matplotlib 3.10.7.

```bash
pip install -r requirements.txt
```

Trên **Google Colab**: clone repository, chạy `pip install -r requirements.txt`, rồi mở notebook. Notebook tự nhận thư mục dự án.

## 6. Các bước chạy lại từ đầu

```bash
# 1) Toàn bộ thực nghiệm (tải dữ liệu, EDA, 3 khảo sát, 6 biến thể RNN × 3 seed, đánh giá test, phân tích lỗi)
python run_all.py            # ~25–30 phút trên CPU
python run_all.py --fast     # bản rút gọn ~10 phút (1 seed, 2 biến thể RNN) để kiểm tra nhanh

# 2) Demo
python demo.py "shop giao hàng chậm mà còn cáu với khách, bực mình thật"
python demo.py               # chế độ tương tác

# 3) Notebook trình bày (đọc lại kết quả; đặt RUN_FULL=True để chạy lại từ đầu)
jupyter notebook notebooks/NLP_BTL_UIT_VSMEC.ipynb

# 4) Sinh lại báo cáo Word/PDF từ results/
python report/build_report.py
```

### Tái tạo số liệu và đồ thị

Mọi số liệu được sinh bởi `run_all.py`. Seed được cố định (42, 43, 44 cho RNN) và op determinism của TensorFlow được bật,
nên chạy lại trên cùng môi trường cho cùng kết quả.

| Nội dung | File |
|---|---|
| Thống kê dữ liệu, phân bố nhãn, độ dài, emoji | `results/tables/eda_*.csv`, `results/figures/eda_*.png` |
| Khảo sát tiền xử lý (P0–P5) | `ablation_preprocessing.csv`, `grid_preprocessing_all.csv`, `ablation_preprocessing.png` |
| Khảo sát n-gram × mô hình | `ablation_features.csv`, `grid_features_all.csv`, `ablation_features.png` |
| Khảo sát RNN (6 biến thể × 3 seed) | `ablation_rnn.csv`, `rnn_runs.csv`, `rnn_histories.csv`, `ablation_rnn.png`, `rnn_learning_curve.png` |
| Thử nghiệm RNN sơ bộ (1 seed, trên valid) | `rnn_exploration_prelim.csv` |
| Chọn trọng số ensemble | `ensemble_weight_valid.csv`, `ensemble_weight.png` |
| **Bảng so sánh chính (test)** | `final_comparison_test.csv`, `final_comparison.png` |
| Kiểm định bootstrap | `bootstrap_test.csv` |
| F1 từng lớp, confusion matrix | `per_class_test.csv`, `per_class_f1.png`, `confusion_*.png` |
| Phân tích lỗi | `error_examples.csv`, `error_top_confusions_*.csv`, `error_agreement.csv`, `error_by_length.csv` |
| Dự đoán từng câu test | `results/predictions/test_predictions.csv` |

## 7. Thiết kế thực nghiệm (tóm tắt)

- **Không rò rỉ dữ liệu**:
  - TF-IDF nằm trong sklearn `Pipeline`, chỉ fit trên train.
  - `TextVectorization` chỉ adapt trên train.
  - Tiền xử lý là quy tắc cố định, không học tham số.
- **Chọn mô hình chỉ theo valid**: tiền xử lý, đặc trưng, siêu tham số, kiến trúc, epoch (early stopping theo macro-F1
  valid) và trọng số ensemble. Tập test chỉ dùng cho bảng cuối.
- **Thước đo**:
  - chính là macro-F1;
  - phụ là macro-P/R, weighted-F1, accuracy, F1 từng lớp và confusion matrix;
  - với RNN, báo cáo TB ± std qua 3 seed; chênh lệch giữa các mô hình được kiểm định bằng paired bootstrap.

## 8. Tài liệu tham khảo chính

- Ho, V. A. et al. (2019). *Emotion Recognition for Vietnamese Social Media Text*. PACLING 2019.
- Pedregosa et al. (2011). *Scikit-learn*. JMLR. Abadi et al. (2015). *TensorFlow*.
- Hochreiter & Schmidhuber (1997). *Long Short-Term Memory*.
- pyvi: https://github.com/trungtv/pyvi
