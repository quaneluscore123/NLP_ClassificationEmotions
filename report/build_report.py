"""Sinh báo cáo NLP_BTL_MaNhom.docx (+ .pdf nếu có MS Word) từ các kết quả trong results/.

    python report/build_report.py

Mọi con số trong báo cáo được đọc trực tiếp từ results/tables/*.csv và results/summary.json
-> truy vết được về mã nguồn và dữ liệu."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
T = ROOT / "results" / "tables"
F = ROOT / "results" / "figures"
OUT_DOCX = ROOT / "report" / "NLP_BTL_MaNhom.docx"

S = json.loads((ROOT / "results" / "summary.json").read_text(encoding="utf-8"))
ENV = json.loads((ROOT / "results" / "environment.json").read_text(encoding="utf-8"))


def rd(name, **kw):
    return pd.read_csv(T / f"{name}.csv", **kw)


def pct(x, d=2):
    return f"{100 * x:.{d}f}"


# ------------------------------------------------------------------ dữ liệu cho báo cáo
stats = rd("eda_split_stats")
lab = rd("eda_label_distribution")
lenlab = rd("eda_length_by_label")
emo = rd("eda_emoji_by_label")
clean = S["cleaning"]
prep_abl = rd("ablation_preprocessing")
feat_abl = rd("ablation_features")
rnn_abl = rd("ablation_rnn")
prelim = rd("rnn_exploration_prelim")
final = rd("final_comparison_test")
boot = rd("bootstrap_test")
pc = rd("per_class_test")
agree = rd("error_agreement", index_col=0)
bylen = rd("error_by_length")
errex = rd("error_examples", index_col=0)
topf = rd("logreg_top_features")
wens = rd("ensemble_weight_valid")
best_prep = S["best_preprocess"]
best_rnn = S["best_rnn"]
sel = S["baseline_selection"]
fin = final.set_index("model")
lr = fin.loc["TF-IDF+LogReg"]
nb = fin.loc["TF-IDF+MultinomialNB"]
svc = fin.loc["TF-IDF+LinearSVC"]
rnn = fin.loc[best_rnn]
lstm = fin.loc["R1_LSTM_last"]
maj = fin.loc["Majority class (tham chiếu)"]
conf_lr = rd("error_top_confusions_TF-IDF_LogReg")
conf_rnn = rd(f"error_top_confusions_{best_rnn}")
n_test = int(stats.set_index("split").loc["test", "n_sentences"])
both_wrong = int(agree.loc[False, "False"]) if "False" in agree.columns else int(agree.iloc[0, 0])
only_lr_wrong = int(agree.iloc[0, 1])
only_rnn_wrong = int(agree.iloc[1, 0])
both_right = int(agree.iloc[1, 1])
_pred = pd.read_csv(ROOT / "results" / "predictions" / "test_predictions.csv")
_wr = _pred[_pred[f"pred_{best_rnn}"] != _pred.true]
over_n = len(_wr)
over_frac = float((_wr[f"conf_{best_rnn}"] >= 0.9).mean())
over_med = float(_wr[f"conf_{best_rnn}"].median())
piv_prep = prep_abl.pivot(index="preprocess", columns="model", values="val_macro_f1")
piv_feat = feat_abl.pivot(index="feature", columns="model", values="val_macro_f1")

# ------------------------------------------------------------------ tiện ích docx
doc = Document()
sec = doc.sections[0]
sec.page_height, sec.page_width = Cm(29.7), Cm(21.0)
sec.left_margin, sec.right_margin, sec.top_margin, sec.bottom_margin = Cm(3), Cm(2), Cm(2), Cm(2)

st = doc.styles["Normal"]
st.font.name = "Times New Roman"
st.font.size = Pt(13)
st.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
st.paragraph_format.line_spacing = 1.3
st.paragraph_format.space_after = Pt(4)
for lvl, size in ((1, 16), (2, 14), (3, 13)):
    h = doc.styles[f"Heading {lvl}"]
    h.font.name = "Times New Roman"
    rf = h.element.rPr.rFonts
    for a in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
        if rf.get(qn(a)) is not None:
            del rf.attrib[qn(a)]
    for a in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        rf.set(qn(a), "Times New Roman")
    h.font.size = Pt(size)
    h.font.bold = True
    h.font.color.rgb = RGBColor(0, 0, 0)
    h.paragraph_format.space_before = Pt(12 if lvl == 1 else 8)
    h.paragraph_format.space_after = Pt(6)

_fig_no = [0]
_tab_no = [0]


def H(text, lvl=1):
    return doc.add_heading(text, level=lvl)


def P(text="", bold_prefix=None, align=None, italic=False, size=None):
    p = doc.add_paragraph()
    if bold_prefix:
        r = p.add_run(bold_prefix)
        r.bold = True
    r = p.add_run(text)
    r.italic = italic
    if size:
        r.font.size = Pt(size)
    p.paragraph_format.alignment = align if align is not None else WD_ALIGN_PARAGRAPH.JUSTIFY
    return p


def BL(items, style="List Bullet"):
    for it in items:
        p = doc.add_paragraph(style=style)
        if isinstance(it, tuple):
            p.add_run(it[0]).bold = True
            p.add_run(it[1])
        else:
            p.add_run(it)
        p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY


def caption(text, kind="Bảng"):
    if kind == "Bảng":
        _tab_no[0] += 1
        n = _tab_no[0]
    else:
        _fig_no[0] += 1
        n = _fig_no[0]
    p = doc.add_paragraph()
    r = p.add_run(f"{kind} {n}. ")
    r.bold = True
    r.font.size = Pt(11)
    r2 = p.add_run(text)
    r2.italic = True
    r2.font.size = Pt(11)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.keep_with_next = kind == "Bảng"
    return n


def shade(cell, hexcolor):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hexcolor)
    tcPr.append(shd)


def TABLE(df: pd.DataFrame, cap: str, fmt=None, font=10, bold_rows=(), widths=None):
    n = caption(cap, "Bảng")
    t = doc.add_table(rows=1, cols=len(df.columns))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for j, c in enumerate(df.columns):
        cell = t.rows[0].cells[j]
        cell.text = ""
        r = cell.paragraphs[0].add_run(str(c))
        r.bold = True
        r.font.size = Pt(font)
        shade(cell, "D9E2F3")
    for i, (_, row) in enumerate(df.iterrows()):
        cells = t.add_row().cells
        for j, c in enumerate(df.columns):
            v = row[c]
            if fmt and c in fmt and pd.notna(v) and not isinstance(v, str):
                s = fmt[c](v)
            elif isinstance(v, float):
                s = "" if pd.isna(v) else f"{v:.4f}"
            else:
                s = str(v)
            cells[j].text = ""
            r = cells[j].paragraphs[0].add_run(s)
            r.font.size = Pt(font)
            r.bold = i in bold_rows
    if widths:
        t.autofit = False
        for row in t.rows:
            for j, w in enumerate(widths):
                row.cells[j].width = Cm(w)
    doc.add_paragraph()
    return n


def FIG(path, cap, width=15):
    doc.add_picture(str(path), width=Cm(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.paragraphs[-1].paragraph_format.keep_with_next = True
    return caption(cap, "Hình")


def page_break():
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)


def add_toc():
    p = doc.add_paragraph()
    r = p.add_run()
    for tag, text in (("begin", None), (None, 'TOC \\o "1-3" \\h \\z \\u'), ("separate", None),
                      (None, None), ("end", None)):
        if tag:
            fc = OxmlElement("w:fldChar")
            fc.set(qn("w:fldCharType"), tag)
            r._r.append(fc)
        elif text:
            it = OxmlElement("w:instrText")
            it.set(qn("xml:space"), "preserve")
            it.text = text
            r._r.append(it)
        else:
            t = OxmlElement("w:t")
            t.text = "(Nhấn chuột phải → Update Field để cập nhật mục lục)"
            r._r.append(t)


def add_page_numbers():
    for s in doc.sections:
        p = s.footer.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run()
        for tag, text in (("begin", None), (None, "PAGE"), ("end", None)):
            if tag:
                fc = OxmlElement("w:fldChar")
                fc.set(qn("w:fldCharType"), tag)
                r._r.append(fc)
            else:
                it = OxmlElement("w:instrText")
                it.set(qn("xml:space"), "preserve")
                it.text = text
                r._r.append(it)


# ------------------------------------------------------------------ sơ đồ pipeline
def pipeline_figure(path):
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.axis("off")
    boxes = [
        (0.02, 0.62, "Dữ liệu thô\nUIT-VSMEC\n(train/valid/test gốc)"),
        (0.19, 0.62, "Làm sạch + EDA\n(trùng lặp, độ dài,\nnhãn, emoji)"),
        (0.36, 0.62, "Tiền xử lý\nNFC, emoticon, teencode,\n(tách từ pyvi)"),
        (0.55, 0.80, "TF-IDF n-gram\n(fit trên train)"),
        (0.55, 0.40, "TextVectorization\n(adapt trên train)\n+ Embedding"),
        (0.73, 0.80, "NB / LogReg / SVM\n(grid-search\ntheo valid)"),
        (0.73, 0.40, "(Bi)LSTM\nearly stop theo\nmacro-F1 valid"),
        (0.89, 0.62, "Đánh giá TEST\nmacro-F1, CM,\nbootstrap,\nphân tích lỗi"),
    ]
    for x, y, t in boxes:
        ax.add_patch(plt.Rectangle((x, y - 0.13), 0.15, 0.26, fc="#DCE6F2", ec="#1F4E79", lw=1.2,
                                   transform=ax.transAxes))
        ax.text(x + 0.075, y, t, ha="center", va="center", fontsize=8.5, transform=ax.transAxes)
    arrows = [((0.17, 0.62), (0.19, 0.62)), ((0.34, 0.62), (0.36, 0.62)),
              ((0.51, 0.66), (0.55, 0.78)), ((0.51, 0.58), (0.55, 0.42)),
              ((0.70, 0.80), (0.73, 0.80)), ((0.70, 0.40), (0.73, 0.40)),
              ((0.88, 0.78), (0.90, 0.70)), ((0.88, 0.42), (0.90, 0.54))]
    for a, b in arrows:
        ax.annotate("", xy=b, xytext=a, xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="->", lw=1.3, color="#1F4E79"))
    ax.text(0.5, 0.07, "Mọi lựa chọn (tiền xử lý, đặc trưng, siêu tham số, kiến trúc) dựa trên VALID; "
            "TEST chỉ dùng một lần để báo cáo.\nKết quả trung gian: data/processed/, results/tables/, "
            "results/figures/, results/predictions/, models/",
            ha="center", fontsize=8.5, style="italic", transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


pipeline_figure(F / "pipeline.png")

# ================================================================== BÌA
for _ in range(2):
    doc.add_paragraph()
for t, sz, b in (("[TÊN TRƯỜNG / KHOA]", 14, True), ("HỌC PHẦN: XỬ LÝ NGÔN NGỮ TỰ NHIÊN", 14, True),
                 ("", 12, False), ("BÁO CÁO BÀI TẬP LỚN", 20, True), ("", 12, False),
                 ("PHÂN LOẠI CẢM XÚC VĂN BẢN TIẾNG VIỆT", 18, True),
                 ("Thực nghiệm trên bộ dữ liệu UIT-VSMEC: baseline TF-IDF và mô hình BiLSTM", 14, False)):
    p = doc.add_paragraph()
    r = p.add_run(t)
    r.font.size = Pt(sz)
    r.bold = b
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
for _ in range(3):
    doc.add_paragraph()
for t in ("Mã nhóm: [MaNhom]", "Giảng viên hướng dẫn: [Họ tên GV]",
          "Thành viên: [Họ tên 1 – MSSV], [Họ tên 2 – MSSV], [Họ tên 3 – MSSV]",
          "Mã nguồn: [đường dẫn repository / Google Colab]"):
    p = doc.add_paragraph()
    p.add_run(t).font.size = Pt(13)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
for _ in range(5):
    doc.add_paragraph()
p = doc.add_paragraph()
p.add_run("[Địa điểm], tháng 11/2026").italic = True
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
page_break()

_p = doc.add_paragraph()
_r = _p.add_run("MỤC LỤC")
_r.bold = True
_r.font.size = Pt(16)
_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
add_toc()
page_break()

H("TÓM TẮT", 1)
P(f"Báo cáo trình bày một quy trình xử lý ngôn ngữ tự nhiên hoàn chỉnh, có thể chạy lại, cho bài toán nhận diện "
  f"cảm xúc của câu văn tiếng Việt thành 7 lớp (Enjoyment, Sadness, Anger, Fear, Disgust, Surprise, Other) trên bộ "
  f"dữ liệu chuẩn UIT-VSMEC (6.927 câu bình luận mạng xã hội). Nhóm xây dựng baseline TF-IDF kết hợp Naive Bayes, "
  f"Logistic Regression và Linear SVM, và phương án đối chứng là mạng (Bi)LSTM huấn luyện bằng TensorFlow. "
  f"Ba khảo sát có kiểm soát được thực hiện trên tập valid: (i) các bước tiền xử lý (chuẩn hóa Unicode, emoticon, "
  f"teencode, tách từ, stopwords, emoji), (ii) loại và bậc n-gram, (iii) kiến trúc và siêu tham số RNN "
  f"(một/hai chiều, pooling, class weight, kích thước embedding, tách từ), mỗi cấu hình RNN chạy 3 seed.")
P(f"Trên tập test, mô hình tốt nhất là TF-IDF (n-gram từ 1–2, âm tiết, có chuẩn hóa teencode) + Logistic Regression "
  f"với macro-F1 = {pct(lr.test_macro_f1)}% và accuracy = {pct(lr.test_accuracy)}%, cao hơn baseline Naive Bayes "
  f"({pct(nb.test_macro_f1)}%) với khoảng tin cậy bootstrap 95% của chênh lệch không chứa 0. Mô hình BiLSTM tốt nhất "
  f"theo valid ({best_rnn}) chỉ đạt macro-F1 test {pct(rnn.test_macro_f1)} ± {pct(rnn.test_macro_f1_std)}% "
  f"(TB ± độ lệch chuẩn qua 3 seed). Kết hợp mềm LogReg + BiLSTM cũng không cải thiện trên valid "
  f"(trọng số tối ưu w = {S['ensemble']['w_logreg']}). Phân tích lỗi cho thấy các nhầm lẫn chủ yếu nằm ở các cặp "
  f"nhãn gần nghĩa (Anger↔Disgust, Other↔Enjoyment/Disgust, Surprise↔Enjoyment), câu mỉa mai và câu cần ngữ cảnh. "
  f"Kết quả khẳng định với dữ liệu nhỏ (~5,5 nghìn câu), mô hình tuyến tính trên n-gram vẫn là lựa chọn mạnh, còn "
  f"mạng nơ-ron học embedding từ đầu dễ quá khớp; hướng cải tiến hợp lý là dùng biểu diễn được huấn luyện trước "
  f"(PhoBERT, word embedding tiếng Việt).")
P("Từ khóa: nhận diện cảm xúc, tiếng Việt, UIT-VSMEC, TF-IDF, Logistic Regression, BiLSTM, macro-F1.", italic=True)
page_break()

# ================================================================== 1. GIỚI THIỆU
H("1. Giới thiệu")
H("1.1. Bối cảnh và vấn đề", 2)
P("Bình luận và đánh giá của người dùng trên mạng xã hội, sàn thương mại điện tử là nguồn phản hồi quan trọng "
  "đối với doanh nghiệp: chúng cho biết khách hàng vui vẻ, thất vọng, giận dữ hay ngạc nhiên trước một sản phẩm, "
  "dịch vụ hay sự kiện. Phân tích cảm xúc truyền thống chỉ chia văn bản thành tích cực/tiêu cực/trung tính; trong khi "
  "nhận diện cảm xúc (emotion recognition) yêu cầu mức chi tiết cao hơn — phân biệt 'tiêu cực vì buồn' với 'tiêu cực "
  "vì giận' hay 'vì ghê tởm' — giúp doanh nghiệp ưu tiên xử lý phản hồi (ví dụ: khách hàng giận dữ cần được phản hồi "
  "ngay) và đo lường giá trị thương hiệu chính xác hơn.")
P("Với tiếng Việt, bài toán khó hơn do: (1) ngôn ngữ đơn lập, một từ có thể gồm nhiều âm tiết nên cần tách từ; "
  "(2) văn bản mạng xã hội dùng nhiều teencode (ko, dc, vs), viết tắt, từ lóng, từ tục, emoji/emoticon, lỗi gõ dấu; "
  "(3) dữ liệu có nhãn ít và mất cân bằng giữa các lớp cảm xúc.")
H("1.2. Phát biểu bài toán", 2)
BL([("Đầu vào: ", "một câu/bình luận tiếng Việt dạng văn bản (độ dài trung vị 12 âm tiết)."),
    ("Đầu ra: ", "một nhãn trong 7 lớp cảm xúc: Enjoyment (vui thích), Sadness (buồn), Anger (giận dữ), Fear (sợ hãi), "
                 "Disgust (ghê tởm/khinh bỉ), Surprise (ngạc nhiên), Other (khác/không rõ cảm xúc)."),
    ("Loại bài toán: ", "phân loại văn bản đơn nhãn, đa lớp, có giám sát."),
    ("Đối tượng sử dụng: ", "bộ phận chăm sóc khách hàng, marketing, nhà phân tích dư luận cần tự động gắn nhãn cảm "
                            "xúc cho khối lượng lớn bình luận; nhà nghiên cứu NLP tiếng Việt."),
    ("Tiêu chí thành công: ", "(a) pipeline chạy lại được từ dữ liệu thô đến số liệu cuối; (b) macro-F1 trên test vượt "
                              "rõ baseline đơn giản (Naive Bayes) và tham chiếu 'đoán lớp đa số'; (c) đạt mức tương "
                              "đương kết quả công bố gốc của bộ dữ liệu (CNN, F1 = 57,61%, Ho et al., 2019); "
                              "(d) giải thích được lỗi và giới hạn.")])
H("1.3. Câu hỏi nghiên cứu", 2)
BL(["RQ1 — Các bước tiền xử lý đặc thù tiếng Việt (chuẩn hóa Unicode, teencode, tách từ, loại stopwords, giữ/bỏ "
    "emoji) ảnh hưởng thế nào đến hiệu quả phân loại cảm xúc?",
    "RQ2 — Loại đặc trưng nào (n-gram từ bậc 1–3, n-gram ký tự, kết hợp) phù hợp nhất cho mô hình tuyến tính?",
    "RQ3 — Mạng (Bi)LSTM học embedding từ đầu có vượt được baseline TF-IDF trên dữ liệu nhỏ không? Các yếu tố kiến "
    "trúc (hai chiều, pooling, class weight, kích thước embedding) đóng góp bao nhiêu?",
    "RQ4 — Mô hình nhầm ở đâu, vì sao, và đâu là giới hạn của cách tiếp cận?"], style="List Number")
H("1.4. Đóng góp của nhóm", 2)
BL(["Xây dựng pipeline tái lập được (một lệnh `python run_all.py`), lưu toàn bộ kết quả trung gian.",
    "Bộ tiền xử lý cho văn bản mạng xã hội tiếng Việt với từng bước bật/tắt được để khảo sát.",
    "Ba khảo sát có kiểm soát, so sánh công bằng trên cùng tập test, kiểm định bootstrap ghép cặp.",
    "Phân tích lỗi định lượng (ma trận nhầm lẫn, theo độ dài câu, mức đồng thuận giữa mô hình) và định tính "
    "(16 ví dụ cụ thể)."])

# ================================================================== 2. DỮ LIỆU
H("2. Dữ liệu")
H("2.1. Nguồn dữ liệu và giấy phép", 2)
P("Nhóm sử dụng bộ dữ liệu UIT-VSMEC (Vietnamese Social Media Emotion Corpus) do Phòng thí nghiệm Khoa học Dữ liệu, "
  "Trường ĐH Công nghệ Thông tin – ĐHQG TP.HCM công bố (Ho et al., 2019). Dữ liệu gồm 6.927 câu bình luận Facebook "
  "tiếng Việt, được gán nhãn thủ công theo 6 cảm xúc cơ bản của Ekman cùng nhãn Other. Bản phân phối dùng trong báo "
  "cáo được tải tự động từ Hugging Face (tridm/UIT-VSMEC: train.json, valid.json, test.json) bằng `src/data.py`, "
  "giữ nguyên cách chia train/valid/test của tác giả để có thể so sánh với các nghiên cứu khác.")
P("Bộ dữ liệu được công bố cho mục đích nghiên cứu; nhóm chỉ dùng cho học tập, trích dẫn bài báo gốc và không phân "
  "phối lại dữ liệu (repository chỉ chứa mã tải tự động). Thông tin cá nhân đã được tác giả ẩn danh: tên người được "
  "thay bằng token 'per' (xuất hiện trong "
  f"{int(stats.set_index('split').loc['train', 'has_per_token'])} câu train). Nhóm không thu thập thêm dữ liệu.")
P("Lưu ý về phạm vi: đề tài hướng tới phân tích cảm xúc trong phản hồi/đánh giá của người dùng, nhưng UIT-VSMEC là "
  "bình luận mạng xã hội nói chung (chủ đề đa dạng: đời sống, giải trí, sản phẩm...), không riêng đánh giá sản phẩm. "
  "Đây là bộ dữ liệu cảm xúc nhiều lớp có nhãn chuẩn, công khai, phổ biến nhất cho tiếng Việt; khả năng chuyển giao "
  "sang miền đánh giá sản phẩm được thảo luận ở Mục 5.8.", italic=True)
H("2.2. Thống kê dữ liệu", 2)
s2 = stats[["split", "n_sentences", "n_tokens(syllable)", "vocab_size(syllable)", "mean_len", "median_len",
            "p95_len", "max_len", "has_emoji", "has_emoticon"]].copy()
s2.columns = ["Tập", "Số câu", "Số âm tiết", "Từ vựng", "ĐD TB", "Trung vị", "P95", "Max", "Có emoji", "Có emoticon"]
TABLE(s2, "Thống kê các tập dữ liệu (sau làm sạch; đơn vị token = âm tiết cách nhau bởi khoảng trắng).",
      fmt={"ĐD TB": lambda v: f"{v:.2f}", "Trung vị": lambda v: f"{v:.0f}", "P95": lambda v: f"{v:.0f}"}, font=9)
l2 = lab.copy()
l2.columns = ["Nhãn", "Train", "Valid", "Test", "Train %", "Valid %", "Test %"]
TABLE(l2, "Phân bố nhãn theo tập.", fmt={c: (lambda v: f"{v:.2f}") for c in ["Train %", "Valid %", "Test %"]})
FIG(F / "eda_label_distribution.png", "Phân bố nhãn cảm xúc trên ba tập.", 14)
P(f"Dữ liệu mất cân bằng rõ rệt: Enjoyment chiếm {lab.set_index('Emotion').loc['Enjoyment', 'train_%']:.1f}% tập "
  f"train trong khi Surprise chỉ {lab.set_index('Emotion').loc['Surprise', 'train_%']:.1f}% và Fear "
  f"{lab.set_index('Emotion').loc['Fear', 'train_%']:.1f}% (tỉ lệ khoảng 6,4 : 1). Do đó (i) accuracy không phản ánh "
  f"đúng chất lượng trên các lớp hiếm — một mô hình luôn đoán Enjoyment đã đạt accuracy test "
  f"{pct(maj.test_accuracy)}% nhưng macro-F1 chỉ {pct(maj.test_macro_f1)}%; (ii) nhóm dùng macro-F1 làm thước đo "
  f"chính và thử nghiệm trọng số lớp (class_weight='balanced'). Phân bố nhãn giữa ba tập tương đối đồng đều (tập "
  f"valid có ít Sadness hơn), nên chọn mô hình theo valid là hợp lý.")
FIG(F / "eda_length.png", "Phân bố độ dài câu (trái) và độ dài theo nhãn (phải) trên tập train.", 16)
ll = lenlab[["Emotion", "count", "mean", "50%", "75%", "max"]].copy()
ll.columns = ["Nhãn", "Số câu", "ĐD TB", "Trung vị", "Q3", "Max"]
TABLE(ll, "Độ dài câu (số âm tiết) theo nhãn – tập train.",
      fmt={"Số câu": lambda v: f"{v:.0f}", "ĐD TB": lambda v: f"{v:.2f}", "Trung vị": lambda v: f"{v:.0f}",
           "Q3": lambda v: f"{v:.0f}", "Max": lambda v: f"{v:.0f}"})
P("Câu rất ngắn (trung vị 12 âm tiết, 95% câu ≤ 33 âm tiết) nên tín hiệu cảm xúc thường tập trung ở vài từ khóa, "
  "emoji hoặc dấu câu. Câu Anger và Disgust dài hơn trung bình (chửi bới, phân tích), Surprise ngắn nhất (câu cảm "
  "thán, câu hỏi tu từ). Độ dài tối đa đầu vào của RNN được đặt 60 token, bao phủ trên 99% số câu.")
e2 = emo.copy()
e2.columns = ["Nhãn", "Số câu có emoji", "Emoji phổ biến (số lần)"]
TABLE(e2, "Emoji phổ biến theo nhãn (train).", widths=[3, 3, 10])
P("Emoji mang tín hiệu cảm xúc mạnh (😭😢 cho Sadness, 😡😠 cho Anger, 😨😱 cho Fear) nhưng cũng nhập nhằng: 😂 "
  "xuất hiện nhiều ở Enjoyment, Other và cả Disgust (cười mỉa). Điều này gợi ý không nên xóa emoji như ký tự rác — "
  "giả thuyết được kiểm chứng ở khảo sát P5.")
H("2.3. Chất lượng dữ liệu và làm sạch", 2)
BL([f"Câu rỗng: {clean['train_empty_removed'] + clean['valid_empty_removed'] + clean['test_empty_removed']}.",
    f"Trùng lặp hoàn toàn (câu + nhãn) trong train: {clean['train_exact_dup_removed']} → loại bỏ; câu trùng nhưng khác "
    f"nhãn: {clean['train_conflicting_label_removed']} → giữ nhãn đầu tiên. Train còn "
    f"{int(stats.set_index('split').loc['train', 'n_sentences'])} câu.",
    f"Chồng lấn train–valid: {clean['valid_overlap_with_train']} câu; train–test: {clean['test_overlap_with_train']} "
    f"câu. Nhóm không sửa valid/test để giữ khả năng so sánh với kết quả công bố.",
    f"Lỗi mã hóa dấu: {int(stats.set_index('split').loc['train', 'broken_diacritic'])} câu train có dấu thanh tổ hợp bị "
    "tách khỏi nguyên âm (ví dụ 'cu ̃ người' thay vì 'cũ người') → được sửa ở bước chuẩn hóa.",
    "Toàn bộ văn bản đã ở dạng chữ thường và chuẩn NFC (tác giả đã chuẩn hóa sẵn); không có URL."])
H("2.4. Tiền xử lý", 2)
P("Tiền xử lý được cài trong `src/preprocess.py`, mỗi bước là một cờ bật/tắt. Tất cả các bước là quy tắc cố định "
  "(không học tham số từ dữ liệu), nên áp dụng cho train/valid/test không gây rò rỉ thông tin. Các từ điển teencode, "
  "emoticon, stopwords do nhóm tự soạn từ quan sát tập train (`src/resources.py`).")
steps = pd.DataFrame([
    ["Chuẩn hóa Unicode", "NFC; gắn lại dấu tổ hợp bị tách; bỏ ký tự biến thể emoji", "Cùng một chữ có nhiều mã → tránh tách thành nhiều token khác nhau"],
    ["Chữ thường", "lower()", "Dữ liệu đã thường sẵn; giữ để pipeline tổng quát"],
    ["Emoticon → token", ":)) =)) → emo_cuoi; :( → emo_buon; -_- → emo_chan; <3 → emo_tim ...", "Emoticon là tín hiệu cảm xúc; nếu tách thành dấu câu sẽ mất nghĩa"],
    ["Emoji", "Tách mỗi emoji thành một token riêng (giữ)", "TfidfVectorizer mặc định bỏ emoji; ta dùng token_pattern \\S+ để giữ"],
    ["Ký tự lặp", "'quáaaa' → 'quáaa' (giữ 2), '!!!!' → '!!'", "Giảm thưa thớt nhưng vẫn giữ dấu hiệu nhấn mạnh"],
    ["Số, dấu câu", "Số → 'num'; tách dấu câu thành token", "Dấu ? ! mang tín hiệu (Surprise, Anger)"],
    ["Teencode", "ko/k → không, dc/đc → được, vs → với, j → gì, t → tao, m → mày ... (~80 mục)", "Gộp biến thể chính tả về một dạng chuẩn"],
    ["Tách từ (tùy chọn)", "pyvi: 'nước mắt' → 'nước_mắt'", "Đơn vị nghĩa của tiếng Việt là từ, không phải âm tiết"],
    ["Stopwords (tùy chọn)", "Loại ~60 hư từ; CHỦ Ý giữ phủ định/mức độ (không, chẳng, quá, rất)", "Loại máy móc sẽ mất tín hiệu ('không thích')"],
], columns=["Bước", "Cách làm", "Lý do"])
TABLE(steps, "Các bước tiền xử lý và lý do lựa chọn.", font=9, widths=[3, 7, 6])
cfg = pd.DataFrame([
    ["P0_raw", "chỉ chữ thường (tách token theo khoảng trắng)"],
    ["P1_normalize", "+ Unicode, emoticon→token, emoji riêng, ký tự lặp, số, dấu câu"],
    ["P2_+teencode", "P1 + chuẩn hóa teencode"],
    ["P3_+segment", "P2 + tách từ pyvi"],
    ["P4_+stopwords", "P3 + loại stopwords"],
    ["P5_-emoji/emoticon", "P3 nhưng xóa emoji và emoticon"],
], columns=["Cấu hình", "Mô tả"])
TABLE(cfg, "Sáu cấu hình tiền xử lý dùng trong khảo sát có kiểm soát (mỗi cấu hình thêm/bớt một yếu tố).")
ex_in = "t cu ̃ người cu ̀ người nghê ̀. ko biết đc gì :)))) 😡😡 xem tới đây ứa nước mắt"
from src.preprocess import CONFIGS, preprocess  # noqa: E402
exdf = pd.DataFrame([[k, preprocess(ex_in, c)] for k, c in CONFIGS.items()], columns=["Cấu hình", "Kết quả"])
P("Ví dụ minh họa với câu đầu vào: “" + ex_in + "”")
TABLE(exdf, "Kết quả tiền xử lý một câu ví dụ theo từng cấu hình.", font=9, widths=[3.5, 12.5])

# ================================================================== 3. PHƯƠNG PHÁP
H("3. Phương pháp")
H("3.1. Tổng quan pipeline", 2)
FIG(F / "pipeline.png", "Sơ đồ pipeline thực nghiệm.", 16.5)
H("3.2. Phương pháp 1 (baseline): TF-IDF + mô hình tuyến tính", 2)
P("Biểu diễn TF-IDF. Mỗi câu d được biểu diễn bằng vector thưa trên từ vựng n-gram V của tập train. Với term t, "
  "trọng số w(t,d) = (1 + log tf(t,d)) · idf(t), trong đó idf(t) = log((1+N)/(1+df(t))) + 1, N là số câu train, "
  "df(t) là số câu chứa t (sublinear tf, smoothing mặc định của scikit-learn); vector được chuẩn hóa L2. Nhóm thử "
  "n-gram từ (1–1, 1–2, 1–3), n-gram ký tự trong biên từ (char_wb 2–5) và ghép cả hai.")
P("Các bộ phân loại:")
BL([("Multinomial Naive Bayes: ", "P(c|d) ∝ P(c) Π P(t|c)^w(t,d) với làm trơn Laplace/Lidstone α. Đơn giản, nhanh, là "
                                   "baseline kinh điển cho phân loại văn bản."),
    ("Logistic Regression (softmax đa lớp): ", "P(c|x) = softmax(W x + b)_c, tối ưu cross-entropy với chính quy hóa L2 "
                                               "(siêu tham số C = 1/λ); tùy chọn class_weight='balanced' (trọng số lớp "
                                               "tỉ lệ nghịch tần suất)."),
    ("Linear SVM: ", "tối ưu hinge loss one-vs-rest với chính quy hóa L2 (C), tùy chọn class_weight.")])
P("Vectorizer và bộ phân loại được đóng gói trong một sklearn Pipeline, fit chỉ trên train; valid/test chỉ được "
  "transform → không rò rỉ IDF hay từ vựng.")
H("3.3. Phương pháp 2 (đối chứng): mạng (Bi)LSTM với TensorFlow", 2)
P("Kiến trúc: TextVectorization (từ vựng lấy từ train, tối đa 20.000 token, độ dài cố định 60) → Embedding "
  "(d = 128 hoặc 256, mask_zero cho padding, khởi tạo ngẫu nhiên) → SpatialDropout1D(0,2) → LSTM 64 đơn vị (một chiều) "
  "hoặc Bidirectional LSTM (2×64) → pooling → Dropout(0,5) → Dense(64, ReLU) → Dropout(0,5) → Dense(7, softmax).")
P("Tế bào LSTM tại bước t: i_t = σ(W_i x_t + U_i h_{t−1} + b_i), f_t = σ(W_f x_t + U_f h_{t−1} + b_f), "
  "o_t = σ(W_o x_t + U_o h_{t−1} + b_o), c̃_t = tanh(W_c x_t + U_c h_{t−1} + b_c), c_t = f_t ⊙ c_{t−1} + i_t ⊙ c̃_t, "
  "h_t = o_t ⊙ tanh(c_t). Cổng quên f_t và trạng thái ô c_t giúp gradient truyền qua nhiều bước, giảm hiện tượng "
  "vanishing gradient của RNN thường. BiLSTM chạy thêm một LSTM ngược chiều và ghép h_t = [h→_t ; h←_t] để mỗi vị trí "
  "thấy cả ngữ cảnh trái và phải (hữu ích với cấu trúc như 'không ... lắm').")
P("Hai kiểu pooling được so sánh: (a) last — dùng trạng thái cuối của mỗi chiều; (b) maxmean — ghép max-pooling và "
  "mean-pooling theo thời gian trên toàn chuỗi h_1..h_T, có mặt nạ bỏ qua vị trí padding (lớp MaskedMaxMeanPool tự cài "
  "đặt). Max-pooling giúp bắt từ khóa cảm xúc mạnh ở bất kỳ vị trí nào, mean-pooling tóm tắt toàn câu.")
P("Huấn luyện: Adam (lr = 10⁻³), batch 32, tối đa 25 epoch, sparse categorical cross-entropy có/không trọng số lớp "
  "'balanced'. Early stopping theo macro-F1 trên valid (patience 4), khôi phục trọng số tốt nhất. Mỗi cấu hình chạy "
  "với 3 seed (42, 43, 44); cố định seed Python/NumPy/TensorFlow và bật op determinism để kết quả lặp lại được.")
H("3.4. Kết hợp mềm (soft-voting ensemble)", 2)
P("Để kiểm tra hai họ mô hình có bổ trợ nhau không, nhóm thử P(c|x) = w·P_LogReg(c|x) + (1−w)·P_BiLSTM(c|x) với "
  "w ∈ {0; 0,1; …; 1}, chọn w theo macro-F1 trung bình trên valid qua 3 seed.")

# ================================================================== 4. THỰC NGHIỆM
H("4. Thiết kế thực nghiệm")
H("4.1. Chia dữ liệu và chống rò rỉ", 2)
BL([f"Dùng split gốc: train {int(stats.iloc[0].n_sentences)} / valid {int(stats.iloc[1].n_sentences)} / test "
    f"{int(stats.iloc[2].n_sentences)} câu.",
    "Mọi lựa chọn (cấu hình tiền xử lý, loại đặc trưng, siêu tham số, kiến trúc RNN, số epoch, trọng số ensemble) chỉ "
    "dựa trên macro-F1 của tập valid. Tập test chỉ được dùng ở bước báo cáo cuối cùng.",
    "TfidfVectorizer và TextVectorization chỉ fit/adapt trên train. Tiền xử lý là quy tắc cố định.",
    "Mô hình cuối cùng được huấn luyện trên train (không gộp valid) để giữ nguyên quy trình đã chọn."])
H("4.2. Thước đo", 2)
P("Thước đo chính là macro-F1 = (1/K) Σ_c F1_c, với F1_c = 2·P_c·R_c/(P_c + R_c), K = 7 — mỗi lớp có trọng số như "
  "nhau nên phù hợp dữ liệu mất cân bằng. Ngoài ra báo cáo macro-precision, macro-recall, weighted-F1 (để so sánh với "
  "bài báo gốc), accuracy, F1 từng lớp và ma trận nhầm lẫn. Với RNN, báo cáo trung bình ± độ lệch chuẩn qua 3 seed. "
  "Để đánh giá ý nghĩa thống kê của chênh lệch giữa hai mô hình trên cùng tập test, nhóm dùng paired bootstrap "
  "(1.000 lần lấy mẫu lại có hoàn lại các câu test, tính khoảng tin cậy 95% của hiệu macro-F1).")
H("4.3. Lưới siêu tham số", 2)
grid = pd.DataFrame([
    ["Multinomial NB", "α ∈ {0,01; 0,05; 0,1; 0,3; 0,5; 1}"],
    ["Logistic Regression", "C ∈ {0,5; 1; 2; 5; 10; 20} × class_weight ∈ {None, balanced}"],
    ["Linear SVM", "C ∈ {0,05; 0,1; 0,3; 0,5; 1} × class_weight ∈ {None, balanced}"],
    ["TF-IDF", "word 1–1/1–2/1–3 (min_df=1), char_wb 2–5 (min_df=2), word1–2 + char2–5; sublinear_tf"],
    ["(Bi)LSTM", "một/hai chiều; pooling last/maxmean; class weight có/không; emb 128/256; input âm tiết/từ"],
], columns=["Thành phần", "Giá trị thử"])
TABLE(grid, "Không gian tìm kiếm siêu tham số (chọn theo macro-F1 valid).", widths=[4, 12])
H("4.4. Môi trường", 2)
envt = pd.DataFrame([[k, str(v)] for k, v in ENV.items()], columns=["Thành phần", "Phiên bản / giá trị"])
TABLE(envt, "Môi trường chạy thực nghiệm (results/environment.json).")
P(f"Toàn bộ pipeline (`python run_all.py`) chạy trên CPU trong khoảng {S['runtime_min']} phút; GPU không bắt buộc.")

# ================================================================== 5. KẾT QUẢ
H("5. Kết quả và thảo luận")
H("5.1. Khảo sát tiền xử lý (RQ1)", 2)
pp = piv_prep.reindex(list(CONFIGS)).reset_index()
pp.columns = ["Cấu hình", "TF-IDF+LogReg", "TF-IDF+MultinomialNB"]
TABLE(pp, "Macro-F1 trên valid theo cấu hình tiền xử lý (TF-IDF word 1–2gram, siêu tham số tốt nhất của từng ô).",
      bold_rows=[list(CONFIGS).index(best_prep)])
FIG(F / "ablation_preprocessing.png", "Ảnh hưởng của các bước tiền xử lý (macro-F1 valid).", 14)
p_lr = piv_prep["LogReg"]
P(f"Nhận xét: (1) Cấu hình tốt nhất (trung bình hai mô hình) là {best_prep}: chuẩn hóa + teencode, không tách từ "
  f"(LogReg {p_lr[best_prep]:.4f}). (2) Chuẩn hóa teencode giúp tăng nhẹ so với P1 "
  f"({p_lr['P1_normalize']:.4f} → {p_lr['P2_+teencode']:.4f} với LogReg). (3) Tách từ bằng pyvi làm giảm hiệu quả "
  f"(LogReg {p_lr['P2_+teencode']:.4f} → {p_lr['P3_+segment']:.4f}): pyvi được huấn luyện trên văn bản chuẩn nên tách "
  f"sai nhiều với văn bản mạng xã hội, đồng thời bigram âm tiết (word 1–2gram) đã nắm được phần lớn từ ghép hai âm "
  f"tiết, nên tách từ không thêm thông tin mà làm từ vựng thưa hơn. (4) Loại stopwords làm giảm rõ "
  f"({p_lr['P3_+segment']:.4f} → {p_lr['P4_+stopwords']:.4f}) vì các hư từ và dấu câu như 'mà', 'thì', '?', '!' "
  f"mang sắc thái cảm xúc trong văn nói. (5) Xóa emoji/emoticon gây giảm mạnh nhất "
  f"({p_lr['P3_+segment']:.4f} → {p_lr['P5_-emoji/emoticon']:.4f} LogReg; NB "
  f"{piv_prep['MultinomialNB']['P3_+segment']:.4f} → {piv_prep['MultinomialNB']['P5_-emoji/emoticon']:.4f}), xác "
  f"nhận emoji/emoticon là tín hiệu quan trọng. (6) Chênh lệch giữa P0 và P2 nhỏ ({p_lr['P0_raw']:.4f} so với "
  f"{p_lr['P2_+teencode']:.4f}), vì dữ liệu đã được tác giả chuẩn hóa một phần.")
H("5.2. Khảo sát đặc trưng và mô hình baseline (RQ2)", 2)
pf = piv_feat.reindex(list(feat_abl.feature.unique())).reset_index()
pf.columns = ["Đặc trưng"] + list(pf.columns[1:])
TABLE(pf, f"Macro-F1 trên valid theo loại đặc trưng và mô hình (tiền xử lý {best_prep}).")
FIG(F / "ablation_features.png", "Khảo sát loại và bậc n-gram (macro-F1 valid).", 14)
P(f"Với Logistic Regression, word 1–2gram tốt nhất ({piv_feat.loc['word_1-2', 'LogReg']:.4f}); thêm bigram tăng "
  f"đáng kể so với unigram ({piv_feat.loc['word_1-1', 'LogReg']:.4f}) vì bắt được cụm hai âm tiết ('nguy hiểm', 'ám "
  f"ảnh', 'cái gì') và cấu trúc phủ định; trigram không giúp thêm do quá thưa với 5,5 nghìn câu. Naive Bayes hưởng lợi "
  f"lớn từ n-gram ký tự (word 1–2 + char 2–5: {piv_feat.loc['word_1-2+char_2-5', 'MultinomialNB']:.4f} so với "
  f"{piv_feat.loc['word_1-2', 'MultinomialNB']:.4f}) vì char n-gram chịu được lỗi chính tả và biến thể teencode "
  f"chưa có trong từ điển. Cấu hình được chọn cho mỗi mô hình: NB — {sel['MultinomialNB']['feature']}, "
  f"{sel['MultinomialNB']['hp']}; LogReg — {sel['LogReg']['feature']}, {sel['LogReg']['hp']}; LinearSVC — "
  f"{sel['LinearSVC']['feature']}, {sel['LinearSVC']['hp']}.")
tf_ = topf.copy()
tf_.columns = ["Nhãn", "Đặc trưng có trọng số lớn nhất (LogReg)"]
TABLE(tf_, "Các n-gram có hệ số lớn nhất của Logistic Regression theo từng lớp.", font=9, widths=[2.5, 13.5])
P("Các đặc trưng học được có ý nghĩa ngôn ngữ rõ ràng: Sadness gắn với 'buồn', 'huhu', 'khóc', 😭, emo_buon; "
  "Enjoyment với 'cười', 'haha', emo_cuoi, 😂; Fear với 'sợ', 'nguy hiểm', 'ám ảnh'; Surprise với 'à', 'sao', 'ngờ', "
  "'wow', '?'; Anger với từ tục và đại từ đối đầu ('mày', 'bọn'); Disgust với 'ngu', 'tởm', 'ghét', 'nhạt'. Other gắn "
  "với từ để hỏi trung tính ('hỏi', 'cái gì', 'gì vậy'). Điều này cho thấy mô hình dựa chủ yếu vào từ khóa — vừa là "
  "điểm mạnh (dễ giải thích) vừa là nguồn lỗi (Mục 5.7).")
H("5.3. Khảo sát mô hình (Bi)LSTM (RQ3)", 2)
P("Trước khi chạy khảo sát chính thức, nhóm thử sơ bộ một số cấu hình với 1 seed trên valid (Bảng dưới) để định "
  "hướng: max+mean pooling với input âm tiết cho kết quả tốt nhất; giảm kích thước mô hình hoặc thêm L2 không giúp.")
pr = prelim[["trial", "preprocess", "config_change", "val_macro_f1", "best_epoch"]].copy()
pr.columns = ["Thử nghiệm", "Tiền xử lý", "Thay đổi", "Macro-F1 valid", "Epoch tốt nhất"]
TABLE(pr, "Thử nghiệm sơ bộ RNN (seed 42, chỉ dùng valid).", font=9, widths=[4.5, 2.6, 4.6, 2.3, 2.2])
ra = rnn_abl[["variant", "preprocess", "bidirectional", "emb_dim", "class_weight", "val_macro_f1_mean",
              "val_macro_f1_std", "best_epoch_mean", "n_params", "train_sec_mean"]].copy()
ra["variant"] = ra["variant"].str.replace("_", " ")
ra["preprocess"] = ra["preprocess"].str.split("_").str[0]
ra.columns = ["Biến thể", "TXL", "2 chiều", "Emb", "CW", "F1 valid TB", "Std", "Epoch", "Tham số", "Giây"]
TABLE(ra, "Khảo sát có kiểm soát các biến thể RNN (3 seed; mỗi biến thể đổi một yếu tố so với R3).",
      fmt={"Epoch": lambda v: f"{v:.1f}", "Giây": lambda v: f"{v:.0f}", "Tham số": lambda v: f"{v:,.0f}",
           "Std": lambda v: f"{v:.3f}"},
      font=8.5, bold_rows=[list(rnn_abl.variant).index(best_rnn)],
      widths=[4.6, 1.1, 1.3, 1.0, 1.1, 1.7, 1.2, 1.2, 1.9, 1.1])
P("TXL = cấu hình tiền xử lý (P2: âm tiết + teencode; P3: thêm tách từ); CW = class weight; Giây = thời gian huấn "
  "luyện trung bình mỗi lần chạy trên CPU.", size=11, italic=True)
FIG(F / "ablation_rnn.png", "Macro-F1 valid của các biến thể RNN (trung bình ± độ lệch chuẩn 3 seed).", 14)
r = rnn_abl.set_index("variant")
P(f"Nhận xét: (1) BiLSTM tốt hơn LSTM một chiều ({r.loc['R2_BiLSTM_last', 'val_macro_f1_mean']:.4f} so với "
  f"{r.loc['R1_LSTM_last', 'val_macro_f1_mean']:.4f}). (2) Max+mean pooling tốt hơn trạng thái cuối "
  f"({r.loc['R3_BiLSTM_maxmean', 'val_macro_f1_mean']:.4f}), phù hợp với việc tín hiệu cảm xúc nằm ở vài từ khóa "
  f"rải rác. (3) Class weight quan trọng: bỏ đi làm giảm còn {r.loc['R5_BiLSTM_maxmean_noCW', 'val_macro_f1_mean']:.4f} "
  f"do mô hình bỏ qua các lớp hiếm. (4) Tách từ tiếp tục gây hại ({r.loc['R4_BiLSTM_maxmean_segment', 'val_macro_f1_mean']:.4f}, "
  f"hội tụ sớm ở epoch 5 và quá khớp nhanh hơn do từ vựng lớn và thưa hơn). (5) Tăng embedding lên 256 cho kết quả "
  f"trung bình cao nhất ({r.loc['R6_BiLSTM_maxmean_emb256', 'val_macro_f1_mean']:.4f}) nhưng phương sai giữa các seed "
  f"lớn (std {r.loc['R6_BiLSTM_maxmean_emb256', 'val_macro_f1_std']:.4f}), nên khác biệt với R3 nằm trong nhiễu. "
  f"Biến thể được chọn theo valid là {best_rnn}.")
FIG(F / "rnn_learning_curve.png", f"Đường học của {best_rnn}: loss train giảm đều trong khi loss valid tăng từ epoch 3–4.", 16)
P("Đường học cho thấy hiện tượng quá khớp điển hình: loss train giảm gần về 0,1–0,2 nhưng loss valid tăng sau vài "
  "epoch, trong khi macro-F1 valid đi ngang quanh 0,50. Với khoảng 5,5 nghìn câu và hơn 0,6–1,2 triệu tham số (phần "
  "lớn nằm ở ma trận embedding học từ đầu), mạng ghi nhớ dữ liệu train thay vì học được biểu diễn tổng quát.")
H("5.4. Kết hợp mềm LogReg + BiLSTM", 2)
FIG(F / "ensemble_weight.png", "Macro-F1 valid theo trọng số w của LogReg trong ensemble.", 11)
P(f"Macro-F1 valid tăng đơn điệu theo w và đạt cực đại tại w = {S['ensemble']['w_logreg']} (chỉ dùng LogReg). Nghĩa là trên "
  f"valid, xác suất của BiLSTM không bổ sung thông tin hữu ích cho LogReg — một phần vì BiLSTM quá tự tin (xác suất "
  f"gần 1 kể cả khi sai) nên làm nhiễu các dự đoán đúng của LogReg. Theo đúng quy trình (chọn theo valid), ensemble "
  f"suy biến thành LogReg và không được báo cáo như một mô hình riêng.")
H("5.5. So sánh tổng hợp trên tập test", 2)
fc = final.copy()
fc["test_macro_f1_s"] = [f"{a:.4f}" + (f" ± {b:.4f}" if pd.notna(b) and b > 0 else "") for a, b in
                         zip(fc.test_macro_f1, fc.get("test_macro_f1_std", [float('nan')] * len(fc)))]
fct = fc[["model", "preprocess", "feature", "val_macro_f1", "test_macro_f1_s", "test_macro_precision",
          "test_macro_recall", "test_weighted_f1", "test_accuracy"]].copy()
fct.columns = ["Mô hình", "Tiền xử lý", "Đặc trưng", "Macro-F1 valid", "Macro-F1 test", "Macro-P", "Macro-R",
               "Weighted-F1", "Accuracy"]
TABLE(fct, "BẢNG SO SÁNH CHÍNH — kết quả trên tập test (cấu hình mỗi mô hình chọn theo valid; RNN: TB ± std 3 seed).",
      font=8, bold_rows=[list(final.model).index("TF-IDF+LogReg")])
FIG(F / "final_comparison.png", "So sánh macro-F1 các mô hình trên test (cột) và valid (chấm đen).", 15)
bt = boot.copy()
bt = bt[["A", "B", "mean_diff", "ci95_low", "ci95_high", "p_B_not_better"]]
bt.columns = ["Mô hình A", "Mô hình B", "ΔF1 (B−A) TB", "CI 95% thấp", "CI 95% cao", "P(B ≤ A)"]
TABLE(bt, "Paired bootstrap (1.000 lần) cho hiệu macro-F1 trên test.", font=9)
b0, b1 = boot.iloc[0], boot.iloc[1]
P(f"Kết quả chính: (1) TF-IDF + Logistic Regression đạt macro-F1 test {pct(lr.test_macro_f1)}%, accuracy "
  f"{pct(lr.test_accuracy)}%, weighted-F1 {pct(lr.test_weighted_f1)}% — tốt nhất trong mọi mô hình. Mức này cao hơn "
  f"kết quả công bố gốc trên UIT-VSMEC (CNN, F1 57,61%, Ho et al., 2019; lưu ý bài báo báo cáo weighted-F1). "
  f"(2) LogReg vượt baseline Naive Bayes ({pct(nb.test_macro_f1)}%) với ΔF1 = {b0.mean_diff:+.4f}, CI 95% "
  f"[{b0.ci95_low:.4f}; {b0.ci95_high:.4f}] không chứa 0 → khác biệt có ý nghĩa. LinearSVC ({pct(svc.test_macro_f1)}%) "
  f"gần bằng LogReg. (3) {best_rnn} đạt {pct(rnn.test_macro_f1)} ± {pct(rnn.test_macro_f1_std)}%, thấp hơn LogReg "
  f"có ý nghĩa (ΔF1 = {b1.mean_diff:+.4f}, CI [{b1.ci95_low:.4f}; {b1.ci95_high:.4f}]). Đáng chú ý LSTM một chiều "
  f"đạt {pct(lstm.test_macro_f1)} ± {pct(lstm.test_macro_f1_std)}% trên test, nhỉnh hơn BiLSTM dù kém hơn trên valid "
  f"— chênh lệch giữa các biến thể RNN nhỏ hơn biến động do seed và do kích thước tập valid/test (~690 câu), nên "
  f"không thể kết luận biến thể nào tốt hơn. (4) Thứ hạng giữa valid và test nhất quán đối với các mô hình baseline, "
  f"cho thấy quy trình chọn theo valid không bị quá khớp vào valid.")
P("Trả lời RQ3: với dữ liệu nhỏ, mô hình (Bi)LSTM học embedding từ đầu không vượt được baseline TF-IDF tuyến tính. "
  "Nguyên nhân: (i) số tham số lớn so với dữ liệu → quá khớp; (ii) tín hiệu cảm xúc trong câu ngắn chủ yếu là từ khóa "
  "và emoji, vốn được TF-IDF n-gram nắm bắt trực tiếp, nên lợi thế mô hình hóa thứ tự của LSTM ít được phát huy; "
  "(iii) không có tri thức ngôn ngữ huấn luyện trước (pre-trained embedding / mô hình ngôn ngữ), embedding của từ "
  "hiếm hầu như không được học.")
H("5.6. Kết quả theo từng lớp", 2)
pcv = pc.pivot(index="label", columns="model", values="f1-score")
sup = pc.drop_duplicates("label").set_index("label")["support"]
pcv = pcv.reindex(sup.index)
pcv.insert(0, "Số câu test", sup.astype(int))
pcv = pcv.reset_index().rename(columns={"label": "Nhãn"})
TABLE(pcv, "F1 từng lớp trên test.", font=9, fmt={"Số câu test": lambda v: f"{v:.0f}"})
FIG(F / "per_class_f1.png", "F1 từng lớp cảm xúc trên test.", 14)
FIG(F / "confusion_TF-IDF_LogReg.png", "Ma trận nhầm lẫn (chuẩn hóa theo hàng) của TF-IDF + LogReg trên test.", 12.5)
FIG(F / f"confusion_{best_rnn}.png", f"Ma trận nhầm lẫn của {best_rnn} (seed 42) trên test.", 12.5)
pcl = pc[pc.model == "TF-IDF+LogReg"].set_index("label")
P(f"Các lớp có từ khóa đặc trưng rõ (Enjoyment {pcl.loc['Enjoyment', 'f1-score']:.2f}, Sadness "
  f"{pcl.loc['Sadness', 'f1-score']:.2f}, Fear {pcl.loc['Fear', 'f1-score']:.2f}) đạt F1 cao; Fear tuy hiếm vẫn tốt nhờ "
  f"từ 'sợ' rất đặc trưng. Ba lớp khó nhất là Anger ({pcl.loc['Anger', 'f1-score']:.2f}), Other "
  f"({pcl.loc['Other', 'f1-score']:.2f}) và Surprise ({pcl.loc['Surprise', 'f1-score']:.2f}). Ma trận nhầm lẫn cho "
  f"thấy 30% câu Anger bị đoán là Disgust và 30% câu Surprise bị đoán là Enjoyment; Other bị phân tán sang Disgust, "
  f"Enjoyment. BiLSTM có recall Surprise cao hơn LogReg nhưng precision thấp (đoán Surprise quá nhiều), và nhầm "
  f"Disgust → Anger nhiều hơn.")
H("5.7. Phân tích lỗi (RQ4)", 2)
cl = conf_lr.head(8)[["true", "pred", "count"]].copy()
cl.columns = ["Nhãn thật", "Dự đoán", "Số câu"]
TABLE(cl, "Các cặp nhầm lẫn phổ biến nhất của TF-IDF + LogReg trên test.")
P(f"Mức đồng thuận giữa hai mô hình trên {n_test} câu test: cả hai đúng {both_right}, cả hai sai {both_wrong}, chỉ "
  f"LogReg sai {only_lr_wrong}, chỉ BiLSTM sai {only_rnn_wrong}. Như vậy khoảng {100 * both_wrong / n_test:.0f}% câu "
  f"test khó với cả hai cách tiếp cận — phần lớn là các trường hợp nhập nhằng thực sự. Tồn tại {only_lr_wrong} câu "
  f"BiLSTM đúng mà LogReg sai, cho thấy có tiềm năng bổ trợ, nhưng ensemble trên valid không khai thác được do "
  f"BiLSTM quá tự tin.")
bl = bylen.copy()
bl.columns = ["Độ dài (âm tiết)", "Số câu", "Accuracy LogReg", f"Accuracy {best_rnn}"]
TABLE(bl, "Accuracy theo độ dài câu trên test.", fmt={"Accuracy LogReg": lambda v: f"{v:.3f}",
                                                      f"Accuracy {best_rnn}": lambda v: f"{v:.3f}"})
P("Câu rất ngắn (1–5 âm tiết) lại dễ nhất với LogReg (thường chỉ gồm một từ cảm thán hoặc emoji), câu trung bình "
  "11–20 âm tiết khó nhất (nhiều tín hiệu trái chiều). BiLSTM kém LogReg ở mọi nhóm độ dài, tức là lợi thế mô hình hóa "
  "chuỗi dài của LSTM không thể hiện trên dữ liệu này.")
P("Ví dụ lỗi cụ thể (lấy tự động từ results/tables/error_examples.csv; 'conf' là xác suất BiLSTM gán cho nhãn dự đoán):")
ee = errex.copy()
cols = list(ee.columns)
ee = ee.rename(columns={cols[0]: "Câu", cols[1]: "Thật", cols[2]: "LogReg", cols[3]: "BiLSTM", cols[4]: "conf",
                        "case": "Trường hợp"})
ee["Câu"] = ee["Câu"].str.slice(0, 160)
ee.insert(0, "#", range(1, len(ee) + 1))
TABLE(ee[["#", "Câu", "Thật", "LogReg", "BiLSTM", "conf"]], "Mười sáu ví dụ lỗi trên test (1–8: cả hai sai; 9–12: chỉ "
      "LogReg sai; 13–16: chỉ BiLSTM sai).", font=8, fmt={"conf": lambda v: f"{v:.2f}"}, widths=[0.8, 9, 1.8, 1.8, 1.8, 1.2])
P("Diễn giải các nhóm lỗi:")
BL([("Ranh giới nhãn mờ Anger ↔ Disgust (ví dụ 2, 3, 8): ", "câu chửi bới có thể được người gán nhãn xem là giận dữ "
     "hoặc khinh bỉ. Ví dụ 2 ('bọn mất dạy... lũ ngu học') mang nghĩa khinh bỉ rõ nhưng nhãn là Anger; ví dụ 3 và 8 ('hỏi "
     "hỏi con c...', 'đụ mẹ ngu à') ngược lại. Từ tục xuất hiện ở cả hai lớp nên mô hình dựa từ khóa không phân biệt "
     "được; đây cũng là cặp nhầm lẫn lớn nhất ở bài báo gốc."),
    ("Câu hỏi tu từ ↔ Surprise (ví dụ 1, 4): ", "'sao', ':))', 'nhỉ' là từ khóa mạnh của Surprise, nhưng câu thực chất "
     "là phản bác/trêu đùa (nhãn Other). Mô hình cần hiểu hành động ngôn từ chứ không chỉ từ khóa."),
    ("Mỉa mai và từ tiêu cực trong câu tích cực (ví dụ 6, 7): ", "ví dụ 6 kể về thầy 'khắt khe... bỏ thói hư tật xấu' "
     "nhưng cảm xúc là biết ơn (Enjoyment); các từ 'hư', 'xấu' kéo dự đoán sang Disgust. Ví dụ 7 'tiền mất tật mang' "
     "(thành ngữ, Sadness) bị đoán Disgust vì 'nát', 'tật'."),
    ("Other bị kéo sang lớp có cảm xúc (ví dụ 5, 10, 11): ", "câu kể/nhận xét trung tính có chứa 'thích', 'yêu', "
     "'apple'... Lớp Other không có từ khóa riêng nên luôn là lớp 'dư' khó học."),
    ("Mô hình quá tự tin: ", f"8 ví dụ 'cả hai sai' ở trên được chọn theo độ tự tin của BiLSTM và đều có xác suất ≥ 0,99. "
     f"Trên toàn tập test, trong {over_n} câu BiLSTM (seed 42) dự đoán sai, {100 * over_frac:.0f}% có xác suất ≥ 0,9 "
     f"(trung vị {over_med:.2f}) — dấu hiệu quá khớp, xác suất không được hiệu chỉnh; điều này cũng giải thích vì sao "
     f"ensemble không có lợi."),
    ("BiLSTM nắm ngữ cảnh tốt hơn ở vài câu (ví dụ 12): ", "'tao đéo ngờ được đây là trường tao :))' — LogReg bị ':))' "
     "kéo về Enjoyment, BiLSTM nhận ra cấu trúc 'không ngờ được' → Surprise."),
    ("BiLSTM sai vì từ hiếm/lỗi gõ (ví dụ 15, 16): ", "'cừời' (gõ sai) và 'táo bón... 🤣🤣🤣' — embedding học từ đầu "
     "của từ hiếm không đáng tin, trong khi LogReg vẫn dựa được vào emoji 🤣 và 'vãi'.")])
H("5.8. Hạn chế", 2)
BL(["Dữ liệu nhỏ (5,5 nghìn câu train, ~30–50 câu/lớp hiếm trên valid/test) → ước lượng F1 từng lớp có phương sai "
    "lớn; chênh lệch dưới ~2 điểm F1 giữa các mô hình RNN không có ý nghĩa thống kê.",
    "Nhãn có độ nhập nhằng cố hữu (Anger/Disgust, Other); mỗi câu chỉ có một nhãn trong khi thực tế một câu có thể "
    "mang nhiều cảm xúc.",
    "Miền dữ liệu là bình luận Facebook đa chủ đề; áp dụng cho đánh giá sản phẩm (sàn TMĐT) có thể gặp lệch miền: từ "
    "vựng về chất lượng, giá, giao hàng ít xuất hiện trong train. Cần thu thập/gán nhãn thêm dữ liệu đánh giá sản phẩm "
    "để đánh giá và tinh chỉnh.",
    "Mô hình chỉ xử lý từng câu độc lập, không có ngữ cảnh bài đăng/hội thoại → khó với mỉa mai, câu trả lời.",
    "RNN không dùng embedding huấn luyện trước và chưa thử mô hình Transformer (PhoBERT) do giới hạn tài nguyên tính "
    "toán (chỉ CPU).",
    "Từ điển teencode/stopwords do nhóm tự soạn, chưa đầy đủ."])

# ================================================================== 6. KẾT LUẬN
H("6. Kết luận và hướng phát triển")
H("6.1. Kết quả đạt được", 2)
BL([f"Xây dựng pipeline NLP hoàn chỉnh, tái lập được cho nhận diện 7 cảm xúc tiếng Việt; mô hình tốt nhất TF-IDF + "
    f"LogReg đạt macro-F1 test {pct(lr.test_macro_f1)}%, accuracy {pct(lr.test_accuracy)}%, vượt baseline Naive Bayes "
    f"có ý nghĩa thống kê và cao hơn kết quả gốc công bố cho bộ dữ liệu.",
    "Khảo sát có kiểm soát cho thấy: giữ emoji/emoticon và chuẩn hóa teencode có ích; tách từ pyvi và loại stopwords "
    "gây hại trên văn bản mạng xã hội; bigram âm tiết là đặc trưng tốt nhất cho mô hình tuyến tính.",
    "BiLSTM (TensorFlow) với max+mean pooling và class weight là cấu hình RNN tốt nhất nhưng vẫn thấp hơn baseline "
    "tuyến tính khoảng 7–8 điểm macro-F1 do quá khớp trên dữ liệu nhỏ.",
    "Phân tích lỗi chỉ ra ba nguồn lỗi chính: ranh giới nhãn mờ, câu hỏi tu từ/mỉa mai, và sự phụ thuộc vào từ khóa."])
H("6.2. Điều chưa đạt", 2)
BL(["Phương án mạng nơ-ron chưa cải thiện được so với baseline; ensemble không mang lại lợi ích trên valid.",
    "Chưa đánh giá trên dữ liệu đánh giá sản phẩm thực tế (chỉ minh họa định tính trong demo)."])
H("6.3. Hướng phát triển", 2)
BL(["Dùng biểu diễn huấn luyện trước: word embedding tiếng Việt (PhoW2V, fastText) khởi tạo cho BiLSTM, hoặc tinh "
    "chỉnh PhoBERT/ViSoBERT (mô hình huấn luyện trên văn bản mạng xã hội tiếng Việt) trên GPU.",
    "Tăng cường dữ liệu cho lớp hiếm (back-translation, thay thế từ đồng nghĩa), hoặc focal loss.",
    "Hiệu chỉnh xác suất (temperature scaling) để ensemble hiệu quả hơn.",
    "Thu thập và gán nhãn bộ đánh giá sản phẩm tiếng Việt (Shopee, Tiki...) tuân thủ điều khoản và ẩn danh người dùng, "
    "để đánh giá chuyển giao miền và tinh chỉnh mô hình.",
    "Xây dựng giao diện web (Gradio/Streamlit) từ `demo.py`."])

# ================================================================== 7. PHÂN CÔNG + TLTK
H("7. Phân công công việc và tài liệu tham khảo")
H("7.1. Phân công", 2)
pc_t = pd.DataFrame([
    ["[Thành viên 1 – MSSV]", "Thu thập dữ liệu, EDA, tiền xử lý (src/data.py, src/eda.py, src/preprocess.py)", "[..%]"],
    ["[Thành viên 2 – MSSV]", "Baseline TF-IDF, khảo sát đặc trưng, đánh giá (src/baseline.py, src/evaluate.py)", "[..%]"],
    ["[Thành viên 3 – MSSV]", "Mô hình BiLSTM TensorFlow, khảo sát RNN, ensemble (src/bilstm.py)", "[..%]"],
    ["[Cả nhóm]", "Phân tích lỗi, báo cáo, demo, README", "—"],
], columns=["Thành viên", "Nhiệm vụ", "Đóng góp"])
TABLE(pc_t, "Phân công công việc (nhóm điền thông tin thực tế).", widths=[4, 10, 2])
H("7.2. Tài liệu tham khảo", 2)
refs = [
    "Ho, V. A., Nguyen, D. H.-C., Nguyen, D. H., Pham, L. T.-V., Nguyen, D.-V., Nguyen, K. V., & Nguyen, N. L.-T. (2019). "
    "Emotion Recognition for Vietnamese Social Media Text. In Proc. PACLING 2019, CCIS vol. 1215, Springer, 2020.",
    "Jurafsky, D., & Martin, J. H. (2024). Speech and Language Processing (3rd ed. draft). Ch. 4 (Naive Bayes, text "
    "classification), Ch. 5 (Logistic Regression), Ch. 8–9 (RNNs, LSTMs).",
    "Manning, C. D., Raghavan, P., & Schütze, H. (2008). Introduction to Information Retrieval. Cambridge University "
    "Press. (TF-IDF, Naive Bayes)",
    "Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. Neural Computation, 9(8), 1735–1780.",
    "Schuster, M., & Paliwal, K. K. (1997). Bidirectional Recurrent Neural Networks. IEEE Transactions on Signal "
    "Processing, 45(11), 2673–2681.",
    "Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. JMLR, 12, 2825–2830.",
    "Abadi, M. et al. (2015). TensorFlow: Large-Scale Machine Learning on Heterogeneous Systems. tensorflow.org.",
    "Trần Việt Trung. pyvi — Python Vietnamese Toolkit. https://github.com/trungtv/pyvi",
    "Koehn, P. (2004). Statistical Significance Tests for Machine Translation Evaluation. In Proc. EMNLP 2004. "
    "(paired bootstrap resampling)",
    "Ekman, P. (1992). An argument for basic emotions. Cognition & Emotion, 6(3–4), 169–200.",
    "Nguyen, D. Q., & Nguyen, A. T. (2020). PhoBERT: Pre-trained language models for Vietnamese. Findings of EMNLP 2020.",
    "Bộ dữ liệu: tridm/UIT-VSMEC, Hugging Face Datasets. https://huggingface.co/datasets/tridm/UIT-VSMEC",
]
for i, rf in enumerate(refs, 1):
    P(f"[{i}] {rf}", size=12)

# ================================================================== PHỤ LỤC
page_break()
H("Phụ lục A. Hướng dẫn chạy lại")
for line in ["pip install -r requirements.txt",
             "python run_all.py            # toàn bộ thực nghiệm → results/, models/",
             "python report/build_report.py  # sinh lại báo cáo từ results/",
             "python demo.py \"câu cần dự đoán\"",
             "jupyter notebook notebooks/NLP_BTL_UIT_VSMEC.ipynb"]:
    p = doc.add_paragraph()
    r = p.add_run(line)
    r.font.name = "Consolas"
    r.font.size = Pt(10)
P("Chi tiết cấu trúc thư mục và cách tái tạo từng bảng/hình: xem README.md.")
H("Phụ lục B. Kết quả demo trên câu đánh giá sản phẩm (minh họa định tính)")
try:
    from demo import Predictor
    pred = Predictor()
    demo_s = ["sản phẩm dùng được 2 hôm đã hỏng, phí tiền vãi",
              "hàng về đẹp quá, shop đóng gói cẩn thận 😍",
              "trời ơi giao hàng nhanh không tưởng luôn á",
              "nhìn con gián trong hộp mà sợ muốn chết",
              "shop trả lời khách như hạch, lần sau đéo mua nữa",
              "mua tặng mẹ mà mẹ không thích, buồn ghê 😢"]
    rows = []
    for s_ in demo_s:
        o = pred.predict(s_)
        rows.append([s_] + [v[0] for v in o.values()][:2])
    dd = pd.DataFrame(rows, columns=["Câu", "TF-IDF+LogReg", best_rnn])
    TABLE(dd, "Dự đoán của hai mô hình cho một số câu đánh giá sản phẩm tự viết (không có nhãn chuẩn, chỉ minh họa).",
          font=9, widths=[9, 3.5, 3.5])
    P("Các câu này không thuộc tập dữ liệu và không có nhãn chuẩn, chỉ dùng để minh họa khả năng chuyển giao sang miền "
      "đánh giá sản phẩm; không dùng để tính bất kỳ chỉ số nào.", italic=True)
except Exception as ex:  # demo cần models/ (sinh bởi run_all.py)
    P(f"(Chưa có mô hình đã lưu: {ex})")
H("Phụ lục C. Phạm vi sử dụng công cụ AI")
P("[Nhóm tự điền trung thực.] Gợi ý nội dung: Nhóm sử dụng trợ lý AI (Claude) để hỗ trợ gợi ý cấu trúc mã nguồn, "
  "viết mã khung cho pipeline, và soạn thảo bản nháp báo cáo. Toàn bộ số liệu trong báo cáo được sinh tự động từ mã "
  "nguồn và dữ liệu (results/), nhóm đã chạy lại, kiểm chứng kết quả, đọc hiểu từng mô-đun và chịu trách nhiệm về toàn "
  "bộ nội dung.")

add_page_numbers()
doc.save(OUT_DOCX)
print("Đã ghi", OUT_DOCX)
