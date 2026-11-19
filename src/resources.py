"""Tài nguyên ngôn ngữ nhỏ dùng cho tiền xử lý.

Các từ điển được nhóm tự soạn từ quan sát dữ liệu train (không nhìn tập test)."""

# Teencode / viết tắt phổ biến trong bình luận mạng xã hội -> dạng chuẩn.
# Chỉ thay khi token đứng riêng (khớp nguyên token), để tránh thay nhầm trong từ.
TEENCODE = {
    "ko": "không", "k": "không", "kh": "không", "khg": "không", "kg": "không",
    "hok": "không", "hong": "không", "hông": "không", "khum": "không", "hem": "không",
    "dc": "được", "đc": "được", "dk": "được", "đk": "được",
    "vs": "với", "mk": "mình", "mik": "mình", "mng": "mọi người", "mn": "mọi người",
    "ns": "nói", "j": "gì", "gi": "gì", "z": "vậy", "v": "vậy", "zậy": "vậy", "dzậy": "vậy",
    "bt": "biết", "bik": "biết", "bjk": "biết", "biet": "biết",
    "r": "rồi", "rùi": "rồi", "ròi": "rồi",
    "t": "tao", "m": "mày", "tui": "tôi",
    "ng": "người", "nguoi": "người",
    "cx": "cũng", "cg": "cũng", "cũg": "cũng",
    "lm": "làm", "wa": "quá", "qá": "quá", "qa": "quá",
    "ntn": "như thế nào", "nt": "nhắn tin", "trc": "trước", "sn": "sinh nhật",
    "a": "anh", "e": "em", "ak": "à", "ah": "à", "uk": "ừ", "uh": "ừ", "ukm": "ừm",
    "thik": "thích", "iu": "yêu", "yeu": "yêu", "hum": "hôm", "nay": "nay",
    "ms": "mới", "mún": "muốn", "lun": "luôn", "nhìu": "nhiều", "bít": "biết",
    "đt": "điện thoại", "dt": "điện thoại", "fb": "facebook", "ok": "ok", "oke": "ok", "okie": "ok",
    "csong": "cuộc sống", "cmt": "bình luận", "ad": "admin", "sp": "sản phẩm",
}

# Emoticon dạng ký tự -> token đặc trưng (giữ tín hiệu cảm xúc thay vì xóa như dấu câu).
# Thứ tự quan trọng: mẫu dài trước.
EMOTICONS = [
    (r"[:;=]\s?\)+|[:;=]\s?\]+|\^\s?\^|\^_\^|=\)+", " emo_cuoi "),
    (r"[:;=]\s?d+\b|x\s?d+\b", " emo_cuoi_lon "),
    (r"[:;=]\s?\(+|t\s?_\s?t|=\(+|:'\(", " emo_buon "),
    (r"[:;=]\s?p+\b", " emo_le_luoi "),
    (r"[:;=]\s?o+\b|o\s?_\s?o|o\.o", " emo_ngac_nhien "),
    (r"-\s?_+\s?-|=\s?\.\s?=", " emo_chan "),
    (r"<3+", " emo_tim "),
    (r"[:;=]\s?\|+", " emo_cam "),
]

# Danh sách stopwords tối giản (hư từ). CHỦ Ý không đưa các từ mang tín hiệu cảm xúc /
# phủ định / mức độ như "không", "chẳng", "quá", "rất", "lắm", "đéo", "vãi" vào đây.
STOPWORDS = {
    "là", "thì", "mà", "của", "và", "có", "cho", "với", "để", "này", "đó", "kia", "ấy",
    "những", "các", "cái", "con", "một", "được", "bị", "đã", "đang", "sẽ", "cũng", "nữa",
    "như", "khi", "lúc", "ở", "tại", "trong", "ngoài", "trên", "dưới", "từ", "đến", "ra",
    "vào", "lên", "xuống", "nên", "vì", "do", "nếu", "thế", "vậy", "rằng", "lại", "đi",
    ".", ",", "?", "!", "...", "..", ":", ";", "(", ")", "-", '"', "'",
}
