"""Tiền xử lý văn bản tiếng Việt cho bài toán nhận diện cảm xúc.

Mỗi bước là một cờ (flag) để có thể làm khảo sát có kiểm soát (ablation).
Tất cả các bước ở đây là QUY TẮC CỐ ĐỊNH (không học tham số từ dữ liệu), nên áp dụng
cho train/valid/test không gây rò rỉ dữ liệu."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, asdict

from .resources import EMOTICONS, STOPWORDS, TEENCODE

# Khối emoji Unicode chính (mặt cười, biểu tượng, ký hiệu...)
EMOJI_RE = re.compile(
    "[\U0001F1E6-\U0001F1FF\U0001F300-\U0001F5FF\U0001F600-\U0001F64F\U0001F680-\U0001F6FF"
    "\U0001F900-\U0001F9FF\U0001FA70-\U0001FAFF☀-⛿✀-➿❤⭐]"
)
VARIATION_RE = re.compile("[️‍]")          # selector biến thể / zero-width joiner
COMBINING_GAP_RE = re.compile(r"\s+([̀-ͯ])")  # "cu ̃" -> "cũ"
URL_RE = re.compile(r"https?://\S+|www\.\S+")
NUM_RE = re.compile(r"\b\d+([.,]\d+)?\b")
REPEAT_RE = re.compile(r"([^\W\d_])\1{2,}")          # "quáaaa" -> "quáa" (giữ 2 để còn dấu hiệu nhấn mạnh)
PUNCT_REPEAT_RE = re.compile(r"([!?.])\1+")          # "!!!!" -> "!!" là token nhấn mạnh
TOKEN_SPLIT_RE = re.compile(r"(\w+|emo_\w+|[^\w\s])", re.UNICODE)


@dataclass(frozen=True)
class PreprocessConfig:
    name: str = "full"
    normalize: bool = True         # NFC, sửa dấu tổ hợp tách rời, số -> num, tách dấu câu
    lowercase: bool = True
    emoticon: str = "token"        # token (":))" -> emo_cuoi) | remove | none (để nguyên)
    emoji: str = "keep"            # keep | remove
    reduce_repeat: bool = True     # kéo dài ký tự / dấu câu
    teencode: bool = True
    segment: bool = True           # tách từ tiếng Việt (pyvi)
    stopwords: bool = False        # loại stopwords

    def as_dict(self):
        return asdict(self)


# Các cấu hình dùng trong khảo sát tiền xử lý (mỗi dòng bật thêm/tắt đúng 1 yếu tố)
CONFIGS = {
    "P0_raw": PreprocessConfig("P0_raw", normalize=False, lowercase=True, emoticon="none",
                               emoji="keep", reduce_repeat=False, teencode=False,
                               segment=False, stopwords=False),
    "P1_normalize": PreprocessConfig("P1_normalize", teencode=False, segment=False),
    "P2_+teencode": PreprocessConfig("P2_+teencode", segment=False),
    "P3_+segment": PreprocessConfig("P3_+segment"),
    "P4_+stopwords": PreprocessConfig("P4_+stopwords", stopwords=True),
    "P5_-emoji/emoticon": PreprocessConfig("P5_-emoji/emoticon", emoji="remove", emoticon="remove"),
}

_segmenter = None


def _segment(text: str) -> str:
    global _segmenter
    if _segmenter is None:
        from pyvi import ViTokenizer
        _segmenter = ViTokenizer
    return _segmenter.tokenize(text)


def preprocess(text: str, cfg: PreprocessConfig) -> str:
    """Trả về chuỗi các token cách nhau bởi khoảng trắng."""
    if text is None:
        return ""
    t = str(text)
    if cfg.normalize:
        t = COMBINING_GAP_RE.sub(r"\1", t)
        t = unicodedata.normalize("NFC", t)
        t = VARIATION_RE.sub("", t)
    if cfg.lowercase:
        t = t.lower()
    t = URL_RE.sub(" url ", t)
    if cfg.emoticon != "none":
        for pat, rep in EMOTICONS:
            t = re.sub(pat, rep if cfg.emoticon == "token" else " ", t)
    # emoji: tách mỗi emoji thành 1 token riêng hoặc xóa
    if cfg.emoji == "remove":
        t = EMOJI_RE.sub(" ", t)
    else:
        t = EMOJI_RE.sub(lambda m: f" {m.group(0)} ", t)
    if cfg.reduce_repeat:
        t = REPEAT_RE.sub(r"\1\1", t)
        t = PUNCT_REPEAT_RE.sub(r"\1\1", t)
    t = NUM_RE.sub(" num ", t) if cfg.normalize else t

    # tách token thô: chữ/số, token emo_, từng dấu câu/emoji
    tokens = []
    for tok in t.split():
        if tok.startswith("emo_") or EMOJI_RE.fullmatch(tok):
            tokens.append(tok)
        else:
            tokens.extend(TOKEN_SPLIT_RE.findall(tok) if cfg.normalize else [tok])
    if cfg.teencode:
        out = []
        for tok in tokens:
            out.extend(TEENCODE.get(tok, tok).split())
        tokens = out
    t = " ".join(tokens)

    if cfg.segment:
        # pyvi ghép âm tiết thành từ bằng "_" (vd: "thất vọng" -> "thất_vọng")
        protected = {}

        def _protect(m):
            key = f"zzemo{len(protected)}zz"
            protected[key] = m.group(0)
            return key
        t = re.sub(r"emo_\w+", _protect, t)
        t = _segment(t)
        for k, v in protected.items():
            t = t.replace(k, v)
    tokens = t.split()
    if cfg.stopwords:
        tokens = [w for w in tokens if w.replace("_", " ") not in STOPWORDS and w not in STOPWORDS]
    return " ".join(tokens)


def preprocess_series(texts, cfg: PreprocessConfig):
    return [preprocess(x, cfg) for x in texts]
