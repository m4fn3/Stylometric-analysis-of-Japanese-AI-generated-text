import neologdn
import unicodedata

def preprocess_text(text: str, length: int) -> str:
    # 正規化
    text = neologdn.normalize(text)
    text = unicodedata.normalize("NFKC", text)
    # 長さを制限
    sentences = text.split("。")
    res = ""
    for sentence in sentences:
        res += sentence + "。"
        if len(res) >= length:
            break
    return res