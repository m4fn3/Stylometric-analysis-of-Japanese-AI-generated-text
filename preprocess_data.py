import json
import re
from tqdm import tqdm
import pyprojroot

from src.preprocessor import preprocess_text

base_path = pyprojroot.find_root(pyprojroot.has_dir(".git"))

length = 400

for split in tqdm(["train", "validation", "test"]):
    data = {} # {title: text}
    with open(base_path / f"data/raw/wiki40b-ja_{split}.json", "r") as f:
        for line in f:
            meta = json.loads(line)
            content = meta["text"]
            content = content.replace("_NEWLINE_", "")
            title_match = re.search(r"_START_ARTICLE_(.+?)_START_", content, re.DOTALL)
            if not title_match:
                continue
            title = title_match.group(1).strip()
            # text
            text_match = re.findall(r'_START_PARAGRAPH_(.+?)(?=_START_SECTION_|$)', content, re.DOTALL)
            text = "".join(text_match)
            # preprocess
            text = preprocess_text(text, length)
            if len(text) >= length:
                data[title] = text
    with open(base_path / f"data/preprocessed/wiki40b-ja/wiki40b-ja_{split}.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
            