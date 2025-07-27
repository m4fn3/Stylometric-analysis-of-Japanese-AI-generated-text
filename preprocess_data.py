import json
import re
from tqdm import tqdm
import pyprojroot

from src.preprocessor import preprocess_text

base_path = pyprojroot.find_root(pyprojroot.has_dir(".git"))

# mode = "wiki40b-ja"
mode = "chatgpt4o"
length = 400

if mode == "wiki40b-ja":
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
            
elif mode == "chatgpt4o":
    for split in tqdm(["test_part1", "test_part2", "validation_part1", "validation_part2", "train_part1", "train_part2", "train_part3", "train_part4", "train_part5", "train_part6", "train_part7", "train_part8", "train_part9", "train_part10", "train_part11", "train_part12", "train_part13", "train_part14", "train_part15", "train_part16", "train_part17", "train_part18", "train_part19", "train_part20", "train_part21", "train_part22", "train_part23", "train_part24", "train_part25", "train_part26", "train_part27", "train_part28", "train_part29", "train_part30", "train_part31", "train_part32", "train_part33", "train_part34", "train_part35", "train_part36", "train_part37", "train_part38", "train_part39", "train_part40"]):
        data = {} # {title: text}
        with open(base_path / f"data/raw/chatgpt4o/parts/chatgpt4o_{split}.json", "r", encoding="utf-8") as f:
            org = json.load(f)
            for title, text in tqdm(org.items()):
                if "502 Bad Gateway\nUnable to reach the origin service. The service may be down or it may not be responding to traffic from cloudflared\n" in text:
                    continue # エラーを含むものを除外
                text = preprocess_text(text, length)
                if len(text) < length-100:
                    continue # 短すぎる文章を除外
                data[title] = text
        with open(base_path / f"data/preprocessed/chatgpt4o/parts/chatgpt4o_{split}.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)