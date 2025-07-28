import json
from tqdm import tqdm
import pyprojroot
import numpy as np
import pickle
import spacy

from src.feature_extractor import FeatureExtractor

base_path = pyprojroot.find_root(pyprojroot.has_dir(".git"))

nlp = spacy.load('ja_ginza')

def build_features(split):
    titles = []
    labels = [] # human: 0, AI: 1
    vectors = []
    chatgpt4o_path = base_path / f"data/preprocessed/chatgpt4o/chatgpt4o_{split}.json"
    wiki40bja_path = base_path / f"data/preprocessed/wiki40b-ja/wiki40b-ja_{split}.json"
    with open(chatgpt4o_path, "r", encoding="utf-8") as f:
        chatgpt4o_data = json.load(f)
    with open(wiki40bja_path, "r", encoding="utf-8") as f:
        wiki40bja_data = json.load(f)
    for key in tqdm(chatgpt4o_data, desc="chatgpt4o"):
        titles.append(key)
        labels.append(1)
        vec = FeatureExtractor(chatgpt4o_data[key], nlp).extract_future_vector()
        vectors.append(vec)
    for key in tqdm(wiki40bja_data, desc="wiki40b-ja"):
        titles.append(key)
        labels.append(0)
        vec = FeatureExtractor(wiki40bja_data[key], nlp).extract_future_vector()
        vectors.append(vec)
    with open(base_path / f"data/features/{split}.json", "wb") as f:
        pickle.dump(
            (
                np.array(titles),
                np.array(labels),
                np.array(vectors)
            )
        )
    tqdm.write(f"[*] Future extraction for {split} completed. ")

splits = ["train", "validation", "test"]
for split in tqdm(splits):
    build_features(split)