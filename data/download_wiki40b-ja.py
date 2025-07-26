from datasets import load_dataset
import pyprojroot

base_path = pyprojroot.find_root(pyprojroot.has_dir(".git"))

wiki = load_dataset("range3/wiki40b-ja")
for split in ["train", "validation", "test"]:
    wiki[split].to_json(base_path / f"data/raw/wiki40b-ja_{split}.json", force_ascii=False)