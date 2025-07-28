import json
import pyprojroot
from tqdm import tqdm

base_path = pyprojroot.find_root(pyprojroot.has_dir(".git"))

def merge_data(split, num):
    data = {}
    for i in tqdm(range(1, num + 1)):
        file_path = base_path / f"data/preprocessed/chatgpt4o/parts/chatgpt4o_{split}_part{i}.json"
        with open(file_path, "r", encoding="utf-8") as f:
            part_data = json.load(f)
            data.update(part_data)
    output_path = base_path / f"data/preprocessed/chatgpt4o/chatgpt4o_{split}.json"
    with open(output_path, "w", encoding="utf-8") as out_file:
        json.dump(data, out_file, ensure_ascii=False, indent=2)
    print(f"[o] Merged {split} data into {output_path}")
    
def main():
    splits = [
        {"file": "test", "num_splits": 2},
        {"file": "validation", "num_splits": 2},
        {"file": "train", "num_splits": 40},
    ]
    
    for split in tqdm(splits):
        merge_data(split["file"], split["num_splits"])

if __name__ == "__main__":
    main()