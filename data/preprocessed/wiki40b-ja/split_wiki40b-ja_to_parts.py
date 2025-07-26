import json
import pyprojroot

base_path = pyprojroot.find_root(pyprojroot.has_dir(".git"))

def split_data(file_path, num_splits, output_prefix):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    items = list(data.items())
    chunk_size = len(items) // num_splits
    for i in range(num_splits):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < num_splits - 1 else len(items)
        chunk = dict(items[start:end])

        output_path = base_path / f"data/preprocessed/wiki40b-ja/parts/{output_prefix}_part{i + 1}.json"
        with open(output_path, "w", encoding="utf-8") as out_file:
            json.dump(chunk, out_file, ensure_ascii=False, indent=2)
        print(f"[o] Saved to {output_path}")

def main():
    splits = [
        {"file": "test", "num_splits": 2},
        {"file": "validation", "num_splits": 2},
        {"file": "train", "num_splits": 40},
    ]
    
    for split in splits:
        file_path = base_path / f"data/preprocessed/wiki40b-ja/wiki40b-ja_{split['file']}.json"
        output_prefix = f"wiki40b-ja_{split['file']}"
        split_data(file_path, split["num_splits"], output_prefix)

if __name__ == "__main__":
    main()