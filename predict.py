import os
import numpy as np
import pyprojroot

from src.predictor import Predictor

base_path = pyprojroot.find_root(pyprojroot.has_dir(".git"))

test_data = base_path / "data/features/test.json"
best_model_path = base_path / "models/best_model.pkl"

my_predictor = Predictor()
my_predictor.load_data(test_data)
my_predictor.load_model(best_model_path)

predictions, labels, titles = my_predictor.predict()

# 結果の出力
with open("result.txt", "w") as f:
    for i, prediction in enumerate(predictions):
        f.write(f"判定結果: {prediction} | 正解: {labels[i]} | 元の文章: {titles[i]}\n")