import os
from tqdm import tqdm

import pyprojroot

from src.trainer import Trainer

base_path = pyprojroot.find_root(pyprojroot.has_dir(".git"))

hyperparameters = {
    "max_depth": [None, 10, 15, 5],
    "n_estimators": [50, 100, 150]
}
features = {
    "1": [1, 1156],
    "2": [1157, 2117],
    "3": [2118, 2133],
    "4": [2134, 2143],
    "5": [2144, 2343],
    "6": [2344, 2545],
}
train_data = base_path / "data/features/train.pkl"
model_output_dir = base_path / "models"

my_trainer = Trainer()
my_trainer.load_data(train_data)

os.makedirs(model_output_dir, exist_ok=True)

# 学習
for n in tqdm(hyperparameters["n_estimators"]):
    for d in tqdm(hyperparameters["max_depth"]):
        model_output_path = os.path.join(model_output_dir, f"RF_d{d}_n{n}.pkl")
        my_trainer.train(d, n)
        my_trainer.dump_model(model_output_path)
        print(f"[o] Model saved to {model_output_path}")
    
# Ablation study: 特徴量ベクトルの一部分のみを利用して学習
for f, r in tqdm(features.items()):
    model_output_path = os.path.join(model_output_dir, f"abltaion/RF_f{f}_dNone_n100.pkl")
    my_trainer.train(None, 100, range=[r[0]-1, r[1]])
    my_trainer.dump_model(model_output_path)
    print(f"[o] Model saved to {model_output_path}")

print(f"[o] Training completed and models are saved to {model_output_dir}.")