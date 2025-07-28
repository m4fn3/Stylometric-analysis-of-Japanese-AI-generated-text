import os
from tqdm import tqdm

import pyprojroot

from src.trainer import Trainer

base_path = pyprojroot.find_root(pyprojroot.has_dir(".git"))

hyperparameters = {
    "max_depth": [None, 10, 15, 5],
    "n_estimators": [50, 100, 150]
}
train_data = base_path / "data/features/train.pkl"
model_output_dir = base_path / "models"

my_trainer = Trainer()
my_trainer.load_data(train_data)

os.makedirs(model_output_dir, exist_ok=True)

# 学習
for h in tqdm(hyperparameters["max_depth"]):
    model_output_path = os.path.join(model_output_dir, f"RF_d{h}_n100.pkl")
    my_trainer.train(h, 100)
    my_trainer.dump_model(model_output_path)
    print(f"[o] Model saved to {model_output_path}")

print(f"[o] Training completed and models are saved to {model_output_dir}.")