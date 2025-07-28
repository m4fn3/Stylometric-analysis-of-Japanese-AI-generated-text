import os
from tqdm import tqdm

import pyprojroot

from src.trainer import Trainer

base_path = pyprojroot.find_root(pyprojroot.has_dir(".git"))

hyperparameters = [1,2,3,4,5]
train_data = base_path / "data/features/train.json"
model_output_dir = base_path / "models"

my_trainer = Trainer()
my_trainer.load_data(train_data)

os.makedirs(model_output_dir, exist_ok=True)

# 学習
for h in tqdm(hyperparameters):
    model_output_path = os.path.join(model_output_dir, f"RF_{h}.pkl")
    my_trainer.train(h)
    my_trainer.dump_model(model_output_path)

print(f"Training completed and models are saved to {model_output_dir}.")