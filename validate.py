import os
import shutil
from tqdm import tqdm
import pyprojroot

from src.validator import Validator

base_path = pyprojroot.find_root(pyprojroot.has_dir(".git"))

validation_data = base_path / "data/features/validation.json"
model_output_dir = base_path / "models"

my_validator = Validator()
my_validator.load_data(validation_data)

best_accuracy = 0
best_model_path = None

for model_file in tqdm(os.listdir(model_output_dir)):
    model_file_path = os.path.join(model_output_dir, model_file)
    my_validator.load_model(model_file_path)
    tqdm.write(f"Validating: {model_file_path}")
    accuracy, precision, recall = my_validator.evaluate_model()
    tqdm.write(f"-> Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}")
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model_path = model_file_path

if best_model_path is None:
    print("No models found for validation. Please check the model dir path.")
else:
    best_model_output_path = os.path.join(model_output_dir, 'best_model.pkl')
    shutil.copy(best_model_path, best_model_output_path)
    print("Validation completed.")
    print(f"The best model: {best_model_path} (accuracy: {best_accuracy}) is saved to {best_model_output_path}")