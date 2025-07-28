import os
import shutil
from tqdm import tqdm
import pyprojroot
import glob

from src.validator import Validator

base_path = pyprojroot.find_root(pyprojroot.has_dir(".git"))

validation_data = base_path / "data/features/validation.pkl"
model_output_dir = base_path / "models"

my_validator = Validator()
my_validator.load_data(validation_data)

best_accuracy = 0
best_model_path = None

for model_file_path in tqdm(glob.glob(os.path.join(model_output_dir, '*.pkl'))):
    my_validator.load_model(model_file_path)
    accuracy, precision, recall, f1_score = my_validator.evaluate_model()
    tqdm.write(f"{model_file_path.split("/")[-1]} -> Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1_score}")
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model_path = model_file_path
        
# ablation study        
features = {
    "1": [1, 1156],
    "2": [1157, 2117],
    "3": [2118, 2133],
    "4": [2134, 2143],
    "5": [2144, 2343],
    "6": [2344, 2545],
}
for model_file_path in tqdm(glob.glob(os.path.join(model_output_dir, 'RF_f*_dNone_n100.pkl'))):
    my_validator.load_model(model_file_path)
    feature_name = model_file_path.split('/')[-1].split('_')[1].replace('f', '')
    r = features[feature_name]
    r[0] -= 1
    accuracy, precision, recall, f1_score = my_validator.evaluate_model(range=r)
    tqdm.write(f"{model_file_path.split('/')[-1]} -> Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1_score}")
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model_path = model_file_path

if best_model_path is None:
    print("[x] No models found for validation. Please check the model dir path.")
else:
    best_model_output_path = os.path.join(model_output_dir, 'best_model.pkl')
    shutil.copy(best_model_path, best_model_output_path)
    print("[o] Validation completed.")
    print(f"[*] The best model: {best_model_path} (accuracy: {best_accuracy}) is saved to {best_model_output_path}")