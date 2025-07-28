import sklearn.metrics as metrics
import pickle

class Validator:
    def __init__(self):
        self.model = None
        self.X = None; self.y = None

    def load_data(self, file_path: str) -> None: # 評価用データセットの読み込み
        with open(file_path, 'rb') as f:
            _, self.y, self.X = pickle.load(f) # Xがデータ、Yがラベル
            
    def load_model(self, file_path: str) -> None: # 学習済モデルの読み込み
        with open(file_path, 'rb') as f:
            self.model = pickle.load(f)
            
    def evaluate_model(self) -> tuple[float, float, float, float]:
        y_pred = self.model.predict(self.X)
        accuracy = metrics.accuracy_score(self.y, y_pred)
        precision = metrics.precision_score(self.y, y_pred, zero_division=0, average="macro")
        recall = metrics.recall_score(self.y, y_pred, zero_division=0, average="macro")
        f1_score = metrics.f1_score(self.y, y_pred, zero_division=0, average="macro")
        return accuracy, precision, recall, f1_score