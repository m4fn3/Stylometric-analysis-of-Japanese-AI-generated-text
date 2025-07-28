import pickle

class Predictor:
    def __init__(self):
        self.model = None
        self.X = None; self.y = None; self.s = None

    def load_data(self, file_path: str) -> None: # テスト用データセットの読み込み
        with open(file_path, 'rb') as f:
            self.s, self.y, self.X = pickle.load(f) # Xがデータ、Yがラベル
            
    def load_model(self, file_path: str) -> None: # 学習済モデルの読み込み
        with open(file_path, 'rb') as f:
            self.model = pickle.load(f)
            
    def predict(self) -> tuple[list, list, list]:
        y_pred = self.model.predict(self.X)
        return y_pred.tolist(), self.y, self.s # 予測ラベル, 正解ラベル, タイトル
            