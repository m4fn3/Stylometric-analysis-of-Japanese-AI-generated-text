import sklearn.ensemble as Ensemble
import pickle

class Trainer:
    def __init__(self):
        self.model = None
        self.X = None; self.y = None
        
    def load_data(self, file_path: str) -> None: # 学習用データセットの読み込み
        with open(file_path, 'rb') as f:
            _, self.y, self.X = pickle.load(f) # Xがデータ、Yがラベル
    
    def dump_model(self, file_path: str) -> None: # 学習済モデルの保存
        with open(file_path, 'wb') as f:
            pickle.dump(self.model, f)
            
    def train(self, max_depth: float, n_estimators: float, range = None) -> None: # range: 特徴量の範囲指定
        X = self.X.copy()
        if range is not None:
            X = X[:,range[0]:range[1]]
        self.model = Ensemble.RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators)
        self.model.fit(X, self.y)