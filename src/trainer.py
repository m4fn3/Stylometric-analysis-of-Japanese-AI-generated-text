import sklearn.svm as SVM
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
            
    def train(self, hyperparameter: float) -> None:
        self.model = Ensemble.RandomForestClassifier(max_depth=hyperparameter)
        self.model.fit(self.X, self.y)