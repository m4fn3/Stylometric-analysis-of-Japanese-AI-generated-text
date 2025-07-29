import pyprojroot
import pickle
import numpy as np

base_path = pyprojroot.find_root(pyprojroot.has_dir(".git"))

path = base_path / "data/features/train.pkl"

with open(path, "rb") as f:
    s, y, X = pickle.load(f)
    
# n番目の特徴量の値を分析
feature_num = 2141 # <- 変数
feature_num -= 1 # indexに変換
chatgpt4o = []
wiki40b = []
for i in range(len(y)):
    if y[i] == 1:
        chatgpt4o.append(X[i, feature_num])
    else:
        wiki40b.append(X[i, feature_num])
        
# 平均値を計算
print(np.mean(chatgpt4o), np.mean(wiki40b))