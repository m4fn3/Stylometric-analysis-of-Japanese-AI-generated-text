import matplotlib.pyplot as plt
import numpy as np
import pyprojroot
import pickle
import json
import japanize_matplotlib

base_path = pyprojroot.find_root(pyprojroot.has_dir(".git"))

model_path = base_path / "models/best_model.pkl"
with open(model_path, 'rb') as f:
    model = pickle.load(f)
    
labels = []
with open(base_path / "src/misc/feature_names.json", 'r') as f:
    labels = json.load(f)

# その値の降順に並べ替え、上位n個を取得
n = 10  # 上位n個の特徴量を取得
sorted_indices = np.argsort(model.feature_importances_)[::-1][:n]
top_labels = [labels[i] for i in sorted_indices]
top_importances = model.feature_importances_[sorted_indices]
print(top_labels)

# plot
plt.figure(figsize=(8, 3.5)) # 30: 10,6 
plt.rcParams['font.size'] = 11
plt.barh(y=range(len(top_importances)), width=top_importances, color="skyblue", height=0.5)
plt.yticks(ticks=range(len(top_labels)), labels=top_labels)
plt.xlabel("特徴量重要度")
plt.ylabel("特徴量")
# plt.title(f"Top {n} Feature Importance")
plt.subplots_adjust(left=0.45, top=0.95, right=0.9, bottom=0.1)
plt.tight_layout() # 余白を最小限に
plt.gca().invert_yaxis()  # 上位が上に来るように反転
plt.show()