import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# === 6. Load Train & Test Dataset ===
train_graphs = torch.load('train_graphs.pt')  # Đã cân bằng bằng SMOTE
test_graphs = torch.load('test_graphs.pt')    # Giữ nguyên, không SMOTE

# Chuyển đổi dữ liệu PyG thành numpy array
def convert_to_numpy(graphs):
    features, labels = [], []
    for data in graphs:
        features.append(data.x.squeeze(0).numpy())  # Chuyển tensor thành numpy
        labels.append(data.y.item())
    return np.array(features), np.array(labels)

# Chuyển train và test thành numpy
X_train, y_train = convert_to_numpy(train_graphs)
X_test, y_test = convert_to_numpy(test_graphs)

# === 7. Train Random Forest ===
clf = RandomForestClassifier(n_estimators=100,max_depth=10,max_features="sqrt",max_samples=0.8, random_state=42)
clf.fit(X_train, y_train)

# === 8. Evaluate Model ===
y_pred = clf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
