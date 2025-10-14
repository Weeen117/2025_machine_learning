import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

# --- Read and process XML ---
file_path = 'O-A0038-003.xml'
tree = ET.parse(file_path)
root = tree.getroot()
ns = {'ns': 'urn:cwa:gov:tw:cwacommon:0.1'}
content_node = root.find('.//ns:Content', ns)
if content_node is None:
    raise ValueError('Content node not found')

raw_data_str = content_node.text.replace('\n', ' ').replace('\r', ' ')
raw_data_list = []
for part in raw_data_str.split(','):
    sub_parts = part.strip().split()
    raw_data_list.extend(sub_parts)
raw_data_list = list(filter(lambda x: x != '', raw_data_list))
raw_data = np.array([float(val) for val in raw_data_list])

dim_lon, dim_lat = 67, 120
data_grid = raw_data.reshape((dim_lat, dim_lon))
lon_start, lat_start, resolution = 120.00, 21.88, 0.03
lons = lon_start + np.arange(dim_lon) * resolution
lats = lat_start + np.arange(dim_lat) * resolution

# --- Feature Engineering ---
classification_records = []
for i in range(dim_lat):
    for j in range(dim_lon):
        temp_val = data_grid[i, j]
        label = 0 if temp_val == -999.0 else 1       
        lon = lons[j]
        lat = lats[i]
        features = {
            'Longitude': lon,
            'Latitude': lat,
            'Lon2': lon ** 2,
            'Lat2': lat ** 2,
            'LonLat': lon * lat,
        }
        center_lon, center_lat = 121.0, 23.5
        features['DistFromCenter'] = np.sqrt((lon - center_lon)**2 + (lat - center_lat)**2)
        features['DistFromWestEdge'] = lon - lon_start
        features['DistFromEastEdge'] = (lon_start + (dim_lon - 1) * resolution) - lon
        features['DistFromSouthEdge'] = lat - lat_start
        features['DistFromNorthEdge'] = (lat_start + (dim_lat - 1) * resolution) - lat
        neighbors = []
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                ni, nj = i + di, j + dj
                if 0 <= ni < dim_lat and 0 <= nj < dim_lon:
                    neighbors.append(data_grid[ni, nj])
        if neighbors:
            features['NeighborMean'] = np.mean(neighbors)
            features['NeighborStd'] = np.std(neighbors)
            features['NumValidNeighbors'] = np.sum(np.array(neighbors) != -999.0)
        else:
            features['NeighborMean'] = 0
            features['NeighborStd'] = 0
            features['NumValidNeighbors'] = 0
        features['Label'] = label
        classification_records.append(features)

classification_df = pd.DataFrame(classification_records)

# --- Gaussian Discriminant Analysis ---
class GaussianDiscriminantAnalysis:
    def __init__(self):
        self.phi = None
        self.mu0 = None
        self.mu1 = None
        self.sigma = None
    def fit(self, X, y):
        n = len(y)
        self.phi = np.mean(y)
        X0 = X[y == 0]
        X1 = X[y == 1]
        self.mu0 = np.mean(X0, axis=0)
        self.mu1 = np.mean(X1, axis=0)
        X0_centered = X0 - self.mu0
        X1_centered = X1 - self.mu1
        self.sigma = (X0_centered.T @ X0_centered + X1_centered.T @ X1_centered) / n
        self.sigma += 1e-6 * np.eye(self.sigma.shape[0])
    def predict_proba(self, X):
        sigma_inv = np.linalg.inv(self.sigma)
        def discriminant(x, mu, prior):
            diff = x - mu
            return -0.5 * diff @ sigma_inv @ diff.T + np.log(prior)
        probs = []
        for x in X:
            d0 = discriminant(x, self.mu0, 1 - self.phi)
            d1 = discriminant(x, self.mu1, self.phi)
            prob1 = 1 / (1 + np.exp(d0 - d1))
            probs.append(prob1)
        return np.array(probs)
    def predict(self, X):
        probs = self.predict_proba(X)
        return (probs >= 0.5).astype(int)

# --- Data Split & Preprocessing ---
feature_cols = [col for col in classification_df.columns if col != 'Label']
X = classification_df[feature_cols].values
y = classification_df['Label'].values

def manual_train_test_split(X, y, train_size=0.7, val_size=0.15, random_state=42):
    np.random.seed(random_state)
    n = len(y)
    indices = np.arange(n)
    np.random.shuffle(indices)
    train_end = int(n * train_size)
    val_end = train_end + int(n * val_size)
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    return (X[train_idx], X[val_idx], X[test_idx],
            y[train_idx], y[val_idx], y[test_idx])

class ManualScaler:
    def __init__(self):
        self.mean = None
        self.std = None
    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        self.std[self.std == 0] = 1
        return self
    def transform(self, X):
        return (X - self.mean) / self.std
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)

def confusion_matrix(y_true, y_pred):
    classes = np.unique(np.concatenate([y_true, y_pred]))
    n_classes = len(classes)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for true_label, pred_label in zip(y_true, y_pred):
        cm[int(true_label), int(pred_label)] += 1
    return cm

def classification_report(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    report = []
    for i in range(len(cm)):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        tn = np.sum(cm) - tp - fp - fn
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        support = tp + fn
        report.append({
            'class': i,
            'precision': precision,
            'recall': recall,
            'f1-score': f1,
            'support': support
        })
    return report

X_train, X_val, X_test, y_train, y_val, y_test = manual_train_test_split(
    X, y, train_size=0.7, val_size=0.15, random_state=42
)

scaler = ManualScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

gda = GaussianDiscriminantAnalysis()
gda.fit(X_train_scaled, y_train)

y_pred_train = gda.predict(X_train_scaled)
y_pred_val = gda.predict(X_val_scaled)
y_pred_test = gda.predict(X_test_scaled)

train_acc = accuracy_score(y_train, y_pred_train)
val_acc = accuracy_score(y_val, y_pred_val)
test_acc = accuracy_score(y_test, y_pred_test)

print(f"\nOverall Accuracy: {test_acc:.4f}")

# --- Visualization without color bar, using legend ---
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Confusion Matrix
cm = confusion_matrix(y_test, y_pred_test)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0], cbar_kws={'label': 'Count'})
axes[0, 0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Predicted Label')
axes[0, 0].set_ylabel('True Label')

# 定義 label 與顏色
label_colors = {0: '#3182bd', 1: '#de2d26'}
legend_patches = [mpatches.Patch(color=color, label=f'Label {lbl}') for lbl, color in label_colors.items()]
cmap = ListedColormap([label_colors[0], label_colors[1]])

# 2. True Labels
true_grid = y.reshape((dim_lat, dim_lon))
axes[0, 1].imshow(true_grid, cmap=cmap, origin='lower',
                  extent=[lons[0], lons[-1], lats[0], lats[-1]], aspect='auto')
axes[0, 1].set_title('True Labels', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Longitude')
axes[0, 1].set_ylabel('Latitude')
axes[0, 1].legend(handles=legend_patches, loc='upper right')

# 3. GDA Predictions
predicted_labels = gda.predict(scaler.transform(X))
pred_grid = predicted_labels.reshape((dim_lat, dim_lon))
axes[1, 0].imshow(pred_grid, cmap=cmap, origin='lower',
                  extent=[lons[0], lons[-1], lats[0], lats[-1]], aspect='auto')
axes[1, 0].set_title(f'GDA Predictions (Acc: {test_acc:.3f})', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Longitude')
axes[1, 0].set_ylabel('Latitude')
axes[1, 0].legend(handles=legend_patches, loc='upper right')

# 4. Prediction Errors
error_colors = {0: 'green', 1: 'red'}
error_grid = (true_grid != pred_grid).astype(int)
error_cmap = ListedColormap([error_colors[0], error_colors[1]])
error_patches = [mpatches.Patch(color=color, label=label) for label, color in zip(['Correct', 'Error'], error_colors.values())]
axes[1, 1].imshow(error_grid, cmap=error_cmap, origin='lower',
                  extent=[lons[0], lons[-1], lats[0], lats[-1]], aspect='auto')
axes[1, 1].set_title('Prediction Errors\n(Red=Error, Green=Correct)', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Longitude')
axes[1, 1].set_ylabel('Latitude')
axes[1, 1].legend(handles=error_patches, loc='upper right')

plt.suptitle('Gaussian Discriminant Analysis Results', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()
