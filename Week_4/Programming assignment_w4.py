import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix
import seaborn as sns

# --- Read XML and prepare datasets ---
file_path = 'O-A0038-003.xml'
tree = ET.parse(file_path)
root = tree.getroot()
ns = {'ns': 'urn:cwa:gov:tw:cwacommon:0.1'}

raw_data_str = content_node.text.replace('\n', ' ').replace('\r', ' ')
raw_data_list = []
for part in raw_data_str.split(','):
    sub_parts = part.strip().split()
    raw_data_list.extend(sub_parts)
raw_data_list = list(filter(lambda x: x != '', raw_data_list))

raw_data = np.array([float(val) for val in raw_data_list])

dim_lon = 67
dim_lat = 120
data_grid = raw_data.reshape((dim_lat, dim_lon))

lon_start = 120.00
lat_start = 21.88
resolution = 0.03
lons = lon_start + np.arange(dim_lon) * resolution
lats = lat_start + np.arange(dim_lat) * resolution

classification_records = []
regression_records = []

for i in range(dim_lat):
    for j in range(dim_lon):
        temp_val = data_grid[i, j]
        label = 0 if temp_val == -999.0 else 1
        classification_records.append((lons[j], lats[i], label))
        if label == 1:
            regression_records.append((lons[j], lats[i], temp_val))

classification_df = pd.DataFrame(classification_records, columns=['Longitude', 'Latitude', 'Label'])
regression_df = pd.DataFrame(regression_records, columns=['Longitude', 'Latitude', 'Value'])

classification_df.to_csv('classification_dataset.csv', index=False)
regression_df.to_csv('regression_dataset.csv', index=False)

# --- Classification result black-and-white map ---
label_grid = classification_df['Label'].values.reshape((dim_lat, dim_lon))
plt.figure(figsize=(8, 10))
plt.imshow(label_grid, cmap='gray_r', origin='lower',
           extent=[lons.min(), lons.max(), lats.min(), lats.max()])
plt.title('Data Validity Map (0=White, 1=Black)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

# --- Regression result temperature heatmap ---
temp_grid = np.full((dim_lat, dim_lon), np.nan)
for idx, row in regression_df.iterrows():
    lon_idx = int(round((row['Longitude'] - lon_start) / resolution))
    lat_idx = int(round((row['Latitude'] - lat_start) / resolution))
    temp_grid[lat_idx, lon_idx] = row['Value']

plt.figure(figsize=(8, 10))
c = plt.imshow(temp_grid, origin='lower',
               extent=[lons.min(), lons.max(), lats.min(), lats.max()],
               cmap='jet')
plt.colorbar(c, label='Temperature (°C)')
plt.title('Temperature Heatmap (Valid Cells Only)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

# --- Data split: 70% train, 15% validation, 15% test ---
def split_three_way(X, y, train_size=0.7, val_size=0.15, test_size=0.15, random_state=42):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(1-train_size), random_state=random_state, stratify=y if len(np.unique(y))<=10 else None)
    val_ratio = val_size / (val_size + test_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1-val_ratio), random_state=random_state, stratify=y_temp if len(np.unique(y_temp))<=10 else None)
    return X_train, X_val, X_test, y_train, y_val, y_test

# Classification data split
X_class = classification_df[['Longitude', 'Latitude']]
y_class = classification_df['Label']
Xc_train, Xc_val, Xc_test, yc_train, yc_val, yc_test = split_three_way(X_class, y_class)

# Regression data split
X_reg = regression_df[['Longitude', 'Latitude']]
y_reg = regression_df['Value']
Xr_train, Xr_val, Xr_test, yr_train, yr_val, yr_test = split_three_way(X_reg, y_reg)

# --- Classification model training (Random Forest + Standardization) ---
clf = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=100, random_state=42))
clf.fit(Xc_train, yc_train)
yc_val_pred = clf.predict(Xc_val)
yc_test_pred = clf.predict(Xc_test)

acc_val = accuracy_score(yc_val, yc_val_pred)
acc_test = accuracy_score(yc_test, yc_test_pred)
print(f'Classification model validation accuracy: {acc_val:.4f}')
print(f'Classification model test accuracy: {acc_test:.4f}')

# --- Regression model training (Random Forest Regression) ---
regressor_rf = RandomForestRegressor(n_estimators=100, random_state=42)
regressor_rf.fit(Xr_train, yr_train)

yr_val_pred = regressor_rf.predict(Xr_val)
yr_test_pred = regressor_rf.predict(Xr_test)

mse_val = mean_squared_error(yr_val, yr_val_pred)
mse_test = mean_squared_error(yr_test, yr_test_pred)
print(f'Random Forest regression model validation MSE: {mse_val:.4f}')
print(f'Random Forest regression model test MSE: {mse_test:.4f}')

# --- Visualization: Classification confusion matrix (validation set) ---
cm_val = confusion_matrix(yc_val, yc_val_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm_val, annot=True, fmt='d', cmap='Blues')
plt.title('Validation Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# --- Visualization: Regression predictions vs true values (validation set) ---
plt.figure(figsize=(8, 6))
plt.scatter(yr_val, yr_val_pred, alpha=0.5, color='green')
plt.plot([yr_val.min(), yr_val.max()], [yr_val.min(), yr_val.max()], 'r--')
plt.title('Random Forest Regression: Validation Predictions vs True Values')
plt.xlabel('True Temperature')
plt.ylabel('Predicted Temperature')
plt.show()
