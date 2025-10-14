import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

# Load datasets
classification_df = pd.read_csv('classification_dataset.csv')
regression_df = pd.read_csv('regression_dataset.csv')

# Grid parameters
lon_start = 120.00
lat_start = 21.88
resolution = 0.03
dim_lon = 67
dim_lat = 120
lons = lon_start + np.arange(dim_lon) * resolution
lats = lat_start + np.arange(dim_lat) * resolution

# Data split: 70% train, 15% validation, 15% test
def split_three_way(X, y, train_size=0.7, val_size=0.15, test_size=0.15, random_state=42):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(1-train_size), random_state=random_state, 
        stratify=y if len(np.unique(y))<=10 else None)
    val_ratio = val_size / (val_size + test_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1-val_ratio), random_state=random_state, 
        stratify=y_temp if len(np.unique(y_temp))<=10 else None)
    return X_train, X_val, X_test, y_train, y_val, y_test

X_class = classification_df[['Longitude', 'Latitude']]
y_class = classification_df['Label']
X_reg = regression_df[['Longitude', 'Latitude']]
y_reg = regression_df['Value']

Xc_train, Xc_val, Xc_test, yc_train, yc_val, yc_test = split_three_way(X_class, y_class)
Xr_train, Xr_val, Xr_test, yr_train, yr_val, yr_test = split_three_way(X_reg, y_reg)

# Train models
clf = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=100, random_state=42))
clf.fit(Xc_train, yc_train)

regressor_rf = RandomForestRegressor(n_estimators=100, random_state=42)
regressor_rf.fit(Xr_train, yr_train)

# Piecewise Model
class PiecewiseModel:
    def __init__(self, classifier, regressor):
        self.C = classifier
        self.R = regressor
    
    def predict(self, X):
        class_pred = self.C.predict(X)
        reg_pred = self.R.predict(X)
        h_pred = np.where(class_pred == 1, reg_pred, -999.0)
        return h_pred

# Apply piecewise model
h_model = PiecewiseModel(clf, regressor_rf)
full_coords = classification_df[['Longitude', 'Latitude']]
h_full_pred = h_model.predict(full_coords)
h_grid = h_full_pred.reshape((dim_lat, dim_lon))

# Visualization: Piecewise Smooth Model h(x)
h_grid_display = np.where(h_grid == -999, np.nan, h_grid)

plt.figure(figsize=(10, 12))
im = plt.imshow(h_grid_display, origin='lower', cmap='jet',
                extent=[lons.min(), lons.max(), lats.min(), lats.max()],
                vmin=10, vmax=35)
plt.colorbar(im, label='Temperature (°C)')
plt.title('Piecewise Smooth Model h(x)\nh(x) = R(x) if C(x)=1, else -999', 
          fontsize=14, fontweight='bold')
plt.xlabel('Longitude', fontsize=12)
plt.ylabel('Latitude', fontsize=12)
plt.tight_layout()
plt.savefig('piecewise_model_h(x).png', dpi=200, bbox_inches='tight')
plt.show()
