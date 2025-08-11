import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from logger_config import log

log.info("California housing dataset loaded successfully.")

# Step 1: Load dataset
california = fetch_california_housing()
df = pd.DataFrame(california.data, columns=california.feature_names)
df['MedHouseVal'] = california.target

# Step 2: Rename columns explicitly
df.rename(columns={
    'MedInc': 'medinc',
    'HouseAge': 'house_age',
    'AveRooms': 'ave_rooms',
    'AveBedrms': 'ave_bedrms',
    'Population': 'population',
    'AveOccup': 'ave_occup',
    'Latitude': 'latitude',
    'Longitude': 'longitude',
    'MedHouseVal': 'medhouseval'
}, inplace=True)
log.info("Columns renamed explicitly.")

# Step 3: Derived Feature — Approximate households
df['households'] = df['population'] / df['ave_occup']
log.info("Derived feature 'households' created.")

# Step 4: Feature Engineering
df['rooms_per_household'] = df['ave_rooms'] / df['households']
df['bedrooms_per_room'] = df['ave_bedrms'] / df['ave_rooms']
df['population_per_household'] = df['population'] / df['households']

# Step 5: Log Transform skewed feature
df['medinc_log'] = np.log1p(df['medinc'])

# Step 6: Feature Scaling (excluding target)
features_to_scale = df.drop('medhouseval', axis=1).columns
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[features_to_scale])

# Handle NaNs, Infs, and extreme values after scaling
pos_inf_val = np.max(scaled_features[np.isfinite(scaled_features)])
neg_inf_val = np.min(scaled_features[np.isfinite(scaled_features)])
scaled_features = np.nan_to_num(
    scaled_features, nan=0.0, posinf=pos_inf_val, neginf=neg_inf_val
)

# Update DataFrame with cleaned scaled features
df[features_to_scale] = scaled_features

# Step 7: Train-Test Split
X = df.drop('medhouseval', axis=1)
y = df['medhouseval']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 8: Recombine and export
train_df = X_train.copy()
train_df['medhouseval'] = y_train

test_df = X_test.copy()
test_df['medhouseval'] = y_test

# Step 9: Export to CSV
df.to_csv("data/california_housing_preprocessed.csv", index=False)
train_df.to_csv("data/california_housing_train.csv", index=False)
test_df.to_csv("data/california_housing_test.csv", index=False)

log.info("✅ Export complete: full, train, and test datasets.")
