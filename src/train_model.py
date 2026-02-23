import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pickle

# --- Load and preprocess the dataset ---
df = pd.read_csv("city_day (1).csv", parse_dates=["Date"])

# Keep only data from 2017–2020
df = df[(df["Date"].dt.year >= 2017) & (df["Date"].dt.year <= 2020)]

# Define features and target
features = ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3"]
target = "AQI"

# Drop rows with missing values in relevant columns
df = df.dropna(subset=features + [target])

# Extract feature matrix and target vector
X = df[features]
y = df[target]

# --- Split and train ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- Save model ---
with open("air_quality_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as 'air_quality_model.pkl'")
