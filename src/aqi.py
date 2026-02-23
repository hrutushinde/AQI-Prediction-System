import pandas as pd

# Load the raw file (replace with your file path)
df = pd.read_csv("city_day (1).csv", parse_dates=["Date"])

# Keep only 2017–2020
df = df[df["Date"].dt.year.between(2017, 2020)]

# Drop rows where AQI is missing
df = df.dropna(subset=["AQI"])

# Optional: Keep only necessary columns for now
df = df[["City", "Date", "PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3", "AQI"]]

# Save cleaned dataset
df.to_csv("aqi_dataset.csv", index=False)
