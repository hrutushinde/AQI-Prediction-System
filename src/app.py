import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import pickle

# --- Load Model and Data ---
with open("air_quality_model.pkl", "rb") as f:
    model = pickle.load(f)

df = pd.read_csv("aqi_dataset.csv", parse_dates=["Date"])
df = df[(df["Date"].dt.year >= 2017) & (df["Date"].dt.year <= 2020)]
df = df.dropna(subset=["AQI"])

# --- City Coordinates for Map ---
city_coords = {
    "Delhi": [28.7041, 77.1025],
    "Mumbai": [19.0760, 72.8777],
    "Ahmedabad": [23.0225, 72.5714],
    "Chennai": [13.0827, 80.2707],
    "Kolkata": [22.5726, 88.3639],
    "Bengaluru": [12.9716, 77.5946],
    "Hyderabad": [17.3850, 78.4867],
    "Pune": [18.5204, 73.8567],
    "Jaipur": [26.9124, 75.7873],
    "Lucknow": [26.8467, 80.9462]
}

# --- Page Config ---
st.set_page_config("AQI Predictor App", layout="centered")

# --- App Title ---
st.markdown("<h1 style='text-align: center;'>🌫️ Air Quality Index (AQI) App</h1>", unsafe_allow_html=True)

# --- Sidebar Inputs ---
st.sidebar.title("📍 Input Pollution Levels")
selected_city = st.sidebar.selectbox("Select City", list(city_coords.keys()))

pm25 = st.sidebar.slider("PM2.5", 0.0, 500.0, 50.0)
pm10 = st.sidebar.slider("PM10", 0.0, 500.0, 50.0)
no = st.sidebar.slider("NO", 0.0, 100.0, 10.0)
no2 = st.sidebar.slider("NO2", 0.0, 200.0, 20.0)
nox = st.sidebar.slider("NOx", 0.0, 300.0, 30.0)
nh3 = st.sidebar.slider("NH3", 0.0, 200.0, 10.0)
co = st.sidebar.slider("CO", 0.0, 10.0, 1.0)
so2 = st.sidebar.slider("SO2", 0.0, 100.0, 10.0)
o3 = st.sidebar.slider("O3", 0.0, 200.0, 20.0)
# --- Prediction ---
if st.sidebar.button("🔮 Predict AQI"):
    input_data = np.array([[pm25, pm10, no, no2, nox, nh3, co, so2, o3]])
    prediction = model.predict(input_data)[0]

    def get_category(aqi):
        if aqi <= 50: return "Good", "#2ecc71"
        elif aqi <= 100: return "Satisfactory", "#f1c40f"
        elif aqi <= 200: return "Moderate", "#e67e22"
        elif aqi <= 300: return "Poor", "#e74c3c"
        elif aqi <= 400: return "Very Poor", "#9b59b6"
        else: return "Severe", "#7f0000"

    category, color = get_category(prediction)

    st.markdown(f"""
    <div style='text-align:center; background-color:{color}; color:white; padding:20px;
                border-radius:12px; margin-top:20px;'>
        <h2>Predicted AQI: {round(prediction, 2)}</h2>
        <h3>Category: {category}</h3>
    </div>
    """, unsafe_allow_html=True)

# --- AQI Trends ---
st.markdown("### 📊 Historical AQI Trends")

city_df = df[df["City"] == selected_city]
tab1, tab2 = st.tabs(["📈 Line Chart", "📊 Yearly Averages"])

with tab1:
    st.line_chart(city_df.set_index("Date")["AQI"])

with tab2:
    yearly_avg = city_df.groupby(city_df["Date"].dt.year)["AQI"].mean()
    st.bar_chart(yearly_avg)

# --- AQI on Specific Day ---
st.markdown("### 📅 AQI on a Specific Day")
selected_date = st.date_input("Choose a Date", value=pd.to_datetime("2020-01-01"),
                              min_value=df["Date"].min().date(),
                              max_value=df["Date"].max().date())

daily_df = df[(df["City"] == selected_city) & (df["Date"].dt.date == selected_date)]
if not daily_df.empty:
    aqi_val = daily_df.iloc[0]["AQI"]
    st.info(f"**AQI in {selected_city} on {selected_date}: {int(aqi_val)}**")
else:
    st.warning("No AQI data available for this date & city.")

# --- Geo Map ---
st.markdown("### 🗺️ AQI Across Cities on Selected Date")
map_date = st.date_input("Select Date for Map", value=pd.to_datetime("2020-01-01"),
                         min_value=df["Date"].min().date(),
                         max_value=df["Date"].max().date())

map_df = df[df["Date"].dt.date == map_date].copy()
map_df["lat"] = map_df["City"].map(lambda x: city_coords.get(x, [None, None])[0])
map_df["lon"] = map_df["City"].map(lambda x: city_coords.get(x, [None, None])[1])
map_df = map_df.dropna(subset=["lat", "lon", "AQI"])

layer = pdk.Layer(
    "ScatterplotLayer",
    data=map_df,
    get_position='[lon, lat]',
    get_color='[255 - AQI, 100, AQI, 160]',
    get_radius=20000,
    pickable=True
)

view_state = pdk.ViewState(latitude=22.5, longitude=80, zoom=4.5)

st.pydeck_chart(pdk.Deck(
    map_style="mapbox://styles/mapbox/light-v9",
    initial_view_state=view_state,
    layers=[layer],
    tooltip={"text": "{City}\nAQI: {AQI}"}
))
