import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# Title
st.title("🏠 California House Price Predictor")
st.markdown("Predict house prices using Machine Learning — "
            "powered by Random Forest")
st.markdown("---")

# Load and train model
@st.cache_resource
def load_model():
    data     = fetch_california_housing(as_frame=True)
    df       = data.frame
    X        = df.drop("MedHouseVal", axis=1)
    y        = df["MedHouseVal"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mse   = mean_squared_error(y_test, preds)
    r2    = r2_score(y_test, preds)
    return model, X, mse, r2

model, X, mse, r2 = load_model()

# Model metrics
col1, col2, col3 = st.columns(3)
col1.metric("Model",        "Random Forest")
col2.metric("R² Score",     f"{r2:.3f}")
col3.metric("MSE",          f"{mse:.3f}")
st.markdown("---")

# Sidebar inputs
st.sidebar.header("🔧 Enter House Details")

med_inc     = st.sidebar.slider("Median Income (x$10,000)",
                                 0.5, 15.0, 5.0, 0.1)
house_age   = st.sidebar.slider("House Age (years)",
                                 1, 52, 20, 1)
avg_rooms   = st.sidebar.slider("Average Rooms per House",
                                 1.0, 15.0, 5.0, 0.1)
avg_bedrms  = st.sidebar.slider("Average Bedrooms per House",
                                 0.5, 5.0, 1.0, 0.1)
population  = st.sidebar.slider("Block Population",
                                 3, 10000, 1500, 10)
avg_occup   = st.sidebar.slider("Average Occupancy",
                                 1.0, 10.0, 3.0, 0.1)
latitude    = st.sidebar.slider("Latitude",
                                 32.0, 42.0, 36.0, 0.1)
longitude   = st.sidebar.slider("Longitude",
                                 -125.0, -114.0, -120.0, 0.1)

# Predict
input_data = np.array([[med_inc, house_age, avg_rooms, avg_bedrms,
                         population, avg_occup, latitude, longitude]])
prediction = model.predict(input_data)[0]

# Show prediction
st.markdown("## 💰 Predicted House Price")
price_usd = prediction * 100000
st.markdown(
    f"<h1 style='color:#2ecc71; text-align:center'>"
    f"${price_usd:,.0f}</h1>",
    unsafe_allow_html=True
)
st.markdown("---")

# Two columns layout
left, right = st.columns(2)

# Feature importance chart
with left:
    st.markdown("### 📊 Feature Importance")
    feat_imp = pd.Series(
        model.feature_importances_, index=X.columns
    ).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = plt.cm.RdYlGn(
        np.linspace(0.3, 1.0, len(feat_imp)))
    ax.barh(feat_imp.index, feat_imp.values,
            color=colors, edgecolor='black')
    ax.set_title("What affects price the most?")
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    st.pyplot(fig)

# Input summary
with right:
    st.markdown("### 📋 Your Input Summary")
    summary = pd.DataFrame({
        'Feature': ['Median Income', 'House Age', 'Avg Rooms',
                    'Avg Bedrooms', 'Population', 'Avg Occupancy',
                    'Latitude', 'Longitude'],
        'Value': [med_inc, house_age, avg_rooms, avg_bedrms,
                  population, avg_occup, latitude, longitude]
    })
    st.dataframe(summary, use_container_width=True)

st.markdown("---")
st.markdown(
    "Built by **Jyotiraditya** | "
    "Data: California Housing Dataset | "
    "Model: Random Forest Regressor"
)