import streamlit as st
import pandas as pd
import numpy as np
import torch
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import seasonal_decompose
import sys
import os

# Ensure src is in path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.patchtst import PatchTST
from src.models.lstm import SimpleLSTM
from src.utils.data import WalmartDataset

# Page Configuration
st.set_page_config(page_title="Walmart Sales Predictor", layout="wide")

# Custom CSS to make the UI look cleaner
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_resources(store_id):
    dataset = WalmartDataset("data/walmart.csv")
    data = dataset.prepare_data(store_id, 52, 4)
    p_model = PatchTST(52, 4)
    l_model = SimpleLSTM(output_dim=4)
    return dataset, data, p_model, l_model

# --- Header Section ---
st.title("🏪 Walmart Sales Intelligence Dashboard")
st.markdown("""
    Welcome! This tool uses **Advanced Artificial Intelligence** to predict future sales for Walmart stores. 
    It compares two different AI "brains" to see which one understands your store's patterns better.
""")

# --- Sidebar Inputs ---
st.sidebar.header("🕹️ Control Panel")
store_id = st.sidebar.number_input("Which Store are we analyzing?", 1, 45, 1)
st.sidebar.info(f"Currently analyzing Store #{store_id}. The AI looks at 52 weeks of history to predict the next 4 weeks.")

dataset, data, p_model, l_model = load_resources(store_id)

# --- Top Level Metrics (Executive Summary) ---
# Simple inference for display
last_window = torch.tensor(data["X_val"][-1]).unsqueeze(0).unsqueeze(-1).float()
with torch.no_grad():
    y_patch_scaled = p_model(last_window).numpy().flatten()
    y_lstm_scaled = l_model(last_window).numpy().flatten()

y_actual = data["scaler"].inverse_transform(data["y_val"][-1].reshape(-1, 1)).flatten()
y_patch = data["scaler"].inverse_transform(y_patch_scaled.reshape(-1, 1)).flatten()
y_lstm = data["scaler"].inverse_transform(y_lstm_scaled.reshape(-1, 1)).flatten()

st.subheader("📊 Performance at a Glance")
col1, col2, col3 = st.columns(3)

# Calculating "Accuracy" in a way non-tech people understand (1 - Percentage Error)
patch_acc = 100 - (np.mean(np.abs(y_actual - y_patch) / y_actual) * 100)
lstm_acc = 100 - (np.mean(np.abs(y_actual - y_lstm) / y_actual) * 100)

col1.metric("Transformer Accuracy", f"{patch_acc:.1f}%", help="How close the Transformer model stayed to actual sales.")
col2.metric("LSTM Accuracy", f"{lstm_acc:.1f}%", help="How close the standard LSTM model stayed to actual sales.")
col3.metric("Better Model", "Transformer" if patch_acc > lstm_acc else "LSTM", delta="Recommended")

st.divider()

# --- Visualization Tabs ---
tab1, tab2, tab3 = st.tabs(["📈 Sales Forecast", "🔎 Pattern Discovery", "❓ Help & Info"])

with tab1:
    st.subheader("Future Sales Prediction")
    st.write("The solid black line is what actually happened. The colored dashed lines are what our AI models thought would happen.")
    
    fig = go.Figure()
    # Historical Context (Last 8 weeks)
    hist_sales = data["raw_sales"][-12:-4]
    weeks_hist = list(range(-8, 0))
    weeks_pred = list(range(0, 4))
    
    fig.add_trace(go.Scatter(x=weeks_hist, y=hist_sales, name="Past Sales", line=dict(color='#BDC3C7')))
    fig.add_trace(go.Scatter(x=weeks_pred, y=y_actual, name="Actual Sales", line=dict(color='black', width=3)))
    fig.add_trace(go.Scatter(x=weeks_pred, y=y_patch, name="AI Prediction (PatchTST)", line=dict(color='#3498DB', dash='dash')))
    fig.add_trace(go.Scatter(x=weeks_pred, y=y_lstm, name="AI Prediction (LSTM)", line=dict(color='#E74C3C', dash='dot')))
    
    fig.update_layout(hovermode="x unified", xaxis_title="Weeks (0 = Today)", yaxis_title="Sales in Dollars ($)")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("What drives your sales?")
    st.write("We break your sales data into two main parts: the **Long-term Trend** and **Seasonal Cycles** (like holidays).")
    
    df_store = dataset.df[dataset.df['Store'] == store_id]
    res = seasonal_decompose(df_store['Weekly_Sales'], model='additive', period=52)
    
    c1, c2 = st.columns(2)
    with c1:
        st.write("**The Trend:** Is the store growing or shrinking over time?")
        st.line_chart(res.trend)
    with c2:
        st.write("**The Seasonality:** Regular ups and downs during the year.")
        st.line_chart(res.seasonal)

with tab3:
    st.info("""
    ### How to read this dashboard:
    1. **MSE/Accuracy**: Lower Error (or Higher Accuracy) means the model is better at learning this specific store's behavior.
    2. **Transformer vs LSTM**: The Transformer (PatchTST) is a newer, more complex "brain" that often catches small details better than the LSTM.
    3. **Seasonality**: This shows you when your busiest times of the year are, helping you plan inventory!
    """)