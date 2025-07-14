import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.express as px
from datetime import datetime

st.set_page_config(page_title="Call Request Forecast", layout="centered")
st.title("📈 Call Request Forecast")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

# Frequency selection
freq = st.selectbox("Select Forecast Frequency", ["Weekly", "Monthly", "Yearly"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.lower()

    # Ensure required columns
    if "date" in df.columns and "call_requests" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df[["date", "call_requests"]]
        df = df.rename(columns={"date": "ds", "call_requests": "y"})

        # Resample based on frequency
        freq_map = {"Weekly": "W", "Monthly": "M", "Yearly": "Y"}
        df = df.resample(freq_map[freq], on="ds").sum().reset_index()

        st.subheader(f"Raw Call Request Data ({freq})")
        st.dataframe(df.tail())

        # Prophet model
        m = Prophet()
        m.fit(df)

        future = m.make_future_dataframe(periods=12, freq=freq_map[freq])
        forecast = m.predict(future)

        st.subheader("Forecast Plot")
        fig = plot_plotly(m, forecast)
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.error("CSV must contain 'date' and 'call_requests' columns.")
