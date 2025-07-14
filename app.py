import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go

st.set_page_config(page_title="Service Request Forecast", layout="centered")
st.title("🔧 Service Request Forecast")

# Input Type
data_type = st.radio("What kind of data are you uploading?", ("Sales Data", "Service Request Data"))

# Upload File
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

# Frequency and Forecast Period
freq = st.selectbox("Select Forecast Frequency", ["Daily", "Weekly", "Monthly", "Yearly"])
freq_map = {"Daily": "D", "Weekly": "W", "Monthly": "M", "Yearly": "Y"}
default_periods = {"Daily": 30, "Weekly": 12, "Monthly": 12, "Yearly": 5}
max_periods = {"Daily": 365, "Weekly": 52, "Monthly": 24, "Yearly": 10}

forecast_period = st.slider(
    f"How many {freq.lower()}s to forecast?",
    min_value=1,
    max_value=max_periods[freq],
    value=default_periods[freq],
    step=1
)

# Button to run forecast
run_forecast = st.button("🚀 Run Forecast")

if uploaded_file and run_forecast:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.lower()

    if data_type == "Sales Data" and "sales" in df.columns:
        df = df.rename(columns={"sales": "y", "date": "ds"})
    elif data_type == "Service Request Data" and "call_requests" in df.columns:
        df = df.rename(columns={"call_requests": "y", "date": "ds"})
    else:
        st.error("CSV must contain 'date' and either 'sales' or 'call_requests'.")
        st.stop()

    df["ds"] = pd.to_datetime(df["ds"])
    df = df[["ds", "y"]].sort_values("ds")
    df = df.resample(freq_map[freq], on="ds").sum().reset_index()

    st.subheader("📊 Historical Data")
    st.dataframe(df.tail())

    # Train and Forecast
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=forecast_period, freq=freq_map[freq])
    forecast = model.predict(future)[["ds", "yhat"]]

    # Split actual vs future
    last_date = df["ds"].max()
    historical = forecast[forecast["ds"] <= last_date]
    future_data = forecast[forecast["ds"] > last_date]

    # Show only recent actuals for better visual
    show_history_points = {"Daily": 30, "Weekly": 12, "Monthly": 12, "Yearly": 5}
    recent_historical = historical.tail(show_history_points[freq])

    # Prepare display table
    future_data["Date"] = future_data["ds"].dt.strftime("%d-%m-%Y")
    future_data["Forecasted Volume"] = future_data["yhat"].round().astype(int)
    display_data = future_data[["Date", "Forecasted Volume"]]

    # Chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=recent_historical["ds"], y=recent_historical["yhat"],
        mode="lines", name="Actual", line=dict(color="royalblue"), fill='tozeroy'
    ))
    fig.add_trace(go.Scatter(
        x=future_data["ds"], y=future_data["Forecasted Volume"],
        mode="lines", name="Forecast", line=dict(color="seagreen"), fill='tozeroy'
    ))
    fig.update_layout(
        title="📈 Service Request Forecast",
        xaxis_title="Date",
        yaxis_title="Forecasted Volume",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.subheader("📊 Forecast Chart")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📋 Forecast Table")
    st.dataframe(display_data)

    csv = display_data.to_csv(index=False).encode("utf-8")
    st.download_button("Download Forecast CSV", csv, "forecast.csv", "text/csv")
