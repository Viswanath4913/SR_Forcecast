import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go

st.set_page_config(page_title="Service Request Forecast", layout="centered")
st.title("🔧 Service Request Forecast")

# Step 1: Select data type
data_type = st.radio("What kind of data are you uploading?", ("Sales Data", "Service Request Data"))

# Step 2: Upload CSV
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

# Step 3: Frequency selection
freq = st.selectbox("Select Forecast Frequency", ["Weekly", "Monthly", "Yearly"])
freq_map = {"Weekly": "W", "Monthly": "M", "Yearly": "Y"}

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.lower()

    # Check columns based on type
    if data_type == "Sales Data" and "sales" in df.columns and "date" in df.columns:
        df = df.rename(columns={"sales": "y", "date": "ds"})
    elif data_type == "Service Request Data" and "call_requests" in df.columns and "date" in df.columns:
        df = df.rename(columns={"call_requests": "y", "date": "ds"})
    else:
        st.error("CSV must contain columns 'date' and either 'sales' or 'call_requests' based on your selection.")
        st.stop()

    df["ds"] = pd.to_datetime(df["ds"])
    df = df[["ds", "y"]]
    df = df.resample(freq_map[freq], on="ds").sum().reset_index()

    st.subheader(f"🗃 Uploaded Data ({freq})")
    st.dataframe(df.tail())

    # Prophet model
    model = Prophet()
    model.fit(df)

    future = model.make_future_dataframe(periods=12, freq=freq_map[freq])
    forecast = model.predict(future)

    # Split forecast
    last_date = df["ds"].max()
    future_data = forecast[forecast["ds"] > last_date]

    # Plotting
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["ds"], y=df["y"],
        mode="lines+markers", name="Actual",
        line=dict(color="blue")
    ))
    fig.add_trace(go.Scatter(
        x=future_data["ds"], y=future_data["yhat"],
        mode="lines+markers", name="Forecast",
        line=dict(color="green", dash="dash")
    ))
    fig.add_trace(go.Scatter(
        x=pd.concat([future_data["ds"], future_data["ds"][::-1]]),
        y=pd.concat([future_data["yhat_upper"], future_data["yhat_lower"][::-1]]),
        fill="toself", fillcolor="rgba(0,255,0,0.2)",
        line=dict(color="rgba(255,255,255,0)"),
        hoverinfo="skip", showlegend=True, name="Confidence Interval"
    ))

    fig.update_layout(
        title="📈 Forecast of Service Requests",
        xaxis_title="Date",
        yaxis_title="Volume",
        template="plotly_white"
    )

    st.subheader("📊 Forecast Plot")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📋 Forecasted Results")
    st.dataframe(future_data[["ds", "yhat", "yhat_lower", "yhat_upper"]])

    # Download forecast
    csv = future_data[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_csv(index=False).encode("utf-8")
    st.download_button("Download Forecast CSV", csv, "forecast.csv", "text/csv")
