import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objects as go

st.set_page_config(page_title="Call Request Forecast", layout="centered")
st.title("📞 Call Request Forecast")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

# Frequency selection
freq = st.selectbox("Select Forecast Frequency", ["Weekly", "Monthly", "Yearly"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.lower()

    if "date" in df.columns and "call_requests" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.rename(columns={"date": "ds", "call_requests": "y"})

        freq_map = {"Weekly": "W", "Monthly": "M", "Yearly": "Y"}
        df = df.resample(freq_map[freq], on="ds").sum().reset_index()

        st.subheader(f"Raw Call Request Data ({freq})")
        st.dataframe(df.tail())

        # Train model
        model = Prophet()
        model.fit(df)

        future = model.make_future_dataframe(periods=12, freq=freq_map[freq])
        forecast = model.predict(future)

        # Split forecast
        historical = forecast[forecast["ds"] <= df["ds"].max()]
        future_data = forecast[forecast["ds"] > df["ds"].max()]

        # Plot
        fig = go.Figure()

        # Actual values
        fig.add_trace(go.Scatter(
            x=df["ds"], y=df["y"],
            mode="lines+markers",
            name="Actual",
            line=dict(color="blue")
        ))

        # Forecasted future
        fig.add_trace(go.Scatter(
            x=future_data["ds"], y=future_data["yhat"],
            mode="lines+markers",
            name="Forecast",
            line=dict(color="green", dash="dash")
        ))

        # Confidence interval
        fig.add_trace(go.Scatter(
            x=pd.concat([future_data["ds"], future_data["ds"][::-1]]),
            y=pd.concat([future_data["yhat_upper"], future_data["yhat_lower"][::-1]]),
            fill="toself",
            fillcolor="rgba(0,255,0,0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            showlegend=True,
            name="Confidence Interval"
        ))

        fig.update_layout(
            title="Forecast of Call Requests",
            xaxis_title="Date",
            yaxis_title="Call Requests",
            template="plotly_white"
        )

        st.subheader("📈 Forecast")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("📋 Forecasted Data Table")
        st.dataframe(future_data[["ds", "yhat", "yhat_lower", "yhat_upper"]])
    else:
        st.error("CSV must contain 'date' and 'call_requests' columns.")
