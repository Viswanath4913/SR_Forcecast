# Service Volume Forecasting App

A professional time series forecasting web application built with Streamlit that predicts service request volumes for consumer electronics. Helps service managers plan resources, staffing, and inventory based on historical data patterns.

## Features

- **Time Series Forecasting**: Prophet and ARIMA models for volume prediction
- **Flexible Data Upload**: Automatic CSV format detection and column mapping
- **Interactive Dashboard**: Real-time charts with confidence intervals
- **Resource Planning**: Actionable insights for manpower and inventory planning
- **Analytics**: Historical trends, seasonal patterns, and performance metrics
- **Export Functions**: Download forecasts and historical data

## Quick Start

### 1. Install Dependencies

```bash
pip install -r deployment_requirements.txt
```

### 2. Run the Application

```bash
streamlit run app.py --server.port 5000
```

### 3. Access the App

Open your browser and go to `http://localhost:5000`

## File Structure

```
service-forecasting-app/
├── app.py                      # Main Streamlit application
├── models/
│   └── ml_models.py           # Forecasting models (Prophet, ARIMA)
├── utils/
│   ├── data_processor.py      # Data handling and CSV processing
│   └── visualizations.py     # Chart and graph functions
├── data/                      # Data storage directory
│   ├── service_history.csv    # Historical service data
│   └── predictions_log.csv    # Forecast history
├── .streamlit/
│   └── config.toml           # Streamlit configuration
└── deployment_requirements.txt # Python dependencies
```

## Usage

### 1. Volume Forecasting
- Generate forecasts for different time periods (days/weeks/months)
- Choose between Prophet, ARIMA, or both models
- Apply filters by product category or region
- View confidence intervals and planning insights

### 2. Data Upload
- Upload any CSV file with time series data
- Automatic column detection for dates and volumes
- Manual mapping for custom formats
- Supports various column names (Date, Time, Count, Requests, etc.)

### 3. Analytics Dashboard
- Historical volume trends and patterns
- Seasonal analysis and decomposition
- Product category and regional breakdowns
- Weekly patterns and peak periods

### 4. Model Performance
- Compare Prophet vs ARIMA accuracy
- View performance metrics (MAE, RMSE)
- Model components and feature importance

## CSV Upload Formats

The app accepts flexible CSV formats:

**Simple Format:**
```csv
date,volume
2024-01-01,45
2024-01-02,52
```

**Detailed Format:**
```csv
date,service_requests,product_category,region
2024-01-01,45,TV,North
2024-01-02,52,AC,South
```

**Custom Columns** (automatically detected):
- Date columns: `Date`, `Time`, `Period`, `Month`, `Year`
- Volume columns: `Count`, `Tickets`, `Calls`, `Demand`, `Requests`
- Category columns: `Product`, `Type`, `Category`
- Location columns: `Region`, `Location`, `Area`, `Zone`

## Deployment Options

### Local Development
```bash
streamlit run app.py
```

### Docker Deployment
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -r deployment_requirements.txt
EXPOSE 5000
CMD ["streamlit", "run", "app.py", "--server.port", "5000", "--server.address", "0.0.0.0"]
```

### Cloud Platforms
- **Streamlit Cloud**: Push to GitHub and deploy directly
- **Heroku**: Use Procfile with `web: streamlit run app.py --server.port $PORT`
- **Railway/Render**: Compatible with default Python deployment

## Configuration

Edit `.streamlit/config.toml` for custom settings:

```toml
[server]
headless = true
address = "0.0.0.0"
port = 5000

[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
```

## Technical Details

- **Framework**: Streamlit for web interface
- **Models**: Facebook Prophet and ARIMA for forecasting
- **Storage**: CSV files (no database required)
- **Visualization**: Plotly for interactive charts
- **Data Processing**: Pandas and NumPy

## Support

For issues or questions:
1. Check the troubleshooting section in the app
2. Verify CSV format matches expected structure
3. Ensure all dependencies are installed correctly

## License

Open source - feel free to modify and distribute as needed.