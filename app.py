import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
from models.ml_models import ServiceVolumeForecaster
from utils.data_processor import DataProcessor
from utils.visualizations import create_forecast_chart, create_trend_analysis, create_seasonal_decomposition, create_volume_trends_chart

# Page configuration
st.set_page_config(
    page_title="Service Volume Forecasting App",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'forecasts_history' not in st.session_state:
    st.session_state.forecasts_history = []

# Initialize components
@st.cache_resource
def load_model():
    return ServiceVolumeForecaster()

@st.cache_resource
def load_data_processor():
    return DataProcessor()

model = load_model()
data_processor = load_data_processor()

# Main app title
st.title("📊 Service Request Volume Forecasting")
st.markdown("Predict future service request volumes to optimize resource planning")
st.markdown("---")

# Sidebar for navigation
st.sidebar.header("Navigation")
page = st.sidebar.selectbox(
    "Select Page",
    ["Volume Forecasting", "Analytics Dashboard", "Data Management", "Model Performance"]
)

if page == "Volume Forecasting":
    st.header("Service Request Volume Forecasting")
    
    # Forecast configuration section
    st.subheader("Forecast Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        forecast_periods = st.selectbox(
            "Forecast Period",
            [7, 14, 30, 60, 90, 180],
            index=2,
            help="Number of days to forecast"
        )
    
    with col2:
        forecast_frequency = st.selectbox(
            "Frequency",
            ["D", "W", "M"],
            format_func=lambda x: {"D": "Daily", "W": "Weekly", "M": "Monthly"}[x],
            help="Forecasting frequency"
        )
    
    with col3:
        model_type = st.selectbox(
            "Model Type",
            ["prophet", "arima", "both"],
            format_func=lambda x: {"prophet": "Prophet", "arima": "ARIMA", "both": "Both Models"}[x],
            help="Choose forecasting model"
        )
    
    # Optional filters section
    st.subheader("Optional Filters")
    
    filter_col1, filter_col2 = st.columns(2)
    
    with filter_col1:
        product_filter = st.multiselect(
            "Product Categories",
            ["TV", "AC", "Refrigerator", "Smartphone", "Laptop"],
            help="Filter by product categories"
        )
    
    with filter_col2:
        region_filter = st.multiselect(
            "Regions",
            ["North", "South", "East", "West", "Central"],
            help="Filter by regions"
        )
    
    # Generate forecast button
    if st.button("Generate Volume Forecast", type="primary"):
        try:
            with st.spinner("Generating forecast..."):
                # Apply filters if any
                filters = {}
                if product_filter:
                    filters['product_category'] = product_filter
                if region_filter:
                    filters['region'] = region_filter
                
                # Generate forecasts
                forecasts = model.forecast_volume(
                    periods=forecast_periods,
                    freq=forecast_frequency,
                    filters=filters
                )
                
                if forecasts:
                    # Display forecast summary
                    st.markdown("---")
                    st.header("Volume Forecast Results")
                    
                    # Show summary metrics
                    summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
                    
                    if 'prophet' in forecasts:
                        prophet_avg = np.mean(forecasts['prophet']['forecast'])
                        prophet_peak = max(forecasts['prophet']['forecast'])
                        
                        with summary_col1:
                            st.metric("Average Volume (Prophet)", f"{prophet_avg:.0f}")
                        
                        with summary_col2:
                            st.metric("Peak Volume (Prophet)", f"{prophet_peak:.0f}")
                    
                    if 'arima' in forecasts:
                        arima_avg = np.mean(forecasts['arima']['forecast'])
                        arima_peak = max(forecasts['arima']['forecast'])
                        
                        with summary_col3:
                            st.metric("Average Volume (ARIMA)", f"{arima_avg:.0f}")
                        
                        with summary_col4:
                            st.metric("Peak Volume (ARIMA)", f"{arima_peak:.0f}")
                    
                    # Create forecast visualization
                    st.subheader("Volume Forecast Chart")
                    
                    fig = go.Figure()
                    
                    # Add Prophet forecast if available
                    if 'prophet' in forecasts and (model_type in ['prophet', 'both']):
                        fig.add_trace(go.Scatter(
                            x=forecasts['prophet']['dates'],
                            y=forecasts['prophet']['forecast'],
                            name='Prophet Forecast',
                            line=dict(color='blue', width=2),
                            mode='lines+markers'
                        ))
                        
                        # Add confidence intervals
                        fig.add_trace(go.Scatter(
                            x=forecasts['prophet']['dates'] + forecasts['prophet']['dates'][::-1],
                            y=forecasts['prophet']['upper_bound'] + forecasts['prophet']['lower_bound'][::-1],
                            fill='tonexty',
                            fillcolor='rgba(0,100,80,0.2)',
                            line=dict(color='rgba(255,255,255,0)'),
                            hoverinfo="skip",
                            showlegend=False,
                            name='Prophet Confidence'
                        ))
                    
                    # Add ARIMA forecast if available
                    if 'arima' in forecasts and (model_type in ['arima', 'both']):
                        fig.add_trace(go.Scatter(
                            x=forecasts['arima']['dates'],
                            y=forecasts['arima']['forecast'],
                            name='ARIMA Forecast',
                            line=dict(color='red', width=2, dash='dash'),
                            mode='lines+markers'
                        ))
                    
                    fig.update_layout(
                        title=f"Service Request Volume Forecast ({forecast_periods} days)",
                        xaxis_title="Date",
                        yaxis_title="Number of Service Requests",
                        hovermode='x unified',
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Generate insights
                    st.subheader("Resource Planning Insights")
                    insights = model.get_volume_insights(forecasts)
                    
                    if insights:
                        for i, insight in enumerate(insights, 1):
                            st.write(f"**{i}.** {insight}")
                    else:
                        st.info("Generate forecast to see resource planning insights")
                    
                    # Log forecast
                    forecast_log = {
                        'timestamp': datetime.now(),
                        'forecast_periods': forecast_periods,
                        'frequency': forecast_frequency,
                        'model_type': model_type,
                        'product_filter': ', '.join(product_filter) if product_filter else 'All',
                        'region_filter': ', '.join(region_filter) if region_filter else 'All'
                    }
                    
                    # Add Prophet metrics if available
                    if 'prophet' in forecasts:
                        forecast_log['prophet_avg_volume'] = np.mean(forecasts['prophet']['forecast'])
                        forecast_log['prophet_peak_volume'] = max(forecasts['prophet']['forecast'])
                    
                    # Add ARIMA metrics if available
                    if 'arima' in forecasts:
                        forecast_log['arima_avg_volume'] = np.mean(forecasts['arima']['forecast'])
                        forecast_log['arima_peak_volume'] = max(forecasts['arima']['forecast'])
                    
                    # Add to session state
                    st.session_state.forecasts_history.append(forecast_log)
                    
                    # Export forecast data
                    st.subheader("Export Forecast")
                    
                    # Prepare export data
                    export_data = []
                    
                    if 'prophet' in forecasts:
                        for i, date in enumerate(forecasts['prophet']['dates']):
                            export_data.append({
                                'date': date,
                                'model': 'Prophet',
                                'forecast': forecasts['prophet']['forecast'][i],
                                'lower_bound': forecasts['prophet']['lower_bound'][i],
                                'upper_bound': forecasts['prophet']['upper_bound'][i]
                            })
                    
                    if 'arima' in forecasts:
                        for i, date in enumerate(forecasts['arima']['dates']):
                            export_data.append({
                                'date': date,
                                'model': 'ARIMA',
                                'forecast': forecasts['arima']['forecast'][i],
                                'lower_bound': forecasts['arima']['lower_bound'][i],
                                'upper_bound': forecasts['arima']['upper_bound'][i]
                            })
                    
                    if export_data:
                        export_df = pd.DataFrame(export_data)
                        csv_data = export_df.to_csv(index=False)
                        
                        st.download_button(
                            label="Download Forecast Data",
                            data=csv_data,
                            file_name=f"volume_forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                
                else:
                    st.error("Unable to generate forecast. Please check if models are properly trained.")
        
        except Exception as e:
            st.error(f"Error generating forecast: {str(e)}")
    
    # Sample data upload section
    st.markdown("---")
    st.subheader("Upload Historical Data for Better Forecasting")
    st.write("Upload your historical service request data to improve forecast accuracy")
    
    sample_format = pd.DataFrame({
        'date': ['2024-01-01', '2024-01-02', '2024-01-03'],
        'service_requests': [45, 52, 38],
        'product_category': ['TV', 'AC', 'Smartphone'],
        'region': ['North', 'South', 'East'],
        'brand': ['Samsung', 'LG', 'Apple']
    })
    
    st.write("**Expected CSV format:**")
    st.dataframe(sample_format)
    
    uploaded_file = st.file_uploader("Choose CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            new_data = pd.read_csv(uploaded_file)
            st.success(f"Successfully loaded {len(new_data)} records")
            st.dataframe(new_data.head())
            
            if st.button("Train Model with New Data"):
                with st.spinner("Training models with new data..."):
                    if model.train_with_new_data(new_data):
                        st.success("Models successfully retrained with new data!")
                        st.rerun()
                    else:
                        st.error("Failed to train models. Please check data format.")
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

elif page == "Analytics Dashboard":
    st.header("Volume Analytics Dashboard")
    
    # Load model's historical data
    if model.historical_data is not None and not model.historical_data.empty:
        historical_data = model.historical_data
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        total_volume = historical_data['service_requests'].sum()
        avg_daily_volume = historical_data.groupby('date')['service_requests'].sum().mean()
        peak_volume = historical_data.groupby('date')['service_requests'].sum().max()
        date_range = (historical_data['date'].max() - historical_data['date'].min()).days
        
        with col1:
            st.metric("Total Volume", f"{total_volume:,}")
        
        with col2:
            st.metric("Average Daily Volume", f"{avg_daily_volume:.0f}")
        
        with col3:
            st.metric("Peak Daily Volume", f"{peak_volume}")
        
        with col4:
            st.metric("Days of Data", f"{date_range}")
        
        # Volume trends chart
        st.subheader("Historical Volume Trends")
        st.plotly_chart(
            create_volume_trends_chart(historical_data),
            use_container_width=True
        )
        
        # Seasonal analysis
        st.subheader("Seasonal Analysis")
        seasonal_data = model.get_seasonal_analysis()
        
        if seasonal_data:
            st.plotly_chart(
                create_seasonal_decomposition(seasonal_data),
                use_container_width=True
            )
        else:
            st.info("Not enough data for seasonal decomposition")
        
        # Product category analysis
        if 'product_category' in historical_data.columns:
            st.subheader("Volume by Product Category")
            category_volume = historical_data.groupby('product_category')['service_requests'].sum().reset_index()
            
            fig_category = px.pie(
                category_volume,
                values='service_requests',
                names='product_category',
                title="Service Requests by Product Category"
            )
            st.plotly_chart(fig_category, use_container_width=True)
        
        # Regional analysis
        if 'region' in historical_data.columns:
            st.subheader("Volume by Region")
            region_volume = historical_data.groupby('region')['service_requests'].sum().reset_index()
            
            fig_region = px.bar(
                region_volume,
                x='region',
                y='service_requests',
                title="Service Requests by Region"
            )
            st.plotly_chart(fig_region, use_container_width=True)
        
        # Weekly patterns
        st.subheader("Weekly Patterns")
        historical_data['weekday'] = pd.to_datetime(historical_data['date']).dt.day_name()
        weekday_volume = historical_data.groupby('weekday')['service_requests'].mean().reindex([
            'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
        ])
        
        fig_weekday = px.bar(
            x=weekday_volume.index,
            y=weekday_volume.values,
            title="Average Service Requests by Day of Week"
        )
        fig_weekday.update_xaxes(title="Day of Week")
        fig_weekday.update_yaxes(title="Average Service Requests")
        st.plotly_chart(fig_weekday, use_container_width=True)
        
    else:
        st.info("No historical data available. Upload data or train the model to see analytics.")

elif page == "Data Management":
    st.header("Time Series Data Management")
    
    # File upload section
    st.subheader("Upload Historical Data")
    st.write("Upload your historical service volume data to improve forecasting accuracy")
    
    # Information about flexible formats
    st.info("📄 **Flexible CSV Upload**: Upload any CSV file with time series data. The app will help you map your columns to the required format.")
    
    # Expected format examples
    tab1, tab2, tab3 = st.tabs(["📊 Simple Format", "📈 Detailed Format", "🔧 Custom Format"])
    
    with tab1:
        simple_data = pd.DataFrame({
            'date': ['2024-01-01', '2024-01-02', '2024-01-03'],
            'volume': [45, 52, 38]
        })
        st.write("**Minimal format (date + volume):**")
        st.dataframe(simple_data, use_container_width=True)
    
    with tab2:
        detailed_data = pd.DataFrame({
            'date': ['2024-01-01', '2024-01-02', '2024-01-03'],
            'service_requests': [45, 52, 38],
            'product_category': ['TV', 'AC', 'Smartphone'],
            'region': ['North', 'South', 'East']
        })
        st.write("**Detailed format (with categories):**")
        st.dataframe(detailed_data, use_container_width=True)
    
    with tab3:
        st.write("**Your CSV can have any column names like:**")
        st.write("- Date columns: `Date`, `Time`, `Period`, `Month`, `Year`")
        st.write("- Volume columns: `Count`, `Tickets`, `Calls`, `Demand`, `Requests`")
        st.write("- Category columns: `Product`, `Type`, `Category`, `Model`")
        st.write("- Location columns: `Region`, `Location`, `Area`, `Zone`")
    
    uploaded_file = st.file_uploader("Choose CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"Successfully loaded {len(df)} records from {uploaded_file.name}")
            
            # Detect CSV format
            format_info = data_processor.detect_csv_format(df)
            
            st.write("**Preview of uploaded data:**")
            st.dataframe(df.head(), use_container_width=True)
            
            # Show detected format
            if format_info['format_type'] == 'time_series':
                st.success("✅ **Time series format detected!** We found these columns:")
                
                col1, col2 = st.columns(2)
                with col1:
                    for key, value in format_info['suggested_mappings'].items():
                        if value:
                            st.write(f"**{key.title()}:** `{value}`")
                
                with col2:
                    st.write("**Sample data:**")
                    for i, row in enumerate(format_info['sample_data'][:2]):
                        st.write(f"Row {i+1}: {row}")
                
                # Allow users to adjust mappings
                st.subheader("Column Mapping")
                st.write("Confirm or adjust the column mappings:")
                
                columns = df.columns.tolist()
                
                col_map1, col_map2 = st.columns(2)
                
                with col_map1:
                    date_col = st.selectbox(
                        "Date Column *",
                        columns,
                        index=columns.index(format_info['suggested_mappings']['date']) if format_info['suggested_mappings'].get('date') in columns else 0,
                        help="Column containing dates/timestamps"
                    )
                    
                    volume_col = st.selectbox(
                        "Volume Column *",
                        columns,
                        index=columns.index(format_info['suggested_mappings']['volume']) if format_info['suggested_mappings'].get('volume') in columns else 1,
                        help="Column containing service request volumes"
                    )
                
                with col_map2:
                    region_col = st.selectbox(
                        "Region Column (optional)",
                        ['None'] + columns,
                        index=columns.index(format_info['suggested_mappings']['region']) + 1 if format_info['suggested_mappings'].get('region') in columns else 0,
                        help="Column containing regions/locations"
                    )
                    
                    product_col = st.selectbox(
                        "Product Column (optional)",
                        ['None'] + columns,
                        index=columns.index(format_info['suggested_mappings']['product']) + 1 if format_info['suggested_mappings'].get('product') in columns else 0,
                        help="Column containing product categories"
                    )
                
                # Process the data with mappings
                if st.button("Process Data with These Mappings", type="primary"):
                    with st.spinner("Processing data..."):
                        mappings = {
                            'date': date_col,
                            'volume': volume_col
                        }
                        
                        if region_col != 'None':
                            mappings['region'] = region_col
                        if product_col != 'None':
                            mappings['product'] = product_col
                        
                        result = data_processor.process_time_series_data(df, mappings)
                        
                        if result['success']:
                            st.success(f"✅ Successfully processed {result['records_processed']} records!")
                            
                            # Show processing summary
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write("**Date Range:**")
                                st.write(f"From: {result['date_range']['start'].strftime('%Y-%m-%d')}")
                                st.write(f"To: {result['date_range']['end'].strftime('%Y-%m-%d')}")
                            
                            with col2:
                                st.write("**Volume Statistics:**")
                                st.write(f"Min: {result['volume_stats']['min']}")
                                st.write(f"Max: {result['volume_stats']['max']}")
                                st.write(f"Average: {result['volume_stats']['mean']:.1f}")
                                st.write(f"Total: {result['volume_stats']['total']:,}")
                            
                            # Train models
                            if st.button("Train Forecasting Models", type="secondary"):
                                with st.spinner("Training models with new data..."):
                                    if model.train_with_new_data(data_processor.load_service_history()):
                                        st.success("Models successfully trained with new data!")
                                        st.rerun()
                                    else:
                                        st.error("Failed to train models. Please check data format.")
                        
                        else:
                            st.error(f"Error processing data: {result['error']}")
            
            elif format_info['format_type'] == 'unknown':
                st.warning("⚠️ **Unknown format detected.** Please verify your CSV has:")
                st.write("- A date/time column")
                st.write("- A numeric volume/count column")
                st.write("- At least 2 columns total")
                
                st.write("**Available columns:**")
                for col in format_info['columns']:
                    st.write(f"- `{col}`")
                
                # Manual mapping interface
                st.subheader("Manual Column Mapping")
                columns = df.columns.tolist()
                
                col_map1, col_map2 = st.columns(2)
                
                with col_map1:
                    date_col = st.selectbox("Date Column *", columns, help="Column containing dates/timestamps")
                    volume_col = st.selectbox("Volume Column *", columns, help="Column containing service request volumes")
                
                with col_map2:
                    region_col = st.selectbox("Region Column (optional)", ['None'] + columns, help="Column containing regions/locations")
                    product_col = st.selectbox("Product Column (optional)", ['None'] + columns, help="Column containing product categories")
                
                if st.button("Process Data with Manual Mapping", type="primary"):
                    with st.spinner("Processing data..."):
                        mappings = {
                            'date': date_col,
                            'volume': volume_col
                        }
                        
                        if region_col != 'None':
                            mappings['region'] = region_col
                        if product_col != 'None':
                            mappings['product'] = product_col
                        
                        result = data_processor.process_time_series_data(df, mappings)
                        
                        if result['success']:
                            st.success(f"✅ Successfully processed {result['records_processed']} records!")
                            
                            # Train models
                            if st.button("Train Forecasting Models"):
                                with st.spinner("Training models..."):
                                    if model.train_with_new_data(data_processor.load_service_history()):
                                        st.success("Models successfully trained!")
                                        st.rerun()
                                    else:
                                        st.error("Failed to train models.")
                        else:
                            st.error(f"Error: {result['error']}")
            
            else:
                st.error(f"Error analyzing file: {format_info.get('error', 'Unknown error')}")
        
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            st.write("**Troubleshooting tips:**")
            st.write("- Ensure the file is a valid CSV")
            st.write("- Check that columns have headers")
            st.write("- Verify the file is not corrupted")
    
    # Current model data overview
    st.subheader("Current Model Data")
    
    if model.historical_data is not None and not model.historical_data.empty:
        historical_data = model.historical_data
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Total records:** {len(historical_data)}")
            st.write(f"**Date range:** {historical_data['date'].min()} to {historical_data['date'].max()}")
            
        with col2:
            st.write(f"**Total volume:** {historical_data['service_requests'].sum():,}")
            st.write(f"**Average daily volume:** {historical_data.groupby('date')['service_requests'].sum().mean():.1f}")
        
        st.dataframe(historical_data.tail(10))
        
        # Data export
        csv_data = historical_data.to_csv(index=False)
        st.download_button(
            label="Download Model Data",
            data=csv_data,
            file_name=f"model_data_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
        
        # Data quality check
        st.subheader("Data Quality")
        data_stats = {
            'Missing values': historical_data.isnull().sum().sum(),
            'Duplicate dates': historical_data.duplicated(subset=['date']).sum(),
            'Zero volume days': (historical_data['service_requests'] == 0).sum(),
            'Negative values': (historical_data['service_requests'] < 0).sum()
        }
        
        for stat, value in data_stats.items():
            color = "red" if value > 0 else "green"
            st.write(f"**{stat}:** :{color}[{value}]")
        
        # Model retrain option
        st.subheader("Model Management")
        if st.button("Retrain Models", type="secondary"):
            with st.spinner("Retraining models..."):
                if model.retrain_models():
                    st.success("Models retrained successfully!")
                    st.rerun()
                else:
                    st.error("Failed to retrain models")
    
    else:
        st.info("No data loaded in model. Upload data to get started.")
    
    # Forecast history
    st.subheader("Forecast History")
    
    if st.session_state.forecasts_history:
        forecast_df = pd.DataFrame(st.session_state.forecasts_history)
        st.dataframe(forecast_df.tail(10))
        
        # Export forecast history
        csv_data = forecast_df.to_csv(index=False)
        st.download_button(
            label="Download Forecast History",
            data=csv_data,
            file_name=f"forecast_history_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    else:
        st.info("No forecasts generated yet.")

elif page == "Model Performance":
    st.header("Forecasting Model Performance")
    
    # Model metrics
    performance_metrics = model.get_performance_metrics()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Prophet Model Metrics")
        st.metric("Mean Absolute Error", f"{performance_metrics['prophet_mae']:.2f}")
        st.metric("Root Mean Square Error", f"{performance_metrics['prophet_rmse']:.2f}")
    
    with col2:
        st.subheader("ARIMA Model Metrics")
        st.metric("Mean Absolute Error", f"{performance_metrics['arima_mae']:.2f}")
        st.metric("Root Mean Square Error", f"{performance_metrics['arima_rmse']:.2f}")
    
    # Overall metrics
    st.subheader("Overall Performance")
    
    col3, col4, col5 = st.columns(3)
    
    with col3:
        st.metric("Data Points", performance_metrics['data_points'])
    
    with col4:
        st.metric("Forecast Accuracy", performance_metrics['forecast_accuracy'])
    
    with col5:
        st.metric("Models Available", "2" if model.prophet_model and model.arima_model else "1" if model.prophet_model or model.arima_model else "0")
    
    # Model components
    st.subheader("Forecasting Components")
    model_components = model.get_model_components()
    
    if not model_components.empty:
        fig_components = px.bar(
            model_components,
            x='importance',
            y='component',
            orientation='h',
            title="Time Series Components Importance",
            hover_data=['description']
        )
        st.plotly_chart(fig_components, use_container_width=True)
        
        # Components details
        st.dataframe(model_components)
    
    # Model comparison
    if model.prophet_model and model.arima_model:
        st.subheader("Model Comparison")
        
        comparison_data = {
            'Model': ['Prophet', 'ARIMA'],
            'MAE': [performance_metrics['prophet_mae'], performance_metrics['arima_mae']],
            'RMSE': [performance_metrics['prophet_rmse'], performance_metrics['arima_rmse']],
            'Complexity': ['High', 'Medium'],
            'Seasonality': ['Automatic', 'Manual']
        }
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df)
        
        # Performance comparison chart
        fig_comparison = go.Figure()
        
        models = ['Prophet', 'ARIMA']
        mae_values = [performance_metrics['prophet_mae'], performance_metrics['arima_mae']]
        rmse_values = [performance_metrics['prophet_rmse'], performance_metrics['arima_rmse']]
        
        fig_comparison.add_trace(go.Bar(
            name='MAE',
            x=models,
            y=mae_values,
            marker_color='blue'
        ))
        
        fig_comparison.add_trace(go.Bar(
            name='RMSE',
            x=models,
            y=rmse_values,
            marker_color='red'
        ))
        
        fig_comparison.update_layout(
            title="Model Performance Comparison",
            yaxis_title="Error Value",
            barmode='group'
        )
        
        st.plotly_chart(fig_comparison, use_container_width=True)
    
    # Model info
    st.subheader("Model Information")
    model_info = model.get_model_info()
    st.json(model_info)
    
    # Model management
    st.subheader("Model Management")
    
    col6, col7 = st.columns(2)
    
    with col6:
        if st.button("Retrain Models", type="primary"):
            with st.spinner("Retraining forecasting models..."):
                if model.retrain_models():
                    st.success("Models retrained successfully!")
                    st.rerun()
                else:
                    st.error("Failed to retrain models")
    
    with col7:
        if st.button("Reset to Sample Data", type="secondary"):
            with st.spinner("Resetting to sample data..."):
                model._initialize_with_sample_data()
                st.success("Models reset to sample data!")
                st.rerun()

# Footer
st.markdown("---")
st.markdown("**Service Volume Forecasting App** | Powered by Time Series Forecasting")
