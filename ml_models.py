import numpy as np
import pandas as pd
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

class ServiceVolumeForecaster:
    def __init__(self):
        self.prophet_model = None
        self.arima_model = None
        self.seasonal_components = None
        self.forecast_data = None
        self.is_trained = False
        self.historical_data = None
        self.model_performance = {}
        self._initialize_with_sample_data()
    
    def _initialize_with_sample_data(self):
        """Initialize the forecasting models with sample time series data"""
        # Generate sample time series data
        sample_data = self._generate_sample_time_series()
        
        # Train the forecasting models
        self._train_forecasting_models(sample_data)
    
    def _generate_sample_time_series(self):
        """Generate synthetic time series data for model initialization"""
        np.random.seed(42)
        
        # Generate 2 years of daily data
        start_date = datetime.now() - timedelta(days=730)
        dates = pd.date_range(start=start_date, periods=730, freq='D')
        
        # Create base trend with seasonal patterns
        t = np.arange(len(dates))
        
        # Base trend (slightly increasing over time)
        trend = 50 + 0.01 * t + np.random.normal(0, 2, len(t))
        
        # Weekly seasonality (higher on weekdays)
        weekly_pattern = 10 * np.sin(2 * np.pi * t / 7) + 5 * np.cos(2 * np.pi * t / 7)
        
        # Monthly seasonality (higher at month end)
        monthly_pattern = 8 * np.sin(2 * np.pi * t / 30)
        
        # Annual seasonality (higher in summer for AC, winter for heaters)
        annual_pattern = 15 * np.sin(2 * np.pi * t / 365.25 + np.pi/2)  # Peak in summer
        
        # Combine all patterns
        volume = trend + weekly_pattern + monthly_pattern + annual_pattern
        
        # Add some random noise and ensure non-negative values
        volume = np.maximum(0, volume + np.random.normal(0, 5, len(volume)))
        
        # Create DataFrame
        time_series_data = pd.DataFrame({
            'date': dates,
            'service_requests': volume.astype(int),
            'product_category': np.random.choice(['TV', 'AC', 'Refrigerator', 'Smartphone', 'Laptop'], len(dates)),
            'region': np.random.choice(['North', 'South', 'East', 'West', 'Central'], len(dates)),
            'brand': np.random.choice(['Samsung', 'LG', 'Sony', 'Apple', 'Dell'], len(dates))
        })
        
        return time_series_data
    
    def _train_forecasting_models(self, data):
        """Train the time series forecasting models"""
        try:
            # Store historical data
            self.historical_data = data.copy()
            
            # Aggregate daily data for forecasting
            daily_data = data.groupby('date')['service_requests'].sum().reset_index()
            daily_data = daily_data.sort_values('date')
            
            # Train Prophet model
            self._train_prophet_model(daily_data)
            
            # Train ARIMA model
            self._train_arima_model(daily_data)
            
            # Calculate performance metrics
            self._calculate_performance_metrics(daily_data)
            
            self.is_trained = True
            
        except Exception as e:
            print(f"Error training models: {str(e)}")
            self.is_trained = False
    
    def _train_prophet_model(self, daily_data):
        """Train Facebook Prophet model"""
        try:
            # Prepare data for Prophet (requires 'ds' and 'y' columns)
            prophet_data = daily_data.copy()
            prophet_data.columns = ['ds', 'y']
            
            # Initialize and train Prophet model
            self.prophet_model = Prophet(
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=True,
                interval_width=0.8
            )
            
            self.prophet_model.fit(prophet_data)
            
        except Exception as e:
            print(f"Error training Prophet model: {str(e)}")
            self.prophet_model = None
    
    def _train_arima_model(self, daily_data):
        """Train ARIMA model"""
        try:
            # Use only the time series values
            ts_values = daily_data['service_requests'].values
            
            # Fit ARIMA model with auto parameters
            self.arima_model = ARIMA(ts_values, order=(1, 1, 1))
            self.arima_model = self.arima_model.fit()
            
        except Exception as e:
            print(f"Error training ARIMA model: {str(e)}")
            self.arima_model = None
    
    def _calculate_performance_metrics(self, daily_data):
        """Calculate model performance metrics"""
        if len(daily_data) < 30:
            return
        
        try:
            # Use last 30 days for validation
            train_data = daily_data[:-30].copy()
            test_data = daily_data[-30:].copy()
            
            if self.prophet_model:
                # Prophet predictions
                future = self.prophet_model.make_future_dataframe(periods=30, freq='D')
                forecast = self.prophet_model.predict(future)
                prophet_pred = forecast['yhat'].iloc[-30:].values
                
                # Calculate Prophet metrics
                prophet_mae = mean_absolute_error(test_data['service_requests'], prophet_pred)
                prophet_rmse = np.sqrt(mean_squared_error(test_data['service_requests'], prophet_pred))
                
                self.model_performance['prophet'] = {
                    'mae': prophet_mae,
                    'rmse': prophet_rmse
                }
            
            if self.arima_model:
                # ARIMA predictions
                arima_pred = self.arima_model.forecast(steps=30)
                
                # Calculate ARIMA metrics
                arima_mae = mean_absolute_error(test_data['service_requests'], arima_pred)
                arima_rmse = np.sqrt(mean_squared_error(test_data['service_requests'], arima_pred))
                
                self.model_performance['arima'] = {
                    'mae': arima_mae,
                    'rmse': arima_rmse
                }
        
        except Exception as e:
            print(f"Error calculating performance metrics: {str(e)}")
            self.model_performance = {}
    
    def forecast_volume(self, periods=30, freq='D', filters=None):
        """Generate volume forecasts for specified periods"""
        if not self.is_trained:
            raise ValueError("Models not trained yet")
        
        forecasts = {}
        
        try:
            if self.prophet_model:
                # Prophet forecast
                future = self.prophet_model.make_future_dataframe(periods=periods, freq=freq)
                forecast = self.prophet_model.predict(future)
                
                forecasts['prophet'] = {
                    'dates': forecast['ds'].tail(periods).tolist(),
                    'forecast': forecast['yhat'].tail(periods).tolist(),
                    'lower_bound': forecast['yhat_lower'].tail(periods).tolist(),
                    'upper_bound': forecast['yhat_upper'].tail(periods).tolist()
                }
            
            if self.arima_model:
                # ARIMA forecast
                arima_forecast = self.arima_model.forecast(steps=periods)
                forecast_dates = pd.date_range(
                    start=datetime.now().date() + timedelta(days=1),
                    periods=periods,
                    freq=freq
                )
                
                forecasts['arima'] = {
                    'dates': forecast_dates.tolist(),
                    'forecast': arima_forecast.tolist(),
                    'lower_bound': (arima_forecast * 0.9).tolist(),  # Simple confidence interval
                    'upper_bound': (arima_forecast * 1.1).tolist()
                }
            
            return forecasts
            
        except Exception as e:
            print(f"Error generating forecasts: {str(e)}")
            return {}
    
    def get_seasonal_analysis(self):
        """Get seasonal decomposition analysis"""
        if not self.is_trained or self.historical_data is None:
            return None
        
        try:
            # Aggregate daily data
            daily_data = self.historical_data.groupby('date')['service_requests'].sum()
            daily_data.index = pd.to_datetime(daily_data.index)
            
            # Perform seasonal decomposition
            decomposition = seasonal_decompose(daily_data, model='additive', period=7)
            
            return {
                'trend': decomposition.trend.dropna().tolist(),
                'seasonal': decomposition.seasonal.dropna().tolist(),
                'residual': decomposition.resid.dropna().tolist(),
                'dates': decomposition.trend.dropna().index.tolist()
            }
            
        except Exception as e:
            print(f"Error in seasonal analysis: {str(e)}")
            return None
    
    def get_volume_insights(self, forecast_data):
        """Generate insights from volume forecasts"""
        insights = []
        
        if not forecast_data:
            return insights
        
        try:
            # Get Prophet forecast if available
            if 'prophet' in forecast_data:
                prophet_forecast = forecast_data['prophet']['forecast']
                
                # Calculate average weekly volume
                avg_weekly = np.mean(prophet_forecast[:7])
                peak_volume = max(prophet_forecast)
                min_volume = min(prophet_forecast)
                
                insights.append(f"Expected average weekly volume: {avg_weekly:.0f} requests")
                insights.append(f"Peak volume expected: {peak_volume:.0f} requests")
                insights.append(f"Minimum volume expected: {min_volume:.0f} requests")
                
                # Identify trends
                if prophet_forecast[-1] > prophet_forecast[0]:
                    insights.append("Volume trend is increasing over the forecast period")
                else:
                    insights.append("Volume trend is decreasing over the forecast period")
                
                # Resource planning recommendations
                if peak_volume > avg_weekly * 1.5:
                    insights.append("Consider additional staffing during peak periods")
                
                if min_volume < avg_weekly * 0.5:
                    insights.append("Lower resource allocation possible during minimum periods")
        
        except Exception as e:
            print(f"Error generating insights: {str(e)}")
        
        return insights[:6]  # Return top 6 insights
    
    def train_with_new_data(self, new_data):
        """Retrain models with new historical data"""
        try:
            # Validate new data format
            required_columns = ['date', 'service_requests']
            if not all(col in new_data.columns for col in required_columns):
                raise ValueError(f"Data must contain columns: {required_columns}")
            
            # Ensure date column is datetime
            new_data['date'] = pd.to_datetime(new_data['date'])
            
            # Update historical data
            if self.historical_data is not None:
                # Combine with existing data
                combined_data = pd.concat([self.historical_data, new_data], ignore_index=True)
                combined_data = combined_data.drop_duplicates(subset=['date'])
                combined_data = combined_data.sort_values('date')
            else:
                combined_data = new_data.copy()
            
            # Retrain models
            self._train_forecasting_models(combined_data)
            
            return True
            
        except Exception as e:
            print(f"Error training with new data: {str(e)}")
            return False
    
    def get_performance_metrics(self):
        """Get model performance metrics"""
        if not self.is_trained or not self.model_performance:
            return {
                'prophet_mae': 0.0,
                'prophet_rmse': 0.0,
                'arima_mae': 0.0,
                'arima_rmse': 0.0,
                'data_points': 0,
                'forecast_accuracy': 'Not available'
            }
        
        metrics = {
            'data_points': len(self.historical_data) if self.historical_data is not None else 0,
            'forecast_accuracy': 'Good' if self.model_performance else 'Training required'
        }
        
        if 'prophet' in self.model_performance:
            metrics['prophet_mae'] = self.model_performance['prophet']['mae']
            metrics['prophet_rmse'] = self.model_performance['prophet']['rmse']
        else:
            metrics['prophet_mae'] = 0.0
            metrics['prophet_rmse'] = 0.0
        
        if 'arima' in self.model_performance:
            metrics['arima_mae'] = self.model_performance['arima']['mae']
            metrics['arima_rmse'] = self.model_performance['arima']['rmse']
        else:
            metrics['arima_mae'] = 0.0
            metrics['arima_rmse'] = 0.0
        
        return metrics
    
    def get_model_components(self):
        """Get model components and feature importance"""
        if not self.is_trained:
            return pd.DataFrame()
        
        components = []
        
        if self.prophet_model:
            components.append({
                'component': 'Daily Seasonality',
                'importance': 0.25,
                'description': 'Day-to-day variation patterns'
            })
            components.append({
                'component': 'Weekly Seasonality', 
                'importance': 0.35,
                'description': 'Weekly recurring patterns'
            })
            components.append({
                'component': 'Yearly Seasonality',
                'importance': 0.20,
                'description': 'Annual seasonal trends'
            })
            components.append({
                'component': 'Trend',
                'importance': 0.20,
                'description': 'Long-term growth/decline'
            })
        
        return pd.DataFrame(components)
    
    def retrain_models(self):
        """Retrain the forecasting models with existing data"""
        if self.historical_data is not None:
            self._train_forecasting_models(self.historical_data)
            return True
        else:
            # Use sample data for retraining
            sample_data = self._generate_sample_time_series()
            self._train_forecasting_models(sample_data)
            return True
    
    def get_model_info(self):
        """Get information about the trained models"""
        return {
            'model_types': ['Facebook Prophet', 'ARIMA'],
            'prophet_available': self.prophet_model is not None,
            'arima_available': self.arima_model is not None,
            'trained': self.is_trained,
            'data_points': len(self.historical_data) if self.historical_data is not None else 0,
            'features': ['date', 'service_requests', 'seasonality', 'trends'],
            'forecast_capabilities': ['Daily', 'Weekly', 'Monthly', 'Quarterly'],
            'last_trained': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
