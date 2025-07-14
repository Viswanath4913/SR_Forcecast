import pandas as pd
import numpy as np
import os
from datetime import datetime
import logging

class DataProcessor:
    def __init__(self):
        self.data_dir = "data"
        self.service_history_file = os.path.join(self.data_dir, "service_history.csv")
        self.predictions_log_file = os.path.join(self.data_dir, "predictions_log.csv")
        self._ensure_data_directory()
    
    def _ensure_data_directory(self):
        """Ensure data directory exists"""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
    
    def log_prediction(self, prediction_data):
        """Log prediction to CSV file"""
        try:
            # Convert prediction data to DataFrame
            df = pd.DataFrame([prediction_data])
            
            # Append to existing file or create new one
            if os.path.exists(self.predictions_log_file):
                df.to_csv(self.predictions_log_file, mode='a', header=False, index=False)
            else:
                df.to_csv(self.predictions_log_file, mode='w', header=True, index=False)
            
            return True
        except Exception as e:
            logging.error(f"Error logging prediction: {str(e)}")
            return False
    
    def load_historical_data(self):
        """Load historical prediction data"""
        try:
            if os.path.exists(self.predictions_log_file):
                df = pd.read_csv(self.predictions_log_file)
                return df
            else:
                return pd.DataFrame()
        except Exception as e:
            logging.error(f"Error loading historical data: {str(e)}")
            return pd.DataFrame()
    
    def load_service_history(self):
        """Load service history data"""
        try:
            if os.path.exists(self.service_history_file):
                df = pd.read_csv(self.service_history_file)
                return df
            else:
                return pd.DataFrame()
        except Exception as e:
            logging.error(f"Error loading service history: {str(e)}")
            return pd.DataFrame()
    
    def detect_csv_format(self, uploaded_df):
        """Detect the format of uploaded CSV and suggest column mappings"""
        try:
            columns = uploaded_df.columns.tolist()
            suggested_mappings = {}
            format_type = "unknown"
            
            # Common column name variations for time series data
            date_variations = ['date', 'time', 'timestamp', 'datetime', 'ds', 'period', 'year', 'month', 'day']
            volume_variations = ['volume', 'count', 'requests', 'service_requests', 'tickets', 'calls', 'y', 'demand', 'quantity', 'amount']
            
            # Try to identify date column
            for col in columns:
                col_lower = col.lower().strip()
                for date_var in date_variations:
                    if date_var in col_lower:
                        suggested_mappings['date'] = col
                        break
                if 'date' in suggested_mappings:
                    break
            
            # Try to identify volume column
            for col in columns:
                col_lower = col.lower().strip()
                for vol_var in volume_variations:
                    if vol_var in col_lower:
                        suggested_mappings['volume'] = col
                        break
                if 'volume' in suggested_mappings:
                    break
            
            # Check if it's time series format
            if 'date' in suggested_mappings and 'volume' in suggested_mappings:
                format_type = "time_series"
            elif len(columns) == 2:
                # Assume first column is date, second is volume for 2-column files
                suggested_mappings['date'] = columns[0]
                suggested_mappings['volume'] = columns[1]
                format_type = "time_series"
            
            # Additional columns detection
            region_variations = ['region', 'location', 'area', 'zone', 'country', 'city']
            product_variations = ['product', 'category', 'type', 'model', 'device']
            
            for col in columns:
                col_lower = col.lower().strip()
                if any(var in col_lower for var in region_variations):
                    suggested_mappings['region'] = col
                elif any(var in col_lower for var in product_variations):
                    suggested_mappings['product'] = col
            
            return {
                'format_type': format_type,
                'columns': columns,
                'suggested_mappings': suggested_mappings,
                'sample_data': uploaded_df.head(3).to_dict('records')
            }
            
        except Exception as e:
            return {
                'format_type': 'error',
                'error': str(e),
                'columns': [],
                'suggested_mappings': {},
                'sample_data': []
            }
    
    def process_time_series_data(self, uploaded_df, column_mappings):
        """Process uploaded time series data with user-defined column mappings"""
        try:
            processed_df = uploaded_df.copy()
            
            # Validate required mappings
            if 'date' not in column_mappings or 'volume' not in column_mappings:
                raise ValueError("Date and Volume columns must be mapped")
            
            # Rename columns to standard format
            rename_dict = {
                column_mappings['date']: 'ds',
                column_mappings['volume']: 'y'
            }
            
            # Add optional columns if mapped
            if 'region' in column_mappings and column_mappings['region']:
                rename_dict[column_mappings['region']] = 'region'
            if 'product' in column_mappings and column_mappings['product']:
                rename_dict[column_mappings['product']] = 'product'
            
            processed_df = processed_df.rename(columns=rename_dict)
            
            # Convert date column
            try:
                processed_df['ds'] = pd.to_datetime(processed_df['ds'])
            except Exception as e:
                raise ValueError(f"Could not convert date column to datetime: {str(e)}")
            
            # Convert volume to numeric
            try:
                processed_df['y'] = pd.to_numeric(processed_df['y'], errors='coerce')
            except Exception as e:
                raise ValueError(f"Could not convert volume column to numeric: {str(e)}")
            
            # Remove rows with invalid data
            processed_df = processed_df.dropna(subset=['ds', 'y'])
            
            # Ensure positive volumes
            processed_df = processed_df[processed_df['y'] >= 0]
            
            # Sort by date
            processed_df = processed_df.sort_values('ds').reset_index(drop=True)
            
            # Add processing timestamp
            processed_df['upload_timestamp'] = datetime.now()
            
            # Save to service history file
            processed_df.to_csv(self.service_history_file, mode='w', header=True, index=False)
            
            return {
                'success': True,
                'records_processed': len(processed_df),
                'date_range': {
                    'start': processed_df['ds'].min(),
                    'end': processed_df['ds'].max()
                },
                'volume_stats': {
                    'min': processed_df['y'].min(),
                    'max': processed_df['y'].max(),
                    'mean': processed_df['y'].mean(),
                    'total': processed_df['y'].sum()
                }
            }
            
        except Exception as e:
            logging.error(f"Error processing time series data: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def process_uploaded_data(self, uploaded_df):
        """Process and validate uploaded data - legacy method for compatibility"""
        try:
            # First try to detect if it's time series format
            format_info = self.detect_csv_format(uploaded_df)
            
            if format_info['format_type'] == 'time_series':
                # Use automatic mapping for time series
                mappings = format_info['suggested_mappings']
                return self.process_time_series_data(uploaded_df, mappings)
            
            # Fall back to original processing for other formats
            # Data validation
            required_columns = ['product_type', 'brand', 'device_age', 'warranty_status', 
                              'location', 'usage_intensity', 'previous_service_count']
            
            missing_columns = [col for col in required_columns if col not in uploaded_df.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Data cleaning
            processed_df = uploaded_df.copy()
            
            # Handle missing values
            processed_df = processed_df.dropna(subset=required_columns)
            
            # Validate data types
            processed_df['device_age'] = pd.to_numeric(processed_df['device_age'], errors='coerce')
            processed_df['previous_service_count'] = pd.to_numeric(processed_df['previous_service_count'], errors='coerce')
            
            # Remove rows with invalid numeric values
            processed_df = processed_df.dropna(subset=['device_age', 'previous_service_count'])
            
            # Validate categorical values
            valid_product_types = ["Smartphone", "Laptop", "Tablet", "Smart TV", "Gaming Console", 
                                 "Headphones", "Smart Watch", "Camera", "Speaker", "Router"]
            valid_brands = ["Apple", "Samsung", "Sony", "LG", "Dell", "HP", "Lenovo", 
                          "Asus", "Nintendo", "Microsoft", "Google", "Xiaomi", "Huawei"]
            valid_warranty_statuses = ["Active", "Expired", "Extended", "Unknown"]
            valid_locations = ["North America", "Europe", "Asia Pacific", "Latin America", "Middle East", "Africa"]
            valid_usage_intensities = ["Light", "Moderate", "Heavy", "Professional"]
            
            # Filter valid data
            processed_df = processed_df[
                (processed_df['product_type'].isin(valid_product_types)) &
                (processed_df['brand'].isin(valid_brands)) &
                (processed_df['warranty_status'].isin(valid_warranty_statuses)) &
                (processed_df['location'].isin(valid_locations)) &
                (processed_df['usage_intensity'].isin(valid_usage_intensities))
            ]
            
            # Add timestamp
            processed_df['timestamp'] = datetime.now()
            
            # Save processed data
            if os.path.exists(self.service_history_file):
                processed_df.to_csv(self.service_history_file, mode='a', header=False, index=False)
            else:
                processed_df.to_csv(self.service_history_file, mode='w', header=True, index=False)
            
            return len(processed_df)
            
        except Exception as e:
            logging.error(f"Error processing uploaded data: {str(e)}")
            raise e
    
    def get_data_statistics(self):
        """Get statistics about the data"""
        try:
            predictions_df = self.load_historical_data()
            service_df = self.load_service_history()
            
            stats = {
                'total_predictions': len(predictions_df),
                'total_service_records': len(service_df),
                'prediction_date_range': None,
                'service_date_range': None,
                'common_issues': [],
                'brand_distribution': {},
                'product_type_distribution': {}
            }
            
            if not predictions_df.empty:
                if 'timestamp' in predictions_df.columns:
                    predictions_df['timestamp'] = pd.to_datetime(predictions_df['timestamp'])
                    stats['prediction_date_range'] = {
                        'start': predictions_df['timestamp'].min(),
                        'end': predictions_df['timestamp'].max()
                    }
                
                # Get distributions
                if 'brand' in predictions_df.columns:
                    stats['brand_distribution'] = predictions_df['brand'].value_counts().to_dict()
                
                if 'product_type' in predictions_df.columns:
                    stats['product_type_distribution'] = predictions_df['product_type'].value_counts().to_dict()
                
                # Get common issues
                if 'issues' in predictions_df.columns:
                    all_issues = []
                    for issues_str in predictions_df['issues'].dropna():
                        if isinstance(issues_str, str):
                            all_issues.extend(issues_str.split(', '))
                    
                    if all_issues:
                        stats['common_issues'] = pd.Series(all_issues).value_counts().head(10).to_dict()
            
            return stats
            
        except Exception as e:
            logging.error(f"Error getting data statistics: {str(e)}")
            return {}
    
    def clear_all_data(self):
        """Clear all stored data"""
        try:
            if os.path.exists(self.predictions_log_file):
                os.remove(self.predictions_log_file)
            
            if os.path.exists(self.service_history_file):
                os.remove(self.service_history_file)
            
            return True
        except Exception as e:
            logging.error(f"Error clearing data: {str(e)}")
            return False
    
    def export_data(self, data_type='all'):
        """Export data in various formats"""
        try:
            if data_type == 'predictions':
                return self.load_historical_data()
            elif data_type == 'service_history':
                return self.load_service_history()
            else:  # all
                predictions_df = self.load_historical_data()
                service_df = self.load_service_history()
                
                if not predictions_df.empty and not service_df.empty:
                    # Combine datasets
                    combined_df = pd.concat([predictions_df, service_df], ignore_index=True, sort=False)
                    return combined_df
                elif not predictions_df.empty:
                    return predictions_df
                elif not service_df.empty:
                    return service_df
                else:
                    return pd.DataFrame()
        except Exception as e:
            logging.error(f"Error exporting data: {str(e)}")
            return pd.DataFrame()
    
    def validate_data_integrity(self):
        """Validate data integrity and consistency"""
        try:
            issues = []
            
            # Check predictions log
            predictions_df = self.load_historical_data()
            if not predictions_df.empty:
                # Check for missing values
                missing_values = predictions_df.isnull().sum()
                if missing_values.sum() > 0:
                    issues.append(f"Missing values in predictions: {missing_values.to_dict()}")
                
                # Check for duplicate entries
                duplicates = predictions_df.duplicated().sum()
                if duplicates > 0:
                    issues.append(f"Duplicate entries in predictions: {duplicates}")
                
                # Check data types
                if 'timestamp' in predictions_df.columns:
                    try:
                        pd.to_datetime(predictions_df['timestamp'])
                    except:
                        issues.append("Invalid timestamp format in predictions")
            
            # Check service history
            service_df = self.load_service_history()
            if not service_df.empty:
                # Similar checks for service history
                missing_values = service_df.isnull().sum()
                if missing_values.sum() > 0:
                    issues.append(f"Missing values in service history: {missing_values.to_dict()}")
            
            return {
                'is_valid': len(issues) == 0,
                'issues': issues
            }
            
        except Exception as e:
            logging.error(f"Error validating data integrity: {str(e)}")
            return {
                'is_valid': False,
                'issues': [f"Error during validation: {str(e)}"]
            }
