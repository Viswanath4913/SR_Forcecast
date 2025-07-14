import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_risk_gauge(risk_score):
    """Create a risk gauge visualization"""
    
    # Determine color based on risk score
    if risk_score < 0.3:
        color = "green"
    elif risk_score < 0.7:
        color = "yellow"
    else:
        color = "red"
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = risk_score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Risk Score"},
        delta = {'reference': 0.5},
        gauge = {
            'axis': {'range': [None, 1]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 0.3], 'color': "lightgray"},
                {'range': [0.3, 0.7], 'color': "gray"},
                {'range': [0.7, 1], 'color': "lightgray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 0.9
            }
        }
    ))
    
    fig.update_layout(
        height=400,
        title_text="Service Risk Assessment",
        title_x=0.5
    )
    
    return fig

def create_forecast_chart(forecast_data):
    """Create a forecast chart showing service probability over time"""
    
    fig = go.Figure()
    
    # Add probability line
    fig.add_trace(go.Scatter(
        x=forecast_data['date'],
        y=forecast_data['service_probability'],
        mode='lines+markers',
        name='Service Probability',
        line=dict(color='blue', width=3),
        marker=dict(size=8)
    ))
    
    # Add threshold lines
    fig.add_hline(y=0.5, line_dash="dash", line_color="orange", 
                  annotation_text="Medium Risk Threshold")
    fig.add_hline(y=0.7, line_dash="dash", line_color="red", 
                  annotation_text="High Risk Threshold")
    
    # Update layout
    fig.update_layout(
        title="Service Probability Forecast",
        xaxis_title="Date",
        yaxis_title="Service Probability",
        yaxis=dict(range=[0, 1]),
        hovermode='x unified',
        height=500
    )
    
    return fig

def create_trend_analysis(historical_data):
    """Create trend analysis visualization"""
    
    # Prepare data for trend analysis
    if 'timestamp' in historical_data.columns:
        historical_data['timestamp'] = pd.to_datetime(historical_data['timestamp'])
        historical_data['date'] = historical_data['timestamp'].dt.date
        
        # Group by date and calculate daily metrics
        daily_stats = historical_data.groupby('date').agg({
            'service_required': 'mean',
            'risk_score': 'mean',
            'estimated_days': 'mean'
        }).reset_index()
        
        # Create subplot
        fig = go.Figure()
        
        # Add service rate trend
        fig.add_trace(go.Scatter(
            x=daily_stats['date'],
            y=daily_stats['service_required'],
            mode='lines+markers',
            name='Service Required Rate',
            yaxis='y',
            line=dict(color='red')
        ))
        
        # Add risk score trend
        fig.add_trace(go.Scatter(
            x=daily_stats['date'],
            y=daily_stats['risk_score'],
            mode='lines+markers',
            name='Average Risk Score',
            yaxis='y2',
            line=dict(color='blue')
        ))
        
        # Update layout with secondary y-axis
        fig.update_layout(
            title="Prediction Trends Over Time",
            xaxis_title="Date",
            yaxis=dict(
                title="Service Required Rate",
                side="left",
                range=[0, 1]
            ),
            yaxis2=dict(
                title="Average Risk Score",
                side="right",
                overlaying="y",
                range=[0, 1]
            ),
            hovermode='x unified',
            height=500
        )
        
        return fig
    
    else:
        # If no timestamp, create a simple histogram
        fig = px.histogram(
            historical_data,
            x='risk_score',
            nbins=20,
            title="Risk Score Distribution"
        )
        
        return fig

def create_brand_comparison(historical_data):
    """Create brand comparison visualization"""
    
    if 'brand' in historical_data.columns and 'risk_score' in historical_data.columns:
        # Calculate brand statistics
        brand_stats = historical_data.groupby('brand').agg({
            'risk_score': ['mean', 'std', 'count'],
            'service_required': 'mean'
        }).round(3)
        
        # Flatten column names
        brand_stats.columns = ['_'.join(col).strip() for col in brand_stats.columns.values]
        brand_stats = brand_stats.reset_index()
        
        # Create comparison chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=brand_stats['brand'],
            y=brand_stats['risk_score_mean'],
            error_y=dict(type='data', array=brand_stats['risk_score_std']),
            name='Average Risk Score',
            marker_color='lightblue'
        ))
        
        fig.update_layout(
            title="Brand Risk Score Comparison",
            xaxis_title="Brand",
            yaxis_title="Average Risk Score",
            xaxis_tickangle=-45,
            height=500
        )
        
        return fig
    
    else:
        # Return empty figure if data not available
        return go.Figure()

def create_issue_distribution(historical_data):
    """Create issue distribution visualization"""
    
    if 'issues' in historical_data.columns:
        # Extract all issues
        all_issues = []
        for issues_str in historical_data['issues'].dropna():
            if isinstance(issues_str, str):
                all_issues.extend(issues_str.split(', '))
        
        if all_issues:
            # Count issues
            issue_counts = pd.Series(all_issues).value_counts()
            
            # Create pie chart
            fig = go.Figure(data=[go.Pie(
                labels=issue_counts.index,
                values=issue_counts.values,
                hole=0.3
            )])
            
            fig.update_layout(
                title="Distribution of Common Issues",
                height=500
            )
            
            return fig
    
    # Return empty figure if no data
    return go.Figure()

def create_service_timeline(historical_data):
    """Create service timeline visualization"""
    
    if 'timestamp' in historical_data.columns and 'service_required' in historical_data.columns:
        historical_data['timestamp'] = pd.to_datetime(historical_data['timestamp'])
        
        # Filter service required cases
        service_cases = historical_data[historical_data['service_required'] == True]
        
        if not service_cases.empty:
            fig = go.Figure()
            
            # Add scatter plot for service cases
            fig.add_trace(go.Scatter(
                x=service_cases['timestamp'],
                y=service_cases['risk_score'],
                mode='markers',
                marker=dict(
                    size=service_cases['estimated_days']/10,
                    color=service_cases['risk_score'],
                    colorscale='Reds',
                    showscale=True,
                    colorbar=dict(title="Risk Score")
                ),
                text=service_cases['product_type'],
                hovertemplate='<b>%{text}</b><br>' +
                            'Risk Score: %{y:.2f}<br>' +
                            'Date: %{x}<br>' +
                            '<extra></extra>',
                name='Service Cases'
            ))
            
            fig.update_layout(
                title="Service Cases Timeline",
                xaxis_title="Date",
                yaxis_title="Risk Score",
                height=500
            )
            
            return fig
    
    return go.Figure()

def create_performance_dashboard(performance_metrics):
    """Create a performance dashboard with multiple metrics"""
    
    # Create subplots
    fig = go.Figure()
    
    # Metrics to display
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    values = [performance_metrics.get(metric, 0) for metric in metrics]
    
    fig.add_trace(go.Bar(
        x=metrics,
        y=values,
        marker_color=['blue', 'green', 'orange', 'red'],
        text=[f"{val:.3f}" for val in values],
        textposition='auto'
    ))
    
    fig.update_layout(
        title="Model Performance Metrics",
        xaxis_title="Metric",
        yaxis_title="Score",
        yaxis=dict(range=[0, 1]),
        height=400
    )
    
    return fig

def create_correlation_matrix(historical_data):
    """Create correlation matrix heatmap"""
    
    # Select numeric columns
    numeric_cols = historical_data.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) > 1:
        # Calculate correlation matrix
        corr_matrix = historical_data[numeric_cols].corr()
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.values,
            texttemplate="%{text:.2f}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title="Feature Correlation Matrix",
            height=500
        )
        
        return fig
    
    return go.Figure()

def create_seasonal_decomposition(seasonal_data):
    """Create seasonal decomposition visualization"""
    
    if not seasonal_data or 'dates' not in seasonal_data:
        return go.Figure()
    
    fig = go.Figure()
    
    # Add trend component
    if 'trend' in seasonal_data:
        fig.add_trace(go.Scatter(
            x=seasonal_data['dates'],
            y=seasonal_data['trend'],
            name='Trend',
            line=dict(color='blue', width=2)
        ))
    
    # Add seasonal component
    if 'seasonal' in seasonal_data:
        fig.add_trace(go.Scatter(
            x=seasonal_data['dates'],
            y=seasonal_data['seasonal'],
            name='Seasonal',
            line=dict(color='green', width=2)
        ))
    
    # Add residual component
    if 'residual' in seasonal_data:
        fig.add_trace(go.Scatter(
            x=seasonal_data['dates'],
            y=seasonal_data['residual'],
            name='Residual',
            line=dict(color='red', width=1)
        ))
    
    fig.update_layout(
        title="Seasonal Decomposition",
        xaxis_title="Date",
        yaxis_title="Value",
        height=500,
        hovermode='x unified'
    )
    
    return fig

def create_volume_trends_chart(historical_data):
    """Create volume trends chart for time series data"""
    
    if historical_data.empty:
        return go.Figure()
    
    fig = go.Figure()
    
    # Check if we have time series data
    if 'date' in historical_data.columns and 'service_requests' in historical_data.columns:
        # Convert date column to datetime
        historical_data['date'] = pd.to_datetime(historical_data['date'])
        
        # Group by date and sum service requests
        daily_data = historical_data.groupby('date')['service_requests'].sum().reset_index()
        
        # Add main trend line
        fig.add_trace(go.Scatter(
            x=daily_data['date'],
            y=daily_data['service_requests'],
            mode='lines+markers',
            name='Daily Volume',
            line=dict(color='blue', width=2),
            marker=dict(size=4)
        ))
        
        # Add 7-day moving average
        if len(daily_data) >= 7:
            daily_data['ma_7'] = daily_data['service_requests'].rolling(window=7).mean()
            fig.add_trace(go.Scatter(
                x=daily_data['date'],
                y=daily_data['ma_7'],
                mode='lines',
                name='7-Day Average',
                line=dict(color='red', width=2, dash='dash')
            ))
        
        fig.update_layout(
            title="Historical Service Request Volume",
            xaxis_title="Date",
            yaxis_title="Number of Requests",
            height=500,
            hovermode='x unified'
        )
    
    else:
        # Fallback: create a simple message
        fig.add_annotation(
            text="No time series data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        
        fig.update_layout(
            title="Volume Trends",
            height=300
        )
    
    return fig
