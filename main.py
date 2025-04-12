import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from meteostat import Point, Daily, Hourly
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import RandomOverSampler
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", message="Support for nested sequences for 'parse_dates'")
warnings.filterwarnings("ignore", message="'H' is deprecated")

# Set page config
st.set_page_config(page_title="Delhi Weather Forecasting Dashboard", layout="wide")

# Global CSS for consistent styling
st.markdown(
    """
    <style>
    /* Base styles */
    body {
        color: #e0e0e0 !important;
        background-color: #0e1117 !important;
    }
    
    /* Title and headers */
    .title {
        text-align: center;
        font-size: 2.5em !important;
        color: #4fc3f7;
        padding: 0.5em;
        border-radius: 8px;
        background-color: #1a1a1a;
        margin-bottom: 0.5em;
        border: 1px solid #444;
        font-weight: bold;
    }
    
    .section-title {
        color: #4fc3f7 !important;
        font-size: 1.25rem !important;
        margin-top: 20px !important;
        margin-bottom: 15px !important;
    }
    
    /* Metrics styling */
    .metric-container {
        background-color: #1a1a1a;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 15px;
        border: 1px solid #444;
    }
    
    .metric-label {
        font-size: 14px;
        color: #b0b0b0;
        margin-bottom: 5px;
    }
    
    .metric-value {
        font-size: 24px;
        color: #4fc3f7;
        font-weight: bold;
    }
    
    .metric-delta {
        font-size: 14px;
        color: #e0e0e0;
    }
    
    /* Text elements */
    .description {
        color: #e0e0e0 !important;
    }
    
    /* Input widgets */
    .stSlider, .stSelectbox, .stRadio {
        color: #e0e0e0 !important;
    }
    
    .st-bb, .st-at, .st-ae {
        background-color: #1a1a1a !important;
    }
    
    /* Plotly chart styling */
    .js-plotly-plot .plotly, .modebar {
        background-color: #1a1a1a !important;
    }
    
    /* Footer */
    footer {
        color: #b0b0b0 !important;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1a1a1a;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #4fc3f7;
        border-radius: 4px;
    }
    
    /* Ensure all text is visible */
    p, div, span, label {
        color: #e0e0e0 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Centered title
st.markdown('<h1 class="title">🌦️ Delhi Weather Forecasting Dashboard</h1>', unsafe_allow_html=True)

# Description with consistent styling
st.markdown(
    """
    <div class="description">
        This dashboard provides weather forecasts for Delhi using machine learning models trained on historical weather data.
        Predict temperature, precipitation probability, and weather conditions for the next day.
    </div>
    """,
    unsafe_allow_html=True
)

# Add a loading spinner while processing
with st.spinner('Loading data and training models...'):
    # [Rest of your existing code for data loading and model training remains the same]
    # Fetch Weather Data (Delhi)
    location = Point(28.6139, 77.2090)  # Delhi
    start = datetime(2015, 1, 1)
    end = datetime(2023, 12, 31)

    # Daily Data
    daily_data = Daily(location, start, end).fetch().reset_index()
    daily_data.rename(columns={
        'time': 'date_time',
        'tavg': 'temperature',
        'prcp': 'precipitation',
        'wspd': 'wind_speed',
        'pres': 'pressure'
    }, inplace=True)

    daily_data = daily_data[['date_time', 'temperature', 'tmin', 'tmax', 'precipitation', 'wind_speed', 'pressure']]
    daily_data.fillna(daily_data.median(numeric_only=True), inplace=True)

    # Hourly Humidity (compute daily average)
    hourly_data = Hourly(location, start, end).fetch().reset_index()
    hourly_data['date_time'] = hourly_data['time'].dt.date
    humidity = hourly_data.groupby('date_time')['rhum'].mean().reset_index()
    humidity.rename(columns={'rhum': 'humidity'}, inplace=True)
    humidity['date_time'] = pd.to_datetime(humidity['date_time'])

    # Merge humidity with daily data
    df = pd.merge(daily_data, humidity, on='date_time', how='left')

    # Add Targets
    df['forecasted_temperature'] = df['temperature'].shift(-1)
    max_precip = df['precipitation'].max()
    df['precipitation_probability'] = df['precipitation'] / (max_precip + 0.001)
    df['forecasted_precip_prob'] = df['precipitation_probability'].shift(-1)
    df['weather_condition'] = np.where(df['precipitation'] > 1.0, 'rainy', 'clear')
    df['forecasted_condition'] = df['weather_condition'].shift(-1)

    # Add Time Features
    df['month'] = df['date_time'].dt.month
    df['day_of_year'] = df['date_time'].dt.dayofyear
    df['season'] = df['month'] % 12 // 3 + 1
    df['is_monsoon'] = df['month'].isin([6, 7, 8, 9]).astype(int)

    # Drop NaNs after shifting
    df.dropna(inplace=True)

    # Model Training & Evaluation
    features = ['temperature', 'tmin', 'tmax', 'wind_speed', 'pressure',
                'precipitation', 'humidity', 'month', 'day_of_year', 'season', 'is_monsoon']

    target_reg_temp = 'forecasted_temperature'
    target_reg_precip = 'forecasted_precip_prob'
    target_cls = 'forecasted_condition'

    X = df[features]

    # Regression - Temperature
    y_temp = df[target_reg_temp]
    X_train_temp, X_test_temp, y_train_temp, y_test_temp = train_test_split(X, y_temp, test_size=0.2, random_state=42)

    # Regression - Precipitation Probability
    y_precip = df[target_reg_precip]
    X_train_precip, X_test_precip, y_train_precip, y_test_precip = train_test_split(X, y_precip, test_size=0.2, random_state=42)

    # Classification - Weather Condition
    y_cond = df[target_cls]
    X_train_cond, X_test_cond, y_train_cond, y_test_cond = train_test_split(X, y_cond, test_size=0.2, random_state=42)

    # Train Models
    reg_temp = RandomForestRegressor(n_estimators=100, random_state=42)
    reg_temp.fit(X_train_temp, y_train_temp)

    reg_precip = RandomForestRegressor(n_estimators=100, random_state=42)
    reg_precip.fit(X_train_precip, y_train_precip)

    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X_train_cond, y_train_cond)

    cls_cond = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    cls_cond.fit(X_resampled, y_resampled)

    # Evaluation
    y_pred_temp = reg_temp.predict(X_test_temp)
    rmse_temp = mean_squared_error(y_test_temp, y_pred_temp, squared=False)

    y_pred_precip = reg_precip.predict(X_test_precip)
    rmse_precip = mean_squared_error(y_test_precip, y_pred_precip, squared=False)

    y_pred_cond = cls_cond.predict(X_test_cond)
    acc_cond = accuracy_score(y_test_cond, y_pred_cond)
    cls_report = classification_report(y_test_cond, y_pred_cond, output_dict=True)

# ======================
# PERFORMANCE METRICS
# ======================

# Centered header
st.markdown('<div class="section-title">Model Performance Overview</div>', unsafe_allow_html=True)

# Metrics cards
metric1, metric2, metric3 = st.columns(3)

with metric1:
    st.markdown(
        f"""
        <div class="metric-container">
            <div class="metric-label">Temperature RMSE</div>
            <div class="metric-value">{rmse_temp:.2f} °C</div>
            <div class="metric-delta">Lower is better</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with metric2:
    st.markdown(
        f"""
        <div class="metric-container">
            <div class="metric-label">Precipitation RMSE</div>
            <div class="metric-value">{rmse_precip:.3f}</div>
            <div class="metric-delta">Lower is better</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with metric3:
    st.markdown(
        f"""
        <div class="metric-container">
            <div class="metric-label">Condition Accuracy</div>
            <div class="metric-value">{acc_cond:.2%}</div>
            <div class="metric-delta">Higher is better</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ======================
# DATA VISUALIZATION
# ======================

st.markdown('<div class="section-title">Weather Data Overview</div>', unsafe_allow_html=True)

# Create a DataFrame for plotting
temp_compare_df = pd.DataFrame({
    'Date': X_test_temp.index,
    'Actual Temperature': y_test_temp.values,
    'Predicted Temperature': y_pred_temp
}).sort_values('Date')

# Plotly line chart with consistent styling
fig_temp = px.line(
    temp_compare_df,
    x='Date',
    y=['Actual Temperature', 'Predicted Temperature'],
    labels={'value': 'Temperature (°C)', 'Date': 'Date'},
    color_discrete_map={
        'Actual Temperature': '#81c784',
        'Predicted Temperature': '#4fc3f7'
    },
    title='Temperature Forecast vs Actual'
)

fig_temp.update_layout(
    template='plotly_dark',
    plot_bgcolor='#1a1a1a',
    paper_bgcolor='#1a1a1a',
    font=dict(color='#e0e0e0'),
    title_font=dict(color='#4fc3f7')
)

st.plotly_chart(fig_temp, use_container_width=True)

# ======================
# PREDICTION INTERFACE
# ======================

st.markdown('<div class="section-title">Weather Forecast Prediction</div>', unsafe_allow_html=True)

# Prediction method selection
prediction_method = st.radio(
    "Prediction Method:",
    ["Select historical date", "Enter parameters manually"],
    horizontal=True,
    label_visibility="visible"
)

if prediction_method == "Select historical date":
    # Date selection
    selected_date = st.date_input(
        "Select historical date",
        min_value=df['date_time'].min().date(),
        max_value=df['date_time'].max().date() - timedelta(days=1),
        value=df['date_time'].max().date() - timedelta(days=30)
    )
    
    # Filter data
    selected_data = df[df['date_time'].dt.date == selected_date]
    
    if not selected_data.empty:
        # Get predictions
        input_data = selected_data[features]
        temp_pred = reg_temp.predict(input_data)[0]
        precip_pred = reg_precip.predict(input_data)[0]
        condition_pred = cls_cond.predict(input_data)[0]
        
        # Current weather metrics
        st.markdown("<div class='section-title'>Current Weather</div>", unsafe_allow_html=True)
        current1, current2, current3 = st.columns(3)
        
        with current1:
            st.markdown(f"""
            <div class='metric-container'>
                <div class='metric-label'>Temperature</div>
                <div class='metric-value'>{selected_data['temperature'].iloc[0]:.1f}°C</div>
            </div>
            """, unsafe_allow_html=True)
        
        with current2:
            st.markdown(f"""
            <div class='metric-container'>
                <div class='metric-label'>Humidity</div>
                <div class='metric-value'>{selected_data['humidity'].iloc[0]:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with current3:
            st.markdown(f"""
            <div class='metric-container'>
                <div class='metric-label'>Wind Speed</div>
                <div class='metric-value'>{selected_data['wind_speed'].iloc[0]:.1f} km/h</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Forecast results
        st.markdown("<div class='section-title'>Forecast Results</div>", unsafe_allow_html=True)
        forecast1, forecast2, forecast3 = st.columns(3)
        
        with forecast1:
            delta_temp = temp_pred-selected_data['temperature'].iloc[0]
            st.markdown(f"""
            <div class='metric-container'>
                <div class='metric-label'>Temperature</div>
                <div class='metric-value'>{temp_pred:.1f}°C</div>
                <div class='metric-delta' style='color: {'#81c784' if delta_temp >= 0 else '#e57373'};'>
                    {delta_temp:+.1f}°C
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with forecast2:
            st.markdown(f"""
            <div class='metric-container'>
                <div class='metric-label'>Precipitation Chance</div>
                <div class='metric-value'>{precip_pred:.1%}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with forecast3:
            condition_icon = "☀️" if condition_pred.lower() == "clear" else "🌧️" if "rain" in condition_pred.lower() else "⛅"
            st.markdown(f"""
            <div class='metric-container'>
                <div class='metric-label'>Expected Condition</div>
                <div class='metric-value'>{condition_icon} {condition_pred.capitalize()}</div>
            </div>
            """, unsafe_allow_html=True)
        
    else:
        st.warning("No data available for selected date")

else:  # Manual input method
    with st.form("manual_input"):
        st.markdown("<div class='section-title'>Enter Weather Parameters</div>", unsafe_allow_html=True)
        
        # Input sliders
        col1, col2 = st.columns(2)
        with col1:
            temperature = st.slider("Temperature (°C)", -10, 50, 25)
            humidity = st.slider("Humidity (%)", 0, 100, 50)
        with col2:
            precipitation = st.slider("Precipitation (mm)", 0, 100, 0)
            month = st.select_slider("Month", options=range(1, 13), value=6)
        
        if st.form_submit_button("Generate Forecast"):
            # Prepare input data
            input_data = pd.DataFrame([[temperature, temperature-5, temperature+5, 10, 1013, 
                                     precipitation, humidity, month, 180, 2, 0]],
                                   columns=features)
            
            # Get predictions
            temp_pred = reg_temp.predict(input_data)[0]
            precip_pred = reg_precip.predict(input_data)[0]
            condition_pred = cls_cond.predict(input_data)[0]
            
            # Display results
            st.markdown("<div class='section-title'>Forecast Results</div>", unsafe_allow_html=True)
            res1, res2, res3 = st.columns(3)
            
            with res1:
                st.markdown(f"""
                <div class='metric-container'>
                    <div class='metric-label'>Temperature</div>
                    <div class='metric-value'>{temp_pred:.1f}°C</div>
                </div>
                """, unsafe_allow_html=True)
            
            with res2:
                st.markdown(f"""
                <div class='metric-container'>
                    <div class='metric-label'>Precipitation Chance</div>
                    <div class='metric-value'>{precip_pred:.1%}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with res3:
                condition_icon = "☀️" if condition_pred.lower() == "clear" else "🌧️" if "rain" in condition_pred.lower() else "⛅"
                st.markdown(f"""
                <div class='metric-container'>
                    <div class='metric-label'>Expected Condition</div>
                    <div class='metric-value'>{condition_icon} {condition_pred.capitalize()}</div>
                </div>
                """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #b0b0b0; font-size: 0.9em;">
        Powered by machine learning models | Data source: Meteostat
    </div>
    """,
    unsafe_allow_html=True
)
