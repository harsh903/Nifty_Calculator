import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# App configuration
st.set_page_config(page_title="Nifty Options Calculator", layout="wide")
st.title("Nifty 50 Options Selling Limits Calculator")
st.markdown("""
Calculate projected upper/lower limits using ATR and Standard Deviation with different confidence levels.
""")

# Date selection in sidebar
with st.sidebar:
    st.header("Data Range Selection")
    start_date = st.date_input("Start Date", value=pd.to_datetime('2018-01-01'))
    end_date = st.date_input("End Date", value=pd.to_datetime('today'))
    
    st.markdown("""
    **Confidence Levels:**
    - 68%: 1 Standard Deviation
    - 80%: 1.28 SD
    - 90%: 1.645 SD
    - 95%: 2 SD
    """)

# Utility functions
def calculate_atr(data, period=14):
    high_low = data['High'] - data['Low']
    high_close = np.abs(data['High'] - data['Close'].shift())
    low_close = np.abs(data['Low'] - data['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def calculate_sd(data, period=14):
    returns = data['Close'].pct_change()
    return returns.rolling(period).std()

# Load data with date selection
@st.cache_data
def load_data(start_date, end_date):
    return yf.download('^NSEI', start=start_date, end=end_date, interval='1wk')

# Main calculation function
def calculate_limits(start_date, end_date):
    data = load_data(start_date, end_date)
    latest_close = data['Close'].iloc[-1]
    
    # Calculate indicators
    data['ATR'] = calculate_atr(data)
    data['SD'] = calculate_sd(data)
    
    # Get latest values
    current_atr = data['ATR'].iloc[-1]
    current_sd = data['SD'].iloc[-1]
    
    return latest_close, current_atr, current_sd

# Confidence levels configuration
confidence_levels = {
    '68% (1σ)': 1,
    '80% (1.28σ)': 1.28,
    '90% (1.645σ)': 1.645,
    '95% (2σ)': 2
}

# Calculate and display results
if st.button('Calculate Projected Limits'):
    with st.spinner('Calculating...'):
        price, atr, sd = calculate_limits(start_date, end_date)
        
        # Create results dataframe
        results = []
        for cl_name, cl in confidence_levels.items():
            atr_upper = price + (atr * cl)
            atr_lower = price - (atr * cl)
            sd_upper = price * (1 + (sd * cl))
            sd_lower = price * (1 - (sd * cl))
            
            results.append({
                'Confidence Level': cl_name,
                'ATR Upper': round(atr_upper, 2),
                'ATR Lower': round(atr_lower, 2),
                'SD Upper': round(sd_upper, 2),
                'SD Lower': round(sd_lower, 2)
            })
            
        df = pd.DataFrame(results)
        
        # Display results
        st.subheader(f"Current Nifty Level: {price:.2f}")
        st.subheader("Projected Limits for Next Week")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### ATR-based Limits")
            st.dataframe(df[['Confidence Level', 'ATR Upper', 'ATR Lower']]
                        .style.format({'ATR Upper': '{:.2f}', 'ATR Lower': '{:.2f}'}))
            
        with col2:
            st.markdown("### SD-based Limits")
            st.dataframe(df[['Confidence Level', 'SD Upper', 'SD Lower']]
                        .style.format({'SD Upper': '{:.2f}', 'SD Lower': '{:.2f}'}))
        
        # Visualization (Fixed plotting code)
        st.subheader("Visual Comparison")
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Prepare data for plotting
        categories = list(confidence_levels.keys())
        atr_ranges = df['ATR Upper'] - df['ATR Lower']
        sd_ranges = df['SD Upper'] - df['SD Lower']

        x = np.arange(len(categories))
        width = 0.35
        
        ax.bar(x - width/2, atr_ranges, width, label='ATR Range', color='#1f77b4')
        ax.bar(x + width/2, sd_ranges, width, label='SD Range', color='#ff7f0e')
        
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.set_ylabel('Price Range')
        ax.set_title('Volatility Range Comparison')
        ax.legend()
        
        st.pyplot(fig)
        
        # Disclaimer
        st.markdown("""
        **Disclaimer:**  
        These projections are based on historical volatility measures and should not be considered as financial advice. 
        Actual market movements may vary significantly.
        """)

# How to Run
st.markdown("""
**How to Use:**
1. Select date range in sidebar
2. Click the 'Calculate Projected Limits' button
3. View results in tables and charts
4. Compare ATR and SD-based projections
""")
