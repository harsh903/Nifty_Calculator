import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# App configuration
st.set_page_config(page_title="Nifty Options Calculator", layout="wide")
st.title("Nifty 50 Weekly Options Calculator")
st.markdown("""
Calculate next week's projected options selling limits using latest volatility measures.
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

# Load data with automatic freshness
@st.cache_data
def load_data():
    df = yf.download('^NSEI', period='5y', interval='1wk')
    df['Last_Update'] = df.index[-1].strftime('%Y-%m-%d')
    return df

# Main calculation function
def calculate_limits():
    data = load_data()
    latest_close = data['Close'].iloc[-1]
    validity_date = (pd.to_datetime(data.index[-1]) + pd.DateOffset(weeks=1)).strftime('%Y-%m-%d')
    
    # Calculate indicators
    data['ATR'] = calculate_atr(data)
    data['SD'] = calculate_sd(data)
    
    # Get latest values
    current_atr = data['ATR'].iloc[-1]
    current_sd = data['SD'].iloc[-1]
    
    return latest_close, current_atr, current_sd, validity_date, data['Last_Update'].iloc[-1]

# Confidence levels configuration
confidence_levels = {
    '68% (1σ)': 1,
    '80% (1.28σ)': 1.28,
    '90% (1.645σ)': 1.645,
    '95% (2σ)': 2
}

# Calculate and display results
if st.button('Calculate Latest Projections'):
    with st.spinner('Analyzing market volatility...'):
        price, atr, sd, validity_date, last_data_date = calculate_limits()
        
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
        
        # Display data freshness
        st.subheader(f"Data Currentness")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Latest Data Date:** {last_data_date}")
        with col2:
            st.markdown(f"**Valid Until:** {validity_date}")
        
        # Display results with simplified layout
        st.subheader("Projected Weekly Limits")
        
        # Combined view of both metrics
        st.dataframe(df[[
            'Confidence Level',
            'ATR Lower', 'ATR Upper',
            'SD Lower', 'SD Upper'
        ]].style.format({
            'ATR Lower': '{:.2f}',
            'ATR Upper': '{:.2f}',
            'SD Lower': '{:.2f}',
            'SD Upper': '{:.2f}'
        }))

        # Display current price for reference
        st.info(f"Current Nifty Price: {price:.2f}")

# Sidebar information
with st.sidebar:
    st.header("Methodology")
    st.markdown("""
    **Calculations based on:**
    - 5 years of weekly Nifty 50 data
    - 14-period ATR (Average True Range)
    - 14-period Standard Deviation
    - Multiple confidence levels
    """)
    
    st.markdown("""
    **Validity:**
    - Projections valid until next weekly close
    - Updated automatically with market data
    """)

# Instructions
st.markdown("""
**How to Use:**
1. Click 'Calculate Latest Projections'
2. Check data currency above results
3. Compare both volatility measures
4. Use values for options strategy planning
""")