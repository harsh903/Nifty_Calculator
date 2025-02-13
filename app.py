import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# App configuration
st.set_page_config(page_title="Nifty Technical Analysis", layout="wide")
st.title("Nifty 50 Technical Analysis")

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["Weekly Volatility", "Technical Levels", "Weekly Levels", "Formula Explanations"])

# Period selection (Global)
period = st.sidebar.number_input("Enter number of weeks:", min_value=1, value=14, step=1)

# Initialize valid_until at the top level
def get_next_thursday(date):
    days_ahead = 3 - date.weekday()  # Thursday is 3
    if days_ahead <= 0:
        days_ahead += 7
    return date + timedelta(days=days_ahead)

def get_previous_thursday(date):
    days_behind = date.weekday() - 3  # Thursday is 3
    if days_behind < 0:
        days_behind += 7
    return date - timedelta(days=days_behind)

valid_until = get_next_thursday(datetime.now().date())

# Confidence levels configuration
confidence_levels = {
    '68% (1σ)': 1,
    '80% (1.28σ)': 1.28,
    '90% (1.645σ)': 1.645,
    '95% (2σ)': 2
}

# Display current date information
current_date = datetime.now().date()
current_thursday = get_next_thursday(current_date)
last_thursday = get_previous_thursday(current_date)

st.sidebar.markdown(f"""
### Date Information
**Current Date:** {current_date}
**Current Week:**
- Start: {last_thursday}
- End: {current_thursday}
""")

@st.cache_data
def load_data(weeks_needed):
    try:
        days_needed = str(int(weeks_needed * 7 + 10)) + 'd'
        df = yf.download('^NSEI', period=days_needed, interval='1d')
        
        if df.empty:
            st.error("Unable to fetch data. Please try again later.")
            return None
            
        # Handle MultiIndex structure
        if isinstance(df.columns, pd.MultiIndex):
            # Select first ticker's data ('^NSEI')
            ticker = df.columns.get_level_values('Ticker')[0]
            
            # Create a new DataFrame with flattened columns
            cleaned_df = pd.DataFrame(index=df.index)
            
            # Map the columns and handle missing data
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                try:
                    cleaned_df[col] = df[(col, ticker)].replace(0, np.nan)
                except KeyError:
                    st.warning(f"Column {col} not found in data")
                    cleaned_df[col] = np.nan
            
            # Add Date column
            cleaned_df = cleaned_df.reset_index()
            cleaned_df.rename(columns={'index': 'Date'}, inplace=True)
            
        else:
            cleaned_df = df.reset_index()
            cleaned_df = cleaned_df.replace(0, np.nan)
        
        # Convert date to datetime
        cleaned_df['Date'] = pd.to_datetime(cleaned_df['Date'])
        
        # Forward fill missing values first
        cleaned_df = cleaned_df.ffill()
        
        # Backward fill any remaining missing values
        cleaned_df = cleaned_df.bfill()
        
        # Handle any remaining NaN values
        cleaned_df = cleaned_df.interpolate(method='linear', limit_direction='both')
        
        # Handle volume separately
        if 'Volume' in cleaned_df.columns:
            cleaned_df['Volume'] = cleaned_df['Volume'].fillna(0)
        
        return cleaned_df
        
    except Exception as e:
        st.error(f"Error in data loading: {str(e)}")
        return None

def calculate_weekly_metrics(daily_data):
    try:
        daily_data = daily_data.copy()
        
        # Verify we have all required data
        required_columns = ['Date', 'Open', 'High', 'Low', 'Close']
        missing_columns = [col for col in required_columns if col not in daily_data.columns]
        if missing_columns:
            st.error(f"Missing required columns: {missing_columns}")
            return None
            
        # Calculate Thursday for grouping
        daily_data['Thursday'] = daily_data['Date'].apply(get_next_thursday)
        
        # Group by Thursday and aggregate
        weekly_data = daily_data.groupby('Thursday').agg({
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Open': 'first'
        }).reset_index()
        
        weekly_data = weekly_data.set_index('Thursday')
        return weekly_data
        
    except Exception as e:
        st.error(f"Error in weekly calculation: {str(e)}")
        return None

def calculate_daily_ma(data, ma_type='EMA', periods=[20, 50, 100, 200]):
    """Calculate Moving Averages on daily data"""
    try:
        ma_dict = {}
        for period in periods:
            if ma_type == 'EMA':
                ma_dict[f'EMA_{period}d'] = data['Close'].ewm(span=period, adjust=False).mean()
            else:  # SMA
                ma_dict[f'SMA_{period}d'] = data['Close'].rolling(window=period).mean()
        return pd.DataFrame(ma_dict).iloc[-1]  # Return only the latest values
    except Exception as e:
        st.error(f"Error in {ma_type} calculation: {str(e)}")
        return None

def calculate_atr(data, period):
    try:
        if len(data) < period:
            return None
        
        high_close = abs(data['High'] - data['Close'].shift(1))
        low_close = abs(data['Low'] - data['Close'].shift(1))
        
        tr = pd.concat([high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=period, min_periods=period).mean()
        
        return atr
    except Exception as e:
        st.error(f"Error in ATR calculation: {str(e)}")
        return None

def calculate_sd(data, period):
    try:
        if len(data) < period:
            return None
        
        returns = data['Close'].pct_change()
        return returns.rolling(window=period, min_periods=period).std()
    except Exception as e:
        st.error(f"Error in SD calculation: {str(e)}")
        return None

def calculate_weekly_fibonacci(data):
    try:
        high = data['High'].max()
        low = data['Low'].min()
        diff = high - low
        
        levels = {
            'Level_0': low,
            'Level_0.236': low + 0.236 * diff,
            'Level_0.382': low + 0.382 * diff,
            'Level_0.5': low + 0.5 * diff,
            'Level_0.618': low + 0.618 * diff,
            'Level_0.786': low + 0.786 * diff,
            'Level_1': high
        }
        return levels
    except Exception as e:
        st.error(f"Error in Fibonacci calculation: {str(e)}")
        return None

def calculate_pivot_points(data):
    """Calculate Pivot Points"""
    try:
        high = data['High'].iloc[-1]
        low = data['Low'].iloc[-1]
        close = data['Close'].iloc[-1]
        
        pivot = (high + low + close) / 3
        r1 = 2 * pivot - low
        r2 = pivot + (high - low)
        r3 = high + 2 * (pivot - low)
        s1 = 2 * pivot - high
        s2 = pivot - (high - low)
        s3 = low - 2 * (high - pivot)
        
        return {
            'R3': r3,
            'R2': r2,
            'R1': r1,
            'Pivot': pivot,
            'S1': s1,
            'S2': s2,
            'S3': s3
        }
    except Exception as e:
        st.error(f"Error in Pivot calculation: {str(e)}")
        return None

# VIX Calculator in sidebar
st.sidebar.header("VIX Weekly Range Calculator")
st.sidebar.markdown("""
#### Formula Used:
```python
Weekly VIX = Annual VIX ÷ √52
Range = Price × (1 ± k × Weekly VIX)
where k = confidence factor
```
""")

vix_value = st.sidebar.number_input("Enter India VIX value:", 
                                   min_value=0.0, 
                                   value=None,
                                   placeholder="Enter VIX...")

if vix_value:
    vix_confidence = st.sidebar.selectbox(
        "Select confidence level:",
        options=list(confidence_levels.keys()),
        key='vix_conf'
    )
    vix_price = st.sidebar.number_input("Enter current Nifty price:", 
                                       min_value=0.0,
                                       value=None,
                                       placeholder="Enter price...")
    
    if vix_price:
        try:
            k = confidence_levels[vix_confidence]
            vix_weekly = vix_value / np.sqrt(52)  # Annual to Weekly
            vix_upper = vix_price * (1 + (k * vix_weekly/100))
            vix_lower = vix_price * (1 - (k * vix_weekly/100))
            
            st.sidebar.markdown(f"""
            **Weekly VIX Range:**
            1. Weekly VIX: {vix_weekly:.4f}%
            2. Upper: {vix_upper:.2f}
            3. Lower: {vix_lower:.2f}
            4. Range: {(vix_upper - vix_lower):.2f}
            """)
        except Exception as e:
            st.sidebar.error(f"Error in VIX calculation: {str(e)}")

# Load data
weeks_needed = period + 2
daily_data = load_data(weeks_needed)

if daily_data is not None:
    weekly_data = calculate_weekly_metrics(daily_data)
    if weekly_data is not None:
        current_price = weekly_data['Close'].iloc[-1]
        latest_date = weekly_data.index[-1]
        valid_until = latest_date + pd.Timedelta(days=7)

        # Display data validity
        st.markdown(f"""
        ### Analysis Information
        - **Latest Data Date:** {latest_date.strftime('%Y-%m-%d')}
        - **Valid Until:** {valid_until.strftime('%Y-%m-%d')}
        - **Trading Days Used:** {period} weeks
        """)

        # Tab 1: Weekly Volatility Analysis
        with tab1:
            st.header("Weekly Volatility Analysis")
            
            # Weekly ATR and SD calculations
            atr_series = calculate_atr(weekly_data, period)
            sd_series = calculate_sd(weekly_data, period)
            
            if atr_series is not None and sd_series is not None:
                current_atr = atr_series.iloc[-1]
                current_sd = sd_series.iloc[-1]
                
                # Display current metrics
                metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                with metrics_col1:
                    st.metric("Current Price", f"₹{current_price:.2f}")
                with metrics_col2:
                    st.metric(f"{period}-Week ATR", f"₹{current_atr:.2f}")
                with metrics_col3:
                    st.metric(f"{period}-Week SD", f"{current_sd*100:.2f}%")

                # ATR ranges
                st.subheader("ATR-Based Weekly Ranges")
                atr_results = []
                for cl_name, cl in confidence_levels.items():
                    atr_upper = round(current_price + (current_atr * cl), 2)
                    atr_lower = round(current_price - (current_atr * cl), 2)
                    atr_move = current_atr * cl
                    atr_move_pct = (current_atr * cl / current_price) * 100
                    
                    atr_results.append({
                        'Confidence': cl_name,
                        'Factor': cl,
                        'Move (₹)': round(atr_move, 2),
                        'Move (%)': f"{round(atr_move_pct, 2)}%",
                        'Lower Limit': atr_lower,
                        'Current': current_price,
                        'Upper Limit': atr_upper,
                    })
                
                st.dataframe(pd.DataFrame(atr_results))
                
                # SD ranges
                st.subheader("SD-Based Weekly Ranges")
                sd_results = []
                for cl_name, cl in confidence_levels.items():
                    sd_move_pct = cl * current_sd * 100
                    sd_move = current_price * (current_sd * cl)
                    sd_upper = round(current_price * (1 + (current_sd * cl)), 2)
                    sd_lower = round(current_price * (1 - (current_sd * cl)), 2)
                    
                    sd_results.append({
                        'Confidence': cl_name,
                        'Factor': cl,
                        'Move (%)': f"{round(sd_move_pct, 2)}%",
                        'Move (₹)': round(sd_move, 2),
                        'Lower Limit': sd_lower,
                        'Current': current_price,
                        'Upper Limit': sd_upper,
                    })
                
                st.dataframe(pd.DataFrame(sd_results))

        # Tab 2: Technical Levels
        with tab2:
            st.header("Technical Levels Analysis")
            
            # Calculate daily moving averages
            ema_values = calculate_daily_ma(daily_data, ma_type='EMA')
            sma_values = calculate_daily_ma(daily_data, ma_type='SMA')
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Daily EMA Levels")
                if ema_values is not None:
                    ema_data = []
                    for name, value in ema_values.items():
                        level_type = "Support" if value < current_price else "Resistance"
                        ema_data.append({
                            'MA Type': name,
                            'Value': round(value, 2),
                            'Type': level_type,
                            'Distance': f"{abs(round((value/current_price - 1) * 100, 2))}%"
                        })
                    st.dataframe(pd.DataFrame(ema_data))
            
            with col2:
                st.subheader("Daily SMA Levels")
                if sma_values is not None:
                    sma_data = []
                    for name, value in sma_values.items():
                        level_type = "Support" if value < current_price else "Resistance"
                        sma_data.append({
                            'MA Type': name,
                            'Value': round(value, 2),
                            'Type': level_type,
                            'Distance': f"{abs(round((value/current_price - 1) * 100, 2))}%"
                        })
                    st.dataframe(pd.DataFrame(sma_data))

            # Pivot Points
            pivot_levels = calculate_pivot_points(weekly_data.tail(1))
            if pivot_levels is not None:
                st.subheader("Weekly Pivot Points")
                pivot_data = []
                for level, value in pivot_levels.items():
                    level_type = "Support" if value < current_price else "Resistance"
                    pivot_data.append({
                        'Level': level,
                        'Value': round(value, 2),
                        'Type': level_type,
                        'Distance': f"{abs(round((value/current_price - 1) * 100, 2))}%"
                    })
                st.dataframe(pd.DataFrame(pivot_data))

        # Tab 3: Weekly Levels
        with tab3:
            st.header("Weekly Levels Analysis")
            
            # Weekly Fibonacci Levels
            fib_levels = calculate_weekly_fibonacci(weekly_data.tail(period))
            if fib_levels is not None:
                st.subheader("Weekly Fibonacci Levels")
                fib_data = []
                for level, value in fib_levels.items():
                    level_type = "Support" if value < current_price else "Resistance"
                    fib_data.append({
                        'Level': level,
                        'Value': round(value, 2),
                        'Type': level_type,
                        'Distance': f"{abs(round((value/current_price - 1) * 100, 2))}%"
                    })
                st.dataframe(pd.DataFrame(fib_data))

            # Summary
            st.subheader("Price Levels Summary")
            st.markdown(f"**Current Price: ₹{current_price:.2f}**")

            # Resistance Levels
            st.markdown("**Resistance Levels:**")
            
            # EMA Resistance
            if ema_values is not None:
                ema_resistances = [v for v in ema_values if v > current_price]
                if ema_resistances:
                    st.markdown(f"- Daily EMA: ₹{min(ema_resistances):.2f}")
                else:
                    st.markdown("- Daily EMA: No resistance found")

            # Pivot Resistance
            if pivot_levels:
                pivot_resistances = [v for v in pivot_levels.values() if v > current_price]
                if pivot_resistances:
                    st.markdown(f"- Pivot: ₹{min(pivot_resistances):.2f}")
                else:
                    st.markdown("- Pivot: No resistance found")

            # Fibonacci Resistance
            if fib_levels:
                fib_resistances = [v for v in fib_levels.values() if v > current_price]
                if fib_resistances:
                    st.markdown(f"- Fibonacci: ₹{min(fib_resistances):.2f}")
                else:
                    st.markdown("- Fibonacci: No resistance found")

            # Support Levels
            st.markdown("\n**Support Levels:**")
            
            # EMA Support
            if ema_values is not None:
                ema_supports = [v for v in ema_values if v < current_price]
                if ema_supports:
                    st.markdown(f"- Daily EMA: ₹{max(ema_supports):.2f}")
                else:
                    st.markdown("- Daily EMA: No support found")

            # Pivot Support
            if pivot_levels:
                pivot_supports = [v for v in pivot_levels.values() if v < current_price]
                if pivot_supports:
                    st.markdown(f"- Pivot: ₹{max(pivot_supports):.2f}")
                else:
                    st.markdown("- Pivot: No support found")

            # Fibonacci Support
            if fib_levels:
                fib_supports = [v for v in fib_levels.values() if v < current_price]
                if fib_supports:
                    st.markdown(f"- Fibonacci: ₹{max(fib_supports):.2f}")
                else:
                    st.markdown("- Fibonacci: No support found")

            # Distance Analysis
            st.markdown("\n**Distance Analysis:**")
            if pivot_resistances and pivot_supports:
                nearest_resistance = min(pivot_resistances)
                nearest_support = max(pivot_supports)
                resistance_distance = abs(round((nearest_resistance/current_price - 1) * 100, 2))
                support_distance = abs(round((nearest_support/current_price - 1) * 100, 2))
                
                st.markdown(f"- Price to Nearest Resistance: {resistance_distance}%")
                st.markdown(f"- Price to Nearest Support: {support_distance}%")
                
                risk_reward = round(resistance_distance/support_distance, 2)
                st.markdown(f"- Risk/Reward Ratio: {risk_reward}")

            # Recent Weekly Data
            st.subheader("Recent Weekly Data")
            display_df = weekly_data.tail(5).copy()
            display_df.index = display_df.index.strftime('%Y-%m-%d')
            display_df = display_df.round(2)
            display_df['Weekly Return%'] = (display_df['Close'].pct_change() * 100).round(2)
            display_df['Weekly Range%'] = ((display_df['High'] - display_df['Low']) / display_df['Low'] * 100).round(2)
            st.dataframe(display_df)

        # Tab 4: Formula Explanations
        with tab4:
            st.header("Technical Analysis Formulas and Methodology")
            
            st.markdown(f"""
            ### Data Information
            - Using {period} weeks of historical data
            - Weekly data from Thursday to Thursday
            - Daily moving averages for better precision
            - Current analysis date: {latest_date.strftime('%Y-%m-%d')}
            - Valid until: {valid_until.strftime('%Y-%m-%d')}
            
            ### Moving Averages (Daily)
            ```python
            # EMA (Exponential Moving Average)
            Multiplier = 2 / (period + 1)
            EMA = Price × Multiplier + Previous_EMA × (1 - Multiplier)

            # SMA (Simple Moving Average)
            SMA = Sum of prices for period / period
            ```
            
            ### Weekly Volatility
            ```python
            # ATR (Average True Range)
            TR = max(
                |High - Previous Close|,
                |Low - Previous Close|
            )
            ATR = Moving Average(TR, period)

            # Standard Deviation (SD)
            Returns = (Close - Previous Close) / Previous Close
            SD = sqrt(sum((Returns - Mean)²) / (period-1))
            Weekly Range = Price × (1 ± k × SD)
            ```
            
            ### Technical Levels
            ```python
            # Fibonacci Retracement
            Range = High - Low
            0% = Low
            23.6% = Low + (Range × 0.236)
            38.2% = Low + (Range × 0.382)
            50.0% = Low + (Range × 0.5)
            61.8% = Low + (Range × 0.618)
            78.6% = Low + (Range × 0.786)
            100% = High

            # Pivot Points
            P = (High + Low + Close) / 3
            R1 = 2P - Low
            R2 = P + (High - Low)
            R3 = High + 2(P - Low)
            S1 = 2P - High
            S2 = P - (High - Low)
            S3 = Low - 2(High - P)
            ```
            
            ### VIX Range
            ```python
            Weekly VIX = Annual VIX / sqrt(52)
            Upper = Price × (1 + k × Weekly VIX)
            Lower = Price × (1 - k × Weekly VIX)
            ```
            """)

# Styling
st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: #4CAF50;
    color: white;
    width: 100%;
}

div.stButton > button:hover {
    background-color: #45a049;
}

div[data-testid="stMetricValue"] {
    font-size: 24px;
}

.formula-box {
    background-color: #f5f5f5;
    padding: 10px;
    border-radius: 5px;
    margin: 10px 0;
}

.date-info {
    background-color: #e1f5fe;
    padding: 10px;
    border-radius: 5px;
    margin: 10px 0;
}
</style>
""", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: #666;'>
Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
<br>
Next update: {valid_until.strftime('%Y-%m-%d')}
</div>
""", unsafe_allow_html=True)