import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression
import streamlit as st
import datetime
from newsapi.newsapi_client import NewsApiClient
import datetime as dt
from dateutil.relativedelta import relativedelta
from dotenv import dotenv_values
import requests
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

config = dotenv_values(".env")
# Function to get data
def get_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)
    data = data['Close'].ffill().bfill()  # Forward-fill NaNs, then backward-fill if needed
    return data

# Function to calculate MRI
def calculate_mri(data, lookback_periods, slope_factor):
    daily_returns = (data - data.shift(1)) / data.shift(1)
    comparison_index = daily_returns[['^VIX', '^MOVE', 'EVZ']]
    mri_list = []

    for lookback in lookback_periods:
        offset_values = {}
        for ticker in comparison_index.columns:
            offset_values[ticker] = -comparison_index[ticker].rolling(window=lookback).sum()
        offset_values_df = pd.DataFrame(offset_values).dropna()

        nm_value = pd.DataFrame(index=offset_values_df.index)
        for ticker in offset_values_df.columns:
            rolling_min = offset_values_df[ticker].rolling(window=lookback).min()
            rolling_max = offset_values_df[ticker].rolling(window=lookback).max()
            nm_value[ticker] = np.where(
                (rolling_max - rolling_min) != 0,
                (offset_values_df[ticker] - rolling_min) / (rolling_max - rolling_min),
                0
            )

        absolute_values = np.abs(2 * nm_value - 1)
        z_scores = pd.DataFrame(index=offset_values_df.index)

        for ticker in comparison_index.columns:
            rolling_mean = data[ticker].rolling(window=lookback).mean()
            rolling_std = data[ticker].rolling(window=lookback).std()
            valid_index = rolling_mean.index.intersection(offset_values_df.index)
            z_scores = z_scores.reindex(valid_index)

            z_scores[ticker] = np.where(
                rolling_std.loc[valid_index] > 0,
                (data[ticker].loc[valid_index] - rolling_mean.loc[valid_index]) / rolling_std.loc[valid_index],
                0
            )

        combo_rhs = -1 * (z_scores * absolute_values).sum(axis=1) / absolute_values.sum(axis=1)
        mri_list.append(combo_rhs)

    mri_avg = pd.concat(mri_list, axis=1).mean(axis=1) * slope_factor
    mri = (0.33 * mri_avg + 0.67 * mri_avg.shift(1)) / 10
    mri.fillna(0, inplace=True)
    return mri

def calculate_mri_slope(mri, lookback_period):
    slopes = pd.Series(index=mri.index, dtype=float)
    for i in range(lookback_period, len(mri)):
        x = np.arange(lookback_period).reshape(-1, 1)
        y = mri.iloc[i-lookback_period:i].values
        
        model = LinearRegression()
        model.fit(x, y)
        slopes.iloc[i] = model.coef_[0]
    slopes.fillna(0, inplace=True)
    return slopes
def calculate_risk_aversion_index(data, lookback_periods):
    daily_returns = (data - data.shift(1)) / data.shift(1)
    comparison_index = daily_returns[['^VIX', '^MOVE', 'EVZ']]
    
    normalisation_indices = []
    
    for lookback in lookback_periods:
        offset_values = {}
        for ticker in comparison_index.columns:
            offset_values[ticker] = -comparison_index[ticker].rolling(window=lookback).sum()
        offset_values_df = pd.DataFrame(offset_values).dropna()

        norm_df = pd.DataFrame(index=offset_values_df.index)
        for ticker in offset_values_df.columns:
            rolling_min = offset_values_df[ticker].rolling(window=lookback).min()
            rolling_max = offset_values_df[ticker].rolling(window=lookback).max()
            norm_df[ticker] = np.where(
                (rolling_max - rolling_min) != 0,
                (offset_values_df[ticker] - rolling_min) / (rolling_max - rolling_min),
                0
            )
        
        combo_values = norm_df.mean(axis=1)
        normalisation_indices.append(combo_values)
    
    combined_normalisation_indices = pd.concat(normalisation_indices, axis=1).mean(axis=1)
    risk_aversion_index = 2 * combined_normalisation_indices
    return risk_aversion_index


def calculate_risk(data, ticker_returns, sign):
    # Initialize only the first value of risk_on and risk_off to 100
    risk_on = pd.Series(index=data.index, dtype=float)
    risk_off = pd.Series(index=data.index, dtype=float)
    risk_on.iloc[0] = 100.0
    risk_off.iloc[0] = 100.0

    # Calculate the risk-on and risk-off values based on the sign and returns
    for i in range(1, len(data)):
        if i >= len(ticker_returns) or i >= len(sign):
            continue

        if sign.iloc[i-1] > 0:
            # Risk-on strategy: increase based on positive returns
            risk_on.iloc[i] = risk_on.iloc[i-1] * (1 + ticker_returns.iloc[i])
            risk_off.iloc[i] = risk_off.iloc[i-1]
        else:
            # Risk-off strategy: decrease based on negative returns
            risk_off.iloc[i] = risk_off.iloc[i-1] * (1 - ticker_returns.iloc[i])
            risk_on.iloc[i] = risk_on.iloc[i-1]
    risk_on = risk_on.ffill().bfill() / 11
    risk_off = risk_off.ffill().bfill() / 5
    return risk_on, risk_off


# Function to calculate ticker performance
def calculate_ticker_performance(data, ticker_returns):
    ticker_performance = pd.Series(100, index=data.index, dtype=float)  # Start with 100
    
    for i in range(1, len(ticker_performance)):
        if i >= len(ticker_returns):
            continue  # Ensure we don't go out of bounds
        current_return = ticker_returns.iloc[i]
        ticker_performance.iloc[i] = ticker_performance.iloc[i-1] * (1 + current_return)
    return ticker_performance

# Function to annotate points
def annotate_points(ax, x, y, label, color):
    highest_idx = y.idxmax()
    highest_val = y.max()
    ax.plot(highest_idx, highest_val, 'o', color=color)
    ax.text(highest_idx, highest_val, f'Highest\n{label}: {highest_val:.2f}\nDate: {highest_idx.strftime("%Y-%m-%d")}', 
            color='black', fontsize=9, ha='right', va='top')
    lowest_idx = y.idxmin()
    lowest_val = y.min()
    ax.plot(lowest_idx, lowest_val, 'o', color=color)
    ax.text(lowest_idx, lowest_val, f'Lowest\n{label}: {lowest_val:.2f}\nDate: {lowest_idx.strftime("%Y-%m-%d")}', 
            color='black', fontsize=9, ha='right', va='bottom')
    today_idx = y.index[-1]
    today_val = y[-1]
    ax.plot(today_idx, today_val, 'o', color=color)
    ax.text(today_idx, today_val, f'Today\n{label}: {today_val:.2f}\nDate: {today_idx.strftime("%Y-%m-%d")}', 
            color='black', fontsize=9, ha='right', va='bottom')

# Function to plot data
def plot_data(data, mri, gspc_ticker='^GSPC', start_date=None, end_date=None):
    if start_date and end_date:
        data = data.loc[start_date:end_date]
        mri = mri.loc[start_date:end_date]
    
    # Determine the date 6 years before the end date
    end_date = pd.to_datetime(end_date)
    start_date_6_years_ago = end_date - pd.DateOffset(years=6)

    # Filter data for the last 6 years
    data = data.loc[start_date_6_years_ago:end_date]
    mri = mri.loc[start_date_6_years_ago:end_date]
    
    plt.figure(figsize=(16, 12))
    
    ax1 = plt.gca()
    ax1.plot(mri.index, mri, label='Keridion MRI', color='blue')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('MRI', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    annotate_points(ax1, mri.index, mri, 'MRI', 'blue')
    
    ax2 = ax1.twinx()
    ax2.plot(data.index, data[gspc_ticker], label='S&P500 Index', color='red')
    annotate_points(ax2, data.index, data[gspc_ticker], 'S&P500 Index', 'red')
    
    ax2.set_ylabel('Price', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    # Center MRI axis around 0
    mri_min, mri_max = mri.min(), mri.max()
    mri_range = max(abs(mri_min), abs(mri_max))
    ax1.set_ylim(-mri_range, mri_range)
    
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    
    plt.title(f"MRI vs S&P500 Price Chart")
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.grid(True)
    plt.tight_layout()
    st.pyplot(plt)

# Function to plot quadrant chart
def plot_quadrant_chart(mri_dif, mri_slope, ticker):
    plt.figure(figsize=(16, 12))
    
    # Create DataFrame for plotting
    df = pd.DataFrame({'MRI': mri_dif - 0.19, 'MRI Slope': mri_slope * 4}, index=pd.to_datetime(mri_dif.index))
    
    # Filter data for the last 12 months
    end_date = df.index.max()
    start_date = end_date - pd.DateOffset(months=12)
    df = df.loc[start_date:end_date]
    
    # Create period-based color mapping
    df['Month'] = df.index.to_period('M')
    months = df['Month'].unique()
    colors = plt.cm.viridis(np.linspace(0, 1, len(months)))
    month_labels = [month.start_time.strftime('%b-%y') for month in months]
    
    # Plot each month's data
    for month, color, label in zip(months, colors, month_labels):
        month_data = df[df['Month'] == month]
        plt.scatter(month_data['MRI'], month_data['MRI Slope'], c=[color], label=label, edgecolors='w')
        plt.plot(month_data['MRI'], month_data['MRI Slope'], color=color, alpha=0.8, linestyle='-', linewidth=1, zorder=1)
    
    # Mark the last entry
    last_entry = df.iloc[-1]
    plt.scatter(last_entry['MRI'], last_entry['MRI Slope'], color='black', s=100, edgecolor='black', zorder=2)
    plt.text(last_entry['MRI'], last_entry['MRI Slope'], 
             f' Last Entry\n({last_entry["MRI"]:.2f}, {last_entry["MRI Slope"]:.2f})', 
             color='Black', fontsize=12, ha='right', va='bottom', bbox=dict(facecolor='white', alpha=0.7))
    
    # Plot axes and labels
    plt.axhline(y=0, color='gray', linestyle='--')
    plt.axvline(x=0, color='gray', linestyle='--')
    plt.xlabel('Market Risk Index (MRI)')
    plt.ylabel('Rising / Falling')
    plt.title(f'MRI vs MRI Slope Quadrant Chart ({end_date.strftime("%Y-%m-%d")})')
    
    # Add grid and legend
    plt.grid(True)
    plt.legend(title='Month', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    st.pyplot(plt)

# Function to plot risk aversion vs returns
def plot_risk_aversion_vs_returns(risk_aversion, returns, ticker):
    valid_mask = ~risk_aversion.isna() & ~returns.isna()
    risk_aversion = risk_aversion[valid_mask] - 0.19
    returns = returns[valid_mask] * 4

    if risk_aversion.empty or returns.empty:
        print("No valid data to plot.")
        return
    
    plt.figure(figsize=(16, 12))
    plt.scatter(risk_aversion, returns, alpha=0.3, label='RAI vs Returns')

    if len(risk_aversion) > 0 and len(returns) > 0:
        model = LinearRegression()
        model.fit(risk_aversion.values.reshape(-1, 1), returns.values.reshape(-1, 1))
        trendline = model.predict(risk_aversion.values.reshape(-1, 1))
        plt.plot(risk_aversion, trendline, color='black', linestyle='--', label='Trendline')

        last_point = len(risk_aversion) - 1
        plt.scatter(risk_aversion.iloc[last_point], returns.iloc[last_point], color='red', label='Last')
        plt.text(risk_aversion.iloc[last_point], returns.iloc[last_point],
                 f'Last\n({risk_aversion.iloc[last_point]:.2f}, {returns.iloc[last_point]:.2f})', 
                 horizontalalignment='right')

    # Set limits to center the origin (0,0)
    xlim = (min(risk_aversion.min(), 0), max(risk_aversion.max(), 0))
    ylim = (min(returns.min(), 0), max(returns.max(), 0))
    
    plt.xlim(xlim)
    plt.ylim(ylim)

    plt.title(f'Risk Aversion Index vs Slope')
    plt.xlabel('Risk Aversion Index')
    plt.ylabel('Rising / Falling')
    plt.grid(True)
    plt.axhline(0, color='black',linewidth=0.5)
    plt.axvline(0, color='black',linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    st.pyplot(plt)

# Function to plot risk on, risk off, and ticker performance
def plot_risk_on_risk_off_performance(risk_on, risk_off, ticker_performance, selected_ticker):
    # Get the current date
    current_date = pd.Timestamp.today()
    
    # Calculate the date 10 years ago from the current date
    ten_years_ago = current_date - pd.DateOffset(years=5)
    
    # Filter the data for the last 10 years
    risk_on_filtered = risk_on[risk_on.index >= ten_years_ago]
    risk_off_filtered = risk_off[risk_off.index >= ten_years_ago]
    ticker_performance_filtered = ticker_performance[ticker_performance.index >= ten_years_ago]
    
    # Plotting
    plt.figure(figsize=(16, 12))
    plt.plot(risk_on_filtered.index, risk_on_filtered, label='Risk On', color='blue')
    plt.plot(risk_off_filtered.index, risk_off_filtered, label='Risk Off', color='orange')
    plt.plot(ticker_performance_filtered.index, ticker_performance_filtered, label=f'{selected_ticker} Performance', color='grey')
    
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title('Risk On, Risk Off, and Ticker Performance (Last 10 Years)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    st.pyplot(plt)
api_key = '4fa92f53a2374022b620325ec7bebe6f'
query = 'S&P 500'

def fetch_news_articles(api_key, query, language='en', page_size=3):
    newsapi = NewsApiClient(api_key=api_key)
    try:
        all_articles = newsapi.get_everything(q=query,
                                              language=language,
                                              sort_by='publishedAt',  # Sort by publication date
                                              page_size=page_size)
        articles = all_articles['articles']
        return articles
    except Exception as e:
        st.error(f"Error fetching news articles: {e}")
        return []

# Function to display news articles
def display_news_articles(articles):
    st.markdown("<style> .news-article { font-size: 12px; } </style>", unsafe_allow_html=True)
    for article in articles:
        st.markdown(
            f"""
            <div class="news-article">
                {article['title']}
                {article['description']}
                <a href="{article['url']}" target="_blank">Read more</a>
                <hr>
            </div>
            """,
            unsafe_allow_html=True
        )

def fetch_news(api_key, query, start_date, prediction_date):
    url = f'https://newsapi.org/v2/everything?q={query}&from={start_date}&to={prediction_date}&apiKey={api_key}'
    response = requests.get(url)
    news_data = response.json()
    return news_data['articles']

def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def get_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

def get_sentiment_ratio(data):
    sentiment_percentage = 0
    sentiment = 'Neutral'
    data['sentiment'] = data['sentiment'].apply(lambda x: 'Positive' if x > 0 else 'Negative')
    positive_sentiment = len(data[data['sentiment']=='Positive'])
    negative_sentiment = len(data[data['sentiment']=='Negative'])
    total_sentiment = positive_sentiment + negative_sentiment
    if(positive_sentiment > negative_sentiment):
        sentiment_percentage = (positive_sentiment / total_sentiment) * 100
        sentiment = 'Positive'
    elif(negative_sentiment > positive_sentiment):
        sentiment_percentage = (negative_sentiment / total_sentiment) * 100
        sentiment = 'Negative'
    return sentiment, sentiment_percentage

def fetch_data(ticker, start_date, prediction_date):
    stock_data = yf.download(tickers=ticker, start=start_date, end=prediction_date)
    stock_data.reset_index(inplace=True)
    return stock_data

def prepare_data(data, feature_cols, target_col):
    input_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_inputs = input_scaler.fit_transform(data[feature_cols])
    scaled_target = target_scaler.fit_transform(data[[target_col]])
    return scaled_inputs, scaled_target, input_scaler, target_scaler

def calculate_sma(data, window=20):
    data[f'SMA_{window}'] = data['Close'].rolling(window=window).mean()
    return data

def calculate_ema(data, window=20):
    data[f'EMA_{window}'] = data['Close'].ewm(span=window, adjust=False).mean()
    return data

def calculate_rsi(data, period=14):
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))
    return data

def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    data['EMA_short'] = data['Close'].ewm(span=short_window, adjust=False).mean()
    data['EMA_long'] = data['Close'].ewm(span=long_window, adjust=False).mean()
    data['MACD'] = data['EMA_short'] - data['EMA_long']
    data['MACD_signal'] = data['MACD'].ewm(span=signal_window, adjust=False).mean()
    data['MACD_histogram'] = data['MACD'] - data['MACD_signal']
    return data

def create_sequences(inputs, targets, sequence_length):
    X, y = [], []
    for i in range(len(inputs) - sequence_length):
        X.append(inputs[i:i + sequence_length])
        y.append(targets[i + sequence_length])
    return np.array(X), np.array(y)

def calculate_performance_metrics(actual, prediction):
    mae = mean_absolute_error(actual, prediction)
    mse = mean_squared_error(actual, prediction)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual, prediction)
    print(f'Mean Absolute Error (MAE): {mae}')
    print(f'Mean Squared Error (MSE): {mse}')
    print(f'Root Mean Squared Error (RMSE): {rmse}')
    print(f'R-squared (R2): {r2}')


def plot_prediction_graph(actual, prediction):
    plt.figure(figsize=(14, 7))
    plt.plot(actual, label='Actual Price')
    plt.plot(prediction, label='Predicted Price')
    plt.title('Actual vs Predicted Stock Prices')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.grid()
    plt.show()

def predict_next_day_close(data, model, input_scaler, target_scaler, sequence_length, features):
    last_sequence = data[-sequence_length:]
    last_sequence_scaled = input_scaler.transform(last_sequence[features])
    last_sequence_scaled = last_sequence_scaled.reshape((1, sequence_length, len(features)))
    predicted_close_scaled = model.predict(last_sequence_scaled)
    predicted_close = target_scaler.inverse_transform(predicted_close_scaled)
    return predicted_close[0][0]

def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(units=100, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(units=80, return_sequences=True),
        Dropout(0.2),
        LSTM(units=50),
        Dropout(0.2),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


# Streamlit layout
st.set_page_config(layout="wide")        
st.header('Keridion Financial Dashboard')
st.sidebar.header('Inputs')
tickers = ['^VIX', '^MOVE', 'EVZ', '^GSPC']

start_date = "2001-09-06"
selected_ticker = '^GSPC'
end_date = st.sidebar.date_input("End Date", datetime.date.today())
user_ticker = '^GSPC'
prediction_date = st.sidebar.date_input("Date to predict the closing price of ^GSPC", datetime.date.today())


# Initialize session state for weights
if 'fast_weight' not in st.session_state:
    st.session_state.fast_weight = 50
if 'slow_weight' not in st.session_state:
    st.session_state.slow_weight = 50

# Define a callback function to update slow weight
def update_slow_weight():
    st.session_state.slow_weight = 100 - st.session_state.fast_weight

# Define a callback function to update fast weight
def update_fast_weight():
    st.session_state.fast_weight = 100 - st.session_state.slow_weight

# Sliders with callback functions
fast_weight = st.sidebar.slider(
    "Fast MRI Weight (%)", 
    min_value=0, max_value=100, 
    value=st.session_state.fast_weight, 
    on_change=update_slow_weight, 
    key='fast_weight'
)
slow_weight = st.sidebar.slider(
    "Slow MRI Weight (%)", 
    min_value=0, max_value=100, 
    value=st.session_state.slow_weight, 
    on_change=update_fast_weight, 
    key='slow_weight'
)


lookback_periods_fast = [5, 10, 22, 66, 90]
lookback_periods_slow = [66, 126, 190, 252, 520]
slope_factor_fast = 10
slope_factor_slow = 22

if st.sidebar.button("Generate Plots"):
    data = get_data(tickers,start_date, end_date)
    
    if data.empty:
        st.error("Failed to load data. Check ticker symbols and date range.")
    else:
        data_mri_fast = calculate_mri(data, lookback_periods_fast, slope_factor_fast)
        data_mri_slow = calculate_mri(data, lookback_periods_slow, slope_factor_slow)

        # Weighted average of fast and slow MRI
        data_mri_combined = (fast_weight * data_mri_fast + slow_weight * data_mri_slow)
        sign = np.sign(data_mri_combined)

        data_mri_combined = data_mri_combined.ffill().bfill()

        data_slope_fast = calculate_mri_slope(data_mri_combined, slope_factor_fast)
        data_slope_slow = calculate_mri_slope(data_mri_combined, slope_factor_slow)
        data_slope_combined = (fast_weight * data_slope_fast + slow_weight * data_slope_slow)

        data_rai_fast = calculate_risk_aversion_index(data, lookback_periods_fast)
        data_rai_slow = calculate_risk_aversion_index(data, lookback_periods_slow)
        data_rai_combined = (data_rai_fast + data_rai_slow) / 2

        daily_returns = data['^GSPC'].pct_change()
        mri_dif = data_mri_combined.diff()

        # Calculate new series
        risk_on, risk_off = calculate_risk(data, daily_returns, sign)
        ticker_performance = calculate_ticker_performance(data, daily_returns)
        # Align columns to the left
        st.markdown("""
        <style>
        .left-align {
            display: flex;
            flex-direction: row;
            justify-content: flex-start;
        }
        </style>
        """, unsafe_allow_html=True)

        # Create columns for layout
        col1, col2 ,col5 = st.columns([2, 2, 2])
        col3, col4 ,col6 = st.columns([2, 2, 2])
     

        # Apply left alignment
        with st.container():
            st.markdown('<div class="left-align">', unsafe_allow_html=True)
            
            with col1:
                plot_data(data, data_mri_combined, gspc_ticker='^GSPC', start_date=start_date, end_date=end_date)
            with col2:
                plot_quadrant_chart(data_mri_combined, data_slope_combined, selected_ticker)

            with col3:
                plot_risk_aversion_vs_returns(data_mri_combined, data_slope_combined, selected_ticker)
            with col4:
                plot_risk_on_risk_off_performance(risk_on, risk_off, ticker_performance, selected_ticker)

            st.markdown('</div>', unsafe_allow_html=True)



    articles = fetch_news_articles(api_key, query)
    if articles:
     with col6:
        # Optional custom CSS for styling
        st.markdown(
            """
            <style>
            .custom-container {
                padding: 10px;
                border: 1px solid #ccc;
                border-radius: 5px;
                background-color: #f9f9f9;
            }
            .expander-text {
               font-size: 12px;  /* Adjust the size as needed */
           }
            </style>
            """,
            unsafe_allow_html=True
        )
        with st.expander('News Articles', expanded=True):
            
            display_news_articles(articles)
            
    else:
     st.write("No articles found.")
 
                
    ticker = '^GSPC'
    today = dt.datetime.today()
    one_week_before = today - relativedelta(days=7)
    ten_year_before = today - relativedelta(years=10)
    stock_start_date = ten_year_before.strftime("%Y-%m-%d")
    news_start_date = one_week_before.strftime("%Y-%m-%d")
    end_date = today.strftime("%Y-%m-%d")
    api_key = '4fa92f53a2374022b620325ec7bebe6f'
    
    news_articles = fetch_news(api_key, 'S&P 500', news_start_date, prediction_date)

    news_df = pd.DataFrame({
        'date': [article['publishedAt'][:10] for article in news_articles],
        'title': [article['title'] for article in news_articles],
        'description': [article['description'] for article in news_articles],
    })

    news_df['text'] = news_df['title'] + ' ' + news_df['description']
    news_df['text'] = news_df['text'].astype(str)
    news_df['sentiment'] = news_df['text'].apply(preprocess_text).apply(get_sentiment)

    news_df['date'] = pd.to_datetime(news_df['date'])
    
    # Compute average sentiment if needed
    news_df['sentiment'] = pd.to_numeric(news_df['sentiment'], errors='coerce')  # Ensure sentiment is numeric
    news_df = news_df.groupby('date').agg({'sentiment': 'mean'}).reset_index()

    data = fetch_data(ticker, stock_start_date, prediction_date)
    data = calculate_sma(data)
    data = calculate_ema(data)
    data = calculate_rsi(data)
    data = calculate_macd(data)
    data.fillna(0, inplace=True)

    features = [i for i in data.columns if i not in ['Close', 'Date']]
    target = 'Close'
    sequence_length = 60

    input_features, input_target, input_scaler, target_scaler = prepare_data(data, features, target)

    X, y = create_sequences(input_features, input_target, sequence_length)

    split = int(len(X)*0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = build_lstm_model((sequence_length, len(features)))
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), callbacks=[EarlyStopping(monitor='val_loss', patience=5)])

    actual = target_scaler.inverse_transform(y_test)
    predictions = model.predict(X_test)
    predictions = target_scaler.inverse_transform(predictions)

    plot_prediction_graph(actual, predictions)
    calculate_performance_metrics(actual, predictions)

    forecast = predict_next_day_close(data, model, input_scaler, target_scaler, sequence_length, features)
    print(f"Next Day Close Price: {forecast:.2f}")

    sentiment, sentiment_percentage = get_sentiment_ratio(news_df)
    print(f"{sentiment_percentage:.2f}% {sentiment}")
    with col5:
              
      st.metric(label="Predicted closing price of stock :", value = f"{forecast:.2f}$")