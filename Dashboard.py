# Import necessary packages
import yfinance as yf  # For fetching financial data from Yahoo Finance
import pandas as pd  # For data manipulation and analysis
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For creating visualisations
import matplotlib.dates as mdates  # For handling dates in plots
from sklearn.linear_model import LinearRegression  # For linear regression modeling
import streamlit as st  # For creating web applications
import datetime  # For handling dates and times
from newsapi.newsapi_client import NewsApiClient  # For fetching news data
import datetime as dt  # For handling dates and times
from dateutil.relativedelta import relativedelta  # For relative date calculations
from dotenv import dotenv_values  # For loading environment variables
import requests  # For making HTTP requests
from sklearn.preprocessing import MinMaxScaler  # For feature scaling
from tensorflow.keras.models import Sequential  # For creating neural network models
from tensorflow.keras.layers import LSTM, Dense, Dropout  # For LSTM and dense layers
from tensorflow.keras.callbacks import EarlyStopping  # For early stopping during training
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # For evaluating model performance
from textblob import TextBlob  # For sentiment analysis
from nltk.corpus import stopwords  # For removing common words
from nltk.tokenize import word_tokenize  # For tokenizing text
from nltk.stem import WordNetLemmatizer  # For lemmatizing words
import string  # For handling string operations
import nltk  # For natural language processing
nltk.download('punkt')  # Download tokenizer models
nltk.download('stopwords')  # Download stopwords list
nltk.download('wordnet')  # Download WordNet lexicon

# Load environment variables from .env file
config = dotenv_values(".env")

# Function to retrieve historical stock data
def get_data(tickers, start_date, end_date):
    # Download historical data from Yahoo Finance
    data = yf.download(tickers, start=start_date, end=end_date)
    # Select the 'Close' price column from the downloaded data
    data = data['Close'].ffill().bfill()
    return data

# Function to calculate the Market Regime Indicator (MRI)
def calculate_mri(data, lookback_periods, slope_factor):
    # Calculate daily returns as the percentage change from the previous day's closing price
    daily_returns = (data - data.shift(1)) / data.shift(1)
    
    # Extract the daily returns for all tickers except the first one
    comparison_index = daily_returns[tickers[1:]]

    mri_list = [] 
    # Loop through each lookback period to calculate MRI
    for lookback in lookback_periods:
        offset_values = {}
        
        # Compute the sum of daily returns over the lookback period for each ticker
        for ticker in comparison_index.columns:
            offset_values[ticker] = -comparison_index[ticker].rolling(window=lookback).sum()
        
        # Create a DataFrame from the computed offset values and drop rows with NaN values
        offset_values_df = pd.DataFrame(offset_values).dropna()

        # Calculate normalised values (NM) based on rolling min and max
        nm_value = pd.DataFrame(index=offset_values_df.index)
        for ticker in offset_values_df.columns:
            rolling_min = offset_values_df[ticker].rolling(window=lookback).min()
            rolling_max = offset_values_df[ticker].rolling(window=lookback).max()
            nm_value[ticker] = np.where(
                (rolling_max - rolling_min) != 0,
                (offset_values_df[ticker] - rolling_min) / (rolling_max - rolling_min),
                0
            )

        # Calculate absolute values based on normalised values
        absolute_values = np.abs(2 * nm_value - 1)
        
        # Compute z-scores for the original data based on rolling mean and standard deviation
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

        # Calculate the MRI value for this lookback period
        combo_rhs = -1 * (z_scores * absolute_values).sum(axis=1) / absolute_values.sum(axis=1)
        mri_list.append(combo_rhs)

    # Compute the average MRI value across all lookback periods and scale it
    mri_avg = pd.concat(mri_list, axis=1).mean(axis=1) * slope_factor
    
    # Smooth the MRI values and normalise
    mri = (0.33 * mri_avg + 0.67 * mri_avg.shift(1)) / 2000
    mri.fillna(0, inplace=True)
    return mri

# Function to calculate the MRI Slope
def calculate_mri_slope(mri, lookback_period):
    
    # Initialise a Series to store the slope values
    slopes = pd.Series(index=mri.index, dtype=float)
    
    # Loop through the MRI values starting from the end of the lookback period
    for i in range(lookback_period, len(mri)):
        # Define the x values (lookback_period indices) and reshape for the LinearRegression model
        x = np.arange(lookback_period).reshape(-1, 1)
        
        # Extract the y values (MRI values) for the current lookback period
        y = mri.iloc[i-lookback_period:i].values
        
        # Create and fit a LinearRegression model to the x and y values
        model = LinearRegression()
        model.fit(x, y)
        
        # Calculate the slope (coefficient of the regression line) and scale it
        slopes.iloc[i] = model.coef_[0] / 50
    
    # Fill any NaN values with 0
    slopes.fillna(0, inplace=True)
    return slopes

# Function to calculate risk values based on the given returns and sign indicators
def calculate_risk(data, ticker_returns, sign):
    
    # Initialise risk-on and risk-off Series with the same index as data
    risk_on = pd.Series(index=data.index, dtype=float)
    risk_off = pd.Series(index=data.index, dtype=float)
    
    # Set the initial values of risk-on and risk-off to 100
    risk_on.iloc[0] = 100.0
    risk_off.iloc[0] = 100.0

    # Loop through the data starting from the second element
    for i in range(1, len(data)):
        # Check if indices are within bounds for ticker_returns and sign
        if i >= len(ticker_returns) or i >= len(sign):
            continue
        
        # Risk-on strategy: Increase risk-on value based on positive returns
        if sign.iloc[i-1] > 0:
            risk_on.iloc[i] = risk_on.iloc[i-1] * (1 + ticker_returns.iloc[i])
            risk_off.iloc[i] = risk_off.iloc[i-1]
        # Risk-off strategy: Decrease risk-off value based on negative returns
        else:
            risk_off.iloc[i] = risk_off.iloc[i-1] * (1 - ticker_returns.iloc[i])
            risk_on.iloc[i] = risk_on.iloc[i-1]
    
    # Forward fill and backward fill missing values, then adjust the risk values by dividing
    risk_on = risk_on.ffill().bfill() / 11
    risk_off = risk_off.ffill().bfill() / 5
    
    return risk_on, risk_off

# Function to calculate ticker performance based on returns
def calculate_ticker_performance(data, ticker_returns):

    # Initialise the ticker performance Series with a starting value of 100
    ticker_performance = pd.Series(100, index=data.index, dtype=float)
    
    # Loop through the data starting from the second element
    for i in range(1, len(ticker_performance)):
        # Check if the index is within bounds for ticker_returns
        if i >= len(ticker_returns):
            continue
        
        # Get the current return for the ticker
        current_return = ticker_returns.iloc[i]
        
        # Update the ticker performance based on the previous performance and current return
        ticker_performance.iloc[i] = ticker_performance.iloc[i-1] * (1 + current_return)
    
    return ticker_performance

# Function to plot data including MRI and S&P500 index
def plot_data(data, mri, gspc_ticker='^GSPC', start_date=None, end_date=None):

    # Filter data and MRI based on the provided date range
    if start_date and end_date:
        data = data.loc[start_date:end_date]
        mri = mri.loc[start_date:end_date]
    
    # Convert end_date to a datetime object
    end_date = pd.to_datetime(end_date)
    
    # Determine the start date 6 years before the end date
    start_date_6_years_ago = end_date - pd.DateOffset(years=6)

    # Filter data and MRI for the last 6 years
    data = data.loc[start_date_6_years_ago:end_date]
    mri = mri.loc[start_date_6_years_ago:end_date]
    
    # Create a new figure with specified size
    plt.figure(figsize=(16, 12))
    
    # Plot MRI on the primary y-axis
    ax1 = plt.gca()
    ax1.plot(mri.index, mri, label='Keridion MRI', color='blue')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('MRI', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    # Create a secondary y-axis to plot the S&P 500 Index
    ax2 = ax1.twinx()
    ax2.plot(data.index, data[gspc_ticker], label='S&P500 Index', color='red')
    
    ax2.set_ylabel('Price', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    # Center the MRI axis around 0 for better visualisation
    mri_min, mri_max = mri.min(), mri.max()
    mri_range = max(abs(mri_min), abs(mri_max))
    ax1.set_ylim(-mri_range, mri_range)
    
    # Set x-axis ticks to show every 6 months and format the date
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    

    plt.title("MRI vs S&P500 Price Chart")
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.grid(True)
    plt.tight_layout()
    st.pyplot(plt)

# Function to plot quadrant chart
def plot_quadrant_chart(mri_dif, mri_slope, ticker):
    plt.figure(figsize=(16, 12))
    
    # Create DataFrame for plotting
    df = pd.DataFrame({'MRI': mri_dif + 0.44, 'MRI Slope': mri_slope * 4}, index=pd.to_datetime(mri_dif.index))
    
    # Filter data for the last 12 months
    end_date = df.index.max()
    start_date = end_date - pd.DateOffset(months=7)
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
        plt.plot(month_data['MRI'], month_data['MRI Slope'], color=color, alpha=0.8, linestyle='-', linewidth=3, zorder=1)
    
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
    
    # Create a mask to filter out NaN values from both risk_aversion and returns
    valid_mask = ~risk_aversion.isna() & ~returns.isna()
    
    # Adjust risk aversion values and scale returns
    risk_aversion = risk_aversion[valid_mask] - 0.19
    returns = returns[valid_mask] * 4

    # Check if there is valid data to plot
    if risk_aversion.empty or returns.empty:
        print("No valid data to plot.")
        return
    
    plt.figure(figsize=(16, 12))
    
    # Plot risk aversion vs returns as a scatter plot with some transparency
    plt.scatter(risk_aversion, returns, alpha=0.3, label='RAI vs Returns')

    # If there is valid data, fit a linear regression model and plot the trendline
    if len(risk_aversion) > 0 and len(returns) > 0:
        model = LinearRegression()
        model.fit(risk_aversion.values.reshape(-1, 1), returns.values.reshape(-1, 1))
        
        # Predict trendline values and adjust by subtracting 0.05
        trendline = model.predict(risk_aversion.values.reshape(-1, 1)) - 0.05
        plt.plot(risk_aversion, trendline, color='black', linestyle='--', label='Trendline')
        
        # Extract the slope and intercept from the model
        slope = model.coef_[0][0]
        intercept = model.intercept_[0]
        
        # Display the equation of the line
        plt.text(0.05, 0.90, f'y = {slope:.2f}x + {intercept:.2f}', transform=plt.gca().transAxes, 
                 fontsize=14, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
        
        # Calculate and display the R^2 value
        r2 = r2_score(returns, model.predict(risk_aversion.values.reshape(-1, 1)))
        plt.text(0.05, 0.95, f'R² = {r2:.2f}', transform=plt.gca().transAxes, 
                 fontsize=14, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
        
        # Mark and label the last data point
        last_point = len(risk_aversion) - 1
        plt.scatter(risk_aversion.iloc[last_point], returns.iloc[last_point], color='red', label='Last')
        plt.text(risk_aversion.iloc[last_point], returns.iloc[last_point],
                 f'Last\n({risk_aversion.iloc[last_point]:.2f}, {returns.iloc[last_point]:.2f})', 
                 horizontalalignment='right')

    # Set limits to center the origin (0,0) for better visualization
    xlim = (min(risk_aversion.min(), 0), max(risk_aversion.max(), 0))
    ylim = (min(returns.min(), 0), max(returns.max(), 0))
    
    plt.xlim(xlim)
    plt.ylim(ylim)

    plt.title(f'Risk Aversion Index Quadrant')
    plt.xlabel('Risk Aversion Index')
    plt.ylabel('Rising / Falling')
    plt.grid(True)
    
    # Add horizontal and vertical lines at y=0 and x=0 to mark the origin
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    
    plt.legend()
    plt.tight_layout()
    st.pyplot(plt)

# Function to plot risk on, risk off, and ticker performance
def plot_risk_on_risk_off_performance(risk_on, risk_off, ticker_performance, selected_ticker):

    # Get the current date
    current_date = pd.Timestamp.today()
    
    # Calculate the date 5 years ago from the current date
    ten_years_ago = current_date - pd.DateOffset(years=10)
    
    # Filter the data to include only the last 10 years
    risk_on_filtered = risk_on[risk_on.index >= ten_years_ago]
    risk_off_filtered = risk_off[risk_off.index >= ten_years_ago]
    ticker_performance_filtered = ticker_performance[ticker_performance.index >= ten_years_ago]
    
    # Create a new figure with a specified size for the plot
    plt.figure(figsize=(16, 12))
    
    # Plot the risk-on values
    plt.plot(risk_on_filtered.index, risk_on_filtered, label='Risk On', color='blue')
    
    # Plot the risk-off values
    plt.plot(risk_off_filtered.index, risk_off_filtered, label='Risk Off', color='orange')
    
    # Plot the performance of the selected ticker
    plt.plot(ticker_performance_filtered.index, ticker_performance_filtered, 
             label=f'{selected_ticker} Performance', color='grey')
    
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title('Risk On, Risk Off, and Ticker Performance (Last 10 Years)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    st.pyplot(plt)

api_key = '4fa92f53a2374022b620325ec7bebe6f'
query = 'S&P 500'

# Function to fetch the S&P 500 news articles
def fetch_news_articles(api_key, query, language='en', page_size=3):
    newsapi = NewsApiClient(api_key=api_key)
    try:
        all_articles = newsapi.get_everything(
            q=query,
            language=language,
            sort_by='publishedAt',  # Sort articles by publication date
            page_size=page_size
        )
        articles = all_articles['articles']
        return articles
    except Exception as e:
        st.error(f"Error fetching news articles: {e}")
        return []

# Function to display the S&P 500 news articles
def display_news_articles(articles):

    with st.container():
        # Display sentiment information (placeholders used here)
        st.markdown(f"<h2 style='font-size: 18px;'>S&P 500 Sentiments are {sentiment_percentage}% ({sentiment})</h2>", unsafe_allow_html=True)
        
        # Define CSS style for news articles
        st.markdown("<style> .news-article { font-size: 12px; } </style>", unsafe_allow_html=True)
        
        for article in articles:
            # Display each article with its title, description, and a link
            st.markdown(
                f"""
                <div class="news-article">
                    <a href="{article['url']}" target="_blank">{article['title']}</a>
                    <br>
                    {article['description']}
                    <hr>
                </div>
                """,
                unsafe_allow_html=True
            )

# Function to fetch the news articles for sentiment analysis
def fetch_news(api_key, query, start_date, prediction_date):
    url = f'https://newsapi.org/v2/everything?q={query}&from={start_date}&to={prediction_date}&apiKey={api_key}'
    response = requests.get(url)
    news_data = response.json()
    return news_data['articles']

# Function to preprocess the text for analysis
def preprocess_text(text):

    lemmatizer = WordNetLemmatizer()  # Initialise the lemmatizer
    stop_words = set(stopwords.words('english'))  # Define the set of stopwords
    
    # Convert text to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Remove stopwords and lemmatize the remaining words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    
    # Join tokens back into a single string
    return ' '.join(tokens)

# Function to get sentiment analysis polarity
def get_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

# Function to get sentiment analysis ration
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

# Function to fetch data for Deep Learning Model
def fetch_data(ticker, start_date, prediction_date):
    stock_data = yf.download(tickers=ticker, start=start_date, end=prediction_date)
    stock_data.reset_index(inplace=True)
    return stock_data

# Function to prepare data for Deep Learning Model
def prepare_data(data, feature_cols, target_col):
    input_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_inputs = input_scaler.fit_transform(data[feature_cols])
    scaled_target = target_scaler.fit_transform(data[[target_col]])
    return scaled_inputs, scaled_target, input_scaler, target_scaler

# Function to calculate SMA
def calculate_sma(data, window=20):
    data[f'SMA_{window}'] = data['Close'].rolling(window=window).mean()
    return data

# Function to calculate EMA
def calculate_ema(data, window=20):
    data[f'EMA_{window}'] = data['Close'].ewm(span=window, adjust=False).mean()
    return data

# Function to calculate RSI
def calculate_rsi(data, period=14):
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))
    return data

# Function to calculate MACD
def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    data['EMA_short'] = data['Close'].ewm(span=short_window, adjust=False).mean()
    data['EMA_long'] = data['Close'].ewm(span=long_window, adjust=False).mean()
    data['MACD'] = data['EMA_short'] - data['EMA_long']
    data['MACD_signal'] = data['MACD'].ewm(span=signal_window, adjust=False).mean()
    data['MACD_histogram'] = data['MACD'] - data['MACD_signal']
    return data

# Function to create sequence
def create_sequences(inputs, targets, sequence_length):
    X, y = [], []
    for i in range(len(inputs) - sequence_length):
        X.append(inputs[i:i + sequence_length])
        y.append(targets[i + sequence_length])
    return np.array(X), np.array(y)

# Function to calculate Performance Metrics
def calculate_performance_metrics(actual, prediction):
    mae = mean_absolute_error(actual, prediction)
    mse = mean_squared_error(actual, prediction)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual, prediction)
    print(f'Mean Absolute Error (MAE): {mae}')
    print(f'Mean Squared Error (MSE): {mse}')
    print(f'Root Mean Squared Error (RMSE): {rmse}')
    print(f'R-squared (R2): {r2}')

# Function to plot Prediction Graph
def plot_prediction_graph(actual, prediction):
    plt.figure(figsize=(14, 7))
    plt.plot(actual, label='Actual Price')
    plt.plot(prediction, label='Predicted Price')
    plt.title('Actual vs Predicted Stock Prices')
    plt.xlabel('Sequence')
    plt.ylabel('Price')
    plt.legend()
    plt.grid()
    st.pyplot(plt)

# Function to predict next day close
def predict_next_day_close(data, model, input_scaler, target_scaler, sequence_length, features):
    last_sequence = data[-sequence_length:]
    last_sequence_scaled = input_scaler.transform(last_sequence[features])
    last_sequence_scaled = last_sequence_scaled.reshape((1, sequence_length, len(features)))
    predicted_close_scaled = model.predict(last_sequence_scaled)
    predicted_close = target_scaler.inverse_transform(predicted_close_scaled)
    return predicted_close[0][0]

# Function to build LSTM Model
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

# Streamlit Setup
# Layout
st.set_page_config(layout="wide")

# Center the header
st.markdown(
    """
    <style>
    .centered-header {
        text-align: center;
        font-size: 2em;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="centered-header">Keridion Financial Dashboard</div>', unsafe_allow_html=True)

# Sidebar and input elements
tickers = ['^GSPC']
st.sidebar.image("logo.jpg", use_column_width=True)
st.sidebar.header('Parameters')
start_date = "2001-09-06"
selected_ticker = '^GSPC'
end_date = st.sidebar.date_input("End Date for MRI Graphs", datetime.date.today())
user_ticker = '^GSPC'
prediction_date = st.sidebar.date_input("Date to predict the closing price of ^GSPC", datetime.date.today())

# Initialise session state if it doesn't exist
if 'fast_weight' not in st.session_state:
    st.session_state.fast_weight = 50

if 'vix_weight' not in st.session_state:
    st.session_state.vix_weight = 33
if 'move_weight' not in st.session_state:
    st.session_state.move_weight = 33
if 'evz_weight' not in st.session_state:
    st.session_state.evz_weight = 34

# Slider for both fast and slow weights
fast_weight = st.sidebar.slider(
    "Look Back Period", 
    min_value=0, 
    max_value=100, 
    value=st.session_state.fast_weight, 
    key='fast_weight'
)

# Calculate the slow_weight
slow_weight = 100 - fast_weight

# Update the session state
st.session_state.slow_weight = slow_weight

# Display the values
st.sidebar.write(f"Fast MRI Weight: {fast_weight}%")
st.sidebar.write(f"Slow MRI Weight: {slow_weight}%")

if st.sidebar.checkbox("Use VIX", value=True):
    tickers.append('^VIX')

if st.sidebar.checkbox("Use MOVE", value=True):
    tickers.append('^MOVE')

if st.sidebar.checkbox("Use EVZ", value=True):
    tickers.append('EVZ')

st.sidebar.write("Selected Tickers:")
for ticker in tickers:
    st.sidebar.write(ticker)

# Set the lookback period and slopes
lookback_periods_fast = [5, 10, 22, 66, 90]
lookback_periods_slow = [66, 126, 190, 252, 520]
slope_factor_fast = 10
slope_factor_slow = 22

# Generate the Dashboard
if st.sidebar.button("Generate Plots"):
    data = get_data(tickers,start_date, end_date)
    
    if tickers == ['^GSPC']:  # Only contains '^GSPC' and no other tickers
        st.sidebar.text("ERROR: No additional tickers selected!")
        st.stop()
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

    # Date Setup and API Key Configuration
    ticker = '^GSPC'
    today = dt.datetime.today()
    one_week_before = today - relativedelta(days=7)
    ten_year_before = today - relativedelta(years=10)
    stock_start_date = ten_year_before.strftime("%Y-%m-%d")
    news_start_date = one_week_before.strftime("%Y-%m-%d")
    end_date = today.strftime("%Y-%m-%d")
    api_key = '4fa92f53a2374022b620325ec7bebe6f'
    
    # Fetch and Process News Articles
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
    
    # Compute Average Sentiment by Date
    news_df['sentiment'] = pd.to_numeric(news_df['sentiment'], errors='coerce')  # Ensure sentiment is numeric
    news_df = news_df.groupby('date').agg({'sentiment': 'mean'}).reset_index()

    # Fetch Stock Data and Calculate Technical Indicators
    data = fetch_data(ticker, stock_start_date, prediction_date)
    data = calculate_sma(data)
    data = calculate_ema(data)
    data = calculate_rsi(data)
    data = calculate_macd(data)
    data.fillna(0, inplace=True)

    # Prepare Data for Model Training
    features = [i for i in data.columns if i not in ['Close', 'Date']]
    target = 'Close'
    sequence_length = 60

    input_features, input_target, input_scaler, target_scaler = prepare_data(data, features, target)

    X, y = create_sequences(input_features, input_target, sequence_length)

    # Split Data and Train the LSTM Model
    split = int(len(X)*0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = build_lstm_model((sequence_length, len(features)))
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), callbacks=[EarlyStopping(monitor='val_loss', patience=5)])

    # Evaluate Model and Forecast Next Day Close
    actual = target_scaler.inverse_transform(y_test)
    predictions = model.predict(X_test)
    predictions = target_scaler.inverse_transform(predictions)

    calculate_performance_metrics(actual, predictions)

    forecast = predict_next_day_close(data, model, input_scaler, target_scaler, sequence_length, features)

    # Display Sentiment Analysis Results
    sentiment, sentiment_percentage = get_sentiment_ratio(news_df)

    # Fetch and Display News Articles
    articles = fetch_news_articles(api_key, query)
    if articles:
        with col6:
            # Custom container to hold news articles
            st.markdown('<div class="custom-container">', unsafe_allow_html=True)
            display_news_articles(articles)
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.write("No articles found.")

    # Visualise and Present Prediction Results
    r2 = r2_score(actual, predictions)
    r2_p = r2*100

    today_value = data['Close'].tail(1).values[0]
    delta_value = (forecast - today_value)

    with col5:
        plot_prediction_graph(actual, predictions)
        st.metric(
            label=f"Predicted closing price of S&P 500 on {prediction_date}:",
            value=f"${forecast:.2f} ({r2_p:.2f}%)",
            delta=f"{delta_value:.2f}$ from Today's Value",
        )