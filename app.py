import streamlit as st
import yfinance as yf
import pandas as pd
from newsapi import NewsApiClient
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="AI Market Sentiment & Price Predictor", layout="wide")
st.title("AI Market Sentiment & Price Predictor")

# USER INPUT
ticker = st.selectbox(
    "Select a stock",
    ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META"]
)

# FETCH STOCK DATA 
data = yf.download(ticker, period="1y")
if data.empty:
    st.error("No data found for this ticker.")
    st.stop()

# Flatten multiindex columns if they exist
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

# HISTORICAL PRICE TABLE
st.subheader(f"{ticker} Historical Price Data (Last 10 Days)")
st.dataframe(data[['Open','High','Low','Close','Volume']].tail(10))

# INTERACTIVE PRICE & VOLUME CHART 
st.subheader("Price & Volume Chart")
chart_data = data[['Close','Volume']].copy()
chart_data['Date'] = chart_data.index
chart_data = chart_data.set_index('Date')
st.line_chart(chart_data)

#  NEWS SENTIMENT 
st.subheader("News Sentiment Analysis")

NEWS_API_KEY = "176819721aca4dcdbe4940d8fbe871f5"
newsapi = NewsApiClient(api_key=NEWS_API_KEY)

try:
    articles = newsapi.get_everything(
        q=ticker,
        language="en",
        sort_by="publishedAt",
        page_size=10
    )
except Exception as e:
    st.error(f"Error fetching news: {e}")
    st.stop()

analyzer = SentimentIntensityAnalyzer()
scores = []

for article in articles["articles"]:
    title = article["title"]
    score = analyzer.polarity_scores(title)["compound"]
    scores.append(score)

if scores:
    avg_score = sum(scores) / len(scores)
    if avg_score > 0.05:
        sentiment = "Positive 📈"
    elif avg_score < -0.05:
        sentiment = "Negative 📉"
    else:
        sentiment = "Neutral ➖"
    st.write(f"Today's Sentiment: **{sentiment}**")
else:
    avg_score = 0
    st.write("No news found for sentiment analysis.")

# SENTIMENT TABLE 
st.subheader("Recent News Sentiment Scores")
if scores:
    df_sentiment = pd.DataFrame({
        'Title':[article['title'] for article in articles['articles']],
        'Sentiment Score': scores
    })
    st.dataframe(df_sentiment)

# MACHINE LEARNING PREDICTION
st.subheader("Tomorrow Price Prediction")


data_ml = data.copy()
data_ml['Sentiment'] = avg_score 
data_ml['Prev_Close'] = data_ml['Close'].shift(1)
data_ml['Target'] = data_ml['Close'].shift(-1)
data_ml = data_ml.dropna()

X = data_ml[['Sentiment', 'Volume', 'Prev_Close']]
y = data_ml['Target']

model = LinearRegression()
model.fit(X, y)

# Predict for next day
last_row = X.iloc[-1].values.reshape(1, -1)
predicted_price = model.predict(last_row)[0]
st.write(f"Predicted Close Price: ${predicted_price:.2f}")