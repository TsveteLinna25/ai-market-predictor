# AI Market Sentiment & Price Predictor

An AI-powered financial dashboard that analyzes stock market sentiment from news and predicts future prices using machine learning.

Built with Python, Streamlit, NLP sentiment analysis, and financial market data.

---

## Features

- Historical stock price data
- Interactive price & volume charts
- News sentiment analysis using NLP
- Machine learning price prediction
- Financial dashboard built with Streamlit

---

## Technologies Used

- Python
- Streamlit
- yfinance API
- NewsAPI
- VADER Sentiment Analyzer
- Scikit-learn
- Pandas
- Machine Learning

---

## How It Works

1. The app fetches historical stock data using the **yfinance API**
2. Financial news is retrieved using **NewsAPI**
3. Headlines are analyzed using **VADER sentiment analysis**
4. Sentiment scores are combined with historical price data
5. A **Linear Regression model** predicts the next day's price

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/TsveteLinna25/ai-market-predictor.git
   cd ai-market-predictor
   

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Mac/Linux
   venv\Scripts\activate      # On Windows

3. Install dependencies:
   ```bash
   pip install -r requirements.txt

4. Run the application:
   ```bash
   streamlit run app.py

## Example Dashboard

- Historical price data
- Interactive price & volume charts
- News sentiment analysis
- Sentiment score table
- Machine learning price prediction

## Future Improvements

- LSTM deep learning price prediction model
- Cryptocurrency market sentiment analysis
- Real-time financial news monitoring
- Portfolio tracking dashboard
- Advanced technical indicators

## Author

Created by Tsvetelina 